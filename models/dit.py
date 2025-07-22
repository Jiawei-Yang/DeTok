"""
Modified from https://github.com/facebookresearch/DiT/blob/main/models.py
    - add support for 1D sequence
    - include samplers inside the model
"""

import logging
from functools import partial

import torch
import torch.nn as nn

from diffusion import create_diffusion

from .layers import (
    Block,
    LabelEmbedder,
    ModulatedLinear,
    PatchEmbed,
    TimestepEmbedder,
    Transformer,
    get_2d_sincos_pos_embed,
)
from .model_utils import SIZE_DICT

logger = logging.getLogger("DeTok")


class DiT(nn.Module):
    """diffusion model with a transformer backbone."""

    def __init__(
        self,
        img_size=256,
        patch_size=1,
        model_size="base",
        tokenizer_patch_size=16,
        token_channels=16,
        label_drop_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        noise_schedule="linear",
        num_sampling_steps=250,
        grad_checkpointing=False,
        force_one_d_seq=0,
        legacy_mode=False,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # basic configuration
        self.learn_sigma = learn_sigma
        self.token_channels = token_channels
        self.out_channels = token_channels * 2 if learn_sigma else token_channels
        self.input_size = img_size // tokenizer_patch_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.force_one_d_seq = force_one_d_seq
        self.grad_checkpointing = grad_checkpointing
        self.legacy_mode = legacy_mode

        # model architecture configuration
        size_dict = SIZE_DICT[model_size]
        num_layers, num_heads, width = size_dict["layers"], size_dict["heads"], size_dict["width"]

        # --------------------------------------------------------------------------
        # embedding layers
        if self.force_one_d_seq:
            self.x_embedder = nn.Linear(token_channels, width)
            self.pos_embed = nn.Parameter(torch.randn(1, self.force_one_d_seq, width) * 0.02)
        else:
            self.x_embedder = PatchEmbed(self.input_size, patch_size, token_channels, width)
            num_patches = self.x_embedder.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, width), requires_grad=False)

        self.t_embedder = TimestepEmbedder(width)
        self.y_embedder = LabelEmbedder(num_classes, width, label_drop_prob)

        # --------------------------------------------------------------------------
        # transformer architecture
        self.transformer = Transformer(
            width,
            num_layers,
            num_heads,
            block_fn=partial(Block, use_modulation=True),
            norm_layer=partial(nn.LayerNorm, elementwise_affine=False, eps=1e-6),
            grad_checkpointing=grad_checkpointing,
        )
        self.final_layer = ModulatedLinear(width, patch_size * patch_size * self.out_channels)

        # --------------------------------------------------------------------------
        # diffusion setup
        self.train_diffusion = create_diffusion("", noise_schedule=noise_schedule)
        self.gen_diffusion = create_diffusion(num_sampling_steps, noise_schedule=noise_schedule)
        self.initialize_weights()

        # log model info
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(
            f"[DiT] params: {num_trainable_params:.2f}M size: {model_size}, num_layers: {num_layers}, width: {width}"
        )

    def initialize_weights(self):
        """initialize model weights."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if hasattr(module, "bias"):
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        """initialize (and freeze) pos_embed by sin-cos embedding"""
        if not self.force_one_d_seq:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        # initialize label embedding table
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # zero-out adaLN modulation layers in DiT blocks
        for block in self.transformer.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """convert patch tokens back to image tensor."""
        c, p = self.out_channels, self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def net(self, x, t, y):
        """core network forward pass."""
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        c = self.t_embedder(t) + self.y_embedder(y, self.training)  # (N, D)
        x = self.transformer(x, condition=c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        if not self.force_one_d_seq:
            x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """forward pass with classifier-free guidance."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.net(combined, t, y)

        if self.legacy_mode:
            eps, rest = model_out[:, :3], model_out[:, 3:]
        else:
            eps, rest = model_out[:, : self.token_channels], model_out[:, self.token_channels :]

        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def forward(self, x, y):
        """forward pass for training."""
        t = torch.randint(0, self.train_diffusion.num_timesteps, (x.shape[0],), device=x.device)
        loss_dict = self.train_diffusion.training_losses(self.net, x, t, dict(y=y))
        return loss_dict["loss"].mean()

    @torch.inference_mode()
    def generate(self, n_samples, labels, cfg=1.0, args=None):
        """generate samples using the model."""
        device = labels.device

        # prepare noise tensor
        if self.force_one_d_seq:
            z = torch.randn(n_samples, self.force_one_d_seq, self.token_channels)
        else:
            z = torch.randn(n_samples, self.token_channels, self.input_size, self.input_size)
        z = z.to(device)

        # setup classifier-free guidance
        if cfg > 1.0:
            z = torch.cat([z, z], 0)
            labels = torch.cat([labels, torch.full_like(labels, self.num_classes)], 0)
            model_kwargs = dict(y=labels, cfg_scale=cfg)
            sample_fn = self.forward_with_cfg
        else:
            model_kwargs = dict(y=labels)
            sample_fn = self.net

        # generate samples
        samples = self.gen_diffusion.p_sample_loop(
            sample_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
        )

        if cfg > 1.0:
            samples, _ = samples.chunk(2, dim=0)  # remove null class samples
        return samples


# model size variants
def DiT_base(**kwargs):
    return DiT(model_size="base", **kwargs)


def DiT_large(**kwargs):
    return DiT(model_size="large", **kwargs)


def DiT_xl(**kwargs):
    return DiT(model_size="xl", **kwargs)


def DiT_huge(**kwargs):
    return DiT(model_size="huge", **kwargs)


DiT_models = {"DiT_base": DiT_base, "DiT_large": DiT_large, "DiT_xl": DiT_xl, "DiT_huge": DiT_huge}
