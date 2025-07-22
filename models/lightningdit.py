"""
Modified from https://github.com/hustvl/LightningDiT/blob/main/models/lightningdit.py

    - add support for 1D sequence
    - include samplers inside the model
    - slightly different cfg conditioning (conditioned on **all channels**)
"""

import logging
from functools import partial

import torch
import torch.nn as nn

from transport import Sampler, create_transport

from .layers import (
    Block,
    LabelEmbedder,
    ModulatedLinear,
    PatchEmbed,
    TimestepEmbedder,
    Transformer,
    VisionRotaryEmbeddingFast,
    get_2d_sincos_pos_embed,
)
from .model_utils import SIZE_DICT

logger = logging.getLogger("DeTok")


class LightningDiT(nn.Module):
    """lightning diffusion transformer model."""

    def __init__(
        self,
        img_size=256,
        patch_size=1,
        model_size="base",
        tokenizer_patch_size=16,
        token_channels=16,
        label_drop_prob=0.1,
        num_classes=1000,
        num_sampling_steps=250,
        sampling_method="euler",
        grad_checkpointing=False,
        force_one_d_seq=0,
        learn_sigma=False,  # no learn_sigma in SiT
        legacy_mode=False,
        qk_norm=False,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # basic configuration
        self.token_channels = self.out_channels = token_channels
        self.input_size = img_size // tokenizer_patch_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.force_one_d_seq = force_one_d_seq
        self.grad_checkpointing = grad_checkpointing
        self.learn_sigma = learn_sigma
        self.legacy_mode = legacy_mode

        # model architecture configuration
        size_dict = SIZE_DICT[model_size]
        num_layers, num_heads, width = size_dict["layers"], size_dict["heads"], size_dict["width"]

        # --------------------------------------------------------------------------
        # embedding layers
        if self.force_one_d_seq > 0:
            self.x_embedder = nn.Linear(token_channels, width)
            # we use learnable positional embeddings for 1D sequence without rope
            self.pos_embed = nn.Parameter(torch.randn(1, self.force_one_d_seq, width) * 0.02)
            self.seq_len = self.force_one_d_seq
        else:
            self.x_embedder = PatchEmbed(self.input_size, patch_size, token_channels, width)
            # use rotary position encoding + abe, borrow from EVA
            num_patches = self.x_embedder.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, width))
            self.rope = VisionRotaryEmbeddingFast(width // num_heads // 2, self.input_size // patch_size)
            self.seq_len = num_patches

        self.t_embedder = TimestepEmbedder(width)
        self.y_embedder = LabelEmbedder(num_classes, width, label_drop_prob)

        # --------------------------------------------------------------------------
        # transformer architecture
        self.transformer = Transformer(
            width,
            num_layers,
            num_heads,
            block_fn=partial(Block, use_modulation=True),
            norm_layer=nn.RMSNorm,
            grad_checkpointing=grad_checkpointing,
            use_swiglu=True,
            qk_norm=qk_norm,
        )
        self.final_layer = ModulatedLinear(width, patch_size**2 * token_channels, use_rmsnorm=True)

        # --------------------------------------------------------------------------
        # transport and sampling setup
        self.transport = create_transport(use_cosine_loss=True, use_lognorm=True)
        self.sampler = Sampler(self.transport)
        self.sample_fn = self.sampler.sample_ode(
            sampling_method=sampling_method,
            num_steps=int(num_sampling_steps),
            timestep_shift=0.3,
        )

        self.initialize_weights()

        # log model info
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(
            f"[LightningDiT] params: {num_trainable_params:.2f}M size: {model_size}, num_layers: {num_layers}, width: {width}"
        )

    def initialize_weights(self):
        """initialize model weights."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # initialize (and freeze) pos_embed by sin-cos embedding
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

        # zero-out adaLN modulation layers in LightningDiT blocks
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

    def net(self, x, t=None, y=None):
        """core network forward pass."""
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        c = self.t_embedder(t) + self.y_embedder(y, self.training)  # (N, D)

        # check if self.pos_embed requires grad
        if not self.force_one_d_seq:
            x = self.transformer(x, condition=c, rope=self.rope)  # (N, T, D)
        else:
            x = self.transformer(x, condition=c)

        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        if not self.force_one_d_seq:
            x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=None, cfg_interval_start=None):
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

        if cfg_interval is True:
            timestep = t[0]
            if timestep < cfg_interval_start:
                half_eps = cond_eps

        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def forward(self, x, y):
        """forward pass for training."""
        loss_dict = self.transport.training_losses(self.net, x, dict(y=y))
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
            y_null = torch.tensor([self.num_classes] * n_samples, device=device)
            labels = torch.cat([labels, y_null], 0)
            model_kwargs = dict(y=labels, cfg_scale=cfg, cfg_interval=True, cfg_interval_start=0.10)
            model_fn = self.forward_with_cfg
        else:
            model_kwargs = dict(y=labels)
            model_fn = self.net

        # generate samples
        samples = self.sample_fn(z, model_fn, **model_kwargs)[-1]
        if cfg > 1.0:
            samples, _ = samples.chunk(2, dim=0)  # remove null class samples
        return samples


# model size variants
def LightningDiT_base(**kwargs) -> LightningDiT:
    return LightningDiT(model_size="base", **kwargs)


def LightningDiT_large(**kwargs) -> LightningDiT:
    return LightningDiT(model_size="large", **kwargs)


def LightningDiT_xl(**kwargs) -> LightningDiT:
    return LightningDiT(model_size="xl", **kwargs)


def LightningDiT_huge(**kwargs) -> LightningDiT:
    return LightningDiT(model_size="huge", **kwargs)


LightningDiT_models = {
    "LightningDiT_base": LightningDiT_base,
    "LightningDiT_large": LightningDiT_large,
    "LightningDiT_xl": LightningDiT_xl,
    "LightningDiT_huge": LightningDiT_huge,
}
