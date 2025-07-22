import logging
from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from .diffloss import DiffLoss
from .layers import Block, Transformer, modulate
from .model_utils import SIZE_DICT

logger = logging.getLogger("DeTok")


class FinalLayer(nn.Module):
    """final layer with adaptive layer normalization."""

    def __init__(self, in_features) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(in_features, 2 * in_features))

    def forward(self, x, condition):
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return x


class ARDiff(nn.Module):
    """decoder-only autoregressive diffusion model."""

    def __init__(
        self,
        img_size=256,
        patch_size=1,
        model_size="base",
        tokenizer_patch_size=16,
        token_channels=16,
        label_drop_prob=0.1,
        num_classes=1000,
        # diffloss parameters
        noise_schedule="cosine",
        diffloss_d=3,
        diffloss_w=1024,
        diffusion_batch_mul=4,
        # sampling parameters
        num_sampling_steps=100,
        grad_checkpointing=False,
        force_one_d_seq=False,
        order="raster",
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # basic configuration
        self.img_size = img_size
        self.patch_size = patch_size
        self.token_channels = token_channels
        self.num_classes = num_classes
        self.label_drop_prob = label_drop_prob
        self.grad_checkpointing = grad_checkpointing
        self.force_one_d_seq = force_one_d_seq
        self.order = order
        self.diffusion_batch_mul = diffusion_batch_mul

        # sequence dimensions
        self.seq_h = self.seq_w = img_size // tokenizer_patch_size // patch_size
        self.seq_len = self.seq_h * self.seq_w + 1  # +1 for BOS token
        self.token_embed_dim = token_channels * patch_size**2

        if force_one_d_seq:
            self.seq_len = force_one_d_seq + 1

        # model architecture configuration
        size_dict = SIZE_DICT[model_size]
        num_layers, num_heads, width = size_dict["layers"], size_dict["heads"], size_dict["width"]

        self.label_drop_prob = label_drop_prob

        scale = width**-0.5

        # class and null token embeddings
        self.class_emb = nn.Embedding(self.num_classes, width)
        self.fake_latent = nn.Parameter(scale * torch.randn(1, width))
        self.bos_token = nn.Parameter(torch.zeros(1, 1, width))

        # input and positional embeddings
        self.x_embedder = nn.Linear(self.token_embed_dim, width)
        self.pos_embed = nn.Parameter(scale * torch.randn((1, self.seq_len, width)))
        self.target_pos_embed = nn.Parameter(scale * torch.randn((1, self.seq_len - 1, width)))
        self.timesteps_embeddings = nn.Parameter(scale * torch.randn((1, self.seq_len, width)))

        # training mask for causal attention
        self.train_mask = torch.tril(torch.ones(self.seq_len, self.seq_len, dtype=torch.bool)).cuda()

        # --------------------------------------------------------------------------
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.ln_pre = norm_layer(width)
        self.transformer = Transformer(
            width,
            num_layers,
            num_heads,
            block_fn=partial(Block, use_modulation=True),
            norm_layer=norm_layer,
            force_causal=True,
            grad_checkpointing=self.grad_checkpointing,
        )
        self.final_layer = FinalLayer(width)
        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=width,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing,
            noise_schedule=noise_schedule,
        )
        self.diffusion_batch_mul = diffusion_batch_mul
        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[ARDiff] params: {params_M:.2f}M, {model_size}-{num_layers}-{width}")

    def initialize_weights(self):
        """initialize model weights."""
        # parameter initialization
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        torch.nn.init.normal_(self.bos_token, std=0.02)
        torch.nn.init.normal_(self.target_pos_embed, std=0.02)
        torch.nn.init.normal_(self.timesteps_embeddings, std=0.02)
        torch.nn.init.normal_(self.class_emb.weight, std=0.02)
        torch.nn.init.normal_(self.fake_latent, std=0.02)

        # apply standard initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """standard weight initialization for layers."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

        # zero-out adaptive modulation layers
        for block in self.transformer.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # zero-out final layer modulation
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    def patchify(self, x):
        """convert image tensor to patch tokens."""
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum("nchpwq->nhwcpq", x)
        x = x.reshape(bsz, h_ * w_, c * p**2)
        return x  # [batch, seq_len, token_dim]

    def unpatchify(self, x):
        """convert patch tokens back to image tensor."""
        bsz = x.shape[0]
        p = self.patch_size
        c = self.token_channels
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum("nhwcpq->nchpwq", x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [batch, channels, height, width]

    def enable_kv_cache(self):
        for block in self.transformer.blocks:
            block.attn.kv_cache = True
            block.attn.reset_kv_cache()
        logger.info("Enable kv_cache for Transformer blocks")

    def disable_kv_cache(self):
        for block in self.transformer.blocks:
            block.attn.kv_cache = False
            block.attn.reset_kv_cache()
        logger.info("Disable kv_cache for Transformer blocks")

    def get_random_orders(self, x):
        """generate random token ordering."""
        batch_size = x.shape[0]
        random_noise = torch.randn(batch_size, self.seq_len - 1, device=x.device)
        shuffled_orders = torch.argsort(random_noise, dim=1)
        return shuffled_orders

    def get_raster_orders(self, x):
        """generate raster (sequential) token ordering."""
        batch_size = x.shape[0]
        raster_orders = torch.arange(self.seq_len - 1, device=x.device)
        shuffled_orders = torch.stack([raster_orders for _ in range(batch_size)])
        return shuffled_orders

    def shuffle(self, x, orders):
        """shuffle tokens according to given orders."""
        batch_size, seq_len = x.shape[:2]
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
        shuffled_x = x[batch_indices, orders]
        return shuffled_x

    def unshuffle(self, shuffled_x, orders):
        """unshuffle tokens to restore original ordering."""
        batch_size, seq_len = shuffled_x.shape[:2]
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
        unshuffled_x = torch.zeros_like(shuffled_x)
        unshuffled_x[batch_indices, orders] = shuffled_x
        return unshuffled_x

    def forward_transformer(self, x, class_embedding, orders=None):
        """forward pass through the transformer."""
        x = self.x_embedder(x)
        bsz = x.shape[0]

        # add BOS token
        bos_token = self.bos_token.expand(bsz, 1, -1)
        x = torch.cat([bos_token, x], dim=1)
        current_seq_len = x.shape[1]

        # add positional embeddings
        pos_embed = self.pos_embed.expand(bsz, -1, -1)
        if orders is not None:
            pos_embed = torch.cat([pos_embed[:, :1], self.shuffle(pos_embed[:, 1:], orders)], dim=1)
        x = x + pos_embed[:, :current_seq_len]

        # add target positional embeddings
        target_pos_embed = self.target_pos_embed.expand(bsz, -1, -1)
        embed_dim = target_pos_embed.shape[-1]
        if orders is not None:
            target_pos_embed = self.shuffle(target_pos_embed, orders)
        target_pos_embed = torch.cat([target_pos_embed, torch.zeros(bsz, 1, embed_dim).to(x.device)], dim=1)
        x = x + target_pos_embed[:, :current_seq_len]

        x = self.ln_pre(x)

        # prepare condition tokens
        condition_token = class_embedding.repeat(1, current_seq_len, 1)
        timestep_embed = self.timesteps_embeddings.expand(bsz, -1, -1)
        condition_token = condition_token + timestep_embed[:, :current_seq_len]

        # handle kv cache for inference
        if self.transformer.blocks[0].attn.kv_cache:
            x = x[:, -1:]
            condition_token = condition_token[:, -1:]

        # transformer forward pass
        for block in self.transformer.blocks:
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, None, None, condition_token)
            else:
                x = block(x, condition=condition_token)

        x = self.final_layer(x, condition=class_embedding)
        return x

    def forward_loss(self, z, target):
        """compute diffusion loss."""
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        return self.diffloss(z=z, target=target)

    def forward(self, x, labels):
        """forward pass for training."""
        # get token ordering
        if self.order == "raster":
            orders = self.get_raster_orders(x)
        elif self.order == "random":
            orders = self.get_random_orders(x)
        else:
            raise NotImplementedError(f"Order '{self.order}' not implemented")

        # prepare class embeddings
        class_embedding = self.class_emb(labels)
        if self.training:
            # randomly drop class embedding during training
            drop_mask = torch.rand(x.shape[0]) < self.label_drop_prob
            drop_mask = drop_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_mask * self.fake_latent + (1 - drop_mask) * class_embedding
        class_embedding = class_embedding.unsqueeze(1)

        # prepare input tokens
        x = self.patchify(x) if not self.force_one_d_seq else x
        x = self.shuffle(x, orders)
        gt_latents = x.clone().detach()

        # forward pass and loss computation
        z = self.forward_transformer(x[:, :-1], class_embedding, orders=orders)
        return self.forward_loss(z=z, target=gt_latents)

    def sample_tokens(
        self,
        bsz,
        cfg=1.0,
        cfg_schedule="linear",
        labels=None,
        temperature=1.0,
        progress=False,
        kv_cache=False,
    ):
        """sample tokens autoregressively."""
        tokens = torch.zeros(bsz, 0, self.token_embed_dim).cuda()
        indices = list(range(self.seq_len - 1))

        # setup kv cache if requested
        if kv_cache:
            self.enable_kv_cache()

        if progress:
            indices = tqdm(indices)

        # get token ordering
        if self.order == "raster":
            orders = self.get_raster_orders(torch.zeros(bsz, self.seq_len - 1, self.token_embed_dim).cuda())
        elif self.order == "random":
            orders = self.get_random_orders(torch.zeros(bsz, self.seq_len - 1, self.token_embed_dim).cuda())
        else:
            raise NotImplementedError(f"Order '{self.order}' not implemented")

        # prepare for classifier-free guidance
        if cfg != 1.0:
            orders = torch.cat([orders, orders], dim=0)

        # generate tokens step by step
        for step in indices:
            cur_tokens = tokens.clone()

            # prepare class embeddings and CFG
            cls_embd = self.fake_latent.repeat(bsz, 1) if labels is None else self.class_emb(labels)

            if cfg != 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                cls_embd = torch.cat([cls_embd, self.fake_latent.repeat(bsz, 1)], dim=0)
            cls_embd = cls_embd.unsqueeze(1)
            z = self.forward_transformer(tokens, cls_embd, orders=orders)[:, -1]

            # apply CFG schedule
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * step / len(indices)
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError(f"CFG schedule '{cfg_schedule}' not implemented")

            # sample next token
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)

            if cfg != 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)

            cur_tokens = torch.cat([cur_tokens, sampled_token_latent.unsqueeze(1)], dim=1)
            tokens = cur_tokens.clone()

        # cleanup
        if kv_cache:
            self.disable_kv_cache()

        if cfg != 1.0:
            orders, _ = orders.chunk(2, dim=0)

        # restore original ordering and convert back to image format
        tokens = self.unshuffle(tokens, orders)
        if not self.force_one_d_seq:
            tokens = self.unpatchify(tokens)

        return tokens

    def generate(self, n_samples, cfg, labels, args):
        """generate samples using the model."""
        return self.sample_tokens(
            n_samples,
            cfg=cfg,
            labels=labels,
            cfg_schedule=args.cfg_schedule,
            temperature=args.temperature,
            progress=True,
            kv_cache=False,
        )


# model size variants
def ARDiff_base(**kwargs):
    return ARDiff(model_size="base", **kwargs)


def ARDiff_large(**kwargs):
    return ARDiff(model_size="large", **kwargs)


def ARDiff_xl(**kwargs):
    return ARDiff(model_size="xl", **kwargs)


def ARDiff_huge(**kwargs):
    return ARDiff(model_size="huge", **kwargs)


ARDiff_models = {
    "ARDiff_base": ARDiff_base,
    "ARDiff_large": ARDiff_large,
    "ARDiff_huge": ARDiff_huge,
    "ARDiff_xl": ARDiff_xl,
}
