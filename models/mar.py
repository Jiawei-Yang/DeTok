"""
Modified from https://github.com/LTH14/mar/blob/main/models/mar.py
    - add support for 1D sequence
    - include samplers inside the model
    - add support for removing dropout in MLPs
"""

import logging
import math
from functools import partial

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from .diffloss import DiffLoss
from .layers import Block

logger = logging.getLogger("DeTok")

MAR_SIZE_DICT = {
    "base": {"width": 768, "layers": 12, "heads": 12},
    "large": {"width": 1024, "layers": 16, "heads": 16},
    "huge": {"width": 1280, "layers": 20, "heads": 16},
}


def mask_by_order(mask_len, order, bsz, seq_len):
    """create masking tensor based on given order and length."""
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(
        masking,
        dim=-1,
        index=order[:, : mask_len.long()],
        src=torch.ones(bsz, seq_len).cuda(),
    ).bool()
    return masking


class MAR(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=1,
        model_size="base",
        tokenizer_patch_size=16,
        token_channels=16,
        mask_ratio_min=0.7,
        label_drop_prob=0.1,
        num_classes=1000,
        attn_dropout=0.1,
        proj_dropout=0.1,
        buffer_size=64,
        diffloss_d=3,
        diffloss_w=1024,
        num_sampling_steps="100",
        noise_schedule="cosine",
        diffusion_batch_mul=4,
        force_one_d_seq=0,
        grad_checkpointing=False,
        no_dropout_in_mlp=False,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.token_channels = token_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // tokenizer_patch_size // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = token_channels * patch_size**2
        self.grad_checkpointing = grad_checkpointing
        self.model_size = model_size
        self.force_one_d_seq = force_one_d_seq
        if force_one_d_seq:
            self.seq_len = force_one_d_seq

        size_dict = MAR_SIZE_DICT[self.model_size]
        num_layers, num_heads, width = size_dict["layers"], size_dict["heads"], size_dict["width"]

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = num_classes
        self.class_emb = nn.Embedding(num_classes, width)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, width))

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, width, bias=True)
        self.z_proj_ln = nn.LayerNorm(width, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, width))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.encoder_blocks = nn.ModuleList(
            [
                Block(
                    width,
                    num_heads,
                    norm_layer=norm_layer,
                    qkv_bias=True,
                    proj_drop=proj_dropout,
                    attn_drop=attn_dropout,
                    no_dropout_in_mlp=no_dropout_in_mlp,
                )
                for _ in range(num_layers)
            ]
        )
        self.encoder_norm = norm_layer(width)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.decoder_embed = nn.Linear(width, width, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, width))

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    width,
                    num_heads,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_dropout,
                    attn_drop=attn_dropout,
                    no_dropout_in_mlp=no_dropout_in_mlp,
                )
                for _ in range(num_layers)
            ]
        )

        self.decoder_norm = norm_layer(width)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, width))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=width,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            noise_schedule=noise_schedule,
            grad_checkpointing=grad_checkpointing,
        )
        self.diffusion_batch_mul = diffusion_batch_mul

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[MAR] params: {params_M:.2f}M, {model_size}-{num_layers}-{width}")
        logger.info(f"[MAR] seq_len: {self.seq_len}, buffer_size: {self.buffer_size}")

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=0.02)
        torch.nn.init.normal_(self.fake_latent, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=0.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=0.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum("nchpwq->nhwcpq", x)
        x = x.reshape(bsz, h_ * w_, c * p**2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.token_channels
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum("nhwcpq->nchpwq", x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, _ = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(
            mask,
            dim=-1,
            index=orders[:, :num_masked_tokens],
            src=torch.ones(bsz, seq_len, device=x.device),
        )
        return mask

    def forward_mae_encoder(self, x, mask, class_embedding):
        x = self.z_proj(x)
        bsz, _, embed_dim = x.shape

        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, : self.buffer_size] = class_embedding.unsqueeze(1)

        # encoder position embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # dropping
        x = x[(1 - mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # apply Transformer blocks
        if self.grad_checkpointing and self.training:
            for i, block in enumerate(self.encoder_blocks):
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)
        return x

    def forward_mae_decoder(self, x, mask):

        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # pad mask tokens
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(
            x.dtype
        )
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(
            x.shape[0] * x.shape[1], x.shape[2]
        )

        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        if self.grad_checkpointing and self.training:
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)

        x = x[:, self.buffer_size :]
        x = x + self.diffusion_pos_embed_learned
        return x

    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz * seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def forward(self, imgs, labels):

        # class embed
        class_embedding = self.class_emb(labels)

        # patchify and mask (drop) tokens
        x = self.patchify(imgs) if not self.force_one_d_seq else imgs
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)

        x = self.forward_mae_encoder(x, mask, class_embedding)
        z = self.forward_mae_decoder(x, mask)
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)
        return loss

    def sample_tokens(
        self,
        bsz,
        num_iter=64,
        cfg=1.0,
        cfg_schedule="linear",
        labels=None,
        temperature=1.0,
        progress=False,
    ):

        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()

            # class embedding and CFG
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            if cfg != 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            # mae encoder
            x = self.forward_mae_encoder(tokens, mask, class_embedding)

            # mae decoder
            z = self.forward_mae_decoder(x, mask)

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(
                torch.Tensor([1]).cuda(),
                torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len),
            )

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if cfg != 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            if cfg != 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        # unpatchify
        if not self.force_one_d_seq:
            tokens = self.unpatchify(tokens)
        return tokens

    @torch.inference_mode()
    def generate(self, n_samples, cfg, labels, args):
        return self.sample_tokens(
            n_samples,
            num_iter=args.num_iter,
            cfg=cfg,
            labels=labels,
            cfg_schedule=args.cfg_schedule,
            temperature=args.temperature,
            progress=True,
        )


def mar_base(**kwargs) -> MAR:
    return MAR(model_size="base", **kwargs)


def mar_large(**kwargs):
    return MAR(model_size="large", **kwargs)


def mar_huge(**kwargs):
    return MAR(model_size="huge", **kwargs)


MAR_models = {
    "MAR_base": mar_base,
    "MAR_large": mar_large,
    "MAR_huge": mar_huge,
}
