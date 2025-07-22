import logging
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor

from models.model_utils import SIZE_DICT

from .autoencoder import DiagonalGaussianDistribution

logger = logging.getLogger("DeTok")


# ================================
# Utility Functions
# ================================

def _to_tensor(x):
    return x.clone().detach() if isinstance(x, torch.Tensor) else torch.tensor(x)

def rotate_half(x: Tensor) -> Tensor:
    """rotate half of the input tensor for rotary position embedding."""
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """apply rotary position embedding to input tensor."""
    freqs_cos, freqs_sin = freqs_cis.unsqueeze(1).chunk(2, dim=-1)
    return x * freqs_cos + rotate_half(x) * freqs_sin


def get_rope_tensor(
    dim: int, seq_h: int, seq_w: int, max_freq: float = 7.0, min_freq: float = 7e-4
) -> Tensor:
    """generate rotary position embedding tensor for 2D sequences."""
    freqs_1d = max_freq * (max_freq / min_freq) ** torch.linspace(0, -1, dim // 4)
    freqs_1d = torch.cat([freqs_1d, freqs_1d])
    freqs_2d = torch.zeros(2, dim)
    freqs_2d[0, : dim // 2] = freqs_1d
    freqs_2d[1, -dim // 2 :] = freqs_1d
    freqs_2d = freqs_2d * 2 * torch.pi
    coord_x = torch.linspace(0, 1, seq_h)
    coord_y = torch.linspace(0, 1, seq_w)
    coords_all = torch.cartesian_prod(coord_x, coord_y)
    angle = coords_all @ freqs_2d
    rope_tensor = torch.cat([angle.cos(), angle.sin()], dim=-1)
    return rope_tensor


# ================================
# Neural Network Components
# ================================


class SwiGLUFFN(nn.Module):
    """Swish-Gated Linear Unit Feed-Forward Network."""

    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class Attention(nn.Module):
    """multi-head attention with rotary position embedding."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim % num_heads !=0, got {dim} and {num_heads}"
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, rope: Tensor) -> Tensor:
        bsz, n_ctx, ch = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.num_heads).unbind(0)
        q, k = apply_rotary_emb(q, rope), apply_rotary_emb(k, rope)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(bsz, n_ctx, ch))


class Block(nn.Module):
    """transformer block with attention and feed-forward layers."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        norm_layer=partial(nn.RMSNorm, eps=1e-6),
    ) -> None:
        super().__init__()
        self.norm1, self.norm2 = norm_layer(dim), norm_layer(dim)
        self.attn = Attention(dim, num_heads)
        self.mlp = SwiGLUFFN(dim, int(2 / 3 * dim * mlp_ratio))

    def forward(self, x: Tensor, rope: Tensor = None) -> Tensor:
        x = x + self.attn(self.norm1(x), rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x


# ================================
# Encoder and Decoder
# ================================


class Encoder(nn.Module):
    """vision Transformer encoder with masked autoencoding capability."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        model_size: str = "base",
        token_channels: int = 16,
        mask_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = self.img_size // self.patch_size
        self.model_size = model_size
        # needs to split into mean and std
        self.token_channels = token_channels * 2
        self.mask_ratio = mask_ratio
        self.seq_len = self.grid_size**2

        size_dict = SIZE_DICT[self.model_size]
        num_layers, num_heads, width = size_dict["layers"], size_dict["heads"], size_dict["width"]
        self.width = width

        # patch embedding layer
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, width, self.patch_size, self.patch_size),
            Rearrange("b c h w -> b (h w) c", h=self.grid_size, w=self.grid_size),
        )

        # learnable embeddings
        scale = width**-0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(1, self.seq_len, width))

        # transformer layers
        norm_layer = partial(nn.RMSNorm, eps=1e-6)
        self.ln_pre = norm_layer(width)
        self.transformer = nn.ModuleList(
            [Block(dim=width, num_heads=num_heads, norm_layer=norm_layer) for _ in range(num_layers)]
        )
        self.ln_post = norm_layer(width)
        self.latent_head = nn.Linear(width, self.token_channels)

        # rotary position embedding
        head_dim = self.transformer[0].attn.head_dim
        rope_tensor = get_rope_tensor(head_dim, self.grid_size, self.grid_size).unsqueeze(0)
        self.register_buffer("rope_tensor", rope_tensor, persistent=False)

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[DeTok-Encoder] params: {params_M:.2f}M, {model_size}-{num_layers}-{width}")

    def unpatchify(self, x: Tensor, chans: int, patch_size: int) -> Tensor:
        """convert patches back to image format."""
        bsz = x.shape[0]
        h_ = w_ = self.grid_size
        x = x.reshape(bsz, h_, w_, chans, patch_size, patch_size)
        x = torch.einsum("nhwcpq->nchpwq", x)
        x = x.reshape(bsz, chans, h_ * patch_size, w_ * patch_size)
        return x

    def mae_random_masking(self, x: Tensor, mask_ratio: float = -1):
        """apply masked autoencoding random masking."""
        bsz, seq_len, chans = x.shape
        # mask: 0 for visible, 1 for masked
        if mask_ratio == 0:
            # no masking
            rope = self.rope_tensor.expand(bsz, -1, -1)
            return x, torch.zeros(bsz, seq_len, device=x.device), None, rope

        if mask_ratio < 0:
            mask_ratio = max(0.0, random.uniform(-0.1, self.mask_ratio))

        len_keep = int(np.ceil(seq_len * (1 - mask_ratio)))
        noise = torch.rand(bsz, seq_len, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(x, 1, ids_keep[..., None].repeat(1, 1, chans))
        rope = self.rope_tensor.expand(bsz, -1, -1)
        rope_visible = torch.gather(rope, 1, ids_keep[..., None].repeat(1, 1, rope.shape[-1]))

        mask = torch.ones(bsz, seq_len, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_visible, mask, ids_restore, rope_visible

    def forward(self, x: Tensor, mask_ratio: float = -1):
        """forward pass through encoder."""
        x = self.patch_embed(x) + self.positional_embedding
        x, _, ids_restore, rope = self.mae_random_masking(x, mask_ratio=mask_ratio)

        x = self.ln_pre(x)
        for block in self.transformer:
            x = block(x, rope)
        x = self.ln_post(x)

        tokens = self.latent_head(x)

        return tokens, ids_restore


class Decoder(nn.Module):
    """vision Transformer decoder with mask tokens for image reconstruction."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        model_size: str = "base",
        token_channels: int = 16,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = self.img_size // self.patch_size
        self.model_size = model_size
        self.token_channels = token_channels
        self.seq_len = self.grid_size**2

        params = SIZE_DICT[self.model_size]
        num_layers, num_heads, width = params["layers"], params["heads"], params["width"]

        # learnable embeddings
        scale = width**-0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(1, self.seq_len, width))
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, width))

        # decoder layers
        self.decoder_embed = nn.Linear(self.token_channels, width)
        norm_layer = partial(nn.RMSNorm, eps=1e-6)
        self.ln_pre = norm_layer(width)
        self.transformer = nn.ModuleList(
            [Block(dim=width, num_heads=num_heads, norm_layer=norm_layer) for _ in range(num_layers)]
        )
        self.ln_post = norm_layer(width)

        # output layers
        self.ffn = nn.Sequential(
            Rearrange("b (h w) c -> b c h w", h=self.grid_size, w=self.grid_size),
            nn.Conv2d(width, self.patch_size * self.patch_size * 3, 1, padding=0),
            Rearrange("b (p1 p2 c) h w -> b c (h p1) (w p2)", p1=self.patch_size, p2=self.patch_size),
        )
        self.conv_out = nn.Conv2d(3, 3, 3, padding=1)

        # rotary position embedding
        head_dim = self.transformer[0].attn.head_dim
        rope_tensor = get_rope_tensor(head_dim, self.grid_size, self.grid_size).unsqueeze(0)
        self.register_buffer("rope_tensor", rope_tensor, persistent=False)

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[DeTok-Decoder] params: {params_M:.2f}M, {model_size}-{num_layers}-{width}")

    def forward(self, z_latents: Tensor, ids_restore: Tensor | None = None) -> Tensor:
        """forward pass through decoder."""
        z = self.decoder_embed(z_latents)
        bsz, seq_len, _ = z.shape

        if ids_restore is not None:
            num_mask_tokens = ids_restore.shape[1] + 1 - seq_len
            mask_tokens = self.mask_token.repeat(bsz, num_mask_tokens, 1)
            z_ = torch.cat([z, mask_tokens], dim=1)
            expanded_ids_restore = ids_restore.unsqueeze(-1).expand(-1, -1, z_.shape[-1])
            z = torch.gather(z_, dim=1, index=expanded_ids_restore)

        z = z + self.positional_embedding

        z = self.ln_pre(z)
        rope = self.rope_tensor.expand(bsz, -1, -1)
        for block in self.transformer:
            z = block(z, rope)
        z = self.ln_post(z)

        z = self.ffn(z)  # embed -> patch
        z = self.conv_out(z)  # final 3x3 conv

        return z


# ================================
# Main DeTok Model
# ================================


class DeTok(nn.Module):
    """
    l-DeTok: latent denoising makes good visual tokenizers.
    """

    _logged = False

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        vit_enc_model_size: str = "small",
        vit_dec_model_size: str = "base",
        token_channels: int = 16,
        mask_ratio: float = 0.75,
        gamma: float = 3.0,
        use_additive_noise: bool = False,
        # normalization parameters used for generative model training
        mean=0.0,
        std=1.0,
        scale_factor: float = 1.0,
    ) -> None:
        super().__init__()

        # initialize encoder and decoder
        self.encoder = Encoder(
            img_size=img_size,
            patch_size=patch_size,
            model_size=vit_enc_model_size,
            token_channels=token_channels,
            mask_ratio=mask_ratio,
        )
        self.decoder = Decoder(
            img_size=img_size,
            patch_size=patch_size,
            model_size=vit_dec_model_size,
            token_channels=token_channels,
        )

        # model configuration
        self.seq_h = img_size // patch_size
        self.width = SIZE_DICT[vit_enc_model_size]["width"]
        self.use_additive_noise = use_additive_noise
        self.gamma = gamma

        self.scale_factor = scale_factor

        # initialize weights
        self.apply(self._init_weights)

        # setup to-posteriors function
        self.to_posteriors = partial(DiagonalGaussianDistribution, channel_dim=-1)

        # logging
        if not DeTok._logged:
            DeTok._logged = True
            logger.info(f"[DeTok] Gamma: {self.gamma}, Max Mask Ratio: {mask_ratio}")

        # setup normalization parameters
        if isinstance(mean, np.ndarray) or isinstance(mean, list):
            mean = np.array(mean).reshape(1, -1, 1, 1)
            std = np.array(std).reshape(1, -1, 1, 1)
        self.register_buffer("mean", torch.tensor(mean), persistent=False)
        self.register_buffer("std", torch.tensor(std), persistent=False)

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[DeTok] params: {params_M:.2f}M")

    def _init_weights(self, module: nn.Module) -> None:
        """initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def freeze_everything_but_decoder(self) -> None:
        """freeze all parameters except the decoder, used for decoder fine-tuning"""
        for param in self.parameters():
            param.requires_grad = False

        for param in self.decoder.parameters():
            param.requires_grad = True

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[DeTok] trainable params: {params_M:.2f}M (after freezing all but decoder)")

    def reset_stats(self, mean: Tensor | np.ndarray | float, std: Tensor | np.ndarray | float) -> None:
        if isinstance(mean, float) and isinstance(std, float) or (mean.ndim == 0 and std.ndim == 0):
            # a single digit global mean and global std
            self.register_buffer("mean", _to_tensor(mean), persistent=False)
            self.register_buffer("std", _to_tensor(std), persistent=False)
        else:
            n_chans = mean.shape[-1]
            self.register_buffer("mean", _to_tensor(mean).reshape(1, 1, n_chans), persistent=False)
            self.register_buffer("std", _to_tensor(std).reshape(1, 1, n_chans), persistent=False)
        logger.info(f"Resetting mean and std ({mean.shape=}, {std.shape=})")
        logger.info(f"Mean: {self.mean}")
        logger.info(f"Std: {self.std}")

    def denormalize_z(self, z: Tensor) -> Tensor:
        """denormalize latent tokens."""
        return z * self.std.to(z) / self.scale_factor + self.mean.to(z)

    def normalize_z(self, z: Tensor) -> Tensor:
        """normalize latent tokens."""
        return (z - self.mean.to(z)) * self.scale_factor / self.std.to(z)

    def encode_into_posteriors(self, x: Tensor):
        """encode image into posterior distributions."""
        z = self.encoder(x, mask_ratio=0.0)[0]
        return self.to_posteriors(z)

    def encode(self, x: Tensor, sampling: bool = False, mask_ratio: float = -1, noise_level: float = -1.0):
        """encode image into latent tokens."""
        z, ids_restore = self.encoder(x, mask_ratio=mask_ratio)

        posteriors = self.to_posteriors(z)
        z_latents = posteriors.sample() if sampling else posteriors.mean

        if self.training and self.gamma > 0.0:
            device = z_latents.device
            bsz, n_tokens, chans = z_latents.shape
            if noise_level > 0.0:
                noise_level_tensor = torch.full((bsz, 1, 1), noise_level, device=device)
            else:
                noise_level_tensor = torch.rand(bsz, 1, 1, device=device)
            noise_level_tensor = noise_level_tensor.expand(-1, n_tokens, chans)
            noise = torch.randn(bsz, n_tokens, chans, device=device) * self.gamma
            if self.use_additive_noise:
                z_latents = z_latents + noise_level_tensor * noise
            else:
                z_latents = (1 - noise_level_tensor) * z_latents + noise_level_tensor * noise

        return z_latents, posteriors, ids_restore

    def forward(self, x: Tensor):
        """forward pass through the entire model."""
        z_latents, result_dict, ids_restore = self.encode(x, sampling=self.training)
        decoded = self.decoder(z_latents, ids_restore=ids_restore)
        return decoded, result_dict

    def tokenize(self, x: Tensor, sampling: bool = False) -> Tensor:
        """tokenize input image and normalize the latent tokens."""
        z = self.encode(x, sampling=sampling, mask_ratio=0.0)[0]
        z = self.normalize_z(z)
        return rearrange(z, "b (h w) c -> b c h w", h=self.seq_h)

    def detokenize(self, z: Tensor) -> Tensor:
        """detokenize latent representation back to image."""
        z = rearrange(z, "b c h w -> b (h w) c")
        z = self.denormalize_z(z)
        decoded_images = self.decoder(z)
        return torch.clamp(decoded_images * 0.5 + 0.5, 0.0, 1.0)

    def sample_from_moments(self, moments: Tensor) -> Tensor:
        """sample from latent moments."""
        z = DiagonalGaussianDistribution(moments, channel_dim=-1).sample()
        z = self.normalize_z(z)
        return rearrange(z, "b (h w) c -> b c h w", h=self.seq_h)

    @torch.inference_mode()
    def reconstruct(self, x: Tensor) -> Tensor:
        """reconstruct input image."""
        return self.detokenize(self.tokenize(x))


# ================================
# Model Factory Functions
# ================================


def detok_SS(**kwargs) -> DeTok:
    return DeTok(vit_enc_model_size="small", vit_dec_model_size="small", **kwargs)


def detok_SB(**kwargs) -> DeTok:
    return DeTok(vit_enc_model_size="small", vit_dec_model_size="base", **kwargs)


def detok_SL(**kwargs) -> DeTok:
    return DeTok(vit_enc_model_size="small", vit_dec_model_size="large", **kwargs)


def detok_BS(**kwargs) -> DeTok:
    return DeTok(vit_enc_model_size="base", vit_dec_model_size="small", **kwargs)


def detok_BB(**kwargs) -> DeTok:
    return DeTok(vit_enc_model_size="base", vit_dec_model_size="base", **kwargs)


def detok_BL(**kwargs) -> DeTok:
    return DeTok(vit_enc_model_size="base", vit_dec_model_size="large", **kwargs)


def detok_LS(**kwargs) -> DeTok:
    return DeTok(vit_enc_model_size="large", vit_dec_model_size="small", **kwargs)


def detok_LB(**kwargs) -> DeTok:
    return DeTok(vit_enc_model_size="large", vit_dec_model_size="base", **kwargs)


def detok_LL(**kwargs) -> DeTok:
    return DeTok(vit_enc_model_size="large", vit_dec_model_size="large", **kwargs)


def detok_XLXL(**kwargs) -> DeTok:
    return DeTok(vit_enc_model_size="xl", vit_dec_model_size="xl", **kwargs)


# ================================
# Model Registry
# ================================

DeTok_models = {
    "detok_SS": detok_SS,
    "detok_SB": detok_SB,
    "detok_SL": detok_SL,
    "detok_BS": detok_BS,
    "detok_BB": detok_BB,
    "detok_BL": detok_BL,
    "detok_LS": detok_LS,
    "detok_LB": detok_LB,
    "detok_LL": detok_LL,
    "detok_XLXL": detok_XLXL,
}
