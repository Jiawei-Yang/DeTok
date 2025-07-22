"""
A collection of commonly used layers for Transformer-based approaches.

This file is heavily based on timm implementations and other seminal works such as DiT/MAE/LlaMA.
"""

import logging
import math
from functools import partial
from math import pi

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger("DeTok")


def to_2tuple(x):
    """Convert input to a 2-tuple. If already a tuple/list, return as tuple. Otherwise duplicate the value."""
    return tuple(x) if isinstance(x, (tuple, list)) else (x, x)


def modulate(x, shift=None, scale=None):
    if shift is None and scale is None:
        return x
    if x.ndim == shift.ndim:
        return x * (1 + scale) + shift
    elif x.ndim == shift.ndim + 1:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    else:
        raise ValueError(f"shift shape {shift.shape} and x shape {x.shape} are not compatible")


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module | None = None,
        bias: bool = True,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(self.norm(x)))
        return x


class ModulatedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        if use_rmsnorm:
            self.norm = nn.RMSNorm(in_features)
        else:
            self.norm = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(in_features, 2 * in_features, bias=bias))

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class Attention(nn.Module):
    _logged = False

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        kv_dim: int | None = None,
        is_cross_attn: bool = False,
        proj_bias: bool = True,
        force_causal: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim % num_heads !=0, got {dim} and {num_heads}"
        self.num_heads = num_heads
        kv_dim = dim if kv_dim is None else kv_dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.is_cross_attn = is_cross_attn
        self.force_causal = force_causal

        self.fused_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if self.fused_attn and not Attention._logged:
            Attention._logged = True
            logger.info(f"[Attention]: Using {torch.__version__} Fused Attention, {force_causal=}")

        if is_cross_attn:
            self.c_q = nn.Linear(dim, dim, bias=qkv_bias)  # context to q
            self.c_kv = nn.Linear(kv_dim, dim * 2, bias=qkv_bias)  # context to kv
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kv_cache = False
        self.k_cache = None
        self.v_cache = None

    def reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, x: Tensor, data: Tensor = None, attn_mask=None, rope=None) -> Tensor:
        # attn_mask: this is actually an bias term. 0 for visible, -inf for invisible
        bs, n_ctx, C = x.shape

        # Get q,k,v - either from cross attention or self attention
        if self.is_cross_attn:
            assert data is not None, "data should not be None for cross attn"
            q = self.c_q(x).view(bs, n_ctx, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = self.c_kv(data)
            k, v = kv.view(bs, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        else:
            qkv = self.qkv(x).reshape(bs, n_ctx, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(dim=0)
        # Apply norms and rotary embeddings
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            q, k = rope(q), rope(k)

        # Handle kv caching for inference
        if self.kv_cache:
            if self.k_cache is None:
                k_cache, v_cache = k, v
            else:
                assert n_ctx in [1, 2], f"x.shape {x.shape}"
                k_cache = torch.cat([self.k_cache, k], dim=-2)
                v_cache = torch.cat([self.v_cache, v], dim=-2)
            self.k_cache, self.v_cache = k_cache, v_cache
            k, v = k_cache, v_cache

        # Compute attention - use fused attention if available
        if self.fused_attn:
            if attn_mask is not None and attn_mask.ndim == 3:
                # in fused_attn, BHMK as (bs, num_heads, n_ctx, head_dim)
                attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None if self.force_causal else attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=self.force_causal,
            )
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            x = self.attn_drop(attn.softmax(dim=-1)) @ v

        # Project output
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(bs, n_ctx, C)))


class Block(nn.Module):
    _logged = False

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        mlp_bias: bool = True,
        attn_proj_bias: bool = True,
        init_values: float | None = None,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        kv_dim: int | None = None,
        use_cross_attn: bool = False,
        use_modulation: bool = False,
        use_swiglu: bool = False,
        force_causal: bool = False,
        no_dropout_in_mlp: bool = False,
    ) -> None:
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.use_modulation = use_modulation
        if not Block._logged:
            Block._logged = True
            logger.info(f"[Block]: {use_modulation=}, {use_cross_attn=}, {use_swiglu=}, {norm_layer=}")

        self.norm1, self.norm2 = norm_layer(dim), norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            kv_dim=kv_dim if kv_dim is None else kv_dim,
            is_cross_attn=use_cross_attn,
            proj_bias=attn_proj_bias,
            force_causal=force_causal,
        )
        if self.use_cross_attn:
            self.data_norm = norm_layer(dim if kv_dim is None else kv_dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        if use_swiglu:
            self.mlp = SwiGLUFFN(dim, int(2 / 3 * dim * mlp_ratio))
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                bias=mlp_bias,
                drop=proj_drop if not no_dropout_in_mlp else 0.0,
            )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.adaLN_modulation = None
        if self.use_modulation:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(
        self,
        x: Tensor,
        data: Tensor = None,
        attn_mask: Tensor = None,
        condition: Tensor = None,
        rope=None,
    ) -> Tensor:
        if self.use_modulation:
            assert condition is not None, "condition should not be None for modulation"
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
                condition
            ).chunk(6, dim=-1)
            if condition.ndim == 2:  # (bsz, dim) -> (bsz, 1, dim)
                gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)
                shift_msa, scale_msa = shift_msa.unsqueeze(1), scale_msa.unsqueeze(1)
                shift_mlp, scale_mlp = shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1)
        else:
            shift_msa, scale_msa, gate_msa = None, None, 1.0
            shift_mlp, scale_mlp, gate_mlp = None, None, 1.0
        if self.use_cross_attn:
            assert data is not None, "data should not be None for cross attn"
            attn = self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa),
                self.data_norm(data),
                attn_mask=attn_mask,
                rope=rope,
            )
        else:
            attn = self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa),
                attn_mask=attn_mask,
                rope=rope,
            )
        x = x + gate_msa * self.ls1(attn)
        x = x + gate_mlp * self.ls2(self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        mlp_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        attn_proj_bias: bool = True,
        init_values: float | None = None,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        block_fn: nn.Module = Block,
        use_swiglu: bool = False,
        force_causal: bool = False,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    init_values=init_values,
                    norm_layer=norm_layer,
                    mlp_bias=mlp_bias,
                    attn_proj_bias=attn_proj_bias,
                    use_swiglu=use_swiglu,
                    force_causal=force_causal,
                )
                for _ in range(depth)
            ]
        )
        self.grad_checkpointing = grad_checkpointing
        logger.info(f"[Transformer]: grad_checkpointing={grad_checkpointing}")

    def forward(
        self,
        x: torch.Tensor,
        data: Tensor = None,
        attn_mask: Tensor = None,
        condition: Tensor = None,
        rope=None,
    ) -> torch.Tensor:
        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, data, attn_mask, condition, rope)
            else:
                x = block(x, data, attn_mask, condition, rope)
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size = to_2tuple(img_size)
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride=patch_size, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return rearrange(self.proj(x), "B C H W -> B (H W) C")


class NeRFPosEmbedder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: int = 0,
        max_freq_exp: int = 10,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies)
        self.register_buffer("freqs", freqs)

    def get_out_dim(self) -> int:
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def forward(self, in_tensor):
        scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        scaled_inputs = scaled_in_tensor[..., None] * self.freqs
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)
        encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)
        return encoded_inputs


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, bias=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=bias),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=bias),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        n_prefix_tokens=0,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        self.n_prefix_tokens = n_prefix_tokens
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, t):
        if self.n_prefix_tokens == 0:
            return t * self.freqs_cos + rotate_half(t) * self.freqs_sin
        # t has shape [batch, L, dim] with L = N + M
        prefix_tokens = t[:, :, : self.n_prefix_tokens]  # prefix tokens leave unchanged
        t = t[:, :, self.n_prefix_tokens :]
        t = t * self.freqs_cos + rotate_half(t) * self.freqs_sin
        return torch.cat([prefix_tokens, t], dim=-2)


def pos_enc(x, min_deg=0, max_deg=10, append_identity=True):
    """The positional encoding used by the original NeRF paper."""
    scales = 2 ** torch.arange(min_deg, max_deg).float()
    scales = scales.to(x.device)
    shape = x.shape[:-1] + (-1,)
    scaled_x = torch.reshape((x[..., None, :] * scales[:, None]), shape)
    # Note that we're not using safe_sin, unlike IPE.
    four_feat = torch.sin(torch.concat([scaled_x, scaled_x + 0.5 * np.pi], dim=-1))
    if append_identity:
        return torch.concat([x] + [four_feat], dim=-1)
    else:
        return four_feat


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, num_extra_tokens=1):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and num_extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([num_extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def resample_abs_pos_embed(
    posemb,
    new_size: list[int],
    old_size: list[int] | None = None,
    n_prefix_tokens: int = 1,
    interpolation: str = "bicubic",  # bicubic is better.
    antialias: bool = True,  # antialias is important.
    verbose: bool = False,
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + n_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - n_prefix_tokens))
        old_size = [hw, hw]

    if n_prefix_tokens:
        posemb_prefix, posemb = (
            posemb[:, :n_prefix_tokens],
            posemb[:, n_prefix_tokens:],
        )
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = torch.nn.functional.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    if not torch.jit.is_scripting() and verbose:
        logger.info(f"Resized position embedding: {old_size} to {new_size}.")

    return posemb


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")
