"""
Modified from LDM's KL-VAE: https://github.com/CompVis/latent-diffusion
"""
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL as DiffusersAutoencoderKLBackbone
from einops import rearrange

from utils.loader import CONSTANTS

logger = logging.getLogger("DeTok")


def nonlinearity(x):  # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        give_pre_end=False,
        grad_checkpointing=True,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.grad_checkpointing = grad_checkpointing

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        logger.info(f"z.shape: {self.z_shape} = {np.prod(self.z_shape)} dimensions.")

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False, channel_dim=1):
        self.parameters = parameters.float()
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=channel_dim)
        self.sum_dims = tuple(range(1, self.mean.dim()))
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    @torch.autocast("cuda", enabled=False)
    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    @torch.autocast("cuda", enabled=False)
    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=self.sum_dims,
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=self.sum_dims,
                )

    @torch.autocast("cuda", enabled=False)
    def nll(self, sample, dims=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims or self.sum_dims,
        )

    @torch.autocast("cuda", enabled=False)
    def mode(self):
        return self.mean


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        embed_dim,
        ch_mult,
        use_variational=True,
        ckpt_path=None,
        scale_factor=1.0,
        mean=0.0,
        std=1.0,
        attn_resolutions=(),
        pixel_shuffle=False,
        name=None,
        gamma=0.0,
    ):
        super().__init__()
        self.name = name if name is not None else "autoencoder"
        logger.info(f"[AutoencoderKL] Initializing {self.name} with {embed_dim} dimensions")
        self.encoder = Encoder(ch_mult=ch_mult, z_channels=embed_dim)
        self.decoder = Decoder(ch_mult=ch_mult, z_channels=embed_dim, attn_resolutions=attn_resolutions)
        self.use_variational = use_variational
        mult = 2 if self.use_variational else 1
        self.quant_conv = torch.nn.Conv2d(2 * embed_dim, mult * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, embed_dim, 1)
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor
        self.pixel_shuffle = pixel_shuffle
        if isinstance(mean, np.ndarray) or isinstance(mean, list):
            mean = np.array(mean).reshape(1, -1, 1, 1)
            std = np.array(std).reshape(1, -1, 1, 1)
        self.register_buffer("mean", torch.tensor(mean), persistent=False)
        self.register_buffer("std", torch.tensor(std), persistent=False)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[AutoencoderKL] {self.name} params: {params_M:.2f}M")
        self.gamma = gamma
        logger.info(f"[AutoencoderKL] {self.name} gamma: {self.gamma}")

    def freeze_everything_but_decoder(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = True
        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(
            f"[AutoencoderKL] After freezing everything but decoder, {self.name} params: {params_M:.2f}M"
        )

    def init_from_ckpt(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        ckpt = torch.load(path, map_location="cpu")
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        # Load the converted state dict with strict=False to allow for partial loading
        logger.info(f"[AutoencoderKL] Loading pre-trained {self.name} from {path}")
        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(f"[AutoencoderKL] Missing keys: {len(msg.missing_keys)}")
        logger.info(f"[AutoencoderKL] Unexpected keys: {len(msg.unexpected_keys)}")

    def reset_stats(self, mean, std) -> None:
        """reset normalization statistics."""
        if mean.ndim == 0:
            self.register_buffer("mean", torch.tensor(mean), persistent=False)
            self.register_buffer("std", torch.tensor(std), persistent=False)
        else:
            n_chans = mean.shape[-1]
            self.register_buffer("mean", torch.tensor(mean).reshape(1, 1, n_chans), persistent=False)
            self.register_buffer("std", torch.tensor(std).reshape(1, 1, n_chans), persistent=False)
        logger.info(f"Resetting mean and std ({mean.shape=}, {std.shape=})")
        logger.info(f"Mean: {self.mean}, Std: {self.std}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        if not self.use_variational:
            moments = torch.cat((moments, torch.ones_like(moments)), 1)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def encode_into_posteriors(self, x):
        # just for naming compatibility with other tokenizers
        return self.encode(x)

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def autoencode(self, x):
        posterior = self.encode(x)
        z = posterior.sample()
        if self.gamma > 0.0 and self.training:
            bsz, c, h, w = z.shape
            device = z.device
            time_input = torch.rand(bsz, 1, 1, 1, device=device)
            time_input = time_input.expand(-1, c, h, w)
            noises = torch.randn(bsz, c, h, w, device=device) * self.gamma
            z = (1 - time_input) * z + time_input * noises
        reconstructions = self.decode(z)
        return reconstructions, posterior, z

    def forward(self, x):
        reconstructions, posterior, z = self.autoencode(x)
        return reconstructions, posterior

    def denormalize_z(self, z):
        z = z * self.std.to(z) / self.scale_factor + self.mean.to(z)
        if self.pixel_shuffle:
            z = torch.nn.functional.pixel_shuffle(z, upscale_factor=2)
        return z

    def normalize_z(self, z):
        z = (z - self.mean.to(z)) * self.scale_factor / self.std.to(z)
        if self.pixel_shuffle:
            assert z.ndim == 4, "B, C, H, W"
            z = torch.nn.functional.pixel_unshuffle(z, downscale_factor=2)
        return z

    def tokenize(self, x, sampling=False):
        sample = self.encode(x).sample() if sampling else self.encode(x).mean
        return self.normalize_z(sample)

    def unpatchify(self, x, p=1):
        c = self.embed_dim
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def detokenize(self, z):
        if z.ndim == 3:  # b, l, c -> b, c, h, w
            z = self.unpatchify(z)
        return torch.clamp(self.decode(self.denormalize_z(z)) * 0.5 + 0.5, 0.0, 1.0)

    def sample_from_moments(self, moments):
        z = DiagonalGaussianDistribution(moments).sample()
        return self.normalize_z(z)

    @torch.inference_mode()
    def reconstruct(self, x):
        return self.detokenize(self.tokenize(x))


class DiffusersAutoencoderKL(nn.Module):
    def __init__(self, name=None, scale_factor=0.18215):
        super().__init__()
        self.name = name if name is not None else "autoencoder"
        if name == "sdvae":
            self.vae: DiffusersAutoencoderKLBackbone = DiffusersAutoencoderKLBackbone.from_pretrained(
                f"stabilityai/sd-vae-ft-ema"
            )
        elif name == "eqvae":
            self.vae: DiffusersAutoencoderKLBackbone = DiffusersAutoencoderKLBackbone.from_pretrained(
                f"zelaki/eq-vae-ema"
            )
        self.vae.eval()
        self.scale_factor = scale_factor

    def forward(self):
        pass

    def denormalize_z(self, z):
        return z / self.scale_factor

    def normalize_z(self, z):
        return z * self.scale_factor

    def tokenize(self, x, sampling=False):
        sample = self.vae.encode(x).latent_dist
        sample = sample.sample() if sampling else sample.mean
        return self.normalize_z(sample)

    def encode(self, x):
        posterior = self.vae.encode(x).latent_dist
        return posterior

    def detokenize(self, z):
        return torch.clamp(self.vae.decode(self.denormalize_z(z)).sample * 0.5 + 0.5, 0.0, 1.0)

    def sample_from_moments(self, moments):
        z = DiagonalGaussianDistribution(moments).sample()
        return self.normalize_z(z)

    @torch.inference_mode()
    def reconstruct(self, x):
        return self.detokenize(self.tokenize(x))


class VectorQuantizer(torch.nn.Module):
    def __init__(
        self,
        codebook_size: int = 1024,
        token_channels: int = 256,
        commitment_cost: float = 0.25,
        use_l2_norm: bool = False,
    ):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.embedding = torch.nn.Embedding(codebook_size, token_channels)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.use_l2_norm = use_l2_norm

    # Ensure quantization is performed using f32
    @torch.autocast("cuda", enabled=False)
    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = z.float()
        if z.ndim == 4:
            z = rearrange(z, "b c h w -> b h w c").contiguous()
            z_flattened = rearrange(z, "b h w c -> (b h w) c")
        else:
            z_flattened = rearrange(z, "b n c -> (b n) c").contiguous()

        if self.use_l2_norm:
            z_flattened = torch.nn.functional.normalize(z_flattened, dim=-1)
            embedding = torch.nn.functional.normalize(self.embedding.weight, dim=-1)
        else:
            embedding = self.embedding.weight
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(embedding**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, embedding.T)
        )

        min_encoding_indices = torch.argmin(d, dim=1)  # num_ele
        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)

        if self.use_l2_norm:
            z = torch.nn.functional.normalize(z, dim=-1)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) ** 2)
        codebook_loss = torch.mean((z_quantized - z.detach()) ** 2)

        loss = commitment_loss + codebook_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match original input shape
        if z.ndim == 4:
            z_quantized = rearrange(z_quantized, "b h w c -> b c h w").contiguous()
            min_encoding_indices = min_encoding_indices.view(
                z_quantized.shape[0], z_quantized.shape[2], z_quantized.shape[3]
            )
        else:
            z_quantized = z_quantized.contiguous()
        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            min_encoding_indices=min_encoding_indices,
        )
        return z_quantized, result_dict

    def get_codebook_entry(self, indices):
        if len(indices.shape) == 1:
            z_quantized = self.embedding(indices)
        elif len(indices.shape) == 2:
            z_quantized = torch.einsum("bd,dn->bn", indices, self.embedding.weight)
        else:
            raise NotImplementedError
        if self.use_l2_norm:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
        return z_quantized


def mar_vae(load_ckpt=True, load_from=None, gamma=0.0) -> AutoencoderKL:
    if load_from is not None:
        ckpt_path = load_from
    else:
        ckpt_path = "pretrained_models/vae/marvae_kl16.ckpt" if load_ckpt else None
    return AutoencoderKL(
        name="marvae",
        embed_dim=16,
        ch_mult=(1, 1, 2, 2, 4),
        scale_factor=0.2325,
        ckpt_path=ckpt_path,
        gamma=gamma,
    )


def va_vae(load_ckpt=True, load_from=None, gamma=0.0) -> AutoencoderKL:
    return AutoencoderKL(
        name="vavae",
        embed_dim=32,
        ch_mult=(1, 1, 2, 2, 4),
        scale_factor=1.0,
        attn_resolutions=(16,),
        mean=CONSTANTS["vavae_mean"],
        std=CONSTANTS["vavae_std"],
        ckpt_path=("pretrained_models/vae/vavae-imagenet256-f16d32-dinov2-clean.pth" if load_ckpt else None),
    )


def sd_vae(load_ckpt=True, load_from=None, gamma=0.0) -> DiffusersAutoencoderKL:
    return DiffusersAutoencoderKL(name="sdvae", scale_factor=0.18215)


def eq_vae(load_ckpt=True, load_from=None, gamma=0.0) -> DiffusersAutoencoderKL:
    return DiffusersAutoencoderKL(name="eqvae", scale_factor=0.18215)


VAE_models = {"marvae": mar_vae, "vavae": va_vae, "sdvae": sd_vae, "eqvae": eq_vae}
