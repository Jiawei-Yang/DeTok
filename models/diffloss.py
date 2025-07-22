"""
Modified from https://github.com/LTH14/mar/blob/main/models/diffloss.py
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from diffusion import create_diffusion
from transport import Sampler, create_transport

from .layers import ModulatedLinear, TimestepEmbedder, modulate


class DiffLoss(nn.Module):
    """diffusion loss module for training."""

    def __init__(
        self,
        target_channels,
        z_channels,
        depth,
        width,
        num_sampling_steps,
        grad_checkpointing=False,
        noise_schedule="cosine",
        use_transport=False,
        timestep_shift=0.3,
        learn_sigma=True,
        sampling_method="euler",
    ):
        super(DiffLoss, self).__init__()

        # --------------------------------------------------------------------------
        # basic configuration
        self.in_channels = target_channels
        self.noise_schedule = noise_schedule
        self.use_transport = use_transport

        # --------------------------------------------------------------------------
        # network architecture
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2 if learn_sigma else target_channels,
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
            use_transport=use_transport,
        )

        # --------------------------------------------------------------------------
        # diffusion/transport setup
        if self.use_transport:
            self.transport = create_transport(use_cosine_loss=True, use_lognorm=True)
            self.sampler = Sampler(self.transport)
            self.sample_fn = self.sampler.sample_ode(
                sampling_method=sampling_method,
                num_steps=int(num_sampling_steps),
                timestep_shift=timestep_shift,
            )
        else:
            self.train_diffusion = create_diffusion("", noise_schedule=noise_schedule)
            self.gen_diffusion = create_diffusion(num_sampling_steps, noise_schedule=noise_schedule)

    def forward(self, target, z, mask=None):
        """forward pass for training."""
        if self.use_transport:
            model_kwargs = dict(c=z)
            loss_dict = self.transport.training_losses(self.net, target, model_kwargs)
        else:
            t = torch.randint(
                0,
                self.train_diffusion.num_timesteps,
                (target.shape[0],),
                device=target.device,
            )
            model_kwargs = dict(c=z)
            loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)

        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z, temperature=1.0, cfg=1.0):
        """sample from the diffusion model."""
        if cfg != 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            if self.use_transport:
                model_kwargs = dict(c=z, cfg_scale=cfg, cfg_interval=True, cfg_interval_start=0.10)
            else:
                model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).cuda()
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        if self.use_transport:
            sampled_token_latent = self.sample_fn(noise, sample_fn, **model_kwargs)[-1]
        else:
            sampled_token_latent = self.gen_diffusion.p_sample_loop(
                sample_fn,
                noise.shape,
                noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                temperature=temperature,
            )
        return sampled_token_latent


class ResBlock(nn.Module):
    """residual block with adaptive layer normalization."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class SimpleMLPAdaLN(nn.Module):
    """simple MLP with adaptive layer normalization for diffusion loss."""

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False,
        use_transport=False,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # basic configuration
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing
        self.use_transport = use_transport

        # --------------------------------------------------------------------------
        # network layers
        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer = ModulatedLinear(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        """initialize model weights."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t=None, c=None):
        """apply the model to an input batch."""
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)
        y = t + c

        for block in self.res_blocks:
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, y)
            else:
                x = block(x, y)
        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale, cfg_interval=None, cfg_interval_start=None):
        """forward pass with classifier-free guidance."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        if cfg_interval is True:
            timestep = t[0]
            if timestep < cfg_interval_start:
                half_eps = cond_eps

        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
