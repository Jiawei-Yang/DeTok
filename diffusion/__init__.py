# Adopted from DiT, which is modified from OpenAI's diffusion repos
#     DiT: https://github.com/facebookresearch/DiT/diffusion
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

import logging

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps

logger = logging.getLogger("DeTok")


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear",
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    channel_last=False,
) -> SpacedDiffusion:
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    if predict_xstart:
        model_mean_type = gd.ModelMeanType.START_X
    else:
        model_mean_type = gd.ModelMeanType.EPSILON

    if learn_sigma:
        model_var_type = gd.ModelVarType.LEARNED_RANGE
    else:
        if sigma_small:
            model_var_type = gd.ModelVarType.FIXED_SMALL
        else:
            model_var_type = gd.ModelVarType.FIXED_LARGE

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        loss_type=loss_type,
        channel_last=channel_last,
    )
    logger.info(f"Created diffusion with timestep respacing {timestep_respacing}")
    return diffusion
