import math
import random
import argparse
import os
import copy
import datetime
from glob import glob
import logging

import numpy as np
import torch
import torch.nn.utils
from torch import inf

import utils.distributed as dist

logger = logging.getLogger("DeTok")


def fix_random_seeds(seed=31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        if args.lr_sched == "constant":
            lr = args.lr
        elif args.lr_sched == "cosine":
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise NotImplementedError
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type,
        )
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, enabled: bool = True):
        self._scaler = torch.GradScaler(device="cuda", enabled=enabled)

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None and clip_grad > 0.0:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def ckpt_resume(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    loss_scaler: NativeScalerWithGradNormCount | None = None,
    model_ema: torch.nn.Module | None = None,
    loss_module: torch.nn.Module | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
    discriminator_loss_scaler: NativeScalerWithGradNormCount | None = None,
):
    if args.resume_from or args.auto_resume:
        if args.resume_from is None:
            # find the latest checkpoint
            checkpoints = [ckpt for ckpt in glob(f"{args.ckpt_dir}/*.pth") if "latest" not in ckpt]
            checkpoints = sorted(checkpoints, key=os.path.getmtime)
            if len(checkpoints) > 0:
                args.resume_from = checkpoints[-1]

        if args.resume_from and os.path.exists(args.resume_from):
            # load the checkpoint
            logger.info(f"[Model-resume] Resuming from: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
            msg = model.load_state_dict(checkpoint["model"])
            logger.info(f"[Model-resume] Loaded model: {msg}")

            if "model_ema" in checkpoint:
                # load the EMA state dict if it exists
                ema_state_dict = checkpoint["model_ema"]
                logger.info(f"[Model-resume] Loaded EMA")
            else:
                # if no EMA state dict, use the model state dict to initialize the EMA state dict
                model_state_dict = model.state_dict()
                param_keys = [k for k, _ in model.named_parameters()]
                ema_state_dict = {k: model_state_dict[k] for k in param_keys}
                logger.info(f"[Model-resume] Loaded EMA with model state dict")

            # load the EMA state dict if it exists
            if model_ema is not None:
                model_ema.load_state_dict(ema_state_dict)
                model_ema.to("cuda")  # move the EMA model to the GPU

            # load the optimizer state dict if it exists
            if "optimizer" in checkpoint and "epoch" in checkpoint and optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
                args.start_epoch = checkpoint["epoch"] + 1
                # load the loss scaler state dict if it exists
                if "loss_scaler" in checkpoint and loss_scaler is not None:
                    loss_scaler.load_state_dict(checkpoint["loss_scaler"])

            # load the last elapsed time if it exists
            if "last_elapsed_time" in checkpoint:
                args.last_elapsed_time = float(checkpoint["last_elapsed_time"])
                elapsed_time_str = str(datetime.timedelta(seconds=int(args.last_elapsed_time)))
                logger.info(f"Loaded elapsed_time: {elapsed_time_str}")

            # load the loss module state dict if it exists
            if "loss_module" in checkpoint and loss_module is not None:
                msg = loss_module.load_state_dict(checkpoint["loss_module"])
                logger.info(f"[Model-resume] Loaded loss_module: {msg}")

            if "discriminator_optimizer" in checkpoint and discriminator_optimizer is not None:
                msg = discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
                logger.info(f"[Model-resume] Loaded discriminator_optimizer: {msg}")

            if "discriminator_loss_scaler" in checkpoint and discriminator_loss_scaler is not None:
                msg = discriminator_loss_scaler.load_state_dict(checkpoint["discriminator_loss_scaler"])
                logger.info(f"[Model-resume] Loaded discriminator_loss_scaler: {msg}")

            # delete the checkpoint to save memory
            del checkpoint
        else:
            logger.info(f"[Model-resume] Could not find checkpoint at {args.resume_from}.")
    else:
        logger.info(f"[Model-resume] Could not find checkpoint at {args.resume_from}.")

    if args.load_from and not args.resume_from:
        # if no checkpoint is provided, load the checkpoint from the load_from path instead
        if os.path.exists(args.load_from):
            logger.info(f"[Model-load] Loading checkpoint from: {args.load_from}")
            checkpoint = torch.load(args.load_from, map_location="cpu", weights_only=False)
            # load the model state dict if it exists
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
            msg = model.load_state_dict(state_dict, strict=False)
            # assert unexpected keys can only start with "loss."
            for key in msg.unexpected_keys:
                assert key.startswith("loss."), f"unexpected key {key} doesn't start with 'loss.'"
            logger.info(f"[Model-load] Loaded model: {msg}")
            if "model_ema" in checkpoint:
                logger.info(f"[Model-load] Loaded EMA")
                ema_state_dict = checkpoint["model_ema"]
            else:
                logger.info(f"[Model-load] Loaded EMA with model state dict")
                ema_state_dict = copy.deepcopy(model.state_dict())
            if model_ema is not None:
                model_ema.load_state_dict(ema_state_dict)
                model_ema.to(device="cuda")  # move the EMA model to the GPU
            del checkpoint  # delete the checkpoint to save memory
        else:
            raise FileNotFoundError(f"Could not find checkpoint at {args.load_from}")


def cleanup_checkpoints(ckpt_dir: str, keep_num: int = 5, milestone_interval: int = 5):
    """
    Clean up older checkpoint files in `ckpt_dir` while keeping the latest `keep_num` checkpoints by epoch number.

    Parameters
    ----------
    ckpt_dir : str
        The directory where checkpoint .pth files are stored.
    keep_num : int, optional
        The number of most recent checkpoints to keep (default=5).
    milestone_interval : int, optional
        The interval used to decide if a checkpoint is a "milestone."
        If (epoch_num + 1) % milestone_interval == 0, it is kept (default=50).
    """
    ckpts = glob(os.path.join(ckpt_dir, "*.pth"))
    ckpts = [ckpt for ckpt in ckpts if "latest" not in ckpt and "best" not in ckpt]

    def get_ckpt_num(path):
        """Extract the epoch number from a checkpoint filename."""
        filename = os.path.basename(path)
        # expecting something like 'epoch_049.pth'
        # we'll parse out the part after the last underscore and before '.pth'
        try:
            return int(filename.rsplit("_", 1)[-1].split(".")[0])
        except ValueError:
            return None

    # sort checkpoints by epoch number
    ckpts.sort(key=lambda x: (get_ckpt_num(x) is None, get_ckpt_num(x)))

    # filter out any that failed to parse an integer epoch (get_ckpt_num == None)
    ckpts = [ckpt for ckpt in ckpts if get_ckpt_num(ckpt) is not None]

    if not ckpts:
        # if no checkpoints remain, nothing to do
        return

    # determine which checkpoints to keep:
    # 1. the newest `keep_num` by epoch number.
    # 2. any milestone checkpoints.
    #    (epoch_num + 1) % milestone_interval == 0
    newest_keep = set(ckpts[-keep_num:])  # handle if keep_num > number of ckpts
    milestone_keep = set(ckpt for ckpt in ckpts if ((get_ckpt_num(ckpt) + 1) % milestone_interval == 0))

    # union of both sets
    keep_set = newest_keep.union(milestone_keep)

    # remove anything not in keep_set
    for ckpt in ckpts:
        if ckpt not in keep_set:
            os.remove(ckpt)
            logger.info(f"Removed checkpoint: {ckpt}")

    # recreate the 'latest.pth' symlink to the newest checkpoint
    if keep_set:
        # we need the absolute newest based on epoch number
        # sort again from keep_set only
        remaining_ckpts_sorted = sorted(keep_set, key=lambda x: (get_ckpt_num(x) is None, get_ckpt_num(x)))
        newest_ckpt = os.path.abspath(remaining_ckpts_sorted[-1])
        latest_symlink = os.path.join(ckpt_dir, "latest.pth")

        # remove the old symlink if it exists
        try:
            os.remove(latest_symlink)
            logger.info(f"Removed old symlink: {latest_symlink}")
        except FileNotFoundError:
            pass

        # create a new symlink
        os.symlink(newest_ckpt, latest_symlink)
        logger.info(f"Created symlink: {latest_symlink} -> {newest_ckpt}")


def save_checkpoint(
    args,
    epoch,
    model,
    optimizer,
    loss_scaler,
    model_ema,
    elapsed_time=0.0,
    loss_module=None,
    discriminator_optimizer=None,
    discriminator_loss_scaler=None,
):
    if not dist.is_main_process():
        return
    checkpoint = {
        "model": model.state_dict(),
        "model_ema": model_ema.state_dict() if model_ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "loss_scaler": loss_scaler.state_dict(),
        "epoch": epoch,
        "last_elapsed_time": elapsed_time,
    }
    if loss_module is not None and isinstance(loss_module, torch.nn.Module):
        checkpoint["loss_module"] = loss_module.state_dict()
    if discriminator_optimizer is not None:
        checkpoint["discriminator_optimizer"] = discriminator_optimizer.state_dict()
    if discriminator_loss_scaler is not None:
        checkpoint["discriminator_loss_scaler"] = discriminator_loss_scaler.state_dict()
    checkpoint_path = os.path.join(args.ckpt_dir, f"epoch_{epoch:04d}.pth")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")
    cleanup_checkpoints(args.ckpt_dir, args.keep_n_ckpts, args.milestone_interval)
