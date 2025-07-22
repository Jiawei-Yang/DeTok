"""
DeTok: Generation model training script.
"""

import argparse
import datetime
import logging
import sys
import time

import torch
import torch.distributed

import models
import utils.distributed as distributed
from utils.builders import create_generation_model, create_optimizer_and_scaler, create_train_dataloader
from utils.misc import ckpt_resume, save_checkpoint
from utils.train_utils import (
    collect_tokenizer_stats,
    evaluate_generator,
    setup,
    train_one_epoch_generator,
    visualize_generator,
    visualize_tokenizer,
)

# performance optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

logger = logging.getLogger("DeTok")


def main(args: argparse.Namespace) -> int:
    global logger
    wandb_logger = setup(args)
    data_loader_train = create_train_dataloader(args)

    # initialize models
    model, tokenizer, ema_model = create_generation_model(args)
    optimizer, loss_scaler = create_optimizer_and_scaler(args, model)
    model_wo_ddp = model

    # handle token caching or tokenizer statistics collection
    if args.collect_tokenizer_stats:
        tmp_data_loader = create_train_dataloader(
            args, should_flip=False, batch_size=args.tokenizer_bsz,
            return_path=True, drop_last=False
        )
        # (B, C, H, W) for chan_dim=2 or (B, seq_len, C) for chan_dim=1
        chan_dim = 2 if args.tokenizer in models.DeTok_models else 1
        
        # collect stats
        result_dict = collect_tokenizer_stats(
            tokenizer, tmp_data_loader, chan_dim=chan_dim,
            stats_dict_key=args.stats_key,
            stats_dict_path=args.stats_cache_path,
            overwrite_stats=args.overwrite_stats,
        )
        # update tokenizer with computed statistics
        mean, std = result_dict["channel"]
        if mean.ndim > 0:
            n_chans = len(mean) // 2
            mean, std = mean[:n_chans], std[:n_chans]
        tokenizer.reset_stats(mean, std)
        
        del tmp_data_loader
        data_dict = next(iter(data_loader_train))
        visualize_tokenizer(args, tokenizer, ema_model=None, data_dict=data_dict)

    # setup distributed training
    if distributed.is_enabled():
        model = torch.nn.parallel.DistributedDataParallel(model)
        model_wo_ddp = model.module

    # resume from checkpoint if needed
    logger.info("Auto-resume enabled")
    ckpt_resume(args, model_wo_ddp, optimizer, loss_scaler, ema_model)

    # evaluation-only mode
    if args.evaluate:
        torch.cuda.empty_cache()
        cfg_list = args.cfg_list if args.cfg_list is not None else [args.cfg]
        for cfg in cfg_list:
            evaluate_generator(
                args,
                model_wo_ddp,
                ema_model,
                tokenizer,
                epoch=args.start_epoch,
                wandb_logger=wandb_logger,
                cfg=cfg,
                use_ema=True, # always use ema model for evaluation
                num_images=args.num_images,
            )
        return 0

    # training loop
    logger.info(f"Start training from {args.start_epoch} to {args.epochs}")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch_generator(
            args, model, data_loader_train, optimizer, loss_scaler, wandb_logger,
            epoch, ema_model, tokenizer
        )

        # progress logging
        elapsed_t = time.time() - start_time + args.last_elapsed_time
        eta = elapsed_t / (epoch + 1) * (args.epochs - epoch - 1)
        logger.info(
            f"[{epoch}/{args.epochs}] "
            f"Accumulated elapsed time: {str(datetime.timedelta(seconds=int(elapsed_t)))}, "
            f"ETA: {str(datetime.timedelta(seconds=int(eta)))}"
        )

        # checkpointing
        should_save = (
            (epoch + 1) % args.save_freq == 0  # save every n epochs
            or (epoch + 1) == args.epochs  # save at the end of training
        )

        if should_save:
            save_checkpoint(args, epoch, model_wo_ddp, optimizer, loss_scaler, ema_model, elapsed_t)
            torch.distributed.barrier()

        # periodic visualization
        if (epoch + 1) % args.vis_freq == 0:
            visualize_generator(args, model_wo_ddp, ema_model, tokenizer, epoch + 1)

        # online evaluation
        if args.online_eval and (epoch + 1) % args.eval_freq == 0:
            torch.cuda.empty_cache()
            evaluate_generator(
                args, model_wo_ddp, ema_model, tokenizer, epoch + 1, wandb_logger,
                use_ema=True, num_images=args.num_images_for_eval_and_search, cfg=args.cfg
            )

    # final evaluation
    total_time = int(time.time() - start_time + args.last_elapsed_time)
    logger.info(f"Training time {str(datetime.timedelta(seconds=total_time))}")


    # determine cfg values for evaluation
    cfg_list = args.cfg_list or [args.cfg]  # use the cfg from the args if not provided
    best_cfg = cfg_list[0]

    if len(cfg_list) > 1:
        # search the best cfg value using 10k images
        fid_dict = {}
        for cfg in cfg_list:
            fid_dict[cfg] = evaluate_generator(
                args, model_wo_ddp, ema_model, tokenizer, args.epochs + 1, wandb_logger,
                use_ema=True, cfg=cfg, num_images=args.num_images_for_eval_and_search
            )
        # find best cfg value and broadcast to all ranks
        if distributed.is_main_process():
            best_fid = 100000
            for cfg in cfg_list:
                if fid_dict[cfg]["fid"] < best_fid:
                    best_fid = fid_dict[cfg]["fid"]
                    best_cfg = cfg
            logger.info(f"Best FID: {best_fid}, Best cfg: {best_cfg}")

    # broadcast best_cfg from rank 0 to all ranks
    if distributed.is_enabled():
        best_cfg_tensor = torch.tensor([best_cfg], dtype=torch.float32, device="cuda")
        torch.distributed.broadcast(best_cfg_tensor, src=0)
        best_cfg = best_cfg_tensor.item()
        torch.distributed.barrier()

    # final comprehensive evaluation with best cfg
    args.num_iter = 128 if args.tokenizer == "maetok-b-128" else 256
    evaluate_generator(
        args, model_wo_ddp, ema_model, tokenizer, args.epochs + 1, wandb_logger,
        use_ema=True, cfg=best_cfg, num_images=args.num_images
    )

    # additional evaluation with cfg=1.0
    evaluate_generator(
        args, model_wo_ddp, ema_model, tokenizer, args.epochs + 1, wandb_logger,
        use_ema=True, cfg=1.0, num_images=args.num_images
    )

    return 0


def get_args_parser():
    parser = argparse.ArgumentParser("Generation model training", add_help=False)

    # basic training parameters
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU for training")

    # model parameters
    parser.add_argument("--model", default="MAR_base", type=str)
    parser.add_argument("--order", default="raster", type=str)
    parser.add_argument("--patch_size", default=1, type=int)
    parser.add_argument("--no_dropout_in_mlp", action="store_true")
    parser.add_argument("--qk_norm", action="store_true")
    parser.add_argument("--force_one_d_seq", type=int, default=0, help="1d tokens, e.g., 128 for MAETok")
    parser.add_argument("--legacy_mode", action="store_true")

    # tokenizer parameters
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--tokenizer", default=None, type=str)
    parser.add_argument("--token_channels", default=16, type=int)
    parser.add_argument("--tokenizer_patch_size", default=16, type=int)
    parser.add_argument("--use_ema_tokenizer", action="store_true")

    # tokenizer cache parameters
    parser.add_argument("--collect_tokenizer_stats", action="store_true")
    parser.add_argument("--tokenizer_bsz", default=256, type=int)
    parser.add_argument("--cached_path", type=str, default="data/imagenet_tokens/")
    parser.add_argument("--stats_key", type=str, default=None)
    parser.add_argument("--overwrite_stats", action="store_true")
    parser.add_argument("--stats_cache_path", type=str, default="work_dirs/stats.pkl")

    # logging parameters
    parser.add_argument("--output_dir", default="./work_dirs")
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=40)
    parser.add_argument("--vis_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--last_elapsed_time", type=float, default=0.0)

    # checkpoint parameters
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--resume_from", default=None, help="resume model weights and optimizer state")
    parser.add_argument("--load_from", type=str, default=None, help="load from pretrained model")
    parser.add_argument("--load_tokenizer_from", type=str, default=None, help="load from pretrained tokenizer")
    parser.add_argument("--keep_n_ckpts", default=1, type=int, help="keep the last n checkpoints")
    parser.add_argument("--milestone_interval", default=100, type=int, help="keep checkpoints every n epochs")

    # evaluation parameters
    parser.add_argument("--num_images_for_eval_and_search", default=10000, type=int)
    parser.add_argument("--num_images", default=50000, type=int)
    parser.add_argument("--online_eval", action="store_true")
    parser.add_argument("--fid_stats_path", type=str, default="data/fid_stats/adm_in256_stats.npz")
    parser.add_argument("--keep_eval_folder", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval_bsz", type=int, default=256)

    # optimization parameters
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_sched", type=str, default="constant", choices=["constant", "cosine"])
    parser.add_argument("--warmup_rate", type=float, default=0.25, help="warmup_ep = warmup_rate * total_ep")
    parser.add_argument("--ema_rate", default=0.9999, type=float)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--grad_clip", type=float, default=3.0)
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--use_aligned_schedule", action="store_true")

    # generation parameters
    parser.add_argument("--num_iter", default=64, type=int, help="number of autoregressive steps for MAR")
    parser.add_argument("--noise_schedule", type=str, default="cosine", help="noise schedule for diffusion")
    parser.add_argument("--cfg", default=4.0, type=float, help="cfg value for diffusion")
    parser.add_argument("--cfg_schedule", default="linear", type=str, help="cfg schedule for diffusion")
    parser.add_argument("--cfg_list", default=None, type=float, nargs="+", help="cfg list for search")

    # mar parameters
    parser.add_argument("--label_drop_prob", default=0.1, type=float)
    parser.add_argument("--mask_ratio_min", type=float, default=0.7)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--proj_dropout", type=float, default=0.1)
    parser.add_argument("--buffer_size", type=int, default=64)

    # diffusion loss parameters
    parser.add_argument("--diffloss_d", type=int, default=3)
    parser.add_argument("--diffloss_w", type=int, default=1024)
    parser.add_argument("--num_sampling_steps", type=str, default="100")
    parser.add_argument("--diffusion_batch_mul", type=int, default=4)
    parser.add_argument("--temperature", default=1.0, type=float)

    # dataset parameters
    parser.add_argument("--use_cached_tokens", action="store_true")
    parser.add_argument("--data_path", default="./data/imagenet/train", type=str)
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--class_of_interest", default=[207, 360, 387, 974, 88, 979, 417, 279], type=int, nargs="+")
    parser.add_argument("--force_class_of_interest", action="store_true",
                        help="generate images of only the class of interest for args.num_images images")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # system parameters
    parser.add_argument("--seed", default=1, type=int)

    # wandb parameters
    parser.add_argument("--project", default="lDeTok", type=str)
    parser.add_argument("--entity", default="YOUR_WANDB_ENTITY", type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--enable_wandb", action="store_true")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    exit_code = main(args)
    sys.exit(exit_code)
