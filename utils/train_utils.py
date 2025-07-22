import argparse
import datetime
import json
import logging
import os
import pickle as pkl
import time
from typing import Any
from functools import partial

import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
import torch.utils.data
from torch import Tensor
import torch_fidelity
import torchvision
from PIL import Image, ImageFile
from torch.distributed import ReduceOp
from tqdm import tqdm, trange

import utils.distributed as dist
import utils.misc as misc
from utils.logger import MetricLogger, SmoothedValue, setup_logging, setup_wandb, WandbLogger

tqdm = partial(tqdm, dynamic_ncols=True)
ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger("DeTok")


def setup(args: argparse.Namespace):
    """setup distributed training, logging, and experiment configuration"""
    dist.enable_distributed()
    global logger

    if args.exp_name is None:
        args.exp_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}_exp"

    base_dir = os.path.join(args.output_dir, args.project, args.exp_name)
    args.log_dir = base_dir
    args.ckpt_dir = os.path.join(base_dir, "checkpoints")
    args.vis_dir = os.path.join(base_dir, "visualization")
    args.eval_dir = os.path.join(base_dir, "eval")

    global_rank, world_size = dist.get_global_rank(), dist.get_world_size()
    args.world_size = world_size
    args.global_bsz = args.batch_size * world_size
    args.print_freq = 100 if args.global_bsz < 512 else args.print_freq

    misc.fix_random_seeds(args.seed + global_rank)

    args.warmup_epochs = int(getattr(args, "warmup_rate", 0) * args.epochs)

    wandb_logger = None
    if global_rank == 0:
        for path in [args.log_dir, args.ckpt_dir, args.vis_dir, args.eval_dir]:
            os.makedirs(path, exist_ok=True)

        if args.enable_wandb:
            wandb_logger = setup_wandb(
                args=args,
                entity=args.entity,
                project=args.project,
                name=args.exp_name,
                log_dir=args.log_dir,
            )

        setup_logging(output=args.log_dir, name="DeTok", rank0_log_only=True)
        logger.info(f"Logging to {args.log_dir}")
        json_config = json.dumps(args.__dict__, indent=4, sort_keys=True)
        logger.info(json_config)

        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        json_path = os.path.join(args.log_dir, f"args_{time_str}.json")
        with open(json_path, "w") as f:
            json.dump(args.__dict__, f, indent=4)
        logger.info(f"Args saved to {json_path}")

    if getattr(args, "use_aligned_schedule", False):
        args.grad_clip = 0
        args.weight_decay = 0
        args.lr = 0.0002
        args.warmup_epochs = 0

    tokenizer = getattr(args, "tokenizer", None)
    if tokenizer:
        token_channels_map = {"vavae": 32, "maetok-b-128": 32, "sdvae": 4, "eqvae": 4}
        args.token_channels = token_channels_map.get(tokenizer, args.token_channels)
    return wandb_logger


def train_one_epoch_generator(
    args: argparse.Namespace,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_scaler: misc.NativeScalerWithGradNormCount,
    wandb_logger: WandbLogger | None,
    epoch: int,
    ema_model: torch.nn.Module,
    tokenizer: torch.nn.Module | None = None,
):
    model.train(True)
    metric_file = os.path.join(args.log_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metric_file, prefetch=True)
    metric_logger.add_meter("lr", SmoothedValue(1, "{value:.6f}"))
    metric_logger.add_meter("samples/s/gpu", SmoothedValue(args.print_freq, "{avg:.2f}"))
    steps_per_epoch = len(data_loader)
    header = f"Epoch: [{epoch}]"
    logger.info(f"log dir: {args.log_dir}")
    start_time = time.perf_counter()

    for step, data_dict in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        # calibrate 1 epoch = 1000 iterations regardless of batch size
        frac_epoch = step / steps_per_epoch + epoch  # fraction of the current epoch
        calib_global_step = int(frac_epoch * 1000)
        tokenization_time = 0.0

        if args.use_cached_tokens:
            # load posterior moments and sample
            moments, labels = data_dict["token"], data_dict["label"]
            x = tokenizer.sample_from_moments(moments)

        elif args.tokenizer is not None:
            # online tokenization
            imgs, labels = data_dict["img"], data_dict["label"]
            # tokenization time estimate is not strictly accurate, but it's a good approximation
            tokenizer_start_time = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                x = tokenizer.tokenize(imgs, sampling=True)
            tokenization_time = time.perf_counter() - tokenizer_start_time

        else:
            # pixel-space inputs, good luck : )
            x, labels = data_dict["img"], data_dict["label"]

        misc.adjust_learning_rate(optimizer, frac_epoch, args)

        # forward pass
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, labels)
            loss_value = loss.item()

        # backward pass
        grad_norm = loss_scaler(loss, optimizer, args.grad_clip, model.parameters())
        optimizer.zero_grad(set_to_none=True)

        # update ema model
        ema_model.step(model)

        torch.cuda.synchronize()

        # log metrics
        loss_value_reduced = dist.all_reduce_mean(loss_value)
        psnr = -10 * np.log10(loss_value_reduced)
        samples_per_second_per_gpu = args.batch_size * (step + 1) / (time.perf_counter() - start_time)
        samples_per_second = samples_per_second_per_gpu * args.world_size
        metric_logger.update(
            loss=loss_value_reduced,
            psnr=psnr,
            grad_norm=grad_norm,
            lr=optimizer.param_groups[0]["lr"],
            tokenization=tokenization_time,
            **{"samples/s/gpu": samples_per_second_per_gpu, "samples/s": samples_per_second},
        )
        if wandb_logger is not None and step % args.print_freq == 0:
            log_dict = {
                "psnr": psnr,
                "loss": loss_value_reduced,
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": grad_norm,
                "tokenization": tokenization_time,
                "samples_per_sec_per_gpu": samples_per_second_per_gpu,
                "samples_per_sec": samples_per_second,
            }
            wandb_logger.update(log_dict, step=calib_global_step)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_tokenizer(
    args: argparse.Namespace,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_scaler: misc.NativeScalerWithGradNormCount,
    wandb_logger: WandbLogger | None,
    epoch: int,
    ema_model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    discriminator_optimizer: torch.optim.Optimizer,
    discriminator_loss_scaler: misc.NativeScalerWithGradNormCount,
):
    model.train(True)
    metric_file = os.path.join(args.log_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metric_file, prefetch=True)
    metric_logger.add_meter("lr", SmoothedValue(1, "{value:.6f}"))
    metric_logger.add_meter("samples/s/gpu", SmoothedValue(args.print_freq, "{avg:.2f}"))
    steps_per_epoch = len(data_loader)
    header = f"Epoch: [{epoch}]"
    logger.info(f"log dir: {args.log_dir}")
    start_time = time.perf_counter()

    for step, data_dict in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        # calibrate 1 epoch = 1000 iterations regardless of batch size
        frac_epoch = step / steps_per_epoch + epoch  # fraction of the current epoch
        calib_global_step = int(frac_epoch * 1000)
        x = data_dict["img"]

        optimizer.zero_grad(set_to_none=True)
        discriminator_optimizer.zero_grad(set_to_none=True)

        # Adjust learning rates
        misc.adjust_learning_rate(optimizer, frac_epoch, args)
        misc.adjust_learning_rate(discriminator_optimizer, frac_epoch, args)

        # Forward pass and generator loss
        with torch.autocast("cuda", dtype=torch.bfloat16):
            results = model(x)
            reconstructions, posteriors = results
            # Normalize inputs to [0, 1] range for loss function
            targets = x * 0.5 + 0.5
            reconstructions = reconstructions * 0.5 + 0.5
            ae_loss, loss_dict = loss_fn(targets, reconstructions, posteriors, epoch, "generator")

            # Process loss dictionary
            autoencoder_logs = {}
            for k, v in loss_dict.items():
                if k in ["discriminator_factor", "d_weight"]:
                    autoencoder_logs[k] = v.cpu().item() if isinstance(v, Tensor) else v
                else:
                    autoencoder_logs[k] = dist.all_reduce_mean(v)

            loss = ae_loss
            loss_dict.update(autoencoder_logs)

        # backward pass for generator
        grad_norm = loss_scaler(loss, optimizer, args.grad_clip, model.parameters())

        # update ema model
        ema_model.step(model)

        # train discriminator if needed
        discriminator_logs = {}
        if epoch >= args.discriminator_start_epoch:
            # this loss module assumes that both x and reconstructed are in [0, 1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                discriminator_loss, loss_dict_discriminator = loss_fn(
                    targets, reconstructions, posteriors, epoch, mode="discriminator"
                )

            # Gather the losses across all processes for logging
            for k, v in loss_dict_discriminator.items():
                if k in ["logits_real", "logits_fake"]:
                    discriminator_logs[k] = v.cpu().item() if isinstance(v, Tensor) else v
                else:
                    discriminator_logs[k] = dist.all_reduce_mean(v)

            loss_dict.update(discriminator_logs)

            discriminator_grad_norm = discriminator_loss_scaler(
                discriminator_loss,
                discriminator_optimizer,
                args.grad_clip,
                loss_fn.parameters(),
            )
        else:
            discriminator_grad_norm = 0.0

        # Synchronize and log metrics
        torch.cuda.synchronize()
        loss_dict_reduced = {k: dist.all_reduce_mean(v) for k, v in loss_dict.items()}
        loss_dict_reduced.pop("total_loss", None)
        total_loss_reduced = sum(loss for k, loss in loss_dict_reduced.items() if "loss" in k)

        # Update metrics
        samples_per_second_per_gpu = args.batch_size * (step + 1) / (time.perf_counter() - start_time)
        samples_per_second = samples_per_second_per_gpu * args.world_size

        metric_logger.update(
            loss=total_loss_reduced,
            grad_norm=grad_norm,
            discriminator_grad_norm=discriminator_grad_norm,
            lr=optimizer.param_groups[0]["lr"],
            **loss_dict_reduced,
            **{"samples/s/gpu": samples_per_second_per_gpu, "samples/s": samples_per_second},
        )

        # Log to writer
        if wandb_logger is not None and step % args.print_freq == 0:
            log_dict = {
                "loss": total_loss_reduced,
                **loss_dict_reduced,
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": grad_norm,
                "discriminator_grad_norm": discriminator_grad_norm,
                "samples_per_sec_per_gpu": samples_per_second_per_gpu,
                "samples_per_sec": samples_per_second,
            }
            wandb_logger.update(log_dict, step=calib_global_step)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_img_save_format(grid, max_pixels=2_000_000):
    grid_height, grid_width = grid.shape[-2:]
    total_pixels = grid_height * grid_width
    return "jpg" if total_pixels > max_pixels else "png"


@torch.inference_mode()
def to_uint8_numpy(tensor: Tensor) -> np.ndarray:
    return (tensor * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

@torch.inference_mode()
def visualize_generator(
    args: argparse.Namespace,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    tokenizer: torch.nn.Module,
    epoch: int,
    use_emas: list[bool] = [True],
):
    model.eval()
    if args.class_of_interest is not None:
        assert all(0 <= c < args.num_classes for c in args.class_of_interest)
        class_labels = torch.tensor(args.class_of_interest, device="cuda", dtype=torch.long)
    else:
        class_labels = torch.randint(args.num_classes, (8,), device="cuda")

    n_samples = len(class_labels)

    for use_ema in use_emas:
        if use_ema:
            ema_model.store(model)
            ema_model.copy_to(model)

        for cfg in [args.cfg, 1.0]:
            logger.info(f"Generating images with cfg={cfg}, n_imgs={n_samples}, ema={use_ema}")
            generated_images = generate_images(args, model, tokenizer, labels=class_labels, cfg=cfg)
            generated_images = dist.concat_all_gather(generated_images).cpu()

            if dist.is_main_process():
                grid = torchvision.utils.make_grid(generated_images, n_samples, 8, pad_value=1)
                format = get_img_save_format(grid)
                outpath = os.path.join(args.vis_dir, f"ep{epoch:04d}_cfg={cfg}_ema={use_ema}.{format}")
                torchvision.utils.save_image(grid, outpath)
                logger.info(f"Saved at {outpath}")

            torch.distributed.barrier()
            torch.cuda.empty_cache()

        if use_ema:
            ema_model.restore(model)

    torch.distributed.barrier()
    torch.cuda.empty_cache()


@torch.inference_mode()
def visualize_tokenizer(
    args: argparse.Namespace,
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
    data_dict: dict[str, Tensor],
    epoch: int = 0,
    split: str = "val",
    use_emas: list[bool] = [True],
):
    world_size = dist.get_world_size()
    if world_size <= 8:
        n_vis = 128 // world_size
    else:
        n_vis = 16 // world_size
    n_vis = max(n_vis, 1)
    if "img" not in data_dict:
        return
    images = data_dict["img"][:n_vis].cuda()
    model.eval()
    for use_ema in use_emas:
        if use_ema and ema_model is not None:
            ema_model.store(model)
            ema_model.copy_to(model)
        logger.info(f"Autoencoding images with ema={use_ema}, n_imgs={len(images)}")
        tokens = model.tokenize(images)
        reconstructed_images = model.detokenize(tokens)
        reconstructed_images = dist.concat_all_gather(reconstructed_images).cpu()
        original_images = images * 0.5 + 0.5
        original_images = dist.concat_all_gather(original_images).cpu()
        # interleave original and reconstructed images
        if dist.is_main_process():
            to_zip = [original_images]
            to_zip.append(reconstructed_images)
            interleaved_images = torch.cat(
                [torch.stack(tensors, dim=0) for tensors in zip(*to_zip)],
                dim=0,
            ).view(-1, *original_images.shape[1:])
            row_mult = 1 if len(to_zip) >= 8 else 4
            grid = torchvision.utils.make_grid(
                interleaved_images, nrow=len(to_zip) * row_mult, padding=8, pad_value=1
            )
            outpath = os.path.join(args.vis_dir, f"ep{epoch:04d}_ema={use_ema}_{split}.jpg")
            torchvision.utils.save_image(grid, outpath)
            logger.info(f"Saved visualization at {outpath}")

        torch.distributed.barrier()
        torch.cuda.empty_cache()

        if use_ema and ema_model is not None:
            ema_model.restore(model)


@torch.inference_mode()
def generate_images(
    args: argparse.Namespace,
    generator: torch.nn.Module,
    tokenizer: torch.nn.Module | None,
    labels: list[int] | Tensor,
    cfg: float = 1.0,
):
    if not isinstance(labels, Tensor):
        labels = torch.tensor(labels, dtype=torch.long).to("cuda")
    generator = generator.eval().to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        generated = generator.generate(n_samples=len(labels), cfg=cfg, labels=labels, args=args)
        if tokenizer is not None:
            generated = tokenizer.detokenize(generated)
    return generated


def get_start_end_indices(total_samples, num_processes, rank):
    """compute the start and end indices for each rank to distribute work evenly"""
    # calculate base number of samples per process
    base = total_samples // num_processes
    # handle remainder samples that need to be distributed
    remainder = total_samples % num_processes
    
    # ranks with index < remainder get one extra sample
    if rank < remainder:
        start_idx = rank * (base + 1)
        end_idx = start_idx + base + 1
    else:
        # remaining ranks get the base number of samples
        start_idx = rank * base + remainder
        end_idx = start_idx + base
    return start_idx, end_idx


@torch.inference_mode()
def evaluate_generator(
    args: argparse.Namespace,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    tokenizer: torch.nn.Module | None,
    epoch: int,
    wandb_logger: WandbLogger | None,
    use_ema: bool = True,
    cfg: float | None = None,
    num_images: int = 50000,
):
    model.eval()
    if tokenizer is not None:
        tokenizer.eval()
    cfg = cfg or args.cfg  # use the cfg from the args if not provided
    eval_dir = f"{args.eval_dir}/epoch_{epoch:03d}_use_ema={use_ema}-cfg={cfg}"
    eval_start_time = time.perf_counter()
    world_size, rank = dist.get_world_size(), dist.get_global_rank()
    per_gpu_bsz = args.eval_bsz
    device = torch.device("cuda")
    if rank == 0:
        os.makedirs(eval_dir, exist_ok=True)

    # get the start and end indices for this rank
    start_idx, end_idx = get_start_end_indices(num_images, world_size, rank)
    samples_per_gpu = end_idx - start_idx

    # calculate the number of batches needed
    n_batches = (samples_per_gpu + per_gpu_bsz - 1) // per_gpu_bsz

    if use_ema:
        logger.info("Using EMA model for evaluation.")
        ema_model.store(model)
        ema_model.copy_to(model)

    # prepare for image generation
    if args.force_class_of_interest:
        all_classes = args.class_of_interest
        num_classes = len(all_classes)
    else:
        all_classes = list(range(args.num_classes))
        num_classes = args.num_classes
    num_repeats = (num_images + num_classes - 1) // num_classes
    all_classes = (all_classes * num_repeats)[: num_images]
    all_classes = np.array(all_classes, dtype=np.int64)

    rank_classes = all_classes[start_idx:end_idx]

    logger.info("Generating images for evaluation...")
    logger.info(f"{world_size=}, {rank=}, {samples_per_gpu=}, {n_batches=}, {per_gpu_bsz=}")
    n_sampling_steps = args.num_sampling_steps
    temperature = args.temperature
    num_iter = args.num_iter
    logger.info(
        f"Setting: {use_ema=}, {cfg=}, {n_sampling_steps=}, {num_iter=} {num_images=}, {temperature=}"
    )
    gen_time, save_time, gen_cnt = 0, 0, 0
    gen_start = time.perf_counter()

    for cur_idx in trange(n_batches, desc=f"Rank{rank}", position=rank):
        # get the start and end indices for this batch
        batch_start = cur_idx * per_gpu_bsz
        batch_end = min(batch_start + per_gpu_bsz, samples_per_gpu)
        y = torch.from_numpy(rank_classes[batch_start:batch_end]).long().to(device)

        # Generate samples
        start_time = time.perf_counter()
        samples = generate_images(args, model, tokenizer, labels=y, cfg=cfg)
        gen_time += time.perf_counter() - start_time
        gen_cnt += len(samples)
        samples = to_uint8_numpy(samples)

        img_per_gpu_per_sec = gen_cnt / gen_time
        elapsed_time = time.perf_counter() - gen_start
        eta = elapsed_time / (cur_idx + 1) * (n_batches - cur_idx - 1)
        logger.info(
            f"[{cur_idx+1}/{n_batches}] Generated {gen_cnt} images in {gen_time:.2f}s. "
            f"Images per second per gpu: {img_per_gpu_per_sec:.4f}. "
            f"Seconds per image: {gen_time / gen_cnt:.4f}. "
            f"Elapsed time: {str(datetime.timedelta(seconds=elapsed_time))} "
            f"ETA (save time included): {str(datetime.timedelta(seconds=eta))}"
        )
        logger.info(f"FIDs will be logged to {args.log_dir}/eval_summary.txt")

        # save generated images
        start_time = time.perf_counter()
        for i, sample in enumerate(samples):
            global_index = start_idx + batch_start + i
            Image.fromarray(sample).save(f"{eval_dir}/{global_index:06d}.png")
        save_time += time.perf_counter() - start_time
        del samples
        torch.cuda.empty_cache()

    # synchronize across processes
    torch.distributed.barrier()
    gen_time_str = str(datetime.timedelta(seconds=gen_time))
    save_time_str = str(datetime.timedelta(seconds=save_time))
    img_per_gpu_per_sec = samples_per_gpu / gen_time if gen_time > 0 else 0
    sec_per_img = gen_time / samples_per_gpu if samples_per_gpu > 0 else 0
    logger.info(
        f"Generation finishes. "
        f"Gen time: {gen_time_str}, Save time: {save_time_str}, "
        f"Images per GPU per second: {img_per_gpu_per_sec:.4f}, "
        f"Seconds per image: {sec_per_img:.4f}, "
        f"Images per second: {img_per_gpu_per_sec * world_size:.4f}"
    )
    if rank == 0:
        num_imgs = len(os.listdir(eval_dir))
        # sanity check to make sure the number of images is correct
        logger.info(f"Final number of images: {num_imgs}")

    # restore EMA parameters
    if use_ema:
        ema_model.restore(model)

    torch.distributed.barrier()
    if rank == 0:
        metrics_dict = evaluate_FID(eval_dir, None, fid_stats_path=args.fid_stats_path)
        fid = metrics_dict["frechet_inception_distance"]
        inception_score = metrics_dict["inception_score_mean"]
        if wandb_logger is not None:
            log_dict = {
                f"eval/FID_ema={use_ema}-nimgs={num_imgs}-cfg={cfg}": fid,
                f"eval/IS_ema={use_ema}-nimgs={num_imgs}-cfg={cfg}": inception_score,
                f"eval/Img_per_sec_per_gpu_ema={use_ema}-nimgs={num_imgs}-cfg={cfg}": img_per_gpu_per_sec,
                f"eval/Sec_per_img_ema={use_ema}-nimgs={num_imgs}-cfg={cfg}": sec_per_img,
            }
            wandb_logger.update(log_dict, step=epoch * 1000)
            logger.info(f"Logged evaluation metrics: {log_dict}")
        log_str = f"Epoch {epoch}, {use_ema=}, {cfg=}, num_iter={num_iter}, temperature={temperature}, num_sampling_steps={n_sampling_steps}, {num_imgs=}, fid={fid}, is={inception_score}"
        with open(f"{args.log_dir}/eval_summary.txt", "a") as f:
            f.write(log_str + "\n")

    # ensure evaluation is done before cleanup
    torch.distributed.barrier()

    # distributed cleanup
    if not args.keep_eval_folder:
        start_time = time.perf_counter()
        # each GPU removes only its own files
        subset_files = [f"{eval_dir}/{index:06d}.png" for index in range(start_idx, end_idx)]
        for file_path in subset_files:
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass

        # ensure all processes wait here before proceeding
        torch.distributed.barrier()

        # rank 0 removes the directories if they are empty
        if rank == 0:
            if not os.listdir(eval_dir):
                os.rmdir(eval_dir)
            logger.info(f"Removed evaluation folder: {eval_dir}")
        logger.info(f"Cleanup time: {time.perf_counter() - start_time:.2f}s")

    # ensure all processes wait here before proceeding
    torch.distributed.barrier()
    torch.cuda.empty_cache()
    time_str = str(datetime.timedelta(seconds=time.perf_counter() - eval_start_time))
    logger.info(f"Total evaluation time (gen+save+cleanup): {time_str}")
    logger.info(f"Results saved in {args.log_dir}/eval_summary.txt")
    return {"fid": fid, "is": inception_score} if rank == 0 else None


@torch.inference_mode()
def evaluate_tokenizer(
    args: argparse.Namespace,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    data_loader_val: torch.utils.data.DataLoader,
    epoch: int = 0,
    wandb_logger: WandbLogger | None = None,
    use_ema: bool = True,
):
    """
    Evaluates the tokenizer (or the reconstruction capability of the model) by:
      1. Reconstructing images
      2. Computing PSNR in PyTorch
      3. Saving reconstructed images as PNG
      4. Gathering and logging metrics (PSNR, FID, IS, etc.)
    """
    model.eval()

    eval_dir = f"{args.eval_dir}/epoch_{epoch:03d}_use_ema={use_ema}"
    eval_start_time = time.perf_counter()
    world_size, rank = dist.get_world_size(), dist.get_global_rank()
    per_gpu_bsz = args.eval_bsz
    n_batches = len(data_loader_val)
    device = torch.device("cuda")

    os.makedirs(eval_dir, exist_ok=True)  # risky though, there's a race condition here
    logger.info(f"Created evaluation directory: {eval_dir}")

    torch.distributed.barrier()
    torch.cuda.empty_cache()

    samples_per_gpu = per_gpu_bsz * n_batches

    if use_ema:
        logger.info("Using EMA model for evaluation.")
        ema_model.store(model)
        ema_model.copy_to(model)

    logger.info(f"Reconstructing images for evaluation, EMA={use_ema}")
    logger.info(f"World size: {world_size}, Rank: {rank}, Batches: {n_batches}, Bsz: {per_gpu_bsz}")

    recon_time, save_time, cnt = 0, 0, 0
    psnr_values_local, img_ids_local = [], []

    recon_start = time.perf_counter()

    for cur_idx, data_dict in tqdm(
        enumerate(data_loader_val), total=n_batches, desc=f"Rank{rank}", position=rank
    ):
        img_ids = data_dict["index"]
        images = data_dict["img"].to(device)

        # reconstruct images as float tensors in [0,1], shape [B, C, H, W]
        start_time = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reconstructed_tensor = model.reconstruct(images)
        recon_time += time.perf_counter() - start_time

        # count how many images we've processed so far
        batch_size = reconstructed_tensor.size(0)
        cnt += batch_size

        # --------------------------------------------------------------
        # Compute PSNR using the newly returned float tensors in [0..1]
        # --------------------------------------------------------------
        cur_psnr = compute_psnr_torch_batch(images * 0.5 + 0.5, reconstructed_tensor, data_range=1.0)
        psnr_values_local.extend(cur_psnr.cpu().tolist())
        img_ids_local.extend(img_ids.cpu().tolist())

        img_per_gpu_per_sec = cnt / recon_time if recon_time > 0 else 0
        elapsed_time = time.perf_counter() - recon_start
        eta = elapsed_time / (cur_idx + 1) * (n_batches - cur_idx - 1)
        logger.info(
            f"[{cur_idx+1}/{n_batches}] Reconstructed {cnt} images in {recon_time:.2f}s. "
            f"Running PSNR: {cur_psnr.mean().item():.4f}. "
            f"Images/sec/gpu: {img_per_gpu_per_sec:.4f}. "
            f"Sec/img: {recon_time / cnt:.4f}. "
            f"Elapsed: {str(datetime.timedelta(seconds=elapsed_time))}, "
            f"ETA: {str(datetime.timedelta(seconds=eta))}"
        )

        # --------------------------------------------------------------
        # Save the reconstructed images as PNG in [0..255]
        # Convert from [B, C, H, W] float in [0..1] to uint8 CPU for PIL
        # --------------------------------------------------------------
        start_time = time.perf_counter()
        reconstructed_uint8 = to_uint8_numpy(reconstructed_tensor)

        for i, sample_np in enumerate(reconstructed_uint8):
            global_index = img_ids[i].item()
            Image.fromarray(sample_np).save(f"{eval_dir}/{global_index:06d}.png")
        # save gt
        # gt_images = data_dict["img"]
        # gt_images = gt_images * 0.5 + 0.5
        # gt_images = (gt_images * 255.0).clamp(0, 255).to(torch.uint8)
        # gt_images = gt_images.permute(0, 2, 3, 1).cpu().numpy()
        # os.makedirs("data/imagenet/gt-image50000", exist_ok=True)
        # for i, sample_np in enumerate(gt_images):
        #     global_index = img_ids[i].item()
        # Image.fromarray(sample_np).save(f"data/imagenet/gt-image50000/{global_index:06d}.png")

        save_time += time.perf_counter() - start_time

        del reconstructed_tensor, reconstructed_uint8
        torch.cuda.empty_cache()

    # --------------------------------------------------------------
    # Synchronize across processes before collecting final metrics
    # --------------------------------------------------------------
    torch.distributed.barrier()

    recon_time_str = str(datetime.timedelta(seconds=recon_time))
    save_time_str = str(datetime.timedelta(seconds=save_time))
    img_per_gpu_per_sec = samples_per_gpu / recon_time if recon_time > 0 else 0
    sec_per_img = recon_time / samples_per_gpu if samples_per_gpu > 0 else 0

    logger.info(
        f"Reconstruction finishes. Recon time: {recon_time_str}, Save time: {save_time_str}, "
        f"Images per GPU per second: {img_per_gpu_per_sec:.4f}, "
        f"Seconds per image: {sec_per_img:.4f}"
    )

    if rank == 0:
        num_imgs = len(os.listdir(eval_dir))
        logger.info(f"Final number of images: {num_imgs}")

    # ----------------------------------------------------------------
    # 1) Convert the local PSNR list to a Tensor on device
    # 2) Use concat_all_gather(...) to gather
    # 3) On rank=0, compute final mean
    # ----------------------------------------------------------------
    psnr_values_local_tensor = torch.tensor(psnr_values_local, device=device, dtype=torch.float32)
    psnr_gathered_tensor = dist.concat_all_gather(psnr_values_local_tensor, gather_dim=0)

    if rank == 0:
        # psnr_gathered_tensor now contains the concatenated PSNR values from all ranks
        mean_psnr = psnr_gathered_tensor.mean().item()
        logger.info(f"Average PSNR (all ranks): {mean_psnr:.4f}")
    else:
        mean_psnr = 0.0

    # Restore EMA parameters
    if use_ema:
        ema_model.restore(model)

    torch.distributed.barrier()

    # Evaluate FID
    if rank == 0:
        metrics_dict = evaluate_FID(eval_dir, fid_stats_path=args.fid_stats_path)
        fid = metrics_dict["frechet_inception_distance"]
        inception_score = metrics_dict["inception_score_mean"]
        if wandb_logger is not None:
            log_dict = {
                f"eval/rFID_ema={use_ema}-nimgs={num_imgs}": fid,
                f"eval/rPSNR_ema={use_ema}-nimgs={num_imgs}": mean_psnr,
                f"eval/Img_per_sec_per_gpu_ema={use_ema}-nimgs={num_imgs}": img_per_gpu_per_sec,
                f"eval/Sec_per_img_ema={use_ema}-nimgs={num_imgs}": sec_per_img,
                f"eval/IS_ema={use_ema}-nimgs={num_imgs}": inception_score,
            }
            wandb_logger.update(log_dict, step=epoch * 1000)
            logger.info(f"Logged evaluation metrics: {log_dict}")
        log_str = f"Epoch {epoch}, {use_ema=}, {num_imgs=}, fid={fid}, psnr={mean_psnr}, is={inception_score}, img_per_gpu_per_sec={img_per_gpu_per_sec}, sec_per_img={sec_per_img}"
        with open(f"{args.log_dir}/eval_summary.txt", "a") as f:
            f.write(log_str + "\n")
    torch.distributed.barrier()

    # Cleanup if needed
    if not args.keep_eval_folder:
        start_time = time.perf_counter()
        subset_files = [f"{eval_dir}/{index:06d}.png" for index in img_ids_local]
        for file_path in subset_files:
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass

        # Ensure all processes wait here before proceeding
        torch.distributed.barrier()

        # Rank 0 removes the directories if they are empty
        if rank == 0:
            if not os.listdir(eval_dir):
                os.rmdir(eval_dir)
            logger.info("Removed evaluation folders.")
        logger.info(f"Cleanup time: {time.perf_counter() - start_time:.2f}s")

    torch.distributed.barrier()
    torch.cuda.empty_cache()

    time_str = str(datetime.timedelta(seconds=time.perf_counter() - eval_start_time))
    logger.info(f"Total evaluation time: {time_str}")
    logger.info(f"Results saved in {args.log_dir}/eval_summary.txt")


@torch.inference_mode()
def compute_psnr_torch_batch(original: Tensor, recon: Tensor, data_range: float = 1.0) -> Tensor:
    """computes psnr for a batch of images using pytorch operations."""
    mse_per_sample = F.mse_loss(original, recon, reduction="none").mean(dim=[1, 2, 3])
    psnr_per_sample = 10.0 * torch.log10(data_range**2 / mse_per_sample)
    return psnr_per_sample


@torch.inference_mode()
def evaluate_FID(
    save_folder: str,
    reference_folder: str | None = None,
    prc: bool = False,
    fid_stats_path: str | None = None,
):
    logger.info(f"Calculating FID for {save_folder}...")
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=save_folder,
        input2=reference_folder,
        fid_statistics_file=fid_stats_path,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=prc,
        verbose=True,
    )
    fid = metrics_dict["frechet_inception_distance"]
    inception_score = metrics_dict["inception_score_mean"]
    logger.info(f"Folder: {save_folder}")
    logger.info(f"Metrics: {metrics_dict}")
    logger.info(f"FID: {fid:.4f}, IS: {inception_score:.4f}")
    return metrics_dict

@torch.inference_mode()
def collect_tokenizer_stats(
    tokenizer: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader[dict[str, Any]],
    stats_dict_key: str,
    stats_dict_path: str = "work_dirs/stats.pkl",
    overwrite_stats: bool = False,
    chan_dim: int = 1,
):
    """
    computes global statistics over latent representations in two ways:
      1. single-digit (overall) statistics: scalar mean and std over all elements
      2. channel-wise statistics: mean and std per channel

    args:
        tokenizer: model that encodes samples into latent representations
        data_loader: yields batches containing at least "img" key
        stats_dict_key: identifier for caching
        stats_dict_path: path to cache file
        overwrite_stats: whether to recompute even if cached
        chan_dim: channel dimension (1 for [B,C,H,W], 2 for [B,seq_len,C])

    returns:
        dict with "single" and "channel" keys containing (mean, std) tuples
    """
    tokenizer.eval()

    if chan_dim not in [1, 2]:
        raise ValueError(
            f"Unsupported chan_dim value: {chan_dim}. "
            f"Supported values: 1 for [B, C, H, W], 2 for [B, seq_len, C]"
        )
    
    if not overwrite_stats and os.path.exists(stats_dict_path):
        try:
            with open(stats_dict_path, "rb") as f:
                cached_stats = pkl.load(f)
            if stats_dict_key in cached_stats:
                logger.info(f"using cached stats for tokenizer: {stats_dict_key}")
                return cached_stats[stats_dict_key]
        except Exception as e:
            logger.warning(f"failed to load cached stats: {e}")

    logger.info(f"computing fresh statistics for tokenizer: {stats_dict_key}")
    start_time = time.perf_counter()

    # statistics accumulators
    total_sum = total_sum_sq = None
    total_count = 0
    channel_sum = channel_sum_sq = None
    channel_count = 0

    metric_logger = MetricLogger(delimiter="  ", prefetch=True)
    
    for batch in metric_logger.log_every(data_loader, 20, "computing stats: "):
        samples = batch["img"]

        # encode samples - handle different tokenizer interfaces
        if hasattr(tokenizer, "encode_into_posteriors"):
            # e.g. shape: [B, 2C, H, W] or [B, seq_len, 2C]
            #########################################################
            # moments is a concatenation of mean and std, so the channel dimension is doubled
            #########################################################
            moments = tokenizer.encode_into_posteriors(samples)
            if hasattr(moments, "parameters"):
                moments = moments.parameters
        elif hasattr(tokenizer, "encode"):
            moments = tokenizer.encode(samples)
        else:
            raise AttributeError("tokenizer must have 'encode_into_posteriors' or 'encode' method")

        device, dtype = moments.device, moments.dtype

        # initialize accumulators on first batch
        if total_sum is None:
            total_sum = torch.tensor(0.0, device=device, dtype=dtype)
            total_sum_sq = torch.tensor(0.0, device=device, dtype=dtype)

        # update statistics based on channel dimension
        if chan_dim == 1:  # [B, 2C, H, W]
            num_channels = moments.size(1) // 2
            relevant_moments = moments[:, :num_channels]
            
            # overall stats
            total_sum += relevant_moments.sum()
            total_sum_sq += (relevant_moments**2).sum()
            total_count += relevant_moments.numel()

            # channel-wise stats
            if channel_sum is None:
                c = moments.size(1)
                channel_sum = torch.zeros(c, device=device, dtype=dtype)
                channel_sum_sq = torch.zeros(c, device=device, dtype=dtype)

            channel_sum += moments.sum(dim=[0, 2, 3])
            channel_sum_sq += (moments**2).sum(dim=[0, 2, 3])
            channel_count += moments.size(0) * moments.size(2) * moments.size(3)

        else:  # chan_dim == 2, [B, seq_len, C]
            num_channels = moments.size(-1) // 2
            relevant_moments = moments[..., :num_channels]
            
            # overall stats
            total_sum += relevant_moments.sum()
            total_sum_sq += (relevant_moments**2).sum()
            total_count += relevant_moments.numel()

            # channel-wise stats
            if channel_sum is None:
                c = moments.size(-1)
                channel_sum = torch.zeros(c, device=device, dtype=dtype)
                channel_sum_sq = torch.zeros(c, device=device, dtype=dtype)

            channel_sum += moments.sum(dim=[0, 1])
            channel_sum_sq += (moments**2).sum(dim=[0, 1])
            channel_count += moments.size(0) * moments.size(1)

        # periodic logging
        if total_count > 0 and total_count % 10000 == 0:
            current_mean = total_sum / total_count
            current_std = ((total_sum_sq / total_count) - current_mean**2).sqrt()
            logger.info(f"processed {total_count:,} elements | mean: {current_mean:.6f}, std: {current_std:.6f}")

    torch.distributed.barrier()

    if total_sum is None:
        logger.error("no valid batches processed")
        return {"single": (None, None), "channel": (None, None)}

    # reduce across processes if distributed
    counts = [torch.tensor(total_count, device=total_sum.device, dtype=torch.long),
              torch.tensor(channel_count, device=channel_sum.device, dtype=torch.long)]
    
    if torch.distributed.get_world_size() > 1:
        for tensor in [total_sum, total_sum_sq, channel_sum, channel_sum_sq] + counts:
            torch.distributed.all_reduce(tensor, op=ReduceOp.SUM)

    global_total_count, global_channel_count = counts[0].item(), counts[1].item()

    # compute final statistics
    def compute_stats(sum_val, sum_sq_val, count):
        if count > 0:
            mean = sum_val / count
            std = ((sum_sq_val / count) - mean**2).sqrt()
            return mean, std
        return None, None

    global_mean_single, global_std_single = compute_stats(total_sum, total_sum_sq, global_total_count)
    global_mean_channel, global_std_channel = compute_stats(channel_sum, channel_sum_sq, global_channel_count)

    global_stats = {
        "single": (global_mean_single, global_std_single),
        "channel": (global_mean_channel, global_std_channel),
    }

    # log results
    computation_time = str(datetime.timedelta(seconds=int(time.perf_counter() - start_time)))
    logger.info(f"statistics computation time: {computation_time}")
    
    if global_mean_single is not None:
        logger.info(f"global stats | mean: {global_mean_single:.6f}, std: {global_std_single:.6f}")
        logger.info(f"channel stats | mean avg: {global_mean_channel[:num_channels].mean():.6f}, "
                   f"std avg: {global_std_channel[:num_channels].mean():.6f}")

    # cache results (main process only)
    if dist.is_main_process():
        try:
            cached_stats = {}
            if os.path.exists(stats_dict_path):
                with open(stats_dict_path, "rb") as f:
                    cached_stats = pkl.load(f)
            else:
                os.makedirs(os.path.dirname(stats_dict_path), exist_ok=True)

            cached_stats[stats_dict_key] = global_stats
            with open(stats_dict_path, "wb") as f:
                pkl.dump(cached_stats, f)
            logger.info(f"cached statistics to {stats_dict_path}")
        except Exception as e:
            logger.error(f"failed to cache statistics: {e}")

    return global_stats
