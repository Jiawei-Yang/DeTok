import logging

import torch.utils.data
import torchvision.transforms as transforms

import models
import utils.distributed as distributed
import utils.losses as losses
from utils.loader import ListDataset, center_crop_arr
from utils.misc import NativeScalerWithGradNormCount

logger = logging.getLogger("DeTok")


def create_train_dataloader(args, should_flip=True, batch_size=-1, return_path=False, drop_last=True):
    transform_train = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    input_transform = transform_train if not args.use_cached_tokens else None
    dataset_train = ListDataset(
        args.data_path,
        data_list="data/train.txt",
        transform=input_transform,
        loader_name="img_loader" if not args.use_cached_tokens else "npz_loader",
        return_label=True,
        return_path=return_path,
        should_flip=should_flip,
    )
    logger.info(f"Train dataset size: {len(dataset_train)}")

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=distributed.get_world_size(),
        rank=distributed.get_global_rank(),
        shuffle=True,
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size if batch_size < 0 else batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=drop_last,
    )
    return data_loader_train


def create_val_dataloader(args):
    transform_val = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset_val = ListDataset(
        args.data_path.replace("train", "val"),
        data_list="data/val.txt",
        transform=transform_val,
        loader_name="img_loader",
        return_label=False,
        return_index=True,
        should_flip=False,
    )
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val,
        num_replicas=distributed.get_world_size(),
        rank=distributed.get_global_rank(),
        shuffle=False,
    )

    logger.info(f"Val dataset size: {len(dataset_val)}")

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.eval_bsz,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return data_loader_val


def create_vis_dataloader(args):
    transform_val = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset_vis = ListDataset(
        args.data_path,
        data_list="data/train.txt",
        transform=transform_val,
        loader_name="img_loader",
        return_label=False,
        return_index=True,
        class_of_interest=args.class_of_interest,
    )
    sampler_vis = torch.utils.data.DistributedSampler(
        dataset_vis,
        num_replicas=distributed.get_world_size(),
        rank=distributed.get_global_rank(),
        shuffle=True,
    )

    logger.info(f"Vis dataset size: {len(dataset_vis)}")

    data_loader_vis = torch.utils.data.DataLoader(
        dataset_vis,
        sampler=sampler_vis,
        batch_size=8,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return data_loader_vis


def create_generation_model(args):
    logger.info("Creating generation models.")
    if args.tokenizer is not None:
        if args.tokenizer in models.VAE_models:
            tokenizer = models.VAE_models[args.tokenizer]()
        elif args.tokenizer in models.DeTok_models:
            tokenizer = models.DeTok_models[args.tokenizer](
                img_size=args.img_size,
                patch_size=args.tokenizer_patch_size,
                token_channels=args.token_channels,
                mask_ratio=0.0,
            )
        else:
            raise ValueError(f"Unsupported tokenizer {args.tokenizer}")
        if args.load_tokenizer_from is not None:
            logger.info(f"[Tokenizer] Loading tokenizer from: {args.load_tokenizer_from}")
            weights = torch.load(args.load_tokenizer_from, weights_only=False, map_location="cpu")
            if args.use_ema_tokenizer and "model_ema" in weights:
                weights = weights["model_ema"]
                msg = tokenizer.load_state_dict(weights, strict=False)
                logger.info(f"[Tokenizer] Missing keys: {msg.missing_keys}")
                logger.info(f"[Tokenizer] Unexpected keys: {msg.unexpected_keys}")
                logger.info("[Tokenizer] Loaded EMA tokenizer.")
            else:
                if args.use_ema_tokenizer:
                    logger.warning("EMA tokenizer is not in the checkpoint, using the model weights")
                weights = weights["model"] if "model" in weights else weights
                msg = tokenizer.load_state_dict(weights, strict=True)
                logger.info(f"[Tokenizer] Missing keys: {msg.missing_keys}")
                logger.info(f"[Tokenizer] Unexpected keys: {msg.unexpected_keys}")
        tokenizer.cuda().eval().requires_grad_(False)
        logger.info("====Tokenizer=====")
        logger.info(tokenizer)
    else:
        tokenizer = None

    if args.model in models.DiT_models:
        model = models.DiT_models[args.model](
            img_size=args.img_size,
            patch_size=args.patch_size,
            tokenizer_patch_size=args.tokenizer_patch_size,
            token_channels=args.token_channels,
            label_drop_prob=args.label_drop_prob,
            num_classes=args.num_classes,
            num_sampling_steps=args.num_sampling_steps,
            force_one_d_seq=args.force_one_d_seq,
            grad_checkpointing=args.grad_checkpointing,
            legacy_mode=args.legacy_mode, # legacy mode: cfg on the first three channels only
        )
    elif args.model in models.SiT_models:
        model = models.SiT_models[args.model](
            img_size=args.img_size,
            patch_size=args.patch_size,
            tokenizer_patch_size=args.tokenizer_patch_size,
            token_channels=args.token_channels,
            label_drop_prob=args.label_drop_prob,
            num_classes=args.num_classes,
            num_sampling_steps=args.num_sampling_steps,
            grad_checkpointing=args.grad_checkpointing,
            force_one_d_seq=args.force_one_d_seq,
            legacy_mode=args.legacy_mode, # legacy mode: cfg on the first three channels only
            qk_norm=args.qk_norm,
        )
    elif args.model in models.LightningDiT_models:
        model = models.LightningDiT_models[args.model](
            img_size=args.img_size,
            patch_size=args.patch_size,
            tokenizer_patch_size=args.tokenizer_patch_size,
            token_channels=args.token_channels,
            label_drop_prob=args.label_drop_prob,
            num_classes=args.num_classes,
            num_sampling_steps=args.num_sampling_steps,
            force_one_d_seq=args.force_one_d_seq,
            grad_checkpointing=args.grad_checkpointing,
            legacy_mode=args.legacy_mode, # legacy mode: cfg on the first three channels only
            qk_norm=args.qk_norm,
        )
    elif args.model in models.ARDiff_models:
        model = models.ARDiff_models[args.model](
            img_size=args.img_size,
            patch_size=args.patch_size,
            tokenizer_patch_size=args.tokenizer_patch_size,
            token_channels=args.token_channels,
            label_drop_prob=args.label_drop_prob,
            num_classes=args.num_classes,
            num_sampling_steps=args.num_sampling_steps,
            diffloss_d=args.diffloss_d,
            diffloss_w=args.diffloss_w,
            diffusion_batch_mul=args.diffusion_batch_mul,
            noise_schedule=args.noise_schedule,
            force_one_d_seq=args.force_one_d_seq,
            grad_checkpointing=args.grad_checkpointing,
            order=args.order,
        )
    elif args.model in models.MAR_models:
        model = models.MAR_models[args.model](
            img_size=args.img_size,
            patch_size=args.patch_size,
            tokenizer_patch_size=args.tokenizer_patch_size,
            token_channels=args.token_channels,
            label_drop_prob=args.label_drop_prob,
            num_classes=args.num_classes,
            num_sampling_steps=args.num_sampling_steps,
            diffloss_d=args.diffloss_d,
            diffloss_w=args.diffloss_w,
            diffusion_batch_mul=args.diffusion_batch_mul,
            noise_schedule=args.noise_schedule,
            attn_dropout=args.attn_dropout,
            proj_dropout=args.proj_dropout,
            buffer_size=args.buffer_size,
            mask_ratio_min=args.mask_ratio_min,
            grad_checkpointing=args.grad_checkpointing,
            force_one_d_seq=args.force_one_d_seq,
            no_dropout_in_mlp=args.no_dropout_in_mlp,
        )
    else:
        raise ValueError(f"Unsupported model {args.model}")

    model.cuda()
    logger.info("====Model=====")
    logger.info(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} Parameters: {n_params / 1e6:.2f}M ({n_params:,})")

    # ema model
    ema = models.SimpleEMAModel(model, decay=args.ema_rate)
    return model, tokenizer, ema


def create_reconstruction_model(args):
    logger.info("Creating reconstruction models.")
    if args.model in models.VAE_models:
        model = models.VAE_models[args.model](
            load_ckpt=not getattr(args, "no_load_ckpt", False),
            gamma=args.gamma,
        )
    elif args.model in models.DeTok_models:
        model = models.DeTok_models[args.model](
            img_size=args.img_size,
            patch_size=args.patch_size,
            token_channels=args.token_channels,
            mask_ratio=args.mask_ratio,
            gamma=args.gamma,
        )
    else:
        raise ValueError(f"Unsupported model {args.model}")

    model.cuda()
    logger.info("====Model=====")
    logger.info(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} Trainable Parameters: {n_params / 1e6:.2f}M ({n_params:,})")
    ema = models.SimpleEMAModel(model, decay=args.ema_rate)
    return model, ema


def create_optimizer_and_scaler(args, model, print_trainable_params=False):
    logger.info("creating optimizers")

    # exclude parameters from weight decay
    exclude = lambda name, p: (
        p.ndim < 2 or any(keyword in name for keyword in 
        ["ln", "bias", "embedding", "norm", "gamma", "embed", "token", "diffloss"])
    )

    named_parameters = list(model.named_parameters())
    no_decay_list = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if not exclude(n, p) and p.requires_grad]
    eff_batch_size = args.batch_size * args.world_size

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    logger.info(f"base lr: {args.lr * 256 / eff_batch_size:.6e}")
    logger.info(f"actual lr: {args.lr:.6e}")
    logger.info(f"effective batch size: {eff_batch_size}")
    logger.info(f"training with {args.world_size} gpus")
    logger.info(f"weight_decay: {args.weight_decay} on {len(rest_params)} weight tensors")
    logger.info(f"no_decay: {len(no_decay_list)} weight tensors")

    optimizer = torch.optim.AdamW(
        [
            {"params": no_decay_list, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": args.weight_decay},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )
    logger.info(f"Optimizer = {str(optimizer)}")
    if print_trainable_params:
        logger.info("trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"\t{name}")

    loss_scaler = NativeScalerWithGradNormCount()
    logger.info(f"Loss Scaler = {str(loss_scaler)}")
    return optimizer, loss_scaler


def create_loss_module(args):
    loss_module = losses.ReconstructionLoss(
        discriminator_start_epoch=getattr(args, "discriminator_start_epoch", 20),
        perceptual_loss=getattr(args, "perceptual_loss", "lpips-convnext_s-1.0-0.1"),
        perceptual_weight=getattr(args, "perceptual_weight", 1.1),
        kl_weight=args.kl_loss_weight,
    )
    loss_module.cuda()
    logger.info("====Loss Module=====")
    # logger.info(loss_module)
    return loss_module
