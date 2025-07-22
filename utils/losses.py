"""
References:
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
    https://github.com/bytedance/1d-tokenizer/blob/main/modeling/modules/perceptual_loss.py
    https://github.com/bytedance/1d-tokenizer/blob/main/modeling/modules/losses.py
"""

import hashlib
import logging
import os
from collections import namedtuple
from typing import Text

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models
from tqdm import tqdm

logger = logging.getLogger("DeTok")

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_LPIPS_MEAN = [-0.030, -0.088, -0.188]
_LPIPS_STD = [0.458, 0.448, 0.450]

URL_MAP = {"vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}
CKPT_MAP = {"vgg_lpips": "vgg.pth"}
MD5_MAP = {"vgg_lpips": "d507d7349b931f0638a25a48a722f98a"}


def download(url: str, local_path: str, chunk_size: int = 1024) -> None:
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path: str) -> str:
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name: str, root: str, check: bool = False) -> str:
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        logger.info("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


def normalize_tensor(x: Tensor, eps: float = 1e-10) -> Tensor:
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x: Tensor, keepdim: bool = True) -> Tensor:
    return x.mean([2, 3], keepdim=keepdim)


def hinge_d_loss(logits_real: Tensor, logits_fake: Tensor) -> Tensor:
    """Hinge loss for discrminator.

    This function is borrowed from
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py#L20
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


class LPIPS(nn.Module):
    def __init__(self, ckpt_pth="work_dirs/ckpts/lpips", use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # VGG16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained(ckpt_pth=ckpt_pth)
        for param in self.parameters():
            param.requires_grad = False

        self._data_range_checked = False

    def load_from_pretrained(self, ckpt_pth="work_dirs/ckpts/lpips", name="vgg_lpips"):
        ckpt = get_ckpt_path(name, ckpt_pth, check=True)
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        logger.info("Loaded pretrained LPIPS loss from {}".format(ckpt))

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer("shift", Tensor(_LPIPS_MEAN)[None, :, None, None])
        self.register_buffer("scale", Tensor(_LPIPS_STD)[None, :, None, None])

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class vgg16(nn.Module):
    def __init__(self, requires_grad: bool = False, pretrained: bool = True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5

        # build feature slices
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class NLayerDiscriminator(nn.Module):
    """patchgan discriminator"""

    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1

        # gradually increase the number of filters
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=False,
                ),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=False,
            ),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.main = nn.Sequential(*sequence)

    def forward(self, input: Tensor) -> Tensor:
        return self.main(input)


class PerceptualLoss(torch.nn.Module):
    # reference: https://github.com/bytedance/1d-tokenizer/blob/main/modeling/modules/perceptual_loss.py
    def __init__(self, model_name: str = "convnext_s"):
        super().__init__()
        self.lpips = None
        self.convnext = None
        self.loss_weight_lpips = None
        self.loss_weight_convnext = None
        self._data_range_checked = False

        # Parsing the model name. We support name formatted in
        # "lpips-convnext_s-{float_number}-{float_number}", where the
        # {float_number} refers to the loss weight for each component.
        # E.g., lpips-convnext_s-1.0-2.0 refers to compute the perceptual loss
        # using both the convnext_s and lpips, and average the final loss with
        # (1.0 * loss(lpips) + 2.0 * loss(convnext_s)) / (1.0 + 2.0).
        if "lpips" in model_name:
            self.lpips = LPIPS().eval()

        if "convnext_s" in model_name:
            self.convnext = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1).eval()

        if "lpips" in model_name and "convnext_s" in model_name:
            loss_config = model_name.split("-")[-2:]
            self.loss_weight_lpips, self.loss_weight_convnext = float(loss_config[0]), float(loss_config[1])
            logger.info(
                f"loss weights - lpips: {self.loss_weight_lpips}, convnext: {self.loss_weight_convnext}"
            )

        self.register_buffer("imagenet_mean", Tensor(_IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer("imagenet_std", Tensor(_IMAGENET_STD)[None, :, None, None])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, inputs: Tensor, pred: Tensor) -> Tensor:
        """Computes the perceptual loss.

        Args:
            inputs: A tensor of shape (B, C, H, W), the gt image. Normalized to [0, 1].
            pred: A tensor of shape (B, C, H, W), the reconstructed image. Normalized to [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        assert inputs.shape == pred.shape, f"{inputs.shape=} != {pred.shape}="

        if not self._data_range_checked:
            assert (
                inputs.min() >= 0.0 and inputs.max() <= 1.0
            ), f"{inputs.min()=} ~ {inputs.max()=}. reminder to normalize input and target to [0, 1]."
            self._data_range_checked = True

        self.eval()
        loss = 0.0
        num_losses = 0.0

        # compute lpips loss, if available
        if self.lpips is not None:
            # lpips expects input in range [-1, 1]
            lpips_loss = self.lpips(inputs * 2 - 1, pred * 2 - 1)
            if self.loss_weight_lpips is None:
                loss += lpips_loss
                num_losses += 1
            else:
                num_losses += self.loss_weight_lpips
                loss += self.loss_weight_lpips * lpips_loss

        if self.convnext is not None:
            inputs_resized = F.interpolate(inputs, size=224, mode="bilinear", antialias=True)
            pred_resized = F.interpolate(pred, size=224, mode="bilinear", antialias=True)
            inputs_norm = (inputs_resized - self.imagenet_mean) / self.imagenet_std
            pred_norm = (pred_resized - self.imagenet_mean) / self.imagenet_std

            input_feats, pred_feats = self.convnext(inputs_norm), self.convnext(pred_norm)
            convnext_loss = F.mse_loss(input_feats, pred_feats, reduction="mean")
            if self.loss_weight_convnext is None:
                num_losses += 1
                loss += convnext_loss
            else:
                num_losses += self.loss_weight_convnext
                loss += self.loss_weight_convnext * convnext_loss

        # weighted average
        loss = loss / num_losses
        return loss


class ReconstructionLoss(nn.Module):
    # reference: https://github.com/bytedance/1d-tokenizer/blob/main/modeling/modules/losses.py
    def __init__(
        self,
        discriminator_weight: float = 0.1,
        discriminator_start_epoch: int = 20,
        perceptual_loss: str = "lpips-convnext_s-1.0-0.1",
        perceptual_weight: float = 1.1,
        reconstruction_loss: str = "l2",
        reconstruction_weight: float = 1.0,
        kl_weight: float = 1e-6,
        logvar_init: float = 0.0,
    ):
        super().__init__()
        self.reconstruction_loss = reconstruction_loss
        self.reconstruction_weight = reconstruction_weight

        self.perceptual_loss = PerceptualLoss(perceptual_loss).eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator()
        self.discriminator_weight = discriminator_weight
        self.discriminator_start_epoch = discriminator_start_epoch

        self.kl_weight = kl_weight
        # `requires_grad` must be false to avoid ddp error. No guarantee this implementationis right though.
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init, requires_grad=False)

        self._data_range_checked = False

        # log hyperparameters
        logger.info("=======ReconstructionLoss=======")
        logger.info(f"reconstruction loss: {self.reconstruction_loss}")
        logger.info(f"reconstruction weight: {self.reconstruction_weight}")
        logger.info(f"perceptual weight: {self.perceptual_weight}")
        logger.info(f"discriminator weight: {self.discriminator_weight}")
        logger.info(f"discriminator start epoch: {self.discriminator_start_epoch}")
        logger.info(f"kl weight: {self.kl_weight}")
        logger.info(f"logvar init: {logvar_init}")
        logger.info("=====================================")

    @torch.autocast("cuda", enabled=False)
    def forward(
        self,
        inputs: Tensor,
        reconstructions: Tensor,
        extra_result_dict,
        epoch: int,
        mode: str = "generator",
        last_layer=None,
    ) -> tuple[Tensor, dict[Text, Tensor]]:
        # both inputs and reconstructions are in range [0, 1]
        inputs = inputs.float()
        reconstructions = reconstructions.float()

        # validate tensor shapes match
        assert (
            inputs.shape == reconstructions.shape
        ), f"shape mismatch: inputs {inputs.shape} != reconstructions {reconstructions.shape}"

        # validate input range is normalized to [0, 1]
        if not self._data_range_checked:
            input_min, input_max = inputs.min(), inputs.max()
            assert input_min >= 0.0 and input_max <= 1.0, (
                f"input values out of range [0, 1]: min={input_min:.4f}, max={input_max:.4f}. "
                "please normalize inputs and targets to [0, 1]."
            )
            self._data_range_checked = True

        if mode == "generator":
            return self._forward_generator(inputs, reconstructions, extra_result_dict, epoch)
        elif mode == "discriminator":
            return self._forward_discriminator(inputs, reconstructions, epoch)
        else:
            raise ValueError(f"unsupported mode {mode}")

    def should_discriminator_be_trained(self, epoch: int):
        return epoch >= self.discriminator_start_epoch

    def _forward_generator(
        self,
        inputs: Tensor,
        reconstructions: Tensor,
        extra_result_dict,
        epoch: int,
    ) -> tuple[Tensor, dict[Text, Tensor]]:
        """generator training step"""
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()

        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2":
            reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2+l1":
            l2_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
            l1_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
            reconstruction_loss = (l2_loss + l1_loss) / 2
        else:
            raise ValueError(f"unsuppored reconstruction_loss {self.reconstruction_loss}")
        reconstruction_loss *= self.reconstruction_weight

        # compute perceptual loss
        perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()

        # compute discriminator loss
        generator_loss = torch.zeros((), device=inputs.device)
        d_factor = 1.0 if self.should_discriminator_be_trained(epoch) else 0
        d_weight = 1.0
        if d_factor > 0.0 and self.discriminator_weight > 0.0:
            # disable discriminator gradients
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        reconstruction_loss = reconstruction_loss / torch.exp(self.logvar)
        kl_loss = torch.zeros((), device=inputs.device)
        if extra_result_dict is not None:
            # assume extra_result_dict contains posteriors with kl method
            posteriors = extra_result_dict
            if hasattr(posteriors, "kl"):
                kl_loss = posteriors.kl()
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        total_loss = (
            reconstruction_loss
            + self.perceptual_weight * perceptual_loss
            + self.kl_weight * kl_loss
            + d_weight * d_factor * generator_loss
        )

        loss_dict = {
            "total_loss": total_loss.clone().detach(),
            "reconstruction_loss": reconstruction_loss.detach(),
            "perceptual_loss": (self.perceptual_weight * perceptual_loss).detach(),
            "kl_loss": (self.kl_weight * kl_loss).detach(),
            "weighted_gan_loss": (d_weight * d_factor * generator_loss).detach(),
            "discriminator_factor": torch.tensor(d_factor),
            "d_weight": torch.tensor(d_weight),
            "gan_loss": generator_loss.detach(),
            "psnr": -10 * torch.log10(reconstruction_loss).detach(),
        }

        return total_loss, loss_dict

    def _forward_discriminator(
        self,
        inputs: Tensor,
        reconstructions: Tensor,
        epoch: int,
    ) -> tuple[Tensor, dict[Text, Tensor]]:
        """discriminator training step"""
        discriminator_factor = 1.0 if self.should_discriminator_be_trained(epoch) else 0

        # turn the gradients on
        for param in self.discriminator.parameters():
            param.requires_grad = True

        real_images = inputs.detach().requires_grad_(True)
        logits_real = self.discriminator(real_images)
        logits_fake = self.discriminator(reconstructions.detach())

        discriminator_loss = discriminator_factor * hinge_d_loss(
            logits_real=logits_real, logits_fake=logits_fake
        )

        loss_dict = {
            "discriminator_loss": discriminator_loss.detach(),
            "logits_real": logits_real.detach().mean(),
            "logits_fake": logits_fake.detach().mean(),
        }

        return discriminator_loss, loss_dict
