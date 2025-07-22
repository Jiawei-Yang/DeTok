import logging
import os
from functools import partial
from typing import Any, Callable

import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader

logger = logging.getLogger("DeTok")

CONSTANTS = {
    # vavae latent statistics from  https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/blob/main/latents_stats.pt
    "vavae_mean": np.array([
        0.5984623, -0.49917176, 0.6440029, -0.0970839, -1.190963, -1.4331622,
        0.46853292, 0.6259252, 0.63195026, -0.4896733, -0.74451625, 1.1595623,
        0.8456217, 0.5008238, 0.22926894, 0.47535565, -0.43787342, 0.8316961,
        -0.0750857, 0.30632293, 0.46645293, -0.09140775, -0.82710165, 0.07807512,
        1.4150785, 1.3792385, 0.2695843, -0.7573224, 0.28129938, -0.30919993,
        0.07785388, 0.34966648,
    ]),
    "vavae_std": np.array([
        3.846138, 4.2699146, 3.5768437, 3.5911105, 3.6230576, 3.481018,
        3.3074617, 3.5092657, 3.5540583, 3.6067245, 3.70579, 3.6314075,
        3.6295316, 3.620502, 3.2590282, 3.186753, 3.8258142, 3.599939,
        3.2966352, 3.226129, 3.2191944, 3.1054573, 3.580496, 4.356914,
        3.308541, 3.2075875, 4.515047, 3.4869924, 3.0415804, 3.4868848,
        4.4310327, 4.0881157,
    ]),
}


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """center cropping implementation from adm.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), 
            resample=Image.Resampling.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), 
        resample=Image.Resampling.BICUBIC
    )
    
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y:crop_y + image_size, crop_x:crop_x + image_size])


def default_np_loader(path: str) -> np.ndarray[Any, np.dtype[Any]]:
    return np.load(path, allow_pickle=True)


class ListDataset:
    def __init__(
        self,
        data_root: str,
        data_list: str,
        transform: Callable[[Any], Any] | None = None,
        loader_name: str = "npz_loader",
        return_path: bool = False,
        return_label: bool = True,
        return_index: bool = False,
        should_flip: bool = True,
        class_of_interest: list[int] | None = None,
    ):
        self.data_root = data_root
        self.transform = transform
        self.return_path = return_path
        self.return_label = return_label
        self.return_index = return_index
        self.should_flip = should_flip
        self.class_of_interest = class_of_interest

        # loader function mapping
        loader_functions = {
            "img_loader": default_loader,
            "npz_loader": partial(np.load, allow_pickle=True),
        }

        if loader_name not in loader_functions:
            raise ValueError(f"Loader '{loader_name}' not supported")

        self.loader = loader_functions[loader_name]
        self.load_vae_latents = loader_name == "npz_loader"
        self.samples = self._load_samples(data_list, loader_name)
        self.targets = [label for _, label in self.samples]

    def _load_samples(self, data_list: str, loader_name: str) -> list[tuple[str, int | None]]:
        samples = []
        with open(data_list, "r") as f:
            for line in f:
                splits = line.strip().split(" ")
                if len(splits) == 2:
                    file_path, label = splits
                    label = int(label)
                else:
                    file_path = line.strip()
                    label = None
                    
                if self.class_of_interest and label not in self.class_of_interest:
                    continue
                    
                # adjust file extensions based on loader
                if loader_name == "npz_loader":
                    file_path = file_path.replace(".JPEG", ".JPEG.npz")
                    
                samples.append((file_path, label))
        return samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._get_item_with_retry(index, 0)

    def _get_item_with_retry(self, index: int, retry_count: int) -> dict[str, Any]:
        if retry_count >= 100:
            raise RuntimeError(f"Failed to load data after 100 retries, last index: {index}")
            
        img_pth, label = self.samples[index]
        img_path_full = os.path.join(self.data_root, img_pth)
        should_flip = np.random.rand() < 0.5 if self.should_flip else False
        to_return = {}

        try:
            img = self.loader(img_path_full)
            if self.load_vae_latents:
                img_data = img  # type: ignore
                img = img_data["moments_flip"] if should_flip else img_data["moments"]
                to_return = {"token": img}
        except Exception as e:
            logger.error(f"Error loading '{img_pth}': {e}")
            return self._get_item_with_retry((index + 1) % len(self.samples), retry_count + 1)

        if self.transform is not None:
            if "token" in to_return:
                # load original image when we have vae latents
                img_path_relative = img_path_full.split("/")[3:]
                img_path_relative = os.path.join(*img_path_relative)
                img_path_relative = img_path_relative.replace(".npz", "")
                img_path_full = os.path.join(self.data_root, img_path_relative)
                img = default_loader(img_path_full)
                
            img = self.transform(img)
            if should_flip:
                img = img.flip(dims=[2])
                
            if len(to_return) > 0:
                to_return["img"] = img
            else:
                to_return = {"img": img}

        if self.return_index:
            to_return["index"] = index
        if self.return_label:
            to_return["label"] = label
        if self.return_path:
            to_return["img_pth"] = img_pth
            
        return to_return

    def __len__(self) -> int:
        return len(self.samples)
