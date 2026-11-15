"""Hash-bound fixed-read inputs for the exploratory feature-control study."""
import numpy as np
from PIL import Image
import torch

from scripts.memory_token_io import digest
from scripts.train_multilevel_refiner_seq4_empty_v2 import read_capture


def read_rgb(item, shape):
    """Load the exact archived input image, with SAM's 1024-pixel RGB resize.

    The color-only contrast map uses unnormalized [0,1] RGB. SAM features use
    the original backbone normalization separately, as in the captured run.
    """
    if digest(item['image']) != item['image_sha256']:
        raise ValueError('RGB input differs from frozen manifest')
    with Image.open(item['image']) as image:
        if (image.height, image.width) != tuple(shape):
            raise ValueError('RGB input geometry differs from captured video')
        # SAM2's _load_img_as_tensor uses PIL RGB conversion and this resize.
        array = np.array(image.convert('RGB').resize((1024, 1024)))
    if array.dtype != np.uint8 or array.shape != (1024, 1024, 3):
        raise ValueError('Unexpected RGB image representation')
    if digest(item['image']) != item['image_sha256']:
        raise ValueError('RGB input changed during decoding')
    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def inputs(item, device, arm):
    early, deep, logits, shape = read_capture(item)
    args = (early.to(device), deep.to(device), logits.to(device))
    if arm == 'rgb_contrast':
        args += (read_rgb(item, shape).to(device),)
    return args, shape
