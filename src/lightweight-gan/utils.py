from math import log2
from PIL import Image
import torch
import os

def is_power_of_two(val):
    return log2(val).is_integer()


def default(val, d):
    if val is None:
        return d
    return val


def image_to_pil(image):
    ndarr = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


def init_folders(model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)