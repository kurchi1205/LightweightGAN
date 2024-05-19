import torch
from PIL import Image

def image_to_pil(image):
    ndarr = image.add_(1).mul(127.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im
