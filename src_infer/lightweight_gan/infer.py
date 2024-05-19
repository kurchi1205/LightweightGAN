import torch
from utils import image_to_pil


def inference(model, num_image, latent_dim):
    latent = torch.randn(num_image, latent_dim)
    with torch.no_grad():
        generated_images = model(latent)
    pil_generated_images = [image_to_pil(image) for image in generated_images]
    return pil_generated_images