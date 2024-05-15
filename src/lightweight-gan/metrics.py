import torch
from torch.nn.functional import adaptive_avg_pool2d
import numpy as np
from scipy import linalg


def get_activations(
    image_batch, model, batch_size=50, dims=2048, device="cpu", num_workers=1
):
    model.eval()
    pred_arr = np.empty((len(image_batch), dims))

    start_idx = 0
    image_batch = image_batch.to(device)

    with torch.no_grad():
        pred = model(image_batch)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    pred = pred.squeeze(3).squeeze(2).cpu().numpy()
    pred_arr[start_idx : start_idx + pred.shape[0]] = pred
    start_idx = start_idx + pred.shape[0]
    return pred_arr


def calculate_activation_statistics(image_batch, model, batch_size=50, dims=2048, device="cpu", num_workers=1):
    act = get_activations(image_batch, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_fid_given_images(generated_images, real_images, batch_size, device, dims, num_workers=1):
    mu_gen, sigma_gen = calculate_activation_statistics(generated_images, batch_size, dims, device, num_workers)
    mu_real, sigma_real = calculate_activation_statistics(real_images, batch_size, dims, device, num_workers)
    return calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
