import cv2
import numpy as np

def pad_or_crop_kernel(psf, target_size=20):
    assert psf.ndim == 2, "Expecting greyscale psf kernel"

    h, w = psf.shape
    
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)

    if pad_h > 0 or pad_w > 0:
        psf = np.pad(
            psf,
            pad_width=(
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2)
            ),
            mode="constant"
        )

    h, w = psf.shape
    crop_h = h - target_size
    crop_w = w - target_size
    h_start = crop_h // 2
    w_start = crop_w // 2

    result = psf[h_start : h_start + target_size, w_start : w_start + target_size]
    return result


def apply_psf_blur(image, psf_kernel):
    assert image.ndim == 2, "Expecting greyscale image"
    assert psf_kernel.ndim == 2, "Expecting greyscale image"

    kernel = psf_kernel / (psf_kernel.sum() + 1e-8)
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred.astype(image.dtype)