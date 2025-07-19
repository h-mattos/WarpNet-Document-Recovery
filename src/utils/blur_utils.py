import cv2
import numpy as np

def pad_or_crop_kernel(psf, target_size = 20):
    h, w, _ = psf.shape
    
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)
    
    if pad_h > 0 or pad_w > 0:
        psf = np.pad(
            psf,
            pad_width = (
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
                (0, 0)
            ),
            mode = "constant"
        )

    h, w, _ = psf.shape
    crop_h = h - target_size
    crop_w = w - target_size
    h_start = crop_h // 2
    w_start = crop_w // 2
    
    result = psf[h_start : h_start + target_size, w_start : w_start + target_size, :]
    return result

def apply_psf_blur(image, psf_kernel):
    blurred_channels = []
    
    # For each channel in RGB, apply blur convolution with PSF kernel
    for c in range(3):
        channel = image[:, :, c]
        kernel = psf_kernel[:, :, c]

        kernel = kernel / (kernel.sum() + 1e-8)

        blurred = cv2.filter2D(channel, -1, kernel)
        blurred_channels.append(blurred)

    blurred_image = np.stack(blurred_channels, axis = -1)
    return blurred_image.astype(image.dtype)