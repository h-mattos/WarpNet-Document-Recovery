import cv2
import numpy as np


def apply_psf_blur(image, psf_kernel):
    blurred_channels = []

    # For each channel in RGB, apply blur convolution with PSF kernel
    for c in range(3):
        channel = image[:, :, c]
        kernel = psf_kernel[:, :, c]

        kernel = kernel / (kernel.sum() + 1e-8)

        blurred = cv2.filter2D(channel, -1, kernel)
        blurred_channels.append(blurred)

    blurred_image = np.stack(blurred_channels, axis=-1)
    return blurred_image.astype(image.dtype)
