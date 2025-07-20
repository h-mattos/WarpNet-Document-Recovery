import cv2
import numpy as np
from scipy.signal import fftconvolve

def pad_or_crop_kernel(psf, target_size=19):
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


# Have to use Wiener deconvolution since deconvolving a blurred image is an ill-posed inverse problem
def wiener_deblur(blurred, psf, K=0.01):
    psf_padded = np.zeros_like(blurred)
    h, w = psf.shape
    psf_padded[:h, :w] = psf
    psf_padded = np.fft.ifftshift(psf_padded)

    psf_fft = np.fft.fft2(psf_padded)
    blurred_fft = np.fft.fft2(blurred)

    psf_power = np.abs(psf_fft) ** 2
    wiener_filter = np.conj(psf_fft) / (psf_power + K)

    deconv_fft = blurred_fft * wiener_filter
    deconv = np.fft.ifft2(deconv_fft).real
    deconv = np.clip(deconv, 0, 1).astype(np.float32)
    
    return deconv