import os
import cv2
import numpy as np
from tqdm import tqdm

from src.utils.blur_utils import wiener_deblur

blurred_dir = "data/blurred"
psf_dir = "data/predicted_psf"
output_dir = "data/deblurred"

os.makedirs(output_dir, exist_ok=True)

for file_name in tqdm(sorted(os.listdir(blurred_dir))):
    img_id = file_name.replace("_blur.png", "")
    blurred_path = os.path.join(blurred_dir, file_name)
    psf_path = os.path.join(psf_dir, f"{img_id}_psf.png")
    output_path = os.path.join(output_dir, f"{img_id}_deblur.png")

    blurred = cv2.imread(blurred_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    psf = cv2.imread(psf_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    psf /= (psf.sum() + 1e-8)

    deblurred = wiener_deblur(blurred, psf, K=0.01)

    cv2.imwrite(output_path, (deblurred * 255).astype(np.uint8))