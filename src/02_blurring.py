import sys
import os
import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.blur_utils import apply_psf_blur, pad_or_crop_kernel

input_image_dir = "data/warped"
input_original_dir = "data/BMVC_image_data/orig"
input_psf_dir = "data/BMVC_image_data/psf"
output_normalized_psf_dir = "data/normalized_psf"
output_blur_dir = "data/blurred"
output_blur_orig_dir = "data/blurred_orig"

os.makedirs(output_normalized_psf_dir, exist_ok=True)
os.makedirs(output_blur_dir, exist_ok=True)
os.makedirs(output_blur_orig_dir, exist_ok=True)

for filename in tqdm(sorted(os.listdir(input_image_dir))):
    img_id = filename.replace("_orig.png", "")
    image_path = os.path.join(input_image_dir, filename)
    original_path = os.path.join(input_original_dir, filename)
    psf_path = os.path.join(input_psf_dir, f"{img_id}_psf.png")
    
    output_path_psf = os.path.join(output_normalized_psf_dir, f"{img_id}_psf.png")
    output_path_blurred = os.path.join(output_blur_dir, f"{img_id}_blur.png")
    output_path_blurred_orig = os.path.join(output_blur_orig_dir, f"{img_id}_blur.png")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    psf_kernel = pad_or_crop_kernel(cv2.imread(psf_path, cv2.IMREAD_GRAYSCALE), target_size=19)

    cv2.imwrite(output_path_psf, psf_kernel)

    blurred = apply_psf_blur(image, psf_kernel)
    blurred_orig = apply_psf_blur(original, psf_kernel)

    cv2.imwrite(output_path_blurred, blurred)
    cv2.imwrite(output_path_blurred_orig, blurred_orig)
