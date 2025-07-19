import sys
import os
import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.blur_utils import apply_psf_blur

input_image_dir = "data/orig"
input_psf_dir = "data/psf"
output_blur_dir = "data/blur"

os.makedirs(output_blur_dir, exist_ok=True)

for filename in tqdm(os.listdir(input_image_dir)):
    img_id = filename.replace("_orig.png", "")
    image_path = os.path.join(input_image_dir, filename)
    psf_path = os.path.join(input_psf_dir, f"{img_id}_psf.png")
    output_path = os.path.join(output_blur_dir, f"{img_id}_blur.png")

    image = cv2.imread(image_path)
    psf_kernel = cv2.imread(psf_path)

    blurred = apply_psf_blur(image, psf_kernel)

    cv2.imwrite(output_path, blurred)
