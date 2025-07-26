import os
import sys
import numpy as np
import h5py
import torch
from torch.nn.functional import mse_loss
import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.functions import compute_ssim, compute_psnr

def eval_metrics(preds, targets):
    preds = torch.from_numpy(preds).unsqueeze(1).float()
    targets = torch.from_numpy(targets).unsqueeze(1).float()
    mse = mse_loss(preds, targets).item()
    ssim = compute_ssim(preds, targets).item()
    psnr = compute_psnr(preds, targets).mean().item()
    return mse, ssim

def load_h5_images(h5_file, ids):
    with h5py.File(h5_file, "r") as f:
        return np.stack([f["imgs"][i] for i in ids]).astype(np.float32) / 255.0
    
def load_original_images(image_dir, ids):
    images = []
    for file_name in [sorted(os.listdir(image_dir))[i] for i in ids]:
        img = cv2.imread(os.path.join(image_dir, file_name), cv2.IMREAD_GRAYSCALE)
        images.append(img.astype(np.float32) / 255.0)
    return np.stack(images)

def main():
    warped_file = "data/warped.h5"
    deblurred_file = "data/deblurred.h5"
    dewarped_file = "data/dewarped.h5"
    blurred_orig_file = "data/blurred_orig.h5"
    recoverd_file = "data/recovered"
    original_file = "data/BMVC_images_data/orig"

    with h5py.File(warped_file, "r") as f:
        total_ids = list(range(len(f["imgs"])))

    train_ids, other_ids = train_test_split(total_ids, test_size=0.3, random_state=7643)
    val_ids, holdout_ids = train_test_split(other_ids, test_size=0.5, random_state=7643)

    splits = {"Train": train_ids, "Validation": val_ids, "Holdout": holdout_ids}

    results = []

    for split_name, ids, in splits.items():
        deblur_mse_total = deblur_ssim_total = deblur_psnr_total = 0.0
        dewarp_mse_total = dewarp_ssim_total = dewarp_psnr_total = 0.0
        recovered_mse_total = recovered_ssim_total = recovered_psnr_total = 0.0
        for idx in tqdm(ids, desc=f"{split_name} split"):
            warped = load_h5_images(warped_file, [idx])[0]
            deblurred = load_h5_images(deblurred_file, [idx])[0]
            deblur_mse, deblur_ssim, deblur_psnr = eval_metrics(deblurred, warped)
            deblur_mse_total += deblur_mse
            deblur_ssim_total += deblur_ssim
            deblur_psnr_total += deblur_psnr

            dewarped = load_h5_images(dewarped_file, [idx])[0]
            blurred_orig = load_h5_images(blurred_orig_file, [idx])[0]
            dewarp_mse, dewarp_ssim, dewarp_psnr = eval_metrics(dewarped, blurred_orig)
            dewarp_mse_total += dewarp_mse
            dewarp_ssim_total += dewarp_ssim
            dewarp_psnr_total += dewarp_psnr
            
            recovered = load_original_images(recoverd_file, [idx])[0]
            original = load_original_images(original_file, [idx])[0]
            recovered_mse, recovered_ssim, recovered_psnr = eval_metrics(recovered, original)
            recovered_mse_total += recovered_mse
            recovered_ssim_total += recovered_ssim
            recovered_psnr_total += recovered_psnr

        N = len(ids)

        results.append({
            "Split": split_name,
            "Deblur MSE": deblur_mse_total / N,
            "Deblur SSIM": deblur_ssim_total / N,
            "Deblur PSNR" : deblur_psnr_total / N,
            "Dewarp MSE": dewarp_mse_total / N,
            "Dewarp SSIM": dewarp_ssim_total / N,
            "Dewarp PSNR" : dewarp_psnr_total / N,
            "Full MSE": recovered_mse_total / N,
            "Full SSIM": recovered_ssim_total / N,
            "Full PSNR" : recovered_psnr_total / N,
        })

    df = pd.DataFrame(results)
    df = df.set_index("Split")
    print(df.round(4))
    print(df.T.round(4))

if __name__ == "__main__":
    main()