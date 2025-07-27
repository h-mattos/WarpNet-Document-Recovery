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
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.functions import compute_ssim#, compute_psnr

def compute_psnr(pred_tensor, target_tensor, max_val=1.0):
    mse = F.mse_loss(pred_tensor, target_tensor, reduction='none').mean(dim=(1,2))
    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr.mean().item()

def eval_metrics(preds, targets):
    preds = torch.from_numpy(preds).unsqueeze(1).float()
    targets = torch.from_numpy(targets).unsqueeze(1).float()
    mse = mse_loss(preds, targets).item()
    ssim = compute_ssim(preds, targets).item()
    psnr = compute_psnr(preds, targets)
    return mse, ssim, psnr

def load_h5_images(h5_file, ids=None):
    with h5py.File(h5_file, "r") as f:
        # return np.stack([f["imgs"][i] for i in ids]).astype(np.float32) / 255.0
        return f['imgs']
    
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
    recoverd_file = "data/dewarped.h5"
    original_file = "data/original.h5"

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
        warped_imgs = h5py.File(warped_file, "r")["imgs"]
        deblurred_imgs = h5py.File(deblurred_file, "r")["imgs"]
        dewarped_imgs = h5py.File(dewarped_file, "r")["imgs"]
        blurred_orig_imgs = h5py.File(blurred_orig_file, "r")["imgs"]
        recovered_imgs = h5py.File(recoverd_file, "r")["imgs"]
        original_imgs = h5py.File(original_file, "r")["imgs"]
        for idx in tqdm(ids, desc=f"{split_name} split"):
            # warped = load_h5_images(warped_file, [idx])[0]
            # deblurred = load_h5_images(deblurred_file, [idx])[0]
            warped = warped_imgs[idx]
            deblurred = deblurred_imgs[idx]
            deblur_mse, deblur_ssim, deblur_psnr = eval_metrics(deblurred, warped)
            deblur_mse_total += deblur_mse
            deblur_ssim_total += deblur_ssim
            deblur_psnr_total += deblur_psnr

            # dewarped = load_h5_images(dewarped_file, [idx])[0]
            # blurred_orig = load_h5_images(blurred_orig_file, [idx])[0]
            dewarped = dewarped_imgs[idx]
            blurred_orig = blurred_orig_imgs[idx]
            dewarp_mse, dewarp_ssim, dewarp_psnr = eval_metrics(dewarped, blurred_orig)
            dewarp_mse_total += dewarp_mse
            dewarp_ssim_total += dewarp_ssim
            dewarp_psnr_total += dewarp_psnr
            
            # recovered = load_h5_images(recoverd_file, [idx])[0]
            # original = load_h5_images(original_file, [idx])[0]
            recovered = recovered_imgs[idx]
            original = original_imgs[idx]
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
    # print(df.round(4))
    print(df.T.round(4))

if __name__ == "__main__":
    main()