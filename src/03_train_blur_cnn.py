import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.functions import compute_ssim, compute_psnr
from src.models.blur_cnn import BlurCNNDataset, CNNDeblurrer
from src.utils.fs import validate_path
import h5py


CHECKPOINT_OUTPUT = "checkpoints"

validate_path(CHECKPOINT_OUTPUT)

def main():
    images_dir = "./data/blurred.h5"
    deblurred_dir = "./data/warped.h5"
    ids = list(range(h5py.File(deblurred_dir, 'r')['imgs'].shape[0]))

    # 70-15-15 train-validation-holdout split
    train_ids, other_ids = train_test_split(ids, test_size=0.3, random_state=7643)
    val_ids, holdout_ids = train_test_split(other_ids, test_size=0.5, random_state=7643)
    print(f"Train size: {len(train_ids)}, Validation size: {len(val_ids)}, Holdout size: {len(holdout_ids)}")

    # Setup dataset and dataloader
    train_dataset = BlurCNNDataset(blurred_h5_file=images_dir, deblurred_h5_file=deblurred_dir, ids=train_ids)
    val_dataset = BlurCNNDataset(blurred_h5_file=images_dir, deblurred_h5_file=deblurred_dir, ids=val_ids)
    # holdout_dataset = BlurCNNDataset(blurred_h5_file=images_dir, deblurred_h5_file=deblurred_dir, ids=holdout_ids)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    # holdout_loader = DataLoader(holdout_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # Training loop sketch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNDeblurrer().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    N_EPOCHS = 5

    train_losses = []
    val_losses = []
    train_ssims = []
    val_ssims = []
    train_psnrs = []
    val_psnrs = []

    for epoch in range(1, N_EPOCHS + 1):
        # Training
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS} - Training")
        model.train()
        train_loss_sum = 0.0
        train_ssim_sum = 0.0
        train_psnr_sum = 0.0
        for imgs, targets in loop:
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * imgs.size(0)
            train_ssim_sum += compute_ssim(preds, targets) * imgs.size(0)
            train_psnr_sum += compute_psnr(preds, targets) * imgs.size(0)
            loop.set_postfix(batch_loss=loss.item())
        avg_train_loss = train_loss_sum / len(train_dataset)
        avg_train_ssim = train_ssim_sum / len(train_dataset)
        avg_train_psnr = train_psnr_sum / len(train_dataset)
        train_losses.append(avg_train_loss)
        train_ssims.append(avg_train_ssim)
        train_psnrs.append(avg_train_psnr)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_ssim_sum = 0.0
        val_psnr_sum = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                preds = model(imgs)
                loss = criterion(preds, targets)
                val_loss_sum += loss.item() * imgs.size(0)
                val_ssim_sum += compute_ssim(preds, targets) * imgs.size(0)
                val_psnr_sum += compute_psnr(preds, targets) * imgs.size(0)
        avg_val_loss = val_loss_sum / len(val_dataset)
        avg_val_ssim = val_ssim_sum / len(val_dataset)
        avg_val_psnr = val_psnr_sum / len(val_dataset)
        val_losses.append(avg_val_loss)
        val_ssims.append(avg_val_ssim)
        val_psnrs.append(avg_val_psnr)
        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | \nTrain SSIM: {avg_train_ssim:.4f} | Val SSIM: {avg_val_ssim:.4f} | \nTrain PSNR: {avg_train_psnr:.4f} | Val PSNR: {avg_val_psnr:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f'{CHECKPOINT_OUTPUT}/deblur_predictor.pth')
    print(f"Saved trained model to {CHECKPOINT_OUTPUT}/deblur_predictor.pth")

    epochs = range(1, N_EPOCHS + 1)
    
    plt.figure(figsize=(9, 3), dpi=300)

    # MSE
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train')
    plt.plot(epochs, val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('')
    plt.title('MSE Loss Curve')
    plt.legend()

    # SSIM
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_ssims, label='Train')
    plt.plot(epochs, val_ssims, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('')
    plt.title('SSIM Curve')
    plt.legend()

    # PSNR
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_psnrs, label='Train')
    plt.plot(epochs, val_psnrs, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('dB')
    plt.title('PSNR Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_OUTPUT, "metrics_plot_deblur.png"))
    print(f"Saved metrics plot to {CHECKPOINT_OUTPUT}/metrics_plot_deblur.png")

if __name__ == "__main__":
    main()
