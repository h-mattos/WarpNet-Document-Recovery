import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    # holdout_loader = DataLoader(holdout_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # Training loop sketch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNDeblurrer().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    N_EPOCHS = 5

    for epoch in range(1, N_EPOCHS + 1):
        # Training
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS} - Training")
        model.train()
        train_loss_sum = 0.0
        for imgs, targets in loop:
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)
            loop.set_postfix(batch_loss=loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * imgs.size(0)
        avg_train_loss = train_loss_sum / len(train_dataset)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                preds = model(imgs)
                loss = criterion(preds, targets)
                val_loss_sum += loss.item() * imgs.size(0)
        avg_val_loss = val_loss_sum / len(val_dataset)
        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f'{CHECKPOINT_OUTPUT}/deblur_predictor.pth')
    print(f"Saved trained model to {CHECKPOINT_OUTPUT}/deblur_predictor.pth")

if __name__ == "__main__":
    main()
