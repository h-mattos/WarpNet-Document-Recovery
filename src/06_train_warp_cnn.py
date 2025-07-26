import os
import sys
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.warp_cnn import ImageToTensorDataset, ConvRegressor


def main():
    warped_image_dir="data/warped"
    displacements_file="data/displacements.npy"
    blurred_file="data/blurred.h5"
    ids = np.arange(len([f for f in os.listdir(warped_image_dir) if f.endswith('.png')]))

    # 70-15-15 train-validation-holdout split
    train_ids, other_ids = train_test_split(ids, test_size=0.3, random_state=7643)
    val_ids, holdout_ids = train_test_split(other_ids, test_size=0.5, random_state=7643)
    print(f"Train size: {len(train_ids)}, Validation size: {len(val_ids)}, Holdout size: {len(holdout_ids)}")


    # Setup transforms
    transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            # optionally normalize, e.g. transforms.Normalize(mean, std)
        ]
    )

    # Setup dataset and dataloader
    train_dataset = ImageToTensorDataset(displacements_file, blurred_file, transform=transform, ids=train_ids)
    val_dataset = ImageToTensorDataset(displacements_file, blurred_file, transform=transform, ids=val_ids)
    # holdout_dataset = ImageToTensorDataset(displacements_file, blurred_file, transform=transform, ids=holdout_ids)

    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    # holdout_loader = DataLoader(holdout_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Training loop sketch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    model = ConvRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # N_EPOCHS = 5
    N_EPOCHS      = 50
    best_val_loss = float('inf')
    patience      = 3
    counter       = 0

    train_losses = []
    val_losses = []

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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(batch_loss=loss.item())
            train_loss_sum += loss.item() * imgs.size(0)
        avg_train_loss = train_loss_sum / len(train_dataset)
        train_losses.append(avg_train_loss)

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
        val_losses.append(avg_val_loss)
        # print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        # early‐stop logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_warp_regressor.pth")
            print(f"Epoch {epoch:02d}: val improved to {avg_val_loss:.4f}, saved best model")
        else:
            counter += 1
            print(f"Epoch {epoch:02d}: val {avg_val_loss:.4f} (no improvement, {counter}/{patience})")
            if counter >= patience:
                print(f"No improvement for {patience} epochs → early stopping")
                break


    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/warp_regressor.pth")
    print("Saved trained model to checkpoints/warp_regressor.pth")

    epochs = range(1, epoch + 1)
    
    plt.figure(figsize=(4, 4), dpi=300)

    # MSE
    plt.plot(epochs, train_losses, label='Train')
    plt.plot(epochs, val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('')
    plt.title('MSE Loss Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig("checkpoints/metrics_plot_dewarp.png")
    print(f"Saved metrics plot to checkpoints/metrics_plot_dewarp.png")    


if __name__ == "__main__":
    main()