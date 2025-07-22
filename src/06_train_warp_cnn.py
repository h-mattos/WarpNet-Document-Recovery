import os
import sys
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.warp_cnn import ImageToTensorDataset, ConvRegressor

def main():
    image_dir="data/warped"
    displacements_file="data/displacements.npy"
    ids = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.png')]

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
    train_dataset = ImageToTensorDataset(image_dir, displacements_file, transform=transform, ids=train_ids)
    val_dataset = ImageToTensorDataset(image_dir, displacements_file, transform=transform, ids=val_ids)
    # holdout_dataset = ImageToTensorDataset(image_dir, displacements_file, transform=transform, ids=holdout_ids)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    # holdout_loader = DataLoader(holdout_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Training loop sketch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvRegressor().to(device)
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
    torch.save(model.state_dict(), "checkpoints/warp_regressor.pth")
    print("Saved trained model to checkpoints/warp_regressor.pth")

if __name__ == "__main__":
    main()