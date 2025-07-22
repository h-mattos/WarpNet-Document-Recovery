import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.blur_cnn import PSFDataset, PSFPredictor

def main():
    # Setup dataset and dataloader
    dataset = PSFDataset(
        images_dir = "data/blurred",
        psf_dir = "data/normalized_psf"
    )
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
    )

    # Training loop sketch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PSFPredictor(psf_size=19).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    N_EPOCHS = 1

    for epoch in range(1, N_EPOCHS + 1):
        # wrap dataloader with progress bar
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{N_EPOCHS}")
        model.train()
        total_loss = 0.0
        for imgs, targets in loop:
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)
            loop.set_postfix(batch_loss=loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:02d}, Loss {avg_loss:.4f}")

    torch.save(model.state_dict(), "checkpoints/psf_predictor.pth")
    print("Saved trained model to checkpoints/psf_predictor.pth")

if __name__ == "__main__":
    main()