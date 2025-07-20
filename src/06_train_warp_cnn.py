import torch
from torch import nn
from src.models.warp_cnn import ImageToTensorDataset, ConvRegressor
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Setup transforms, dataset, dataloader
transform = transforms.Compose(
    [
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        # optionally normalize, e.g. transforms.Normalize(mean, std)
    ]
)

dataset = ImageToTensorDataset(
    image_dir="data/warped",
    displacements_file="data/displacements.npy",
    transform=transform,
)
dataloader = DataLoader(
    dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
)

# Training loop sketch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvRegressor().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

N_EPOCHS = 5

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
