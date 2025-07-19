import os
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 1. Custom Dataset
class ImageToTensorDataset(Dataset):
    def __init__(self, image_dir: str, tensor_dir: str, transform=None):
        self.image_dir = image_dir
        self.tensor_dir = tensor_dir
        self.transform = transform
        # assume identical basenames, e.g. img_001.png  ↔  img_001.npy
        self.ids = [
            os.path.splitext(f)[0]
            for f in os.listdir(image_dir)
            if f.lower().endswith(".png")
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        # load image
        img_path = os.path.join(self.image_dir, id_ + ".png")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # load target tensor
        tensor_path = os.path.join(self.tensor_dir, id_ + ".npy")
        target = np.load(tensor_path).astype(np.float32)  # shape (5, 5)
        target = torch.from_numpy(target)  # tensor shape (5, 5)
        return img, target


# 2. Simple CNN that outputs a 5×5 map
class ConvRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # two conv blocks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # reduce spatial to 5×5 regardless of input (300→150→75)
        self.adapt_pool = nn.AdaptiveAvgPool2d((5, 5))
        # project 64 channels → 1 channel
        self.conv_out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.adapt_pool(x)  # shape (batch, 64, 5, 5)
        x = self.conv_out(x)  # shape (batch, 1, 5, 5)
        return x.squeeze(1)  # shape (batch, 5, 5)


# 3. Setup transforms, dataset, dataloader
transform = transforms.Compose(
    [
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        # optionally normalize, e.g. transforms.Normalize(mean, std)
    ]
)

dataset = ImageToTensorDataset(
    image_dir="path/to/images", tensor_dir="path/to/tensors", transform=transform
)
dataloader = DataLoader(
    dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
)

# 4. Training loop sketch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvRegressor().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 6):  # e.g. 5 epochs
    model.train()
    total_loss = 0.0
    for imgs, targets in dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        preds = model(imgs)
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch:02d}, Loss {avg_loss:.4f}")
