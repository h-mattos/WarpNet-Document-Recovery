import os
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset


# 1. Custom Dataset
class ImageToTensorDataset(Dataset):
    def __init__(self, image_dir: str, displacements_file: str, transform=None, ids=None):
        self.image_dir = image_dir
        self.transform = transform
        # load all displacement tensors in one array
        self.displacements = np.load(displacements_file).astype(np.float32)
        # assume one displacement per image, order matches directory listing
        self.all_ids = [
            os.path.splitext(f)[0]
            for f in sorted(os.listdir(image_dir))
            if f.lower().endswith(".png")
        ]

        if ids is not None:
            self.ids = ids
            self.ids_to_idx = {id : idx for idx, id in enumerate(self.all_ids)}
            self.indices = [self.ids_to_idx[id] for id in ids]
        else:
            self.ids = self.all_ids
            self.indices = list(range(len(self.all_ids)))

        assert self.displacements.shape[0] == len(self.all_ids), (
            f"Number of tensors ({self.displacements.shape[0]}) does not match number of images ({len(self.all_ids)})"
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        # load image
        img_path = os.path.join(self.image_dir, id_ + ".png")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # get target tensor from preloaded array
        target = torch.from_numpy(self.displacements[self.indices[idx]])  # tensor shape (5, 5)
        return img, target


# 2. Simple CNN that outputs a 2×5×5 map for x and y displacements
class ConvRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # two conv blocks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # reduce spatial to 5×5 regardless of input (300→150→75)
        self.adapt_pool = nn.AdaptiveAvgPool2d((5, 5))
        # project 64 channels → 2 channels for x and y displacements
        self.conv_out = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.adapt_pool(x)  # shape (batch, 64, 5, 5)
        x = self.conv_out(x)  # shape (batch, 2, 5, 5)
        return x  # shape (batch, 2, 5, 5)
