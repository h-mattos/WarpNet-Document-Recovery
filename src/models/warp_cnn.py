import os
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
import h5py


# 1. Custom Dataset
class ImageToTensorDataset(Dataset):
    def __init__(self, displacements_file: str, blurred_h5_file: str, transform=None, ids=None):
        self.transform = transform
        # load all displacement tensors in one array
        self.displacements = np.load(displacements_file).astype(np.float32)

        # open HDF5 file and reference the image dataset (assumes dataset named 'imgs')
        self.blurred_h5_file = blurred_h5_file
        self.h5 = None

        if ids is not None:
            self.ids = ids
            self.displacements = self.displacements[self.ids]
        else:
            self.ids = self.all_ids
            self.indices = list(range(len(self.all_ids)))


    def __len__(self):
        return self.displacements.shape[0]


    def __getitem__(self, idx):
        # lazily open HDF5 in worker process
        if self.h5 is None:
            self.h5 = h5py.File(self.blurred_h5_file, 'r')
            self.images = self.h5['imgs']
        # load image array from HDF5
        arr = self.images[self.ids[idx]]  # shape (300, 300)
        # convert to PIL image and RGB
        img = Image.fromarray(arr).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # get target tensor from preloaded array
        target = torch.from_numpy(self.displacements[idx])  # tensor shape (5, 5)
        return img, target


# 2. Simple CNN that outputs a 2×5×5 map for x and y displacements
class ConvRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder path
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)  # 300->150

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)  # 150->75

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Decoder path
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 75->150
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # 150->300
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Final projection to 2 displacement channels and downsample to 5×5 grid
        self.conv_out = nn.Conv2d(64, 2, kernel_size=1)
        self.adapt_pool = nn.AdaptiveAvgPool2d((5, 5))

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        # Bottleneck
        b = self.bottleneck(p2)
        # Decoder
        u2 = self.up2(b)
        d2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(d2)
        u1 = self.up1(d2)
        d1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(d1)
        # Output projection
        out = self.conv_out(d1)
        # Downsample to displacement grid
        out = self.adapt_pool(out)
        return out  # shape (batch, 2, 5, 5)
