import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import h5py


# 1. Custom Dataset with ability to subset by ID for train-validation-holdout split
class BlurCNNDataset(Dataset):
    def __init__(self, blurred_h5_file: str, deblurred_h5_file: str, ids=None):
        self.blurred_h5_file = blurred_h5_file
        self.deblurred_h5_file = deblurred_h5_file
        self.ids = ids
        self.blurred = None
        self.deblurred = None

    def _lazy_init(self):
        if self.blurred is None:
            self.blurred = h5py.File(self.blurred_h5_file, 'r')['imgs']
        if self.deblurred is None:
            self.deblurred = h5py.File(self.deblurred_h5_file, 'r')['imgs']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        self._lazy_init()
        img_arr = self.blurred[self.ids[idx]]  # shape (300, 300)
        out_arr = self.deblurred[self.ids[idx]]  # shape (300, 300)
        
        image = torch.from_numpy(np.array(img_arr, dtype=np.float32) / 255.0).unsqueeze(0)
        out = torch.from_numpy(np.array(out_arr, dtype=np.float32) / 255.0).unsqueeze(0)

        return image, out

# 2. CNN that takes in a blurred image and outputs a deblurred one
# Adapting the U-Net architecture from: https://arxiv.org/pdf/1505.04597
# Borrowing from the warp_cnn architecture since that was having success
class CNNDeblurrer(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder path
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

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
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv_out = nn.Conv2d(64, 1, kernel_size=1)
        
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
       
        return out  # shape (batch, 1, 300, 300)
