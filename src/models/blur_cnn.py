import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import h5py


# 1. Custom Dataset with ability to subset by ID for train-validation-holdout split
class PSFDataset(Dataset):
    def __init__(self, images_h5_file: str, psf_h5_file: str, ids=None):
        self.images_h5_file = images_h5_file
        self.psf_h5_file = psf_h5_file
        self.images = None
        self.psf = None

        self.all_ids = list(range(h5py.File(self.psf_h5_file, 'r')['imgs'].shape[0]))

        if ids is not None:
            self.ids = ids
        else:
            self.ids = self.all_ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        if self.images is None:
            self.images = h5py.File(self.images_h5_file, 'r')['imgs']
        if self.psf is None:
            self.psf = h5py.File(self.psf_h5_file, 'r')['imgs']
        img_arr = self.images[self.ids[idx]]  # shape (300, 300)
        psf_arr = self.psf[self.ids[idx]]  # shape (19, 19)
        
        image = torch.from_numpy(np.array(img_arr, dtype=np.float32) / 255.0).unsqueeze(0)
        psf = torch.from_numpy(np.array(psf_arr, dtype=np.float32) / 255.0).unsqueeze(0)

        return image, psf

# 2. CNN that outputs a psf kernel corresponding to a blurred image
class PSFPredictor(nn.Module):
    def __init__(self, psf_size=19):
        super(PSFPredictor, self).__init__()
        self.psf_size = psf_size

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2), # 300 to 150
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 150 to 75
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 75 to 38
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 38 to 19
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, psf_size * psf_size)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        psf_flat = self.decoder(x)
        psf = psf_flat.view(-1, 1, self.psf_size, self.psf_size)
        psf = torch.relu(psf)
        psf = psf / (psf.sum(dim=[2,3], keepdim=True) + 1e-8)
        # psf = F.softmax(psf.view(batch_size, -1), dim=-1).view_as(psf)
        return psf
