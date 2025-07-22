import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# 1. Custom Dataset with ability to subset by ID for train-validation-holdout split
class PSFDataset(Dataset):
    def __init__(self, images_dir, psf_dir, ids=None):
        self.image_dir = images_dir
        self.psf_dir = psf_dir

        if ids is not None:
            self.ids = ids
        else:
            self.ids = [os.path.splitext(f)[0].replace('_blur', '') for f in os.listdir(images_dir) if f.endswith('blur.png')]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}_blur.png")
        psf_path = os.path.join(self.psf_dir, f"{img_id}_psf.png")

        image = Image.open(img_path).convert('L')
        psf = Image.open(psf_path).convert('L')

        image = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).unsqueeze(0)
        psf = torch.from_numpy(np.array(psf, dtype=np.float32) / 255.0).unsqueeze(0)

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
        psf = F.softmax(psf.view(batch_size, -1), dim=-1).view_as(psf)
        return psf
