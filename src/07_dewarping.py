# Perform dewarping using trained model like in 05_deblurring.py
import os
import sys
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.ndimage import map_coordinates
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.warp_cnn import ImageToTensorDataset, ConvRegressor
from src.utils import fs, functions as func

warped_image_dir="data/warped"
displacements_file="data/displacements.npy"
blurred_file="data/blurred.h5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        # optionally normalize, e.g. transforms.Normalize(mean, std)
    ]
)

# Now that we have a trained model, let's make use of it by loading the best-performing model and using it to predict displacements for new images
best_model_state_dict = torch.load("checkpoints/warp_regressor.pth")
best_model = ConvRegressor().to(device)
best_model.load_state_dict(best_model_state_dict)
best_model.eval()

# Load the first dimension of the blurred images (i.e. each image is a 3D array with shape (height, width, channels))
dataset = ImageToTensorDataset(displacements_file, blurred_file, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

preds = None
with torch.no_grad():
    for imgs, targets in tqdm(data_loader, desc="Predicting"):
        imgs = imgs.to(device)
        targets = targets.to(device)
        tmp_preds = best_model(imgs).cpu().numpy()
        if preds is None:
            preds = tmp_preds
        else:
            preds = np.append(preds, tmp_preds, axis=0)

np.load(displacements_file)[0]
preds[0]
# Apply inverse warp (dewarping)
GRID_SIZE = 5, 5
IMG_SHAPE = 300, 300
IMAGES_PATH = "data/deblurred"
OUTPUT_PATH = "data/dewarped"
xs = np.linspace(0, IMG_SHAPE[0], GRID_SIZE[0])
ys = np.linspace(0, IMG_SHAPE[1], GRID_SIZE[1])
image_file = "0000002_deblur.png"

coords_x, coords_y = func.disp2coords(-preds[2], IMG_SHAPE, xs, ys)
image = (
    np.array(
        Image.open(f"{IMAGES_PATH}/{image_file}").convert("L"), dtype=np.float32
    )
    / 255.0
)
warped = map_coordinates(
    image, [coords_y, coords_x], order=3, mode="constant", cval=1
).reshape(IMG_SHAPE)
fs.validate_path(OUTPUT_PATH)
Image.fromarray(
    (np.minimum(np.maximum(warped, 0), 1) * 255).astype(np.uint8)
).save(f"{OUTPUT_PATH}/{image_file}")