import os
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

from src.models.blur_cnn import PSFPredictor

blurred_dir = "data/blurred"
out_psf_dir = "data/predicted_psf"
model_checkpoint = "checkpoints/psf_predictor.pth"

os.makedirs(out_psf_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PSFPredictor()
model.load_state_dict(torch.load(model_checkpoint, map_location=device))
model.to(device)
model.eval()

with torch.no_grad():
    for file_name in tqdm(sorted(os.listdir(blurred_dir))):
        img_id = file_name.replace("_blur.png", "")
        img_path = os.path.join(blurred_dir, file_name)

        image = Image.open(img_path).convert("L")
        image = torch.from_numpy(
            np.array(image, dtype=np.float32) / 255.0
        ).unsqueeze(0).unsqueeze(0).to(device)

        psf = model(image)[0, 0].cpu().numpy()
        psf_img = (psf / (psf.max() + 1e-8) * 255).astype(np.uint8)
        Image.fromarray(psf_img).save(os.path.join(out_psf_dir, f"{img_id}_psf.png"))