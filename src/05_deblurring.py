import sys
import os
import cv2
import numpy as np
from tqdm import tqdm
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.blur_cnn import CNNDeblurrer

blurred_dir = "data/blurred"
model_checkpoint = "checkpoints/deblur_predictor.pth"
output_dir = "data/deblurred"

os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNDeblurrer().to(device)
model.load_state_dict(torch.load(model_checkpoint, map_location=device))
model.eval()

for file_name in tqdm(sorted(os.listdir(blurred_dir))):
    img_id = file_name.replace("_blur.png", "")
    blurred_path = os.path.join(blurred_dir, file_name)
    output_path = os.path.join(output_dir, f"{img_id}_deblur.png")

    blurred = cv2.imread(blurred_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    tensor = torch.from_numpy(blurred).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        output = output.squeeze().cpu().numpy()
        output = np.clip(output, 0, 1)

    cv2.imwrite(output_path, (output * 255).astype(np.uint8))