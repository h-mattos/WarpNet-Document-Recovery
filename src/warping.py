import numpy as np
from scipy.ndimage import map_coordinates
from scipy.stats import truncnorm
import numpy as np
from PIL import Image
import os
from math import prod
from tqdm import tqdm
from utils import fs, functions as func


IMAGES_PATH = "data/BMVC_image_data/orig"
OUTPUT_PATH = "data/warped"
GRID_SIZE = 5, 5
IMG_SHAPE = 300, 300


def main():
    # Pre-generate displacement values
    SIGMA = 5
    N_SIGMAS = 2

    assert fs.validate_path(OUTPUT_PATH)

    images_files = sorted(os.listdir(IMAGES_PATH))
    n_images = len(images_files)
    rng = np.random.default_rng(42)
    displacements = np.reshape(
        truncnorm.rvs(
            -N_SIGMAS,
            N_SIGMAS,
            loc=0,
            scale=SIGMA,
            size=n_images * prod(GRID_SIZE) * 2,
            random_state=rng,
        ),
        (n_images, 2, *GRID_SIZE),
    ).astype(np.float32)
    xs = np.linspace(0, IMG_SHAPE[0], GRID_SIZE[0])
    ys = np.linspace(0, IMG_SHAPE[1], GRID_SIZE[1])

    np.save("./data/displacements.npy", displacements)

    # Apply displacements
    for i, image_file in tqdm(
        enumerate(images_files), total=n_images, desc="Processing Warping"
    ):
        coords_x, coords_y = func.disp2coords(displacements[i], IMG_SHAPE, xs, ys)
        image = (
            np.array(
                Image.open(f"{IMAGES_PATH}/{image_file}").convert("L"), dtype=np.float32
            )
            / 255.0
        )
        warped = map_coordinates(
            image, [coords_y, coords_x], order=3, mode="constant", cval=1
        ).reshape(IMG_SHAPE)
        Image.fromarray(
            (np.minimum(np.maximum(warped, 0), 1) * 255).astype(np.uint8)
        ).save(f"{OUTPUT_PATH}/{image_file}")


if __name__ == "__main__":
    main()

# Apply inverse warp (dewarping)
# coords_y_inv = (YY - disp_y).ravel()
# coords_x_inv = (XX - disp_x).ravel()
# dewarped = map_coordinates(warped, [coords_y_inv, coords_x_inv], order=3, mode='reflect').reshape((h, w))
