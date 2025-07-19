import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates
from scipy.stats import truncnorm
import numpy as np
from PIL import Image
import os
from math import prod
from tqdm import tqdm
import argparse


IMAGES_PATH = 'data/BMVC_image_data/orig'
OUTPUT_PATH = 'data/warped'
GRID_SIZE = 5, 5
IMG_SHAPE = 300, 300

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_coords', default=False, type=bool)
    args = parser.parse_args()
    if args.compute_coords:
        # Pre-generate displacement values
        SIGMA = 5
        N_SIGMAS = 2
        images_files = os.listdir(IMAGES_PATH)
        n_images = len(images_files)
        rng = np.random.default_rng(42)
        displacements = np.reshape(truncnorm.rvs(-N_SIGMAS, N_SIGMAS, loc=0, scale=SIGMA, size=n_images*prod(GRID_SIZE)*2, random_state=rng), (n_images, 2, *GRID_SIZE)).astype(np.float32)
        xs = np.linspace(0, IMG_SHAPE[0], GRID_SIZE[0])
        ys = np.linspace(0, IMG_SHAPE[1], GRID_SIZE[1])
        # coords = np.zeros(shape=(n_images, 2, prod(IMG_SHAPE)), dtype=np.float32)
        coords = np.memmap(
            './data/coords.dat',
            dtype=np.float32,
            mode='w+',
            shape=(n_images,2,prod(IMG_SHAPE))
        )
        for i in tqdm(range(displacements.shape[0]), total=n_images, desc='Processing Displacements'):
            spline_dx = RectBivariateSpline(ys, xs, displacements[i][0])
            spline_dy = RectBivariateSpline(ys, xs, displacements[i][1])
            XX, YY = np.meshgrid(np.arange(IMG_SHAPE[0]), np.arange(IMG_SHAPE[1]))
            disp_x = spline_dx(YY[:,0], XX[0,:])
            disp_y = spline_dy(YY[:,0], XX[0,:])
            coords[i][0] = (XX + disp_x).ravel()
            coords[i][1] = (YY + disp_y).ravel()

        np.save('./data/displacements.npy', displacements)

    coords = np.memmap(
        './data/coords.dat',
        dtype=np.float32,
        mode='r',
        shape=(n_images,2,prod(IMG_SHAPE))
    )

    # Apply displacements
    for i, image_file in tqdm(enumerate(images_files), total=n_images, desc='Processing Warping'):
        image = np.array(
            Image.open(f'{IMAGES_PATH}/{image_file}').convert('L'),
            dtype=np.float32
        ) / 255.0
        warped = map_coordinates(image, [coords[i][0], coords[i][1]], order=3, mode='constant', cval=1).reshape(IMG_SHAPE)
        Image.fromarray((np.minimum(np.maximum(warped, 0), 1) * 255).astype(np.uint8), mode='L').save(f'{OUTPUT_PATH}/{image_file}')


if __name__ == '__main__':
    main()

# Apply inverse warp (dewarping)
# coords_y_inv = (YY - disp_y).ravel()
# coords_x_inv = (XX - disp_x).ravel()
# dewarped = map_coordinates(warped, [coords_y_inv, coords_x_inv], order=3, mode='reflect').reshape((h, w))

