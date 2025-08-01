import h5py, numpy as np
from PIL import Image
import os
from tqdm import tqdm
from utils.fs import validate_path


def png2h5(img_dir:str, output:str, w:int, h:int) -> None:
    images_files = os.listdir(img_dir)
    n_images = len(images_files)
    validate_path(os.path.split(output)[0])

    with h5py.File(output, 'w') as f:
        dset = f.create_dataset(
            'imgs',
            shape=(n_images, h, w),
            dtype='uint8',
            chunks=(1, h, w),         # one image per chunk
            compression='lzf'            # fast, lightweight compression
        )
        for i, fname in tqdm(enumerate(sorted(images_files)), total=n_images, desc='Compressing'):
            img = Image.open(os.path.join(img_dir, fname))
            # if img is not grayscale, convert to grayscale
            if img.mode != 'L':
                img = img.convert('L')
            arr = np.array(img)[np.newaxis, :, :]
            dset[i] = arr