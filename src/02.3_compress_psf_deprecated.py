import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.compress import png2h5


IMG_DIR = './data/normalized_psf'
OUTPUT = './data/normalized_psf.h5'
W, H = 19, 19

def main():
    png2h5(IMG_DIR, OUTPUT, W, H)

if __name__ == '__main__':
    main()