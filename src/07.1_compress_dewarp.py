import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.compress import png2h5


IMG_DIR = './data/dewarped'
OUTPUT = './data/dewarped.h5'
W, H = 300, 300

def main():
    png2h5(IMG_DIR, OUTPUT, W, H)

if __name__ == '__main__':
    main()