# THIS IS A CLONE OF THE [ORIGINAL REPO](https://github.gatech.edu/hmattos3/WarpNet-Document-Recovery).

Gradescope did not let us connect to the original repo, so we made a clone of it.

# WarpNet-Document-Recovery

Joint deblurring and dewarping of document images using deep learning, for improved OCR readiness. CS7643 Deep Learning project by Joshua Batkhan and Hector Mattos.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [Clone Repository](#clone-repository)
  - [Environment Setup](#environment-setup)
- [Data](#data)
- [Usage](#usage)
  - [Synthetic Distortion Generation](#synthetic-distortion-generation)
  - [Training Deblurring Model](#training-deblurring-model)
  - [Training Dewarping Model](#training-dewarping-model)
  - [Running the Full Pipeline](#running-the-full-pipeline)
  - [Evaluating Performance](#evaluating-performance)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Authors](#authors)
- [License](#license)
- [References](#references)

---

## Overview

This repository implements a two-stage pipeline for recovering clean document images from synthetically degraded ones. The pipeline first removes blur using a CNN-based deblurring model, then reverses geometric warping using a displacement-grid-based dewarping model. Each model is trained independently using custom datasets constructed from clean document images.

---

## Features

- **Synthetic Degradation Pipeline**: Applies spline-parameterized warping and PSF-based blurring to clean documents.
- **U-Net Inspired Deblurring Network**: Recovers clean but still-warped images from blurred and warped inputs.
- **Dewarping Network**: Predicts displacement grids from distorted inputs to restore geometric structure.
- **Independent Training**: Each model is trained separately with specialized targets.
- **Image Quality Evaluation**: Tracks MSE, SSIM, and PSNR across train/val/holdout splits.
- **Full Pipeline Recovery**: Applies both models sequentially to restore clean documents.

---

## Installation

### Clone Repository
```bash
git clone https://github.gatech.edu/hmattos3/WarpNet-Document-Recovery.git
cd WarpNet-Document-Recovery
```

### Environment Setup
```bash
conda env create -f environment.yml
conda activate warpnet_env
```

---

## Data

1. Register on Kaggle.
2. Download the [Text Deblurring Dataset with PSF for OCR](https://www.kaggle.com/datasets/anggadwisunarto/text-deblurring-dataset-with-psf-for-ocr).
3. Place extracted images and kernels into `data/BMVC_images_data/`.
4. Run preprocessing scripts to generate warped, blurred, and recovered datasets:
```bash
python 01_warping.py
python 02_blurring.py
python 02.1_compress_warp.py
python 02.2_compress_blur.py
python 02.4_compress_blur_orig.py
```

---

## Usage

### Synthetic Distortion Generation
```bash
python 01_warping.py           # Applies geometric warping
python 02_blurring.py          # Applies PSF-based blur
```

### Training Deblurring Model
```bash
python 03_train_blur_cnn.py
```

### Training Dewarping Model
```bash
python 06_train_warp_cnn.py
```

### Running the Full Pipeline
```bash
python 07_dewarping.py         # Apply predicted displacement fields to deblurred images
```

### Evaluating Performance
```bash
python 08_evaluate_performance.py
```
Prints MSE, SSIM, and PSNR across Train, Validation, and Holdout splits for:
- Deblurring only
- Dewarping only
- Full pipeline recovery

---

## Project Structure

```
├── data/
│   ├── BMVC_images_data/        # Clean images + PSF kernels
│   ├── warped.h5                # Warped images
│   ├── blurred_orig.h5          # Blurred originals
│   ├── deblurred.h5             # Outputs from deblurring model
│   ├── dewarped.h5              # Outputs from dewarping model
│   ├── recovered/               # Final recovered PNGs
├── src/
│   ├── models/
│   │   ├── blur_cnn.py          # U-Net for deblurring
│   │   ├── warp_cnn.py          # U-Net for displacement prediction
│   │   ├── train_blur_cnn.py    # Training loop for blur model
│   │   ├── train_warp_cnn.py    # Training loop for warp model
│   ├── utils/
│       ├── blur_utils.py
│       ├── compress.py
│       ├── fs.py
│       ├── functions.py         # SSIM, PSNR, warping utilities
├── 01_warping.py
├── 02_blurring.py
├── 02.1_compress_warp.py
├── 02.2_compress_blur.py
├── 02.4_compress_blur_orig.py
├── 03_train_blur_cnn.py
├── 05_deblurring.py
├── 05.1_compress_deblur.py
├── 06_train_warp_cnn.py
├── 07_dewarping.py
├── 08_evaluate_performance.py
├── environment.yml
```

---

## Contributing

Contributions are welcome. Please open an issue to discuss changes. Include examples/tests with pull requests.

### Linting and Formatting
```bash
ruff .        # Check
ruff --fix .  # Auto-fix
```

---

## Authors
- Joshua Batkhan
- Hector Mattos

---

## License

This project is released under the MIT License.

---

## References

1. Ma, K. et al. DocUNet: Document image unwarping via a stacked U-Net. CVPR (2018).
2. Das, S. et al. DewarpNet: Single-image document unwarping with 3D+2D regression. ICCV (2019).
3. Jiang, X. et al. Revisiting document image dewarping by grid regularization. CVPR (2022).
4. Li, H. et al. Foreground & text-lines aware document rectification. ICCV (2023).
5. Ronneberger, O. et al. U-Net: Convolutional networks for biomedical image segmentation. MICCAI (2015).
6. Dong, C. et al. Learning a deep convolutional network for image super-resolution. ECCV (2014).
7. Dan Stowell. Wiener deconvolution in Python (gist). https://gist.github.com/danstowell/f2d81a897df9e23cc1da, 2015. Accessed: 2025-07-21.
