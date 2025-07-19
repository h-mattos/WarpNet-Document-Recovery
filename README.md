# WarpNet-Document-Recovery
Joint deblurring and dewarping of document images using deep learning, for improved OCR readiness. CS7643 Deep Learning project by Joshua Batkhan and Hector Mattos.

---

## Table of Contents

- [WarpNet-Document-Recovery](#warpnet-document-recovery)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Installation](#installation)
    - [Clone repository](#clone-repository)
  - [Environment setup](#environment-setup)
    - [Create and activate environment](#create-and-activate-environment)
  - [Data](#data)
  - [Usage](#usage)
    - [Training](#training)
    - [Evaluation](#evaluation)
  - [Project Structure](#project-structure)
  - [Contributing](#contributing)
    - [Linting and Formatting](#linting-and-formatting)
  - [Authors](#authors)
  - [License](#license)
  - [References](#references)

---

## Overview

This repository implements a unified pipeline for document image recovery.  
It synthesizes geometric warps and blur on clean page scans, then trains a deep network to jointly  
rectify warping and remove blur. Final outputs are optimized for downstream OCR accuracy.

---

## Features

- **Synthetic Distortion Generation**  
  Create warped + blurred inputs via OpenCV remapping, SciPy splines, and custom PSF kernels.  
- **Page Rectification**  
  Spatial transformer network based on ResNet to predict dewarping control points (Kornia TPS).  
- **Deblurring Module**  
  U-Net or DeblurGAN-v2 restores sharp text from blurred inputs.  
- **End-to-End Training**  
  Joint L₁ + perceptual loss optimizes both sub‐modules concurrently.  
- **Evaluation Suite**  
  PSNR/SSIM for image fidelity; CER/WER (via pretrained TrOCR) for OCR performance.

---

## Installation

### Clone repository

```sh
git clone https://github.gatech.edu/hmattos3/WarpNet-Document-Recovery.git
cd WarpNet-Document-Recovery
```

---

## Environment setup

We use conda to manage Python and GPU dependencies. An example environment.yml is provided.

### Create and activate environment

```sh
conda env create -f environment.yml
conda activate warpnet_env
# Update environment
conda env update -f environment.yml
```

---

## Data

1. Register or login to Kaggle.
2. Download the “Text Deblurring Dataset with PSF for OCR” (66 K images) from:
https://www.kaggle.com/datasets/anggadwisunarto/text-deblurring-dataset-with-psf-for-ocr
1. Unzip into data/raw/

---

## Usage

### Training

Train joint dewarping + deblurring

```sh
python scripts/train.py \
  --data_dir data/raw \
  --output_dir experiments/exp01 \
  --batch_size 16 \
  --epochs 100 \
  --learning_rate 1e-4
```

### Evaluation

Compute PSNR/SSIM on test set

```sh
python scripts/evaluate_image_metrics.py \
  --pred_dir experiments/exp01/predictions \
  --gt_dir data/raw/original \
  --metrics psnr ssim
```

Compute CER/WER using TrOCR

```sh
python scripts/evaluate_ocr.py \
  --pred_dir experiments/exp01/predictions \
  --gt_text data/annotations/gt.txt
```

---

## Project Structure



---

## Contributing

Contributions are welcome. Please open an issue to discuss any major changes before submitting a pull request.
Ensure that new code includes tests or examples demonstrating its functionality.

### Linting and Formatting

This section describes how to enforce consistent code style and catch errors early using Ruff. Run Ruff to check all Python files:

```sh
ruff .
```

Apply automatic fixes for supported rules:

```sh
ruff --fix .
```

---

## Authors
- Joshua Batkhan
- Hector Mattos

---

## License

This project is released under the MIT License. See LICENSE for details.

---

## References

1.	Ma, K. et al. DocUNet: Document image unwarping via a stacked U-Net. CVPR (2018).
2.	Das, S. et al. DewarpNet: Single-image document unwarping with 3D+2D regression. ICCV (2019).
3.	Kupyn, O. et al. DeblurGAN: Blind motion deblurring using conditional adversarial nets. arXiv (2017).
4.	Jiang, X. et al. Revisiting document image dewarping by grid regularization. CVPR (2022).
5.	Li, H. et al. Foreground & text-lines aware document rectification. ICCV (2023).

