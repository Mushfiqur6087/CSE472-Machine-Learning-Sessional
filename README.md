# CSE472 Machine Learning Sessional

Coursework repository for CSE472 (Machine Learning Sessional), including assignments on data preprocessing, tree-based methods, and convolutional neural networks.

## Overview

This repository contains:

- Data preprocessing and feature engineering work in notebooks.
- From-scratch implementations and analysis of Decision Trees, Random Forests, and Extra Trees.
- Multiple CNN online assignment sets (NiN, custom ResNet-style model, Inception-style model, simple CNN, and other architecture exercises).

## Repository Structure

```text
CSE472-Machine-Learning-Sessional/
|-- CNN Online/
|   |-- 20/
|   |   |-- A1/
|   |   `-- B1/
|   |-- 21/
|   |   |-- A1-A2/
|   |   |-- B1-B2/
|   |   `-- C1-C2/
|   |-- A1/
|   |-- A2/
|   |-- B1/
|   `-- B2/
|-- Data Preprocessing & Feature Engineering/
`-- Decision Trees, Random Forests, and Extra Trees/
```

## Main Folders

### `Data Preprocessing & Feature Engineering/`

- Assignment notebook and report for preprocessing and feature engineering tasks.

### `Decision Trees, Random Forests, and Extra Trees/`

- Notebook-based implementation and experiments.
- Detailed write-up in `Report.md`.
- Generated result figures for model comparisons.

### `CNN Online/`

Contains several online assignment sets:

- `20/A1` and `20/B1`: NiN-style CNN experiments on CIFAR-10.
- `21/A1-A2`: Custom residual-network style image classifier.
- `21/B1-B2`: Inception-style network with a custom optimizer.
- `21/C1-C2`: Multi-block custom CNN classifier.
- `A2`: Tiny U-Net exercise template (`unet.py`).
- `B1`: MobileNet-style exercise template (`Online-B1.py`).
- `B2`: SqueezeNet-like assignment (`2005107.py`) and question file.

## Setup

Use Python 3.9+ (3.10+ recommended).

1. Create and activate a virtual environment.
2. Install required packages:

```bash
pip install torch torchvision numpy scikit-learn matplotlib jupyter sympy
```

Some notebooks may additionally use common data science packages such as `pandas` or `seaborn`.

## How To Run

Run scripts from their own folder so relative paths (for datasets and image folders) resolve correctly.

Examples:

```bash
cd "CNN Online/20/A1"
python3 solution_20_A1.py

cd "../B1"
python3 2005080.py

cd "../../21/A1-A2"
python3 solution_21_A.py

cd "../B1-B2"
python3 solution_21_B.py

cd "../C1-C2"
python3 solution_21_C.py

cd "../../B2"
python3 2005107.py
```

To work with notebooks:

```bash
jupyter notebook
```

Then open the `.ipynb` files in the relevant directories.

## Dataset Notes

- Several scripts use `torchvision.datasets` and will automatically download data to a local `data/` directory.
- The folders `CNN Online/21/B1-B2/images/` and `CNN Online/21/C1-C2/images/` contain two classes (`NORMAL` and `PNEUMONIA`) used by `ImageFolder`-based scripts.
- Keep directory names unchanged, because multiple scripts depend on relative folder names such as `images/`.

## Important Notes

- Some files are starter templates and include `TODO` sections (for example `CNN Online/A2/unet.py` and `CNN Online/B1/Online-B1.py`).
- Other files are solution versions and are more directly runnable.
