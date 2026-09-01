# Skin Cancer Multimodal Classification (HAM10000)

This repository trains binary **benign vs. malignant** models on the **HAM10000** dermoscopic dataset using **images only**, **clinical metadata only**, or a **multimodal** combination (ResNet-18 + encoded tabular features).

## Prerequisites

- Python 3.10+ recommended  
- [PyTorch](https://pytorch.org/) with CUDA optional (CPU runs are supported but slower)  
- Dependencies used by the training scripts include: `torch`, `torchvision`, `pandas`, `scikit-learn`, `Pillow`, `matplotlib`, `seaborn`, `tqdm`, `numpy`

You can install packages manually or start from `environment.txt`. 

## Dataset: download and folder layout

Download the dataset from https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data
and place files under the project’s `data` folder.

### Expected directory structure

From the **project root** (`skin_cancer_multimodal/`), create and fill `data/raw/` like this:

```text
data/
  raw/
    HAM10000_metadata.csv
    HAM10000_images_part_1/
      ISIC_xxxxxxx.jpg
      ...
    HAM10000_images_part_2/
      ISIC_xxxxxxx.jpg
      ...
  processed/          # created by data preparation (see below)
    train.csv
    val.csv
    test.csv
```

Paths are defined in `src/config.py`: metadata at `data/raw/HAM10000_metadata.csv`, images under `data/raw/HAM10000_images_part_1` and `data/raw/HAM10000_images_part_2`.

## Quick start

Run all commands from the **project root** so imports resolve (`python -m src.<module>`).

1. **Prepare splits** — reads metadata, resolves image paths, maps diagnoses to binary labels, and writes stratified `train.csv` / `val.csv` / `test.csv` under `data/processed/`:

   ```bash
   python -m src.data_prep
   ```

2. **Train models** (after `data_prep`):

   - Image-only (ResNet-18):

     ```bash
     python -m src.train_image
     ```

   - Metadata-only (Random Forest on age, sex, localization, dx_type):

     ```bash
     python -m src.train_metadata
     ```

   - Multimodal (images + metadata):
    Switch code in dataset.py for ensemble model processing
     ```bash
     python -m src.train_multimodal
     ```

3. **Exploratory analysis** — `src/eda.py` generates figures; it currently uses **hardcoded absolute paths** at the top of the file. Update `BASE_PATH` and `OUT_DIR` to match your machine before running, or refactor it to use `src.config` like the other modules.

## Outputs

Training scripts write under `outputs/` (e.g. figures such as confusion matrices and learning curves). Exact subfolders depend on the script (for example `outputs/figures/image_model/` for the image baseline).

## Configuration

Shared settings (image size, batch size, epochs, learning rate, paths) live in `src/config.py`. Adjust there if you change hardware or training budget.

## Project layout (high level)

| Path | Role |
|------|------|
| `src/config.py` | Paths and hyperparameters |
| `src/data_prep.py` | Build processed CSV splits from raw HAM10000 |
| `src/dataset.py` | Image-only PyTorch dataset |
| `src/multimodal_dataset.py` | Image + metadata dataset |
| `src/models.py` | Multimodal model definition |
| `src/utils.py` | Metadata encoding helpers |
| `src/plot_utils.py` | Saving curves, confusion matrices, sample predictions |

## License and attribution

This **code repository** is separate from the **HAM10000** dataset license. You are responsible for complying with HAM10000’s terms and for citing the dataset appropriately in any publication or coursework.

## Academic integrity

See `statement.txt` in this repository for a plagiarism statement. 
