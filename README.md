# Marine Oil Spill Detection with SegNet

A semantic segmentation pipeline for detecting marine oil spills in SAR (Synthetic Aperture Radar) satellite imagery using a custom SegNet architecture trained on PALSAR and Sentinel data.

---

## Overview

This project implements a binary segmentation model that classifies each pixel in a SAR image as either **oil spill** or **background**. SegNet's encoder–decoder structure with max-pooling indices makes it well suited for this task - preserving spatial resolution during upsampling without needing skip connections.

<img width="2153" height="772" alt="image" src="https://github.com/user-attachments/assets/4b4a3253-9719-420f-8832-e90655dd7c73" />

The model is trained on grayscale SAR images from two sensors and evaluated with standard segmentation metrics including IoU, Dice, F1, ROC-AUC, and Precision-Recall.

---

## Model Architecture

The model is a custom **SegNet** with:

- **Encoder**: 5 convolutional blocks (64 → 128 → 256 → 512 → 512 channels), each followed by MaxPool2d with stored indices
- **Decoder**: 5 mirrored upsampling blocks using `max_unpool2d` with the stored indices, preserving spatial information
- **Output**: 1×1 convolution producing a single-channel logit map (binary segmentation)
- **Loss**: `BCEWithLogitsLoss`
- **Optimizer**: Adam with `ReduceLROnPlateau` scheduler (patience=5, factor=0.5)
- **Input**: Single-channel (grayscale) SAR images resized to 256×256

---

## Dataset

The dataset contains SAR satellite imagery with paired ground-truth masks from two sensors:

```
dataset/
├── train/
│   ├── palsar/        # *_sat.jpg + *_mask.png (co-located)
│   └── sentinel/
└── test/
    ├── palsar/
    │   ├── sat/       # *_sat.jpg
    │   └── gt/        # *_mask.png
    └── sentinel/
        ├── sat/
        └── gt/
```

- **PALSAR**: L-band SAR from ALOS-2 — 776 test image/mask pairs
- **Sentinel**: C-band SAR from Sentinel-1
- Images are grayscale JPEGs; masks are binary PNGs (oil spill = white, background = black)

---

## Results

Evaluation on the combined PALSAR + Sentinel test set:

| Metric       | Score  |
|--------------|--------|
| Accuracy     | 88.65% |
| Precision    | 81.31% |
| Recall       | 73.43% |
| Specificity  | 94.04% |
| F1 Score     | 77.17% |
| IoU          | 62.82% |
| Dice         | 77.17% |
| ROC-AUC      | 94.10% |
| Avg Precision| 86.03% |

---

## Project Structure

```
Oilspill_DetectionSegNet/
├── dataset/                   # SAR images and ground-truth masks
├── models/
│   └── saved_models/          # Saved .pth checkpoints
├── outputs/
│   ├── training_plots/        # Loss, F1/IoU, Precision/Recall curves + CSV
│   ├── evaluation/            # Confusion matrix, ROC, PR curves + CSV
│   └── Sample outputs/        # Example prediction visuals
├── dataset.py                 # OilSpillDataset — PyTorch Dataset class
├── model.py                   # SegNet architecture
├── train.py                   # Training loop with metrics and plotting
├── evaluate.py                # Full evaluation on test set
├── predict.py                 # Single-image inference with overlay
└── requirements.txt
```

---

## Setup

**Python 3.9+** recommended.

```bash
pip install -r requirements.txt
```

GPU acceleration is supported on CUDA, Apple MPS, and DirectML. The scripts auto-detect the best available device.

---

## Usage

### Train

```bash
python train.py --epochs 100 --batch-size 2 --lr 1e-3 --sensor both
```

| Argument       | Default | Description                                      |
|----------------|---------|--------------------------------------------------|
| `--epochs`     | 100     | Number of training epochs                        |
| `--batch-size` | 2       | Batch size                                       |
| `--lr`         | 1e-3    | Initial learning rate                            |
| `--sensor`     | `both`  | Sensor to train on: `palsar`, `sentinel`, `both` |

Training saves:
- `models/saved_models/segnet_best_model.pth` — best checkpoint by loss
- `models/saved_models/segnet_epoch{N}.pth` — every 10 epochs
- `models/saved_models/segnet_final_model.pth` — end of training
- `outputs/training_plots/` — loss, F1/IoU, precision/recall curves and a `training_history.csv`

### Evaluate

```bash
python evaluate.py --checkpoint models/saved_models/segnet_best_model.pth --sensor both
```

| Argument       | Default                                       | Description                          |
|----------------|-----------------------------------------------|--------------------------------------|
| `--checkpoint` | `models/saved_models/segnet_best_model.pth`   | Path to model checkpoint             |
| `--sensor`     | `both`                                        | Sensor subset to evaluate            |
| `--output`     | `outputs/evaluation`                          | Directory for evaluation outputs     |

Produces: confusion matrix, ROC curve, precision-recall curve, and `metrics_summary.csv`.

### Predict (single image)

```bash
python predict.py --image path/to/image.jpg --checkpoint models/saved_models/segnet_best_model.pth
```

Saves a 3-panel figure to `outputs/` showing the original SAR image, predicted binary mask, and a detection overlay with red contours marking the detected spill regions.


---

## Dependencies

| Package        | Purpose                              |
|----------------|--------------------------------------|
| torch          | Model, training, inference           |
| torchvision    | Transforms, dataset utilities        |
| numpy          | Array operations                     |
| opencv-python  | Contour drawing for prediction overlay|
| scikit-learn   | ROC, PR curves, confusion matrix     |
| matplotlib     | Plot generation                      |
| tqdm           | Training progress bars               |
| albumentations | (Available for augmentation)         |


### Run Instructions

1. **Create and activate a virtual environment (Windows):**
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model:**
   ```bash
   # Default: use both PALSAR and Sentinel
   python train.py

   # Only PALSAR
   python train.py --sensor palsar

   # Only Sentinel
   python train.py --sensor sentinel
   ```

4. **Generate a mask for a JPG image:**
   ```bash
   python predict.py --image dataset/test/palsar/sat/10003_sat.jpg --output outputs/
   ```


