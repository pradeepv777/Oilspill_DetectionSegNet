# Satellite Image Segmentation Project

This project implements a SegNet-based model for satellite image segmentation, specifically for oil spill detection using SAR (Synthetic Aperture Radar) data.

## Quick Start

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


### Training
```bash
# Start training
# Default: both sensors
python train.py

# Only PALSAR
python train.py --sensor palsar

# Only Sentinel
python train.py --sensor sentinel

# Checkpoints are written to:
# models/saved_models/ (best, every 10 epochs, and final)
```

### Prediction
```bash
# Basic usage: generate mask for one JPG
python predict.py --image dataset/test/palsar/sat/10003_sat.jpg --output outputs/

# Use a specific checkpoint
python predict.py --image dataset/test/sentinel/sat/20015_sat.jpg \
  --output outputs/ \
  --checkpoint models/saved_models/segnet_best_model.pth

# Arguments
# --image       Path to input JPG image (required)
# --output      Directory to save PNG mask (default: outputs)
# --checkpoint  Path to model weights (default: models/saved_models/segnet_best_model.pth)
```

## Model Architecture

- **SegNet**: Encoder-decoder architecture with skip connections
- **Input**: Grayscale satellite images (1 channel)
- **Output**: Binary masks (1 channel)
- **Loss**: Binary Cross Entropy with Logits

## Requirements

## Environment Setup (Windows)

Create and activate a virtual environment before installing dependencies:

```bash
# Create venv
python -m venv venv

# Activate (PowerShell)
.\venv\Scripts\Activate.ps1

# Or activate (Command Prompt)
venv\Scripts\activate.bat

# Or activate (Git Bash)
source venv/Scripts/activate

# Deactivate when done
deactivate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- torch>=2.8.0
- torchvision>=0.17.0
- numpy
- opencv-python
- albumentations
- tqdm
- PIL (Pillow)

## Output

The prediction script will:
1. Load your JPG satellite image
2. Convert it to grayscale
3. Resize to 256x256 for processing
4. Generate a binary mask
5. Resize back to original dimensions
6. Save as PNG mask in the output directory

## Troubleshooting

- If you get "Model checkpoint not found", run `python train.py` first
- For memory issues, reduce batch_size in train.py
- Make sure your input image is a valid JPG file