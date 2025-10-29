# Satellite Image Segmentation Project

oil spill detection using SegNet model.

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


