import os
import torch
from PIL import Image
from torchvision import transforms
from model import SegNet
import argparse
import numpy as np

def predict(image_path, output_dir, checkpoint_path="models/saved_models/segnet_best_model.pth"):
    # -------------------
    # Device setup
    # -------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")   # NVIDIA GPU
    elif torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        device = torch.device("mps")    # Apple Silicon (Mac)
    elif hasattr(torch.backends, "directml") and torch.backends.directml.is_available():
        device = torch.device("directml")  # AMD / Intel GPU (Windows)
    else:
        device = torch.device("cpu")    # CPU fallback
    print(f"Using device: {device}")

    # -------------------
    # Load model
    # -------------------
    model = SegNet(in_channels=1, num_classes=1).to(device)
    
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}")
        print("Please train the model first using: python train.py")
        return
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Model loaded from {checkpoint_path}")

    # -------------------
    # Load image
    # -------------------
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    original_size = img.size
    print(f"Original image size: {original_size}")
    
    # Resize to a standard size if needed (optional)
    target_size = (256, 256)  # You can adjust this
    img_resized = img.resize(target_size, Image.LANCZOS)
    
    transform = transforms.ToTensor()
    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    # -------------------
    # Run inference
    # -------------------
    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu()
        pred_mask = (pred_mask > 0.5).float() * 255

    # -------------------
    # Resize back to original size and save prediction
    # -------------------
    pred_mask_np = pred_mask.byte().numpy()
    pred_mask_pil = Image.fromarray(pred_mask_np, mode='L')
    pred_mask_resized = pred_mask_pil.resize(original_size, Image.NEAREST)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_mask.png")
    
    pred_mask_resized.save(output_path)
    print(f"Prediction saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mask from satellite image")
    parser.add_argument("--image", type=str, required=True, help="Path to input JPG satellite image")
    parser.add_argument("--output", type=str, default="outputs", help="Directory to save output PNG mask")
    parser.add_argument("--checkpoint", type=str, default="models/saved_models/segnet_best_model.pth", help="Path to model checkpoint")
    args = parser.parse_args()

    predict(args.image, args.output, args.checkpoint)
