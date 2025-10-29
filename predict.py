import os
import torch
from PIL import Image
from torchvision import transforms
from model import SegNet
import argparse

def get_device():
    """Gets the best available device for PyTorch."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    if hasattr(torch.backends, "directml") and torch.backends.directml.is_available():
        return torch.device("directml")
    return torch.device("cpu")

def predict(image_path, output_dir, checkpoint_path, device):
    model = SegNet(in_channels=1, num_classes=1).to(device)
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}. Please train the model first.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Model loaded from {checkpoint_path}")

    img = Image.open(image_path).convert("L")
    original_size = img.size
    target_size = (256, 256)
    img_resized = img.resize(target_size, Image.LANCZOS)

    transform = transforms.ToTensor()
    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = (torch.sigmoid(output).squeeze().cpu() > 0.5).byte() * 255

    pred_mask_pil = Image.fromarray(pred_mask.numpy(), mode='L')
    pred_mask_resized = pred_mask_pil.resize(original_size, Image.NEAREST)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_mask.png")
    pred_mask_resized.save(output_path)
    print(f"Prediction saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate mask from satellite image")
    parser.add_argument("--image", type=str, required=True, help="Path to input satellite image")
    parser.add_argument("--output", type=str, default="outputs", help="Directory to save output mask")
    parser.add_argument("--checkpoint", type=str, default="models/saved_models/segnet_best_model.pth", help="Path to model checkpoint")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    predict(args.image, args.output, args.checkpoint, device)

if __name__ == "__main__":
    main()
