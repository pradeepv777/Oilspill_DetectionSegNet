import os
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import SegNet
import argparse


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    if hasattr(torch.backends, "directml") and torch.backends.directml.is_available():
        return torch.device("directml")
    return torch.device("cpu")


def predict(image_path, output_dir, checkpoint_path, device):
    # ── Load model ───────────────────────────────────────
    model = SegNet(in_channels=1, num_classes=1).to(device)
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}. Please train the model first.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Model loaded from {checkpoint_path}")

    # ── Load and preprocess image ────────────────────────
    img_pil = Image.open(image_path).convert("L")
    img_resized = img_pil.resize((256, 256), Image.LANCZOS)

    transform = transforms.ToTensor()
    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    # ── Run inference ────────────────────────────────────
    with torch.no_grad():
        output = model(img_tensor)
        prob_map = torch.sigmoid(output).squeeze().cpu().numpy()   # (256, 256) float
        pred_mask = (prob_map > 0.5).astype(np.uint8) * 255       # binary 0/255

    # ── Build overlay (original + red contours) ──────────
    img_np = np.array(img_resized, dtype=np.uint8)                 # (256, 256) gray
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = img_bgr.copy()
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    cv2.addWeighted(overlay, 0.6, img_bgr, 0.4, 0, overlay)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)        

    # ── Build 3panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#1a1a2e")

    panels = [
        (img_np,      "SAR Satellite Image", "gray"),
        (pred_mask,   "Predicted Mask",       "gray"),
        (overlay_rgb, "Detection Overlay",    None),
    ]

    for ax, (img_data, label, cmap) in zip(axes, panels):
        if cmap:
            ax.imshow(img_data, cmap=cmap)
        else:
            ax.imshow(img_data)
        ax.set_title(label, fontsize=14, fontweight="bold",
                     color="white", pad=10)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
            spine.set_linewidth(1.5)

    plt.suptitle("Oil Spill Detection - SegNet", fontsize=16,
                 fontweight="bold", color="white", y=1.02)
    plt.tight_layout()

    # ── Save outputs ─────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    composite_path = os.path.join(output_dir, f"{base_name}_result.png")
    plt.savefig(composite_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Result saved to {composite_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate oil spill detection result")
    parser.add_argument("--image",      type=str, required=True,
                        help="Path to input satellite image")
    parser.add_argument("--output",     type=str, default="outputs",
                        help="Directory to save output")
    parser.add_argument("--checkpoint", type=str,
                        default="models/saved_models/segnet_best_model.pth",
                        help="Path to model checkpoint")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    predict(args.image, args.output, args.checkpoint, device)


if __name__ == "__main__":
    main()
