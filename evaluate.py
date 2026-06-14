import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)
from torchvision import transforms
from PIL import Image
from dataset import OilSpillDataset
from model import SegNet
import argparse


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")


def plot_confusion_matrix(cm, save_path):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Background", "Oil Spill"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")


def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ROC curve saved        → {save_path}")


def plot_pr_curve(precision, recall, ap, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="darkorange", lw=2, label=f"PR (AP = {ap:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  PR curve saved         → {save_path}")


def save_metrics_csv(metrics, save_path):
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    print(f"  Metrics CSV saved      → {save_path}")


def evaluate(checkpoint_path, sensor, output_dir, batch_size):
    device = get_device()
    print(f"\nUsing device: {device}")

    model = SegNet(in_channels=1, num_classes=1).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Model loaded from {checkpoint_path}\n")

    resize = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    resize_mask = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    sensors = ["palsar", "sentinel"] if sensor == "both" else [sensor]

    all_probs = []
    all_labels = []

    for s in sensors:
        sat_dir = os.path.join("dataset", "test", s, "sat")
        gt_dir  = os.path.join("dataset", "test", s, "gt")
        image_paths = sorted([os.path.join(sat_dir, f) for f in os.listdir(sat_dir) if f.endswith((".jpg", ".png"))])
        mask_paths  = sorted([os.path.join(gt_dir,  f) for f in os.listdir(gt_dir)  if f.endswith((".jpg", ".png"))])
        print(f"Evaluating {len(image_paths)} samples from {s}...")

        for img_path, mask_path in zip(image_paths, mask_paths):
            img  = resize(Image.open(img_path).convert("L")).unsqueeze(0).to(device)
            mask = resize_mask(Image.open(mask_path).convert("L"))

            with torch.no_grad():
                prob = torch.sigmoid(model(img)).squeeze().cpu().numpy().flatten()

            all_probs.append(prob)
            all_labels.append((mask.numpy().flatten() > 0.5).astype(np.uint8))

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds  = (all_probs > 0.5).astype(np.uint8)

    smooth = 1e-7
    tp = int(((all_preds == 1) & (all_labels == 1)).sum())
    fp = int(((all_preds == 1) & (all_labels == 0)).sum())
    fn = int(((all_preds == 0) & (all_labels == 1)).sum())
    tn = int(((all_preds == 0) & (all_labels == 0)).sum())

    accuracy    = (tp + tn) / (tp + tn + fp + fn + smooth)
    precision   = tp / (tp + fp + smooth)
    recall      = tp / (tp + fn + smooth)
    specificity = tn / (tn + fp + smooth)
    f1          = 2 * precision * recall / (precision + recall + smooth)
    iou         = tp / (tp + fp + fn + smooth)
    dice        = 2 * tp / (2 * tp + fp + fn + smooth)

    metrics = {
        "accuracy": round(accuracy, 6), "precision": round(precision, 6),
        "recall": round(recall, 6), "specificity": round(specificity, 6),
        "f1": round(f1, 6), "iou": round(iou, 6), "dice": round(dice, 6),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }

    print("─" * 45)
    for k, v in metrics.items():
        print(f"  {k:12s}: {v}")
    print("─" * 45)

    fpr, tpr, _            = roc_curve(all_labels, all_probs)
    roc_auc                = auc(fpr, tpr)
    prec_c, rec_c, _       = precision_recall_curve(all_labels, all_probs)
    ap                     = average_precision_score(all_labels, all_probs)
    cm                     = confusion_matrix(all_labels, all_preds)
    metrics["roc_auc"]     = round(float(roc_auc), 6)
    metrics["avg_precision"] = round(float(ap), 6)

    os.makedirs(output_dir, exist_ok=True)
    print("Saving evaluation outputs:")
    plot_confusion_matrix(cm, os.path.join(output_dir, "confusion_matrix.png"))
    plot_roc_curve(fpr, tpr, roc_auc, os.path.join(output_dir, "roc_curve.png"))
    plot_pr_curve(prec_c, rec_c, ap, os.path.join(output_dir, "pr_curve.png"))
    save_metrics_csv(metrics, os.path.join(output_dir, "metrics_summary.csv"))
    print("\nEvaluation complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/saved_models/segnet_best_model.pth")
    parser.add_argument("--sensor", type=str, choices=["palsar", "sentinel", "both"], default="both")
    parser.add_argument("--output", type=str, default="outputs/evaluation")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.sensor, args.output, args.batch_size)

if __name__ == "__main__":
    main()
