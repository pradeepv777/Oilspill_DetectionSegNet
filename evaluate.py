import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for all environments

from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from dataset import OilSpillDataset
from model import SegNet
import argparse


# ─────────────────────────────────────────────
# Device helper (same logic as train.py)
# ─────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    if hasattr(torch.backends, "directml") and torch.backends.directml.is_available():
        return torch.device("directml")
    return torch.device("cpu")


# ─────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────
def plot_confusion_matrix(cm, save_path):
    """Plots and saves a labelled confusion matrix."""
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Background", "Oil Spill"],
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")


def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """Plots and saves the ROC curve."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ROC curve saved        → {save_path}")


def plot_pr_curve(precision, recall, ap, save_path):
    """Plots and saves the Precision-Recall curve."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="darkorange", lw=2, label=f"PR (AP = {ap:.4f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  PR curve saved         → {save_path}")


def save_metrics_csv(metrics: dict, save_path: str):
    """Saves a flat metrics dict as a single-row CSV."""
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    print(f"  Metrics CSV saved      → {save_path}")


# ─────────────────────────────────────────────
# Main evaluation routine
# ─────────────────────────────────────────────
def evaluate(checkpoint_path: str, sensor: str, output_dir: str, batch_size: int):
    device = get_device()
    print(f"\nUsing device: {device}")

    # ── Load model ──────────────────────────────
    model = SegNet(in_channels=1, num_classes=1).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Please train the model first (python train.py)."
        )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Model loaded from {checkpoint_path}\n")

    # ── Load test dataset ────────────────────────
    sensors = ["palsar", "sentinel"] if sensor == "both" else [sensor]
    datasets = [OilSpillDataset(split="test", sensor=s) for s in sensors]
    test_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Evaluating on {len(test_dataset)} test samples...\n")

    # ── Collect raw predictions ──────────────────
    all_probs = []   # sigmoid probabilities  (flat, pixel-level)
    all_labels = []  # ground-truth binary     (flat, pixel-level)

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            outputs = model(images)                          # logits
            probs = torch.sigmoid(outputs).cpu().numpy()     # [B,1,H,W]
            labels = masks.cpu().numpy()                     # [B,1,H,W]

            all_probs.append(probs.flatten())
            all_labels.append((labels > 0.5).astype(np.uint8).flatten())

    all_probs = np.concatenate(all_probs)    # shape: (N_pixels,)
    all_labels = np.concatenate(all_labels)  # shape: (N_pixels,)
    all_preds = (all_probs > 0.5).astype(np.uint8)

    # ── Scalar metrics ───────────────────────────
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
        "accuracy":    round(accuracy,    6),
        "precision":   round(precision,   6),
        "recall":      round(recall,      6),
        "specificity": round(specificity, 6),
        "f1":          round(f1,          6),
        "iou":         round(iou,         6),
        "dice":        round(dice,        6),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }

    print("─" * 45)
    print(f"  Accuracy    : {accuracy:.4f}")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall      : {recall:.4f}")
    print(f"  Specificity : {specificity:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  IoU         : {iou:.4f}")
    print(f"  Dice        : {dice:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print("─" * 45 + "\n")

    # ── Curve data ───────────────────────────────
    fpr, tpr, _       = roc_curve(all_labels, all_probs)
    roc_auc           = auc(fpr, tpr)
    prec_curve, rec_curve, _ = precision_recall_curve(all_labels, all_probs)
    ap                = average_precision_score(all_labels, all_probs)
    cm                = confusion_matrix(all_labels, all_preds)

    metrics["roc_auc"] = round(float(roc_auc), 6)
    metrics["avg_precision"] = round(float(ap), 6)

    # ── Save everything ──────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    print("Saving evaluation outputs:")
    plot_confusion_matrix(cm,  os.path.join(output_dir, "confusion_matrix.png"))
    plot_roc_curve(fpr, tpr, roc_auc,
                              os.path.join(output_dir, "roc_curve.png"))
    plot_pr_curve(prec_curve, rec_curve, ap,
                              os.path.join(output_dir, "pr_curve.png"))
    save_metrics_csv(metrics,  os.path.join(output_dir, "metrics_summary.csv"))

    print("\nEvaluation complete.")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate SegNet on the test set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/saved_models/segnet_best_model.pth",
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        choices=["palsar", "sentinel", "both"],
        default="both",
        help="Which sensor's test data to evaluate on",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation",
        help="Directory to save graphs and CSV",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.sensor, args.output, args.batch_size)


if __name__ == "__main__":
    main()
