import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from dataset import OilSpillDataset
from model import SegNet
from tqdm import tqdm
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

def calculate_metrics(preds, labels, smooth=1e-6):
    preds = (torch.sigmoid(preds) > 0.5).float()
    labels = labels.float()

    # Flatten tensors
    preds = preds.view(-1)
    labels = labels.view(-1)

    # True Positives, False Positives, True Negatives, False Negatives
    tp = (preds * labels).sum()
    fp = ((1 - labels) * preds).sum()
    fn = (labels * (1 - preds)).sum()
    tn = ((1 - labels) * (1 - preds)).sum()

    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn + smooth)
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    specificity = tn / (tn + fp + smooth)
    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    iou = tp / (tp + fp + fn + smooth)

    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "specificity": specificity.item(),
        "f1": f1.item(),
        "iou": iou.item(),
    }

def main():
    parser = argparse.ArgumentParser(description="Train SegNet on marine oil spill dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--sensor", type=str, choices=["palsar", "sentinel", "both"], default="both", help="Sensor data to train on")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    checkpoint_dir = "models/saved_models"
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = SegNet(in_channels=1, num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    print("Loading training dataset...")
    sensors = ["palsar", "sentinel"] if args.sensor == "both" else [args.sensor]
    datasets = [OilSpillDataset(split="train", sensor=s) for s in sensors]
    train_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"Training on {len(train_dataset)} samples")

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_specificity = 0.0
        total_f1 = 0.0
        total_iou = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')

        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            metrics = calculate_metrics(outputs, masks)
            total_accuracy += metrics["accuracy"]
            total_precision += metrics["precision"]
            total_recall += metrics["recall"]
            total_specificity += metrics["specificity"]
            total_f1 += metrics["f1"]
            total_iou += metrics["iou"]
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_loss = running_loss / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)
        avg_precision = total_precision / len(train_loader)
        avg_recall = total_recall / len(train_loader)
        avg_specificity = total_specificity / len(train_loader)
        avg_f1 = total_f1 / len(train_loader)
        avg_iou = total_iou / len(train_loader)

        scheduler.step(avg_loss)
        print(f"Epoch [{epoch}/{args.epochs}] Avg Loss: {avg_loss:.4f}, "
              f"Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, "
              f"Recall: {avg_recall:.4f}, Specificity: {avg_specificity:.4f}, "
              f"F1: {avg_f1:.4f}, IoU: {avg_iou:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(checkpoint_dir, "segnet_best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved: {best_model_path}")

        if epoch % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"segnet_epoch{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    final_model_path = os.path.join(checkpoint_dir, "segnet_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")
    print("Training finished!")

if __name__ == "__main__":
    main()
