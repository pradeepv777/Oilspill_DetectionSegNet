import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from dataset import OilSpillDataset
from model import SegNet
from tqdm import tqdm
import argparse

# ------------------- Device Setup -------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device = torch.device("mps")
elif hasattr(torch.backends, "directml") and torch.backends.directml.is_available():
    device = torch.device("directml")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# ------------------- Arguments & Hyperparameters -------------------
parser = argparse.ArgumentParser(description="Train SegNet on marine oil spill dataset")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
parser.add_argument("--sensor", type=str, choices=["palsar", "sentinel", "both"], default="both", help="Which sensor split to train on (default: both)")
args = parser.parse_args()

batch_size = args.batch_size  # Reduced batch size for memory efficiency
learning_rate = args.lr
num_epochs = args.epochs
checkpoint_dir = "models/saved_models"
os.makedirs(checkpoint_dir, exist_ok=True)

# ------------------- Model, Loss, Optimizer -------------------
# Grayscale input, single-channel mask output
model = SegNet(in_channels=1, num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# ------------------- Dataset & DataLoader -------------------
print("Loading training dataset...")
if args.sensor == "both":
    palsar_ds = OilSpillDataset(split="train", sensor="palsar")
    sentinel_ds = OilSpillDataset(split="train", sensor="sentinel")
    train_dataset = ConcatDataset([palsar_ds, sentinel_ds])
else:
    train_dataset = OilSpillDataset(split="train", sensor=args.sensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

if isinstance(train_dataset, ConcatDataset):
    total_len = sum(len(ds) for ds in train_dataset.datasets)
else:
    total_len = len(train_dataset)
print(f"Training on {total_len} samples")

# ------------------- Training Loop -------------------
best_loss = float('inf')
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    num_batches = 0
    # Metrics accumulators (binary segmentation)
    epoch_true_positive = 0
    epoch_false_positive = 0
    epoch_true_negative = 0
    epoch_false_negative = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
    
    for batch_idx, (images, masks) in enumerate(progress_bar):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # ----- Metrics (F1, Pixel Accuracy) -----
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).to(torch.int)
            targets = (masks > 0.5).to(torch.int)

            tp = ((preds == 1) & (targets == 1)).sum().item()
            tn = ((preds == 0) & (targets == 0)).sum().item()
            fp = ((preds == 1) & (targets == 0)).sum().item()
            fn = ((preds == 0) & (targets == 1)).sum().item()

            epoch_true_positive += tp
            epoch_true_negative += tn
            epoch_false_positive += fp
            epoch_false_negative += fn

    avg_loss = running_loss / num_batches
    scheduler.step(avg_loss)
    
    # Compute epoch metrics
    total_pixels = epoch_true_positive + epoch_true_negative + epoch_false_positive + epoch_false_negative
    pixel_accuracy = ((epoch_true_positive + epoch_true_negative) / total_pixels) if total_pixels > 0 else 0.0

    precision_den = (epoch_true_positive + epoch_false_positive)
    recall_den = (epoch_true_positive + epoch_false_negative)
    precision = (epoch_true_positive / precision_den) if precision_den > 0 else 0.0
    recall = (epoch_true_positive / recall_den) if recall_den > 0 else 0.0
    f1_denom = (precision + recall)
    f1_score = (2 * precision * recall / f1_denom) if f1_denom > 0 else 0.0

    iou_den = (epoch_true_positive + epoch_false_positive + epoch_false_negative)
    iou = (epoch_true_positive / iou_den) if iou_den > 0 else 0.0

    dice_den = (2 * epoch_true_positive + epoch_false_positive + epoch_false_negative)
    dice = (2 * epoch_true_positive / dice_den) if dice_den > 0 else 0.0

    print(
        f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f} | "
        f"Acc: {pixel_accuracy:.4f} F1: {f1_score:.4f} Prec: {precision:.4f} Rec: {recall:.4f} IoU: {iou:.4f} Dice: {dice:.4f}"
    )

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_path = os.path.join(checkpoint_dir, "segnet_best_model.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved: {best_model_path}")

    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"segnet_epoch{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

# Save final model
final_model_path = os.path.join(checkpoint_dir, "segnet_final_model.pth")
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved: {final_model_path}")
print("Training finished!")
