import numpy as np
import torch
import cv2
import os

def to_numpy(tensor):
    """Converts a tensor to a numpy array."""
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.array(tensor)

def _prepare_binary_maps(pred, target, threshold=0.5):
    """Converts prediction and target tensors to binary numpy arrays."""
    pred_bin = (to_numpy(pred) > threshold).astype(np.uint8)
    target_bin = (to_numpy(target) > 0.5).astype(np.uint8)
    return pred_bin, target_bin

def iou_score(pred, target, threshold=0.5, eps=1e-7):
    pred_bin, target_bin = _prepare_binary_maps(pred, target, threshold)
    inter = (pred_bin & target_bin).sum()
    union = (pred_bin | target_bin).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return (inter + eps) / (union + eps)

def dice_score(pred, target, threshold=0.5, eps=1e-7):
    pred_bin, target_bin = _prepare_binary_maps(pred, target, threshold)
    tp = (pred_bin & target_bin).sum()
    return (2 * tp + eps) / (pred_bin.sum() + target_bin.sum() + eps)

def save_overlay(image, mask, outpath):
    img = (to_numpy(image) * 255).astype('uint8')
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask_col = (to_numpy(mask) > 0.5).astype('uint8') * 255
    contours, _ = cv2.findContours(mask_col, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = img_color.copy()
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    cv2.addWeighted(overlay, 0.5, img_color, 0.5, 0, overlay)
    cv2.imwrite(outpath, overlay)
