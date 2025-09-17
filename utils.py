import numpy as np
import torch
import cv2
import os

def iou_score(pred, target, threshold=0.5, eps=1e-7):
    pred = (pred > threshold).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    inter = (pred & target).sum()
    union = (pred | target).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / (union + eps)

def dice_score(pred, target, threshold=0.5, eps=1e-7):
    pred = (pred > threshold).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    tp = (pred & target).sum()
    return (2*tp + eps) / (pred.sum() + target.sum() + eps)

def save_overlay(image, mask, outpath):
    img = (image*255).astype('uint8')
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask_col = (mask>0.5).astype('uint8')*255
    contours, _ = cv2.findContours(mask_col, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = img_color.copy()
    cv2.drawContours(overlay, contours, -1, (0,0,255), 2)
    cv2.addWeighted(overlay, 0.5, img_color, 0.5, 0, overlay)
    cv2.imwrite(outpath, overlay)

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)
