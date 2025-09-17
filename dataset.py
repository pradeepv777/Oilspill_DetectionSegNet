import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class OilSpillDataset(Dataset):
    def __init__(self, split="train", sensor="palsar"):
        """
        Dataset loader for SAR-based Oil Spill Detection
        split: 'train' or 'test'
        sensor: 'palsar' or 'sentinel'
        """
        self.split = split
        self.sensor = sensor

        if split == "train":
            # For training, images and masks are in the same directory
            self.data_dir = os.path.join("dataset", "train", sensor)
        else:
            # For test, images and masks are in separate directories
            self.sat_dir = os.path.join("dataset", "test", sensor, "sat")
            self.gt_dir = os.path.join("dataset", "test", sensor, "gt")

        if split == "train":
            # Collect all jpg files for training
            all_files = [f for f in os.listdir(self.data_dir) if f.endswith(".jpg")]
            self.image_paths = [os.path.join(self.data_dir, f) for f in all_files]
            
            # Find corresponding mask files
            self.mask_paths = []
            for img_path in self.image_paths:
                mask_name = os.path.basename(img_path).replace("_sat.jpg", "_mask.png")
                mask_path = os.path.join(self.data_dir, mask_name)
                if os.path.exists(mask_path):
                    self.mask_paths.append(mask_path)
                else:
                    self.mask_paths.append(None)
        else:
            # For test data
            self.image_paths = sorted([os.path.join(self.sat_dir, f) 
                                       for f in os.listdir(self.sat_dir) 
                                       if f.endswith((".jpg", ".png"))])
            self.mask_paths = sorted([os.path.join(self.gt_dir, f) 
                                      for f in os.listdir(self.gt_dir) 
                                      if f.endswith((".jpg", ".png"))]) if os.path.exists(self.gt_dir) else None

        # Filter out images without corresponding masks for training
        if split == "train":
            valid_pairs = [(img, mask) for img, mask in zip(self.image_paths, self.mask_paths) if mask is not None]
            self.image_paths = [pair[0] for pair in valid_pairs]
            self.mask_paths = [pair[1] for pair in valid_pairs]

        print(f"Found {len(self.image_paths)} image-mask pairs for {split} {sensor}")

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        img = self.transform(img)

        if self.mask_paths and self.mask_paths[idx] is not None:
            mask = Image.open(self.mask_paths[idx]).convert("L")
            mask = self.transform(mask)
            return img, mask
        return img
