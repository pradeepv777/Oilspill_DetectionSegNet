#!/usr/bin/env python3
"""
Test script to verify the complete pipeline works
"""
import os
import sys
from PIL import Image
import torch
from dataset import OilSpillDataset
from model import SegNet

def test_dataset():
    """Test if dataset loads correctly"""
    print("Testing dataset loading...")
    try:
        dataset = OilSpillDataset(split="train", sensor="palsar")
        print(f"✓ Dataset loaded successfully with {len(dataset)} samples")
        
        # Test loading a sample
        if len(dataset) > 0:
            img, mask = dataset[0]
            print(f"✓ Sample loaded: Image shape {img.shape}, Mask shape {mask.shape}")
            return True
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False

def test_model():
    """Test if model can be instantiated and run"""
    print("\nTesting model...")
    try:
        model = SegNet(in_channels=1, num_classes=1)
        print("✓ Model instantiated successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Forward pass successful: Input {dummy_input.shape} -> Output {output.shape}")
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_training_setup():
    """Test if training can start"""
    print("\nTesting training setup...")
    try:
        from torch.utils.data import DataLoader
        
        dataset = OilSpillDataset(split="train", sensor="palsar")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        
        model = SegNet(in_channels=1, num_classes=1)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Test one training step
        for images, masks in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            print(f"✓ Training step successful: Loss {loss.item():.4f}")
            break
            
        return True
    except Exception as e:
        print(f"✗ Training setup failed: {e}")
        return False

def main():
    print("=" * 50)
    print("SATELLITE IMAGE SEGMENTATION PIPELINE TEST")
    print("=" * 50)
    
    tests = [
        test_dataset,
        test_model,
        test_training_setup
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"TESTS COMPLETED: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("✓ All tests passed! The pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Train the model: python train.py")
        print("2. Generate masks: python predict.py --image path/to/image.jpg")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()