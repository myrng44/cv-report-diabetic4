"""
Test segmentation model với 1 ảnh cụ thể - FIXED VERSION
"""
import torch
import cv2
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from advanced_segmentation_model import create_advanced_segmentation_model

print("Testing segmentation model with CORRECT preprocessing...")

device = torch.device('cpu')
model_path = 'outputs/models/best_seg_model.pth'

# Load model
model = create_advanced_segmentation_model(in_channels=3, out_channels=3)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded from {model_path}")
print(f"Checkpoint keys: {checkpoint.keys()}")
if 'best_iou' in checkpoint:
    print(f"Best IoU during training: {checkpoint['best_iou']:.4f}")
if 'best_dice' in checkpoint:
    print(f"Best Dice during training: {checkpoint['best_dice']:.4f}")

# Test với 1 ảnh - USING CORRECT PREPROCESSING!
image_path = r"data\B. Disease Grading\1. Original Images\b. Testing Set\IDRiD_001.jpg"

# Read image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ✅ CRITICAL FIX: Use same preprocessing as training!
transform = A.Compose([
    A.Resize(512, 512),
    # ⚠️ This is what was missing! Training used ImageNet normalization
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

transformed = transform(image=image)
image_tensor = transformed['image'].unsqueeze(0).to(device)

print(f"\n✅ Using CORRECT preprocessing (ImageNet normalization)")
print(f"Input shape: {image_tensor.shape}")
print(f"Input range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
print(f"Input mean: {image_tensor.mean():.3f}")

with torch.no_grad():
    output = model(image_tensor)
    print(f"\n📊 Model Output:")
    print(f"Output shape: {output.shape}")
    print(f"Output range (logits): [{output.min():.3f}, {output.max():.3f}]")
    print(f"Output mean: {output.mean():.3f}")
    print(f"Output std: {output.std():.3f}")

    probs = torch.sigmoid(output)
    print(f"\n📈 After Sigmoid:")
    print(f"Probs range: [{probs.min():.6f}, {probs.max():.6f}]")
    print(f"Probs mean: {probs.mean():.6f}")
    print(f"Probs std: {probs.std():.6f}")

    # Test different thresholds
    print("\n🎯 Pixels > threshold:")
    for thresh in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        count = (probs > thresh).sum().item()
        percentage = count / probs.numel() * 100
        print(f"  {thresh}: {count} pixels ({percentage:.4f}%)")

    # Check each channel
    print("\n📋 Per-channel analysis:")
    lesion_names = ['Microaneurysms', 'Haemorrhages', 'Hard Exudates']
    for i in range(3):
        channel = probs[0, i]
        print(f"  Channel {i} ({lesion_names[i]}):")
        print(f"    mean={channel.mean():.6f}, max={channel.max():.6f}, min={channel.min():.6f}")
        for thresh in [0.1, 0.15, 0.2]:
            count = (channel > thresh).sum().item()
            percentage = count / channel.numel() * 100
            print(f"    > {thresh}: {count} pixels ({percentage:.4f}%)")

print("\n" + "="*80)
print("✅ If you see pixels > threshold now, the fix worked!")
print("If still 0 pixels, model may need retraining with better loss functions.")
print("="*80)

