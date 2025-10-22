"""
Test script to verify mask loading
"""
import os
import cv2
import numpy as np
from dataset import DRSegmentationDataset, get_segmentation_transforms
import config

# Create dataset
dataset = DRSegmentationDataset(
    image_dir=config.SEG_TRAIN_IMG_DIR,
    mask_dir=config.SEG_TRAIN_MASK_DIR,
    transform=None  # No transform for testing
)

print(f"Total images: {len(dataset)}")
print(f"Lesion types: {dataset.lesion_types}")

# Test first few samples
for i in range(min(3, len(dataset))):
    image, mask = dataset[i]
    img_name = dataset.image_files[i]

    print(f"\n{'='*60}")
    print(f"Sample {i}: {img_name}")
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")

    # Check each channel
    for j, (folder, suffix) in enumerate(dataset.lesion_types):
        mask_channel = mask[:, :, j]
        unique_values = np.unique(mask_channel)
        non_zero_pixels = np.count_nonzero(mask_channel)

        # Check if mask file exists
        base_name = img_name.replace('.jpg', '')
        mask_path = os.path.join(config.SEG_TRAIN_MASK_DIR, folder, f"{base_name}_{suffix}.tif")
        exists = "✓" if os.path.exists(mask_path) else "✗"

        print(f"  {suffix:3s} - File {exists} | Shape: {mask_channel.shape} | "
              f"Non-zero pixels: {non_zero_pixels:6d} | "
              f"Unique values: {len(unique_values)} | "
              f"Min: {mask_channel.min():.4f}, Max: {mask_channel.max():.4f}")

        if not os.path.exists(mask_path):
            print(f"      Missing: {mask_path}")

print(f"\n{'='*60}")
print("Testing with transforms...")

dataset_with_transform = DRSegmentationDataset(
    image_dir=config.SEG_TRAIN_IMG_DIR,
    mask_dir=config.SEG_TRAIN_MASK_DIR,
    transform=get_segmentation_transforms(is_train=False, img_size=384)
)

image, mask = dataset_with_transform[0]
print(f"Image tensor shape: {image.shape}")
print(f"Mask tensor shape: {mask.shape}")
print(f"Image dtype: {image.dtype}")
print(f"Mask dtype: {mask.dtype}")
print(f"Mask value range: [{mask.min():.4f}, {mask.max():.4f}]")
print(f"Mask sum: {mask.sum():.4f}")

# Count non-zero pixels per channel
for j, (_, suffix) in enumerate(dataset_with_transform.lesion_types):
    non_zero = (mask[j] > 0).sum().item()
    print(f"  {suffix}: {non_zero} non-zero pixels")
