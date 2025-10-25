"""
Dataset and DataLoader for DR Classification and Segmentation
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional
import config


class DRClassificationDataset(Dataset):
    """Dataset for DR classification"""

    def __init__(self, image_dir: str, labels_csv: str, transform=None, is_train: bool = True):
        self.image_dir = image_dir
        self.transform = transform
        self.is_train = is_train

        # Read labels
        df = pd.read_csv(labels_csv)
        self.image_names = df['Image name'].values
        self.labels = df['Retinopathy grade'].values

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load image
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, f"{img_name}.jpg")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # Get label
        label = int(self.labels[idx])

        return image, label


class DRSegmentationDataset(Dataset):
    """Dataset for DR lesion segmentation"""

    def __init__(self, image_dir: str, mask_dir: str, transform=None,
                 lesion_types: list = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Define lesion types with folder names and file suffixes
        if lesion_types is None:
            self.lesion_types = [
                ('1. Microaneurysms_', 'MA'),
                ('2. Haemorrhages_', 'HE'),
                ('3. Hard Exudates_', 'EX')
            ]
        else:
            self.lesion_types = lesion_types

        # Get all image files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load masks for all lesion types
        base_name = img_name.replace('.jpg', '')
        masks = []

        for folder_name, suffix in self.lesion_types:
            mask_path = os.path.join(self.mask_dir, folder_name, f"{base_name}_{suffix}.tif")

            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                # Create empty mask if not exists
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            masks.append(mask)

        # Stack masks (channels: MA, HEM, EX)
        mask = np.stack(masks, axis=-1)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
            # Normalize mask to [0, 1]
            mask = mask.float() / 255.0
            
            # Permute mask from [H, W, C] to [C, H, W] to match PyTorch convention
            mask = mask.permute(2, 0, 1)
        else:
            # No transform: convert to tensor manually
            mask = torch.from_numpy(mask).float() / 255.0
            mask = mask.permute(2, 0, 1)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, mask


def get_classification_transforms(is_train: bool = True, img_size: int = 256):
    """Get augmentation transforms for classification - Enhanced for >75% accuracy"""

    if is_train:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.6),  # Increased rotation
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=30, p=0.6),

            # Color augmentations - stronger
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),

            # Noise and blur for robustness
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.5),

            # Advanced augmentations
            A.CoarseDropout(max_holes=8, max_height=int(img_size*0.1), max_width=int(img_size*0.1), p=0.4),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.3),

            # CLAHE for better contrast
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    return transform


def get_segmentation_transforms(is_train: bool = True, img_size: int = 256):
    """Get augmentation transforms for segmentation - ENHANCED for tiny lesions"""

    if is_train:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            # Geometric augmentations - moderate to preserve lesion locations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),

            # Color augmentations - important for DR images
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),

            # Elastic deformation for medical images
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    return transform


def get_classification_loaders(batch_size: int = 4, num_workers: int = 2, img_size: int = 256):
    """Create data loaders for classification"""

    train_dataset = DRClassificationDataset(
        image_dir=config.CLASS_TRAIN_IMG_DIR,
        labels_csv=config.CLASS_TRAIN_LABELS,
        transform=get_classification_transforms(is_train=True, img_size=img_size),
        is_train=True
    )

    # Check if test labels exist
    if os.path.exists(config.CLASS_TEST_LABELS):
        test_dataset = DRClassificationDataset(
            image_dir=config.CLASS_TEST_IMG_DIR,
            labels_csv=config.CLASS_TEST_LABELS,
            transform=get_classification_transforms(is_train=False, img_size=img_size),
            is_train=False
        )
    else:
        # Use validation split from training data
        from sklearn.model_selection import train_test_split
        train_indices, val_indices = train_test_split(
            range(len(train_dataset)), test_size=0.2, random_state=config.SEED
        )

        train_dataset_split = torch.utils.data.Subset(train_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(train_dataset, val_indices)
        train_dataset = train_dataset_split

    # Detect if CUDA is available for pin_memory
    import torch
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory  # Only use pin_memory if CUDA available
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory  # Only use pin_memory if CUDA available
    )

    return train_loader, test_loader


def get_segmentation_loaders(batch_size: int = 4, num_workers: int = 2, img_size: int = 256):
    """Create data loaders for segmentation"""

    train_dataset = DRSegmentationDataset(
        image_dir=config.SEG_TRAIN_IMG_DIR,
        mask_dir=config.SEG_TRAIN_MASK_DIR,
        transform=get_segmentation_transforms(is_train=True, img_size=img_size)
    )

    # Use validation split
    from sklearn.model_selection import train_test_split
    train_indices, val_indices = train_test_split(
        range(len(train_dataset)), test_size=0.2, random_state=config.SEED
    )

    train_dataset_split = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(train_dataset, val_indices)

    # Detect if CUDA is available for pin_memory
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset_split,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory  # Only use pin_memory if CUDA available
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory  # Only use pin_memory if CUDA available
    )

    return train_loader, val_loader
