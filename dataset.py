"""
Dataset và DataLoader cho Phân loại và Phân đoạn DR
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
from preprocessing import CLAHEEnhancer  # Import CLAHE tùy chỉnh


# Khởi tạo bộ tăng cường CLAHE tùy chỉnh toàn cục để tăng hiệu suất
_clahe_enhancer = CLAHEEnhancer(clip_limit=4.0, tile_grid_size=(8, 8))


class CustomCLAHE(A.ImageOnlyTransform):
    """
    Transform CLAHE tùy chỉnh sử dụng triển khai preprocessing.py của chúng ta
    Tích hợp với pipeline Albumentations
    """
    
    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5):
        super(CustomCLAHE, self).__init__(always_apply, p)
        self.enhancer = CLAHEEnhancer(clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    
    def apply(self, img, **params):
        """Áp dụng CLAHE tùy chỉnh cho ảnh RGB"""
        return self.enhancer.apply_rgb(img)


class DRClassificationDataset(Dataset):
    """Dataset cho phân loại DR"""

    def __init__(self, image_dir: str, labels_csv: str, transform=None, is_train: bool = True):
        self.image_dir = image_dir
        self.transform = transform
        self.is_train = is_train

        # Đọc nhãn
        df = pd.read_csv(labels_csv)
        self.image_names = df['Image name'].values
        self.labels = df['Retinopathy grade'].values

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Tải ảnh
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, f"{img_name}.jpg")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Áp dụng các phép biến đổi
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # Lấy nhãn
        label = int(self.labels[idx])

        return image, label


class DRSegmentationDataset(Dataset):
    """Dataset cho phân đoạn tổn thương DR"""

    def __init__(self, image_dir: str, mask_dir: str, transform=None,
                 lesion_types: list = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Định nghĩa các loại tổn thương với tên thư mục và hậu tố file
        if lesion_types is None:
            self.lesion_types = [
                ('1. Microaneurysms_', 'MA'),
                ('2. Haemorrhages_', 'HE'),
                ('3. Hard Exudates_', 'EX')
            ]
        else:
            self.lesion_types = lesion_types

        # Lấy tất cả các file ảnh
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Tải ảnh
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Tải mask cho tất cả các loại tổn thương
        base_name = img_name.replace('.jpg', '')
        masks = []

        for folder_name, suffix in self.lesion_types:
            mask_path = os.path.join(self.mask_dir, folder_name, f"{base_name}_{suffix}.tif")

            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                # Tạo mask rỗng nếu không tồn tại
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            masks.append(mask)

        # Xếp chồng các mask (channels: MA, HEM, EX)
        mask = np.stack(masks, axis=-1)

        # Áp dụng các phép biến đổi
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
            # Chuẩn hóa mask về [0, 1]
            mask = mask.float() / 255.0
            
            # Hoán vị mask từ [H, W, C] sang [C, H, W] theo quy ước PyTorch
            mask = mask.permute(2, 0, 1)
        else:
            # Không có transform: chuyển sang tensor thủ công
            mask = torch.from_numpy(mask).float() / 255.0
            mask = mask.permute(2, 0, 1)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, mask


def get_classification_transforms(is_train: bool = True, img_size: int = 256):
    """Lấy các phép biến đổi tăng cường cho phân loại - Tăng cường để đạt >75% accuracy"""

    if is_train:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            # Tăng cường hình học
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.6),
            A.Affine(scale=(0.85, 1.15), translate_percent=(0.15, 0.15), rotate=(-30, 30), p=0.6),

            # Tăng cường màu sắc - mạnh hơn
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),

            # Nhiễu và làm mờ để tăng độ bền vững
            A.OneOf([
                A.GaussNoise(var_limit=50, p=1.0),  # Fixed: sử dụng var_limit không phải (min, max)
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.5),

            # Tăng cường nâng cao - SỬA tên tham số
            A.CoarseDropout(max_holes=8, max_height=int(img_size*0.1), max_width=int(img_size*0.1), fill_value=0, p=0.4),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.OpticalDistortion(distort_limit=0.3, p=0.3),  # Đã xóa shift_limit không hợp lệ

            # CLAHE để tăng độ tương phản - SỬ DỤNG TRIỂN KHAI TÙY CHỈNH
            CustomCLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),

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
    """Lấy các phép biến đổi tăng cường cho phân đoạn - TĂNG CƯỜNG cho các tổn thương nhỏ"""

    if is_train:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            # Tăng cường hình học - vừa phải để bảo toàn vị trí tổn thương
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-15, 15), p=0.5),

            # Tăng cường màu sắc - quan trọng cho ảnh DR
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4),
            
            # CLAHE để tăng độ tương phản - SỬ DỤNG TRIỂN KHAI TÙY CHỈNH
            CustomCLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),

            # Biến dạng đàn hồi cho ảnh y tế - ĐÃ SỬA tham số
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),  # Đã xóa alpha_affine không hợp lệ
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
    """Tạo các data loader cho phân loại"""

    train_dataset = DRClassificationDataset(
        image_dir=config.CLASS_TRAIN_IMG_DIR,
        labels_csv=config.CLASS_TRAIN_LABELS,
        transform=get_classification_transforms(is_train=True, img_size=img_size),
        is_train=True
    )

    # Kiểm tra xem nhãn test có tồn tại không
    if os.path.exists(config.CLASS_TEST_LABELS):
        test_dataset = DRClassificationDataset(
            image_dir=config.CLASS_TEST_IMG_DIR,
            labels_csv=config.CLASS_TEST_LABELS,
            transform=get_classification_transforms(is_train=False, img_size=img_size),
            is_train=False
        )
    else:
        # Sử dụng chia validation từ dữ liệu huấn luyện
        from sklearn.model_selection import train_test_split
        train_indices, val_indices = train_test_split(
            range(len(train_dataset)), test_size=0.2, random_state=config.SEED
        )

        train_dataset_split = torch.utils.data.Subset(train_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(train_dataset, val_indices)
        train_dataset = train_dataset_split

    # Phát hiện xem CUDA có khả dụng cho pin_memory không
    import torch
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory  # Chỉ sử dụng pin_memory nếu CUDA khả dụng
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory  # Chỉ sử dụng pin_memory nếu CUDA khả dụng
    )

    return train_loader, test_loader


def get_segmentation_loaders(batch_size: int = 4, num_workers: int = 2, img_size: int = 256):
    """Tạo các data loader cho phân đoạn"""

    train_dataset = DRSegmentationDataset(
        image_dir=config.SEG_TRAIN_IMG_DIR,
        mask_dir=config.SEG_TRAIN_MASK_DIR,
        transform=get_segmentation_transforms(is_train=True, img_size=img_size)
    )

    # Sử dụng chia validation
    from sklearn.model_selection import train_test_split
    train_indices, val_indices = train_test_split(
        range(len(train_dataset)), test_size=0.2, random_state=config.SEED
    )

    train_dataset_split = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(train_dataset, val_indices)

    # Phát hiện xem CUDA có khả dụng cho pin_memory không
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset_split,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory  # Chỉ sử dụng pin_memory nếu CUDA khả dụng
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory  # Chỉ sử dụng pin_memory nếu CUDA khả dụng
    )

    return train_loader, val_loader
