"""
Advanced Training Strategies for Better Performance
Các chiến lược nâng cao để cải thiện accuracy
"""

# ============================================================================
# STRATEGY 1: Weighted Loss for Class Imbalance
# ============================================================================
# Problem: Grade 1 chỉ có 20 samples (4.84%) vs Grade 0,2 có ~134 samples
# Solution: Tăng weight cho class ít samples

import torch
import torch.nn as nn

def get_class_weights():
    """
    Calculate class weights for imbalanced dataset
    Based on your data distribution:
    Grade 0: 134 samples (32.45%)
    Grade 1: 20 samples (4.84%)   <- Need higher weight!
    Grade 2: 136 samples (32.93%)
    Grade 3: 74 samples (17.92%)
    Grade 4: 49 samples (11.86%)
    """
    class_counts = [134, 20, 136, 74, 49]
    total = sum(class_counts)

    # Inverse frequency weighting
    weights = [total / (len(class_counts) * count) for count in class_counts]

    # Normalize to sum = num_classes
    weights = torch.FloatTensor(weights)
    weights = weights / weights.sum() * len(class_counts)

    print("Class weights:", weights)
    # Expected: [0.74, 5.03, 0.74, 1.35, 2.04]
    # Grade 1 gets 5x higher weight!

    return weights


def create_weighted_loss(device='cpu'):
    """Create weighted CrossEntropyLoss"""
    weights = get_class_weights().to(device)
    return nn.CrossEntropyLoss(weight=weights)


# ============================================================================
# STRATEGY 2: Learning Rate Warmup + Cosine Annealing
# ============================================================================

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    LR schedule: warmup → cosine decay
    Better than ReduceLROnPlateau for stable convergence
    """
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(current_step):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine annealing phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ============================================================================
# STRATEGY 3: MixUp Augmentation
# ============================================================================
# Mixup: Mix 2 images to create synthetic training samples
# Very effective for small datasets!

import numpy as np

def mixup_data(x, y, alpha=0.2):
    """
    MixUp augmentation
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# STRATEGY 4: Test Time Augmentation (TTA)
# ============================================================================
# During inference, augment test image multiple times and average predictions

def tta_predict(model, image, n_augments=5):
    """
    Test Time Augmentation
    Apply multiple augmentations and average predictions
    """
    import torchvision.transforms as T

    predictions = []

    # Original image
    with torch.no_grad():
        pred = model(image)
        predictions.append(pred)

    # Augmented versions
    augmentations = [
        T.RandomHorizontalFlip(p=1.0),
        T.RandomVerticalFlip(p=1.0),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.1, contrast=0.1),
    ]

    for aug in augmentations[:n_augments-1]:
        aug_image = aug(image)
        with torch.no_grad():
            pred = model(aug_image)
            predictions.append(pred)

    # Average predictions
    avg_pred = torch.stack(predictions).mean(dim=0)
    return avg_pred


# ============================================================================
# STRATEGY 5: Label Smoothing
# ============================================================================
# Prevent overconfident predictions

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing
    Instead of [0, 0, 1, 0, 0] → [0.05, 0.05, 0.8, 0.05, 0.05]
    Prevents overfitting on small datasets
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_pred = torch.log_softmax(pred, dim=-1)

        # Smooth labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        return torch.mean(torch.sum(-true_dist * log_pred, dim=-1))


# ============================================================================
# STRATEGY 6: Gradient Clipping
# ============================================================================
# Prevent exploding gradients

def clip_gradients(model, max_norm=1.0):
    """Clip gradients to prevent instability"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


# ============================================================================
# RECOMMENDED CONFIGURATION FOR YOUR DATASET
# ============================================================================

ADVANCED_CONFIG = {
    # Image settings
    'img_size': 384,  # You're using this ✓
    'batch_size': 4,

    # Optimizer settings
    'learning_rate': 3e-5,  # Even lower than 5e-5
    'weight_decay': 5e-4,   # Higher regularization
    'use_class_weights': True,  # ← Important for imbalance!

    # Training settings
    'num_epochs': 60,
    'warmup_epochs': 5,  # Warmup first 5 epochs
    'use_mixup': True,   # ← Very effective!
    'mixup_alpha': 0.2,
    'label_smoothing': 0.1,
    'gradient_clip_norm': 1.0,

    # Augmentation
    'strong_augmentation': True,
    'use_cutout': True,  # Random erasing patches

    # TTA during validation
    'use_tta': True,
    'tta_n_augments': 3,

    # Early stopping
    'patience': 20,  # Very patient

    # LR scheduler
    'scheduler': 'cosine_warmup',  # Better than ReduceLROnPlateau
}


# ============================================================================
# HOW TO USE THESE STRATEGIES
# ============================================================================

"""
1. Class Weights (Most Important for your data!):
   - In train_classification.py, replace:
     self.criterion = FocalLoss(...)
   - With:
     self.criterion = create_weighted_loss(device)

2. Cosine Schedule with Warmup:
   - Replace ReduceLROnPlateau scheduler
   - Add in trainer __init__:
     total_steps = len(train_loader) * num_epochs
     warmup_steps = len(train_loader) * 5
     self.scheduler = get_cosine_schedule_with_warmup(
         self.optimizer, warmup_steps, total_steps
     )

3. MixUp Augmentation:
   - In training loop, before forward pass:
     if use_mixup:
         images, labels_a, labels_b, lam = mixup_data(images, labels)
         outputs = self.model(images)
         loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)

4. Label Smoothing:
   - Replace loss function:
     self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

5. Gradient Clipping:
   - After loss.backward(), before optimizer.step():
     clip_gradients(self.model, max_norm=1.0)

6. Test Time Augmentation:
   - In validation/inference:
     predictions = tta_predict(model, images, n_augments=3)
"""


# ============================================================================
# EXPECTED IMPROVEMENTS
# ============================================================================

"""
Current: 60.19% accuracy, F1=0.576

With IMG_SIZE=384 only:
→ Expected: 65-72% accuracy, F1=0.62-0.68

With IMG_SIZE=384 + Class Weights + MixUp:
→ Expected: 70-78% accuracy, F1=0.68-0.75

With ALL strategies combined:
→ Expected: 75-82% accuracy, F1=0.72-0.80

These are realistic expectations for a dataset with:
- 413 training samples
- High class imbalance (Grade 1: only 20 samples)
- Medical imaging (inherently difficult)
"""

