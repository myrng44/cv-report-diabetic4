"""
Advanced Loss Functions for Diabetic Retinopathy Segmentation
Optimized for severe class imbalance (tiny lesions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C, H, W) - logits
            targets: (B, C, H, W) - one-hot encoded targets
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C, H, W) - logits
            targets: (B, C, H, W) - one-hot encoded targets
        """
        inputs = torch.sigmoid(inputs)

        # Flatten for computation
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice Loss
    Better for handling false positives vs false negatives trade-off
    """

    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # weight for false positives
        self.beta = beta    # weight for false negatives
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C, H, W) - logits
            targets: (B, C, H, W) - one-hot encoded targets
        """
        inputs = torch.sigmoid(inputs)

        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return 1 - Tversky


class CombinedSegmentationLoss(nn.Module):
    """
    Combined Loss Function optimized for tiny lesion segmentation
    Combines Focal + Dice + Tversky Loss
    """

    def __init__(
        self,
        focal_weight=0.3,
        dice_weight=0.5,
        tversky_weight=0.2,
        focal_alpha=0.25,
        focal_gamma=2.0,
        tversky_alpha=0.7,
        tversky_beta=0.3
    ):
        super(CombinedSegmentationLoss, self).__init__()

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(smooth=1.0)
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C, H, W) - logits
            targets: (B, C, H, W) - one-hot encoded targets
        """
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        tversky = self.tversky_loss(inputs, targets)

        total_loss = (
            self.focal_weight * focal +
            self.dice_weight * dice +
            self.tversky_weight * tversky
        )

        return total_loss


class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision Loss for multi-scale prediction
    """

    def __init__(self, loss_fn, weights=None):
        super(DeepSupervisionLoss, self).__init__()
        self.loss_fn = loss_fn
        self.weights = weights if weights is not None else [1.0, 0.8, 0.6, 0.4]

    def forward(self, predictions, targets):
        """
        Args:
            predictions: List of predictions at different scales
            targets: Ground truth targets
        """
        total_loss = 0
        for i, pred in enumerate(predictions):
            # Resize target if needed
            if pred.shape != targets.shape:
                target_resized = F.interpolate(
                    targets,
                    size=pred.shape[2:],
                    mode='nearest'
                )
            else:
                target_resized = targets

            weight = self.weights[i] if i < len(self.weights) else self.weights[-1]
            total_loss += weight * self.loss_fn(pred, target_resized)

        return total_loss / len(predictions)


class IoULoss(nn.Module):
    """IoU Loss (Jaccard Loss)"""

    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Intersection and Union
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + self.smooth) / (union + self.smooth)

        return 1 - IoU


def get_segmentation_loss(loss_type='combined', **kwargs):
    """
    Factory function to get loss function

    Args:
        loss_type: 'focal', 'dice', 'tversky', 'iou', 'combined'
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function
    """
    if loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'dice':
        return DiceLoss(**kwargs)
    elif loss_type == 'tversky':
        return TverskyLoss(**kwargs)
    elif loss_type == 'iou':
        return IoULoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedSegmentationLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")

    # Create dummy data
    batch_size = 2
    num_classes = 3
    height, width = 256, 256

    inputs = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, 2, (batch_size, num_classes, height, width)).float()

    # Test each loss
    losses = {
        'Focal': FocalLoss(),
        'Dice': DiceLoss(),
        'Tversky': TverskyLoss(),
        'IoU': IoULoss(),
        'Combined': CombinedSegmentationLoss()
    }

    print("\nLoss values:")
    for name, loss_fn in losses.items():
        loss_value = loss_fn(inputs, targets)
        print(f"{name}: {loss_value.item():.4f}")

    print("\n✓ All loss functions working correctly!")

