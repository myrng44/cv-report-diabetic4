"""
Improved Segmentation Model with Advanced Loss Functions
Optimized for tiny lesions with severe class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveBatchNorm2d(nn.Module):
    """Adaptive Batch Normalization"""

    def __init__(self, num_features: int):
        super(AdaptiveBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.bn(x)
        return self.alpha * normalized + self.beta


class ConvBlock(nn.Module):
    """Convolutional block with Adaptive BN"""

    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.abn1 = AdaptiveBatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.abn2 = AdaptiveBatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.abn1(self.conv1(x)))
        x = self.relu(self.abn2(self.conv2(x)))
        return x


class DownBlock(nn.Module):
    """Encoder block with max pooling"""

    def __init__(self, in_channels: int, out_channels: int):
        super(DownBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple:
        skip = self.conv_block(x)
        x = self.pool(skip)
        return x, skip


class UpBlock(nn.Module):
    """Decoder block with upsampling"""

    def __init__(self, in_channels: int, out_channels: int):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class ModifiedUNet(nn.Module):
    """Modified U-Net with Adaptive Batch Normalization"""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_filters: int = 32):
        super(ModifiedUNet, self).__init__()

        # Encoder
        self.down1 = DownBlock(in_channels, base_filters)
        self.down2 = DownBlock(base_filters, base_filters * 2)
        self.down3 = DownBlock(base_filters * 2, base_filters * 4)
        self.down4 = DownBlock(base_filters * 4, base_filters * 8)

        # Bottleneck
        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 16)

        # Decoder
        self.up1 = UpBlock(base_filters * 16, base_filters * 8)
        self.up2 = UpBlock(base_filters * 8, base_filters * 4)
        self.up3 = UpBlock(base_filters * 4, base_filters * 2)
        self.up4 = UpBlock(base_filters * 2, base_filters)

        # Output layer
        self.out_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        # Output
        x = self.out_conv(x)
        return x


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""

    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)

        # Calculate per channel (important for multi-class)
        pred = pred.view(pred.size(0), pred.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)

        intersection = (pred * target).sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (pred.sum(dim=2) + target.sum(dim=2) + self.smooth)

        # Average across batch and channels
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss - Handles severe class imbalance (99% background vs 1% lesion)"""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Higher weight for positive class (lesions)
        self.gamma = gamma  # Focus on hard examples

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)

        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # p_t: probability of correct class
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)

        # Focal weight: (1 - p_t)^gamma - focuses on misclassified/hard examples
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weight: balances positive vs negative samples
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        focal_loss = alpha_t * focal_weight * bce

        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss - Better than Dice for imbalanced data"""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives (HIGHER = penalize missed lesions more!)
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)

        # Calculate per channel
        pred = pred.view(pred.size(0), pred.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)

        # True Positives, False Positives & False Negatives
        TP = (pred * target).sum(dim=2)
        FP = ((1 - target) * pred).sum(dim=2)
        FN = (target * (1 - pred)).sum(dim=2)

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        # Average across batch and channels
        return 1 - tversky.mean()


class CombinedLoss(nn.Module):
    """
    Combined Focal + Tversky + Dice Loss
    Optimized for tiny lesions with severe class imbalance

    Why this works:
    - Focal: Handles 99% background vs 1% lesion imbalance
    - Tversky: Penalizes missing lesions (false negatives)
    - Dice: Overall region overlap metric
    """

    def __init__(self, focal_weight: float = 1.0, tversky_weight: float = 1.0, dice_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.dice_weight = dice_weight

        # Focal: alpha=0.75 gives 75% weight to lesions (positive class)
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)

        # Tversky: beta=0.7 > alpha=0.3 → penalize missed lesions more than false alarms
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)

        # Dice: standard overlap metric
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(pred, target)
        tversky_loss = self.tversky(pred, target)
        dice_loss = self.dice(pred, target)

        total_loss = (self.focal_weight * focal_loss +
                     self.tversky_weight * tversky_loss +
                     self.dice_weight * dice_loss)

        return total_loss


def create_segmentation_model(in_channels: int = 3,
                              out_channels: int = 3,
                              base_filters: int = 32) -> ModifiedUNet:
    """Factory function to create Modified U-Net model"""
    return ModifiedUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_filters=base_filters
    )


# Alias for improved model (same as ModifiedUNet but with better name)
class ImprovedSegmentationModel(ModifiedUNet):
    """Improved Segmentation Model - Optimized for tiny lesions"""
    pass


if __name__ == "__main__":
    # Test model
    print("Testing Improved Segmentation Model...")

    model = ImprovedSegmentationModel(in_channels=3, out_channels=3, base_filters=32)

    # Test with dummy input
    dummy_input = torch.randn(2, 3, 1024, 1024)  # High resolution for tiny lesions
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("\n✓ Model test successful!")
