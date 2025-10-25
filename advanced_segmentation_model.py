"""
Advanced Segmentation Model with Attention and Multi-Scale Features
Optimized for tiny lesion detection in DR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SpatialAttention(nn.Module):
    """Spatial attention module to focus on lesion regions"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x


class ChannelAttention(nn.Module):
    """Channel attention module"""

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return out * x


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ConvBlock(nn.Module):
    """Convolutional block with attention"""

    def __init__(self, in_channels, out_channels, use_attention=True):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        if self.use_attention:
            x = self.attention(x)

        return x


class UNetWithAttention(nn.Module):
    """U-Net with Attention Mechanisms for tiny lesion segmentation"""

    def __init__(self, in_channels=3, out_channels=3, base_filters=64):
        super(UNetWithAttention, self).__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_filters, use_attention=True)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_filters, base_filters * 2, use_attention=True)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4, use_attention=True)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(base_filters * 4, base_filters * 8, use_attention=True)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 16, use_attention=True)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_filters * 16, base_filters * 8, use_attention=True)

        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_filters * 8, base_filters * 4, use_attention=True)

        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_filters * 4, base_filters * 2, use_attention=True)

        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = ConvBlock(base_filters * 2, base_filters, use_attention=True)

        # Output
        self.out = nn.Conv2d(base_filters, out_channels, 1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Output
        out = self.out(dec1)

        return out


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - Better for highly imbalanced tiny lesions
    Combines Tversky (generalization of Dice) with Focal (hard example mining)
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        """
        Args:
            alpha: weight of false positives
            beta: weight of false negatives (higher = more recall)
            gamma: focal parameter (higher = focus on hard examples)
        """
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        # Flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)

        # True Positives, False Positives & False Negatives
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()

        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Focal Tversky Loss
        focal_tversky = torch.pow((1 - tversky), self.gamma)

        return focal_tversky


class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss for better segmentation of tiny lesions
    = Focal Tversky + Dice + BCE
    """

    def __init__(self, ft_weight=1.0, dice_weight=0.5, bce_weight=0.3):
        super(CombinedSegmentationLoss, self).__init__()
        self.focal_tversky = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
        self.bce = nn.BCEWithLogitsLoss()

        self.ft_weight = ft_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def dice_loss(self, pred, target, smooth=1e-6):
        """Dice loss"""
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

        return 1 - dice

    def forward(self, pred, target):
        """
        Args:
            pred: model output [B, C, H, W] (logits)
            target: ground truth [B, C, H, W] (0 or 1)
        """
        ft_loss = self.focal_tversky(pred, target)
        dice_loss = self.dice_loss(pred, target)
        bce_loss = self.bce(pred, target)

        total_loss = (self.ft_weight * ft_loss +
                     self.dice_weight * dice_loss +
                     self.bce_weight * bce_loss)

        return total_loss


def create_advanced_segmentation_model(in_channels=3, out_channels=3, base_filters=64):
    """Create advanced segmentation model with attention"""
    model = UNetWithAttention(
        in_channels=in_channels,
        out_channels=out_channels,
        base_filters=base_filters
    )
    return model


def create_advanced_loss():
    """Create advanced combined loss for tiny lesion segmentation"""
    return CombinedSegmentationLoss(
        ft_weight=1.0,  # Focal Tversky - most important for tiny lesions
        dice_weight=0.5,  # Dice - for overall structure
        bce_weight=0.3  # BCE - for pixel-wise accuracy
    )


if __name__ == "__main__":
    # Test model
    model = create_advanced_segmentation_model()
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Test loss
    loss_fn = create_advanced_loss()
    target = torch.randint(0, 2, (2, 3, 512, 512)).float()
    loss = loss_fn(y, target)
    print(f"Loss: {loss.item():.4f}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

