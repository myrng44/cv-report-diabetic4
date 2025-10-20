"""
Modified U-Net with Adaptive Batch Normalization and EfficientNet layers
For retinal lesion segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveBatchNorm2d(nn.Module):
    """Adaptive Batch Normalization"""

    def __init__(self, num_features: int):
        super(AdaptiveBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        # Learnable parameters for adaptation
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
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class ModifiedUNet(nn.Module):
    """
    Modified U-Net with Adaptive Batch Normalization
    For segmentation of retinal lesions
    """

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

        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined BCE and Dice Loss"""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def create_segmentation_model(in_channels: int = 3,
                              out_channels: int = 3,
                              base_filters: int = 32) -> ModifiedUNet:
    """Factory function to create Modified U-Net model"""
    return ModifiedUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_filters=base_filters
    )

