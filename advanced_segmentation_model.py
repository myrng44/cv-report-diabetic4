"""
Mô hình Phân đoạn Nâng cao với Attention và Đặc trưng Đa tỷ lệ
Được tối ưu hóa cho phát hiện tổn thương nhỏ trong DR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SpatialAttention(nn.Module):
    """Module attention không gian để tập trung vào vùng tổn thương"""

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
    """Module attention kênh"""

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
    """Module Attention Khối Tích chập (Convolutional Block Attention Module)"""

    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ConvBlock(nn.Module):
    """Khối tích chập với attention"""

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
    """U-Net với Cơ chế Attention cho phân đoạn tổn thương nhỏ"""

    def __init__(self, in_channels=3, out_channels=3, base_filters=64):
        super(UNetWithAttention, self).__init__()

        # Bộ mã hóa (Encoder)
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

        # Bộ giải mã (Decoder)
        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_filters * 16, base_filters * 8, use_attention=True)

        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_filters * 8, base_filters * 4, use_attention=True)

        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_filters * 4, base_filters * 2, use_attention=True)

        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = ConvBlock(base_filters * 2, base_filters, use_attention=True)

        # Đầu ra
        self.out = nn.Conv2d(base_filters, out_channels, 1)

    def forward(self, x):
        # Bộ mã hóa
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Bộ giải mã
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

        # Đầu ra
        out = self.out(dec1)

        return out


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - Tốt hơn cho các tổn thương nhỏ mất cân bằng cao
    Kết hợp Tversky (tổng quát hóa của Dice) với Focal (khai thác ví dụ khó)
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        """
        Args:
            alpha: trọng số của false positives
            beta: trọng số của false negatives (cao hơn = recall nhiều hơn)
            gamma: tham số focal (cao hơn = tập trung vào ví dụ khó)
        """
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        # Làm phẳng tensors
        pred = pred.view(-1)
        target = target.view(-1)

        # True Positives, False Positives & False Negatives
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()

        # Chỉ số Tversky
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Focal Tversky Loss
        focal_tversky = torch.pow((1 - tversky), self.gamma)

        return focal_tversky


class CombinedSegmentationLoss(nn.Module):
    """
    Loss kết hợp để phân đoạn tốt hơn các tổn thương nhỏ
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
        """Loss Dice"""
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

        return 1 - dice

    def forward(self, pred, target):
        """
        Args:
            pred: đầu ra mô hình [B, C, H, W] (logits)
            target: ground truth [B, C, H, W] (0 hoặc 1)
        """
        ft_loss = self.focal_tversky(pred, target)
        dice_loss = self.dice_loss(pred, target)
        bce_loss = self.bce(pred, target)

        total_loss = (self.ft_weight * ft_loss +
                     self.dice_weight * dice_loss +
                     self.bce_weight * bce_loss)

        return total_loss


def create_advanced_segmentation_model(in_channels=3, out_channels=3, base_filters=64):
    """Tạo mô hình phân đoạn nâng cao với attention"""
    model = UNetWithAttention(
        in_channels=in_channels,
        out_channels=out_channels,
        base_filters=base_filters
    )
    return model


def create_advanced_loss():
    """Tạo loss kết hợp nâng cao cho phân đoạn tổn thương nhỏ"""
    return CombinedSegmentationLoss(
        ft_weight=1.0,  # Focal Tversky - quan trọng nhất cho tổn thương nhỏ
        dice_weight=0.5,  # Dice - cho cấu trúc tổng thể
        bce_weight=0.3  # BCE - cho độ chính xác từng pixel
    )


if __name__ == "__main__":
    # Kiểm tra mô hình
    model = create_advanced_segmentation_model()
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Kiểm tra loss
    loss_fn = create_advanced_loss()
    target = torch.randint(0, 2, (2, 3, 512, 512)).float()
    loss = loss_fn(y, target)
    print(f"Loss: {loss.item():.4f}")

    # Đếm tham số
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

