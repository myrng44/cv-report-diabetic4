"""
Classification model: DenseNet + Attention + Optimized GRU with SANGO
"""

import torch
import torch.nn as nn
import torchvision.models as models
from sango_optimizer import OptimizedGRU


class AttentionBlock(nn.Module):
    """Self-attention mechanism"""

    def __init__(self, in_features: int):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Tanh(),
            nn.Linear(in_features // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        attention_weights = self.attention(x)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Apply attention
        weighted = x * attention_weights
        return weighted.sum(dim=1)  # (batch, features)


class DRClassificationModel(nn.Module):
    """
    Complete DR Classification Model:
    DenseNet backbone + Attention + Optimized GRU
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True,
                 gru_hidden_size: int = 128, gru_num_layers: int = 2):
        super(DRClassificationModel, self).__init__()

        # DenseNet backbone (memory efficient) - Fixed deprecated pretrained parameter
        if pretrained:
            from torchvision.models import DenseNet121_Weights
            densenet = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        else:
            densenet = models.densenet121(weights=None)

        # Remove classifier
        self.features = nn.Sequential(*list(densenet.children())[:-1])

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)
            feature_output = self.features(dummy_input)
            self.feature_dim = feature_output.shape[1]

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        # Reshape for sequence processing
        self.seq_len = 64  # 8x8 grid

        # Attention mechanism
        self.attention = AttentionBlock(self.feature_dim)

        # Optimized GRU
        self.gru = OptimizedGRU(
            input_size=self.feature_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=0.3,
            num_classes=num_classes
        )

        # Alternative path: Direct classification (for ablation study)
        self.direct_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.use_gru = True  # Flag to switch between GRU and direct classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch, 3, H, W)

        Returns:
            Logits (batch, num_classes)
        """
        # Extract features
        features = self.features(x)  # (batch, feature_dim, H', W')

        # Adaptive pooling
        features = self.adaptive_pool(features)  # (batch, feature_dim, 8, 8)

        # Reshape to sequence
        batch_size = features.size(0)
        features = features.view(batch_size, self.feature_dim, -1)  # (batch, feature_dim, 64)
        features = features.permute(0, 2, 1)  # (batch, 64, feature_dim)

        if self.use_gru:
            # GRU-based classification with attention
            output = self.gru(features)
        else:
            # Direct classification
            pooled = features.mean(dim=1)  # (batch, feature_dim)
            output = self.direct_classifier(pooled)

        return output


class MultiTaskModel(nn.Module):
    """
    Multi-task model for both classification and segmentation
    """

    def __init__(self, num_classes: int = 5, seg_classes: int = 3):
        super(MultiTaskModel, self).__init__()

        # Shared encoder (DenseNet) - Fixed deprecated pretrained parameter
        from torchvision.models import DenseNet121_Weights
        densenet = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(densenet.children())[:-1])

        # Classification head
        self.classification_head = DRClassificationModel(num_classes=num_classes)

        # Segmentation head (will be separate model)
        # Note: In practice, segmentation and classification are often trained separately
        # to avoid memory issues

    def forward(self, x: torch.Tensor, task: str = 'classification') -> torch.Tensor:
        """
        Forward pass for specified task

        Args:
            x: Input tensor
            task: 'classification' or 'segmentation'
        """
        if task == 'classification':
            return self.classification_head(x)
        else:
            raise NotImplementedError("Use separate segmentation model")


def create_classification_model(num_classes: int = 5,
                                pretrained: bool = True,
                                gru_hidden_size: int = 128,
                                gru_num_layers: int = 2) -> DRClassificationModel:
    """
    Factory function to create classification model

    Args:
        num_classes: Number of DR severity classes (default: 5)
        pretrained: Use pretrained DenseNet weights
        gru_hidden_size: GRU hidden dimension
        gru_num_layers: Number of GRU layers

    Returns:
        Classification model
    """
    model = DRClassificationModel(
        num_classes=num_classes,
        pretrained=pretrained,
        gru_hidden_size=gru_hidden_size,
        gru_num_layers=gru_num_layers
    )
    return model


# Loss functions for classification
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(pred, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
