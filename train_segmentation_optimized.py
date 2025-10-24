"""
Optimized Segmentation Training with Advanced Loss Functions
Target: IoU >0.40, Dice >0.55
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import config
from dataset import get_segmentation_loaders
from segmentation_model_improved import ImprovedSegmentationModel
from improved_losses import CombinedSegmentationLoss


def calculate_iou(pred, target, threshold=0.5):
    """Calculate IoU (Intersection over Union) - per channel average"""
    pred = (pred > threshold).float()
    target = target.float()

    # Calculate per channel
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection

    # Avoid division by zero
    iou = (intersection + 1e-6) / (union + 1e-6)

    # Return mean across batch and channels
    return iou.mean().item()


def calculate_dice(pred, target, threshold=0.5):
    """Calculate Dice Similarity Coefficient - per channel average"""
    pred = (pred > threshold).float()
    target = target.float()

    # Calculate per channel
    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2.0 * intersection + 1e-6) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-6)

    # Return mean across batch and channels
    return dice.mean().item()


class OptimizedSegmentationTrainer:
    """Optimized trainer for DR lesion segmentation"""

    def __init__(self, model, device, learning_rate=5e-5, use_amp=True):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp

        # GradScaler for mixed precision
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        self.scaler = GradScaler(device_type) if use_amp and device.type == 'cuda' else None
        self.device_type = device_type

        # Advanced combined loss for tiny lesions
        self.criterion = CombinedSegmentationLoss(
            focal_weight=config.FOCAL_WEIGHT,
            dice_weight=config.DICE_WEIGHT,
            tversky_weight=config.TVERSKY_WEIGHT,
            focal_alpha=config.FOCAL_ALPHA,
            focal_gamma=config.FOCAL_GAMMA
        )

        # Optimizer - AdamW with higher LR
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler
        self.scheduler = None  # Will be set in train()

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        self.val_dices = []

        self.best_val_iou = 0.0
        self.patience_counter = 0

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')

        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp and self.device_type == 'cuda':
                with autocast(device_type=self.device_type):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Update learning rate for OneCycleLR
            if self.scheduler and hasattr(self.scheduler, 'step') and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()

            running_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        avg_loss = running_loss / num_batches
        return avg_loss

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                if self.use_amp and self.device_type == 'cuda':
                    with autocast(device_type=self.device_type):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                # Calculate metrics
                pred_probs = torch.sigmoid(outputs)
                iou = calculate_iou(pred_probs, masks)
                dice = calculate_dice(pred_probs, masks)

                running_loss += loss.item()
                running_iou += iou
                running_dice += dice
                num_batches += 1

        avg_loss = running_loss / num_batches
        avg_iou = running_iou / num_batches
        avg_dice = running_dice / num_batches

        return avg_loss, avg_iou, avg_dice

    def train(self, train_loader, val_loader, num_epochs):
        """Complete training loop"""

        # Initialize scheduler based on number of epochs
        total_steps = len(train_loader) * num_epochs
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.SEG_LEARNING_RATE * 2,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos'
        )

        print(f"Starting segmentation training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp and self.device_type == 'cuda'}")
        print(f"✓ Using OneCycleLR (max_lr={config.SEG_LEARNING_RATE * 2:.0e}, warm-up=30%)")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_iou, val_dice = self.validate(val_loader)

            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_ious.append(val_iou)
            self.val_dices.append(val_dice)

            # Print metrics
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}")

            # Save best model
            if val_iou > self.best_val_iou:
                self.best_val_iou = val_iou
                self.patience_counter = 0
                torch.save(
                    self.model.state_dict(),
                    os.path.join(config.MODEL_DIR, 'best_seg_model.pth')
                )
                print(f"✓ Best model saved! (IoU: {val_iou:.4f})")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

        print(f"\n✓ Segmentation training completed!")
        print(f"Best validation IoU: {self.best_val_iou:.4f}")

        # Plot metrics
        self.plot_metrics()

        return self.best_val_iou

    def plot_metrics(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss plot
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # IoU plot
        axes[1].plot(self.val_ious, label='Val IoU', color='green')
        axes[1].axhline(y=0.40, color='r', linestyle='--', label='Target (0.40)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('IoU')
        axes[1].set_title('Validation IoU')
        axes[1].legend()
        axes[1].grid(True)

        # Dice plot
        axes[2].plot(self.val_dices, label='Val Dice', color='orange')
        axes[2].axhline(y=0.55, color='r', linestyle='--', label='Target (0.55)')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Dice Score')
        axes[2].set_title('Validation Dice Score')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULT_DIR, 'segmentation_metrics.png'), dpi=300)
        print(f"Metrics plot saved to {config.RESULT_DIR}")


def train_segmentation_model(num_epochs=100, batch_size=4, img_size=1024, learning_rate=5e-5, device='cuda'):
    """Main function to train optimized segmentation model"""

    # Set random seed
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Device configuration
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loaders with HIGH image size for tiny lesions
    print("Loading datasets...")
    train_loader, val_loader = get_segmentation_loaders(
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        img_size=img_size
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Create improved model
    print("\nCreating segmentation model...")
    model = ImprovedSegmentationModel(
        in_channels=3,
        out_channels=config.SEG_CLASSES,
        base_filters=32
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = OptimizedSegmentationTrainer(model, device, learning_rate=learning_rate, use_amp=config.USE_AMP)

    # Train
    best_iou = trainer.train(train_loader, val_loader, num_epochs)

    return best_iou


if __name__ == "__main__":
    train_segmentation_model()

