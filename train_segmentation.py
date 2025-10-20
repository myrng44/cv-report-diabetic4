"""
Training script for DR Lesion Segmentation
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import config
from dataset import get_segmentation_loaders
from segmentation_model import create_segmentation_model, CombinedLoss


def calculate_iou(pred, target, threshold=0.5):
    """Calculate IoU (Intersection over Union)"""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def calculate_dice(pred, target, threshold=0.5):
    """Calculate Dice Similarity Coefficient"""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    return dice.item()


class SegmentationTrainer:
    """Trainer for DR lesion segmentation"""

    def __init__(self, model, device, use_amp=True):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None

        # Loss function
        self.criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=1e-6
        )

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

        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader)
        return epoch_loss

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_ious = []
        all_dices = []

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                masks = masks.to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                running_loss += loss.item()

                # Calculate metrics for each sample
                outputs_sig = torch.sigmoid(outputs)
                for i in range(outputs.size(0)):
                    iou = calculate_iou(outputs_sig[i], masks[i])
                    dice = calculate_dice(outputs_sig[i], masks[i])
                    all_ious.append(iou)
                    all_dices.append(dice)

        val_loss = running_loss / len(val_loader)
        val_iou = np.mean(all_ious)
        val_dice = np.mean(all_dices)

        return val_loss, val_iou, val_dice

    def train(self, train_loader, val_loader, num_epochs):
        """Complete training loop"""
        print(f"Starting segmentation training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_iou, val_dice = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step()

            # Track metrics
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
                self.save_checkpoint(epoch, 'best_seg_model.pth')
                print(f"✓ Best model saved! (IoU: {val_iou:.4f})")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, f'seg_checkpoint_epoch_{epoch}.pth')

        print("\n✓ Segmentation training completed!")
        print(f"Best validation IoU: {self.best_val_iou:.4f}")

        return self.best_val_iou

    def save_checkpoint(self, epoch, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_iou': self.best_val_iou,
        }

        path = os.path.join(config.MODEL_DIR, filename)
        torch.save(checkpoint, path)

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
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('IoU')
        axes[1].set_title('Validation IoU')
        axes[1].legend()
        axes[1].grid(True)

        # Dice plot
        axes[2].plot(self.val_dices, label='Val Dice', color='orange')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Dice Score')
        axes[2].set_title('Validation Dice Score')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULT_DIR, 'segmentation_metrics.png'), dpi=300)
        print(f"Metrics plot saved to {config.RESULT_DIR}")


def train_segmentation_model():
    """Main function to train segmentation model"""

    # Set random seed
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loaders
    print("Loading datasets...")
    train_loader, val_loader = get_segmentation_loaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        img_size=config.IMG_SIZE
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Create model
    print("\nCreating segmentation model...")
    model = create_segmentation_model(
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
    trainer = SegmentationTrainer(model, device, use_amp=config.USE_AMP)

    # Train
    best_iou = trainer.train(train_loader, val_loader, config.NUM_EPOCHS)

    # Plot metrics
    trainer.plot_metrics()

    return best_iou


if __name__ == '__main__':
    train_segmentation_model()

