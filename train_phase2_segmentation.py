"""
PHASE 2: SEGMENTATION TRAINING - Ultra Optimized
Chạy sau khi Phase 1 hoàn thành để tránh OOM

Usage trong Kaggle:
    !python train_phase2_segmentation.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

import config
from dataset import get_segmentation_loaders
from advanced_segmentation_model import create_advanced_segmentation_model, create_advanced_loss


def calculate_metrics(pred, target, smooth=1e-6):
    """
    Calculate IoU and Dice metrics for segmentation

    Args:
        pred: predicted masks [B, C, H, W] (binary 0/1)
        target: ground truth masks [B, C, H, W] (0/1)
        smooth: smoothing factor

    Returns:
        iou: Intersection over Union
        dice: Dice coefficient
    """
    pred = pred.float()
    target = target.float()

    # Flatten
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection

    # IoU
    iou = (intersection + smooth) / (union + smooth)

    # Dice
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    return iou.item(), dice.item()


def train_segmentation_phase(epochs=100, batch_size=4, img_size=1024, device='cuda'):
    """Ultra-optimized segmentation training for tiny lesions"""

    print("\n" + "="*80)
    print("🎯 PHASE 2: SEGMENTATION TRAINING")
    print("="*80)
    print(f"Target: IoU >0.40, Dice >0.50")
    print(f"Strategy: Attention U-Net + Focal Tversky Loss + High Resolution")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Image Size: {img_size}")
    print("="*80 + "\n")

    # Create output directories
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)

    # Load data
    print("Loading segmentation data...")
    train_loader, val_loader = get_segmentation_loaders(
        batch_size=batch_size,
        num_workers=2,  # Giảm workers cho Kaggle
        img_size=img_size
    )

    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}\n")

    # Create advanced model with attention
    print("Creating Attention U-Net model...")
    model = create_advanced_segmentation_model(
        in_channels=3,
        out_channels=3,
        base_filters=64
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}\n")

    # Advanced loss for tiny lesions
    criterion = create_advanced_loss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )

    # OneCycle LR for faster convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # Mixed precision
    device_type = 'cuda' if device == 'cuda' else 'cpu'
    scaler = GradScaler(device_type) if device == 'cuda' else None

    # Training state
    best_iou = 0.0
    best_dice = 0.0
    patience = 0
    max_patience = 25

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': []
    }

    print("="*80)
    print("🚀 Starting Training...")
    print("="*80 + "\n")

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # ============ Training Phase ============
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            if scaler and device == 'cuda':
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)

        # ============ Validation Phase ============
        model.eval()
        val_loss = 0.0
        val_iou_sum = 0.0
        val_dice_sum = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation", leave=False):
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()

                # Calculate metrics
                preds = torch.sigmoid(outputs) > 0.5
                iou, dice = calculate_metrics(preds, masks)
                val_iou_sum += iou
                val_dice_sum += dice

        val_loss /= len(val_loader)
        val_iou = val_iou_sum / len(val_loader)
        val_dice = val_dice_sum / len(val_loader)

        # Log metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = (time.time() - epoch_start) / 60

        # Print results
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs} - Time: {epoch_time:.1f}min - LR: {current_lr:.2e}")
        print(f"{'='*80}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val IoU:    {val_iou:.4f} | Val Dice: {val_dice:.4f}")

        # Save best model
        is_best = val_iou > best_iou
        if is_best:
            best_iou = val_iou
            best_dice = val_dice
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
                'best_dice': best_dice,
            }, 'outputs/models/best_segmentation_model.pth')
            print(f"✅ Best model saved! (IoU: {best_iou:.4f}, Dice: {best_dice:.4f})")
            patience = 0
        else:
            patience += 1
            print(f"⏳ Patience: {patience}/{max_patience}")

        print(f"{'='*80}\n")

        # Early stopping
        if patience >= max_patience:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            print(f"Best IoU: {best_iou:.4f}, Best Dice: {best_dice:.4f}")
            break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f'outputs/models/segmentation_checkpoint_epoch{epoch+1}.pth')

    training_time = (time.time() - start_time) / 60

    # Final Summary
    print("\n" + "="*80)
    print("✅ PHASE 2: SEGMENTATION TRAINING COMPLETED!")
    print("="*80)
    print(f"📊 Best IoU:  {best_iou:.4f}")
    print(f"📊 Best Dice: {best_dice:.4f}")
    print(f"⏱️  Training Time: {training_time:.1f} minutes")
    print(f"💾 Model saved: outputs/models/best_segmentation_model.pth")
    print("="*80 + "\n")

    # Plot training curves
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Metrics plot
        axes[1].plot(history['val_iou'], label='Val IoU', color='green')
        axes[1].plot(history['val_dice'], label='Val Dice', color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Validation Metrics')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('outputs/logs/segmentation_training_curves.png', dpi=150, bbox_inches='tight')
        print("📈 Training curves saved: outputs/logs/segmentation_training_curves.png\n")
    except Exception as e:
        print(f"⚠️  Could not save plots: {e}\n")

    return best_iou, best_dice, history


if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*80)
    print("🚀 DIABETIC RETINOPATHY - PHASE 2: SEGMENTATION")
    print("="*80)
    print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  Device: {device}")

    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("="*80 + "\n")

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ CUDA cache cleared\n")

    # Train
    best_iou, best_dice, history = train_segmentation_phase(
        epochs=100,
        batch_size=4,
        img_size=1024,
        device=device
    )

    print("\n" + "="*80)
    print("🎉 ALL PHASES COMPLETE!")
    print("="*80)
    print("Both models trained successfully:")
    print("  ✅ Classification model: outputs/models/best_classification_model.pth")
    print("  ✅ Segmentation model:   outputs/models/best_segmentation_model.pth")
    print("="*80 + "\n")

