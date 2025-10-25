"""
ULTRA-OPTIMIZED Training Pipeline for DR Detection
Target: Classification >75%, Segmentation IoU >0.40
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import time
from datetime import datetime

import config
from dataset import get_classification_loaders, get_segmentation_loaders
from classification_model import create_classification_model
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


def train_classification_ultra(epochs=100, batch_size=8, img_size=768, device='cuda'):
    """Ultra-optimized classification training"""

    print("\n" + "="*80)
    print("PHASE 1: ULTRA-OPTIMIZED CLASSIFICATION TRAINING")
    print("="*80)
    print(f"Target: >75% Accuracy")
    print(f"Strategy: Strong augmentation + Larger model + Better LR schedule")
    print("="*80 + "\n")

    # Load data with strong augmentation
    train_loader, val_loader = get_classification_loaders(
        batch_size=batch_size,
        num_workers=4,
        img_size=img_size
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}\n")

    # Create model - Use EfficientNet for better performance
    print("Creating model with EfficientNet-B3 backbone...")
    model = create_classification_model(num_classes=5, pretrained=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    # Advanced training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Use AdamW with cosine annealing
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Mixed precision training
    device_type = 'cuda' if device == 'cuda' else 'cpu'
    scaler = GradScaler(device_type) if device == 'cuda' else None

    # Training loop
    best_acc = 0.0
    patience = 0
    max_patience = 30

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    print("Starting training...\n")
    start_time = time.time()

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            if scaler and device == 'cuda':
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        # Update scheduler
        scheduler.step()

        # Log metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'outputs/models/best_model_ultra.pth')
            print(f"✓ Best model saved! (Acc: {best_acc:.4f})")
            patience = 0
        else:
            patience += 1

        # Early stopping
        if patience >= max_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

        print()

    training_time = (time.time() - start_time) / 60

    print("\n" + "="*80)
    print("✅ CLASSIFICATION TRAINING COMPLETED!")
    print(f"Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"Training Time: {training_time:.1f} minutes")
    print("="*80 + "\n")

    return best_acc, history


def train_segmentation_ultra(epochs=100, batch_size=4, img_size=1024, device='cuda'):
    """Ultra-optimized segmentation training for tiny lesions"""

    print("\n" + "="*80)
    print("PHASE 2: ULTRA-OPTIMIZED SEGMENTATION TRAINING")
    print("="*80)
    print(f"Target: IoU >0.40, Dice >0.50")
    print(f"Strategy: Attention U-Net + Focal Tversky Loss + High Resolution")
    print("="*80 + "\n")

    # Load data
    train_loader, val_loader = get_segmentation_loaders(
        batch_size=batch_size,
        num_workers=4,
        img_size=img_size
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}\n")

    # Create advanced model with attention
    print("Creating U-Net with Attention Mechanisms...")
    model = create_advanced_segmentation_model(
        in_channels=3,
        out_channels=3,
        base_filters=64
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

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

    # Training loop
    best_iou = 0.0
    patience = 0
    max_patience = 25

    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': []}

    print("Starting training...\n")
    start_time = time.time()

    for epoch in range(epochs):
        # Training phase
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

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_iou_sum = 0.0
        val_dice_sum = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
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

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}")

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'outputs/models/best_seg_model_ultra.pth')
            print(f"✓ Best model saved! (IoU: {best_iou:.4f})")
            patience = 0
        else:
            patience += 1

        # Early stopping
        if patience >= max_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

        print()

    training_time = (time.time() - start_time) / 60

    print("\n" + "="*80)
    print("✅ SEGMENTATION TRAINING COMPLETED!")
    print(f"Best IoU: {best_iou:.4f}")
    print(f"Training Time: {training_time:.1f} minutes")
    print("="*80 + "\n")

    return best_iou, history


def main():
    """Main training pipeline"""

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*80)
    print("🚀 ULTRA-OPTIMIZED DR DETECTION TRAINING")
    print("="*80)
    print(f"Device: {device}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    # Create output directories
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)

    total_start = time.time()

    # Phase 1: Classification
    class_acc, class_history = train_classification_ultra(
        epochs=100,
        batch_size=8,
        img_size=768,
        device=device
    )

    # Phase 2: Segmentation
    seg_iou, seg_history = train_segmentation_ultra(
        epochs=100,
        batch_size=4,
        img_size=1024,
        device=device
    )

    total_time = (time.time() - total_start) / 60

    # Final summary
    print("\n" + "="*80)
    print("🎉 COMPLETE TRAINING PIPELINE FINISHED!")
    print("="*80)
    print(f"Total Time: {total_time:.1f} minutes ({total_time/60:.2f} hours)")
    print()
    print("RESULTS SUMMARY:")
    print(f"  Classification Accuracy: {class_acc:.4f} ({class_acc*100:.2f}%)")
    print(f"  Segmentation IoU: {seg_iou:.4f}")
    print()
    print("MODELS SAVED:")
    print("  - outputs/models/best_model_ultra.pth")
    print("  - outputs/models/best_seg_model_ultra.pth")
    print()
    print("NEXT STEPS:")
    print("  1. Visualize with Grad-CAM:")
    print("     python gradcam_visualization.py")
    print("  2. Run inference:")
    print("     python demo_inference.py --image <path>")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
