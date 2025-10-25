"""
PHASE 1: CLASSIFICATION TRAINING - Ultra Optimized
Chạy riêng phase này trên Kaggle để tránh OOM

Usage trong Kaggle:
    !python train_phase1_classification.py
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
from dataset import get_classification_loaders
from classification_model import create_classification_model


def train_classification_phase(epochs=100, batch_size=8, img_size=768, device='cuda'):
    """Ultra-optimized classification training"""

    print("\n" + "="*80)
    print("🎯 PHASE 1: CLASSIFICATION TRAINING")
    print("="*80)
    print(f"Target: >75% Accuracy")
    print(f"Strategy: EfficientNet + Strong Augmentation + Cosine LR")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Image Size: {img_size}")
    print("="*80 + "\n")

    # Create output directories
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)

    # Load data
    print("Loading classification data...")
    train_loader, val_loader = get_classification_loaders(
        batch_size=batch_size,
        num_workers=2,  # Giảm workers cho Kaggle
        img_size=img_size
    )

    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}\n")

    # Create model
    print("Creating EfficientNet-B3 model...")
    model = create_classification_model(num_classes=5, pretrained=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}\n")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

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

    # Mixed precision
    device_type = 'cuda' if device == 'cuda' else 'cpu'
    scaler = GradScaler(device_type) if device == 'cuda' else None

    # Training state
    best_acc = 0.0
    best_f1 = 0.0
    patience = 0
    max_patience = 30

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
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

        # ============ Validation Phase ============
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", leave=False):
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
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        epoch_time = (time.time() - epoch_start) / 60

        # Print results
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs} - Time: {epoch_time:.1f}min - LR: {current_lr:.2e}")
        print(f"{'='*80}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"Val F1:     {val_f1:.4f}")

        # Save best model
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_f1 = val_f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'best_f1': best_f1,
            }, 'outputs/models/best_classification_model.pth')
            print(f"✅ Best model saved! (Acc: {best_acc:.4f}, F1: {best_f1:.4f})")
            patience = 0
        else:
            patience += 1
            print(f"⏳ Patience: {patience}/{max_patience}")

        print(f"{'='*80}\n")

        # Early stopping
        if patience >= max_patience:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            print(f"Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
            break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f'outputs/models/classification_checkpoint_epoch{epoch+1}.pth')

    training_time = (time.time() - start_time) / 60

    # Final Summary
    print("\n" + "="*80)
    print("✅ PHASE 1: CLASSIFICATION TRAINING COMPLETED!")
    print("="*80)
    print(f"📊 Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"📊 Best F1 Score: {best_f1:.4f}")
    print(f"⏱️  Training Time: {training_time:.1f} minutes")
    print(f"💾 Model saved: outputs/models/best_classification_model.pth")
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

        # Accuracy plot
        axes[1].plot(history['val_acc'], label='Val Accuracy', color='green')
        axes[1].plot(history['val_f1'], label='Val F1', color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Validation Metrics')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('outputs/logs/classification_training_curves.png', dpi=150, bbox_inches='tight')
        print("📈 Training curves saved: outputs/logs/classification_training_curves.png\n")
    except Exception as e:
        print(f"⚠️  Could not save plots: {e}\n")

    # Print confusion matrix
    try:
        print("Confusion Matrix (Validation Set):")
        cm = confusion_matrix(val_labels, val_preds)
        print(cm)
        print()
    except:
        pass

    return best_acc, best_f1, history


if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*80)
    print("🚀 DIABETIC RETINOPATHY - PHASE 1: CLASSIFICATION")
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

    # Train
    best_acc, best_f1, history = train_classification_phase(
        epochs=100,
        batch_size=8,
        img_size=768,
        device=device
    )

    print("\n" + "="*80)
    print("🎉 PHASE 1 COMPLETE - Ready for Phase 2!")
    print("="*80)
    print(f"Next step: Run train_phase2_segmentation.py")
    print("="*80 + "\n")

