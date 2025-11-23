"""
GIAI ĐOẠN 2: HUẤN LUYỆN PHÂN ĐOẠN - PHIÊN BẢN ĐA GPU
Sử dụng 2x T4 GPU trên Kaggle để train với cấu hình cao

Cách sử dụng trong Kaggle:
    !python train_phase2_segmentation_multigpu.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import gc

import config
from dataset import get_segmentation_loaders
from advanced_segmentation_model import create_advanced_segmentation_model, create_advanced_loss


def calculate_metrics(pred, target, smooth=1e-6):
    """
    Tính toán metrics IoU và Dice cho phân đoạn

    Args:
        pred: masks dự đoán [B, C, H, W] (nhị phân 0/1)
        target: masks ground truth [B, C, H, W] (0/1)
        smooth: hệ số làm mượt

    Returns:
        iou: Intersection over Union
        dice: Hệ số Dice
    """
    pred = pred.float()
    target = target.float()

    # Làm phẳng
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Tính giao và hợp
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection

    # IoU
    iou = (intersection + smooth) / (union + smooth)

    # Dice
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    return iou.item(), dice.item()


def train_segmentation_multigpu(epochs=100, batch_size=4, img_size=1024, device='cuda'):
    """
    Huấn luyện phân đoạn đa GPU

    Với 2x T4 GPU:
    - Batch size: 4 → 8 (mỗi GPU 4)
    - Image size: 1024 (giữ nguyên)
    - Model: Attention U-Net (phiên bản đầy đủ)
    """

    print("\n" + "="*80)
    print("GIAI ĐOẠN 2: HUẤN LUYỆN PHÂN ĐOẠN (ĐA GPU)")
    print("="*80)
    print(f"Sử dụng 2x T4 GPU cho huấn luyện ĐỘ PHÂN GIẢI CAO")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Kích thước Ảnh: {img_size}")
    print("="*80 + "\n")

    # Kiểm tra GPU khả dụng
    if not torch.cuda.is_available():
        print("LỖI: Không có GPU khả dụng!")
        return None, None, None

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s):")
    for i in range(num_gpus):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    print()

    if num_gpus < 2:
        print("WARNING: Only 1 GPU detected. Multi-GPU training disabled.")
        print("Consider using train_phase2_segmentation_lite.py instead.")
        use_multigpu = False
    else:
        print(f"Multi-GPU training enabled with {num_gpus} GPUs!")
        use_multigpu = True
    print()

    # Memory cleanup
    print("Cleaning memory...")
    gc.collect()
    for i in range(num_gpus):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    print("✓ Memory cleaned on all GPUs\n")

    # Create output directories
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)

    # Load data
    # Với multi-GPU, batch_size sẽ được chia đều cho các GPU
    effective_batch_size = batch_size
    print(f"Loading segmentation data (image size: {img_size})...")
    print(f"Effective batch size: {effective_batch_size} ({'×'.join([str(effective_batch_size//num_gpus)]*num_gpus)} per GPU)")

    train_loader, val_loader = get_segmentation_loaders(
        batch_size=effective_batch_size,
        num_workers=4,  # Tăng workers vì có nhiều GPU
        img_size=img_size
    )

    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}\n")

    # Create FULL Attention U-Net model
    print("Creating Attention U-Net model...")
    model = create_advanced_segmentation_model(
        in_channels=3,
        out_channels=3,
        base_filters=64  # Full version
    )

    # Wrap model with DataParallel for multi-GPU
    if use_multigpu and num_gpus > 1:
        print(f"Wrapping model with DataParallel ({num_gpus} GPUs)...")
        model = DataParallel(model, device_ids=list(range(num_gpus)))
        print(f"✓ Model will use GPUs: {list(range(num_gpus))}")

    model = model.to(device)

    # Count parameters
    if use_multigpu:
        total_params = sum(p.numel() for p in model.module.parameters())
        trainable_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}\n")

    # Loss function
    criterion = create_advanced_loss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )

    # OneCycle LR scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # Mixed precision - use on GPU 0
    scaler = GradScaler('cuda')

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
    print("Starting Phase 2 with Multi-GPU Training...")
    print("="*80 + "\n")

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # ============ Training Phase ============
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, (images, masks) in enumerate(pbar):
            # DataParallel will automatically split batch across GPUs
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            # Mixed precision training
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Clear cache periodically
            if batch_idx % 20 == 0:
                for i in range(num_gpus):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()

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

            # Save model (handle DataParallel)
            model_to_save = model.module if use_multigpu else model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
                'best_dice': best_dice,
                'num_gpus': num_gpus,
            }, 'outputs/models/best_seg_model.pth')  # ✅ SỬA TÊN FILE CHO KHỚP
            print(f"Best model saved! (IoU: {best_iou:.4f}, Dice: {best_dice:.4f})")
            patience = 0
        else:
            patience += 1
            print(f"Patience: {patience}/{max_patience}")

        print(f"{'='*80}\n")

        # GPU Memory info
        if (epoch + 1) % 5 == 0:
            print("GPU Memory Usage:")
            for i in range(num_gpus):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                print(f"   GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            print()

        # Early stopping
        if patience >= max_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best IoU: {best_iou:.4f}, Best Dice: {best_dice:.4f}")
            break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            model_to_save = model.module if use_multigpu else model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f'outputs/models/segmentation_checkpoint_epoch{epoch+1}.pth')

        # Memory cleanup
        if (epoch + 1) % 5 == 0:
            gc.collect()
            for i in range(num_gpus):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()

    training_time = (time.time() - start_time) / 60

    # Final Summary
    print("\n" + "="*80)
    print("PHASE 2: SEGMENTATION TRAINING COMPLETED!")
    print("="*80)
    print(f"Best IoU:  {best_iou:.4f}")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Training Time: {training_time:.1f} minutes")
    print(f"Used {num_gpus} GPU(s)")
    print(f"Model saved: outputs/models/best_seg_model.pth")
    print("="*80 + "\n")

    # Plot training curves
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Metrics plot
        axes[1].plot(history['val_iou'], label='Val IoU', color='green', linewidth=2)
        axes[1].plot(history['val_dice'], label='Val Dice', color='orange', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Validation Metrics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('outputs/logs/segmentation_training_curves_multigpu.png', dpi=150, bbox_inches='tight')
        print("Training curves saved: outputs/logs/segmentation_training_curves_multigpu.png\n")
    except Exception as e:
        print(f"Could not save plots: {e}\n")

    return best_iou, best_dice, history


if __name__ == '__main__':
    print("\n" + "="*80)
    print("DIABETIC RETINOPATHY - PHASE 2: SEGMENTATION (MULTI-GPU)")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check GPUs
    if not torch.cuda.is_available():
        print("ERROR: No GPU available!")
        exit(1)

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    for i in range(num_gpus):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

    print("="*80 + "\n")

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)  # Seed all GPUs

    # Memory cleanup
    print("Initial memory cleanup...")
    gc.collect()
    for i in range(num_gpus):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # Set memory config
    try:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("✓ Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    except:
        pass

    print("✓ Ready to train!\n")

    # Train with HIGH settings (Multi-GPU)
    device = torch.device('cuda:0')  # Primary device

    best_iou, best_dice, history = train_segmentation_multigpu(
        epochs=100,
        batch_size=4,      # Tổng batch size (mỗi GPU 4)
        img_size=1024,     # HIGH resolution
        device=device
    )

    print("\n" + "="*80)
    print("ALL PHASES COMPLETE!")
    print("="*80)
    print("Both models trained successfully:")
    print("Classification model: outputs/models/best_classification_model.pth")
    print("Segmentation model:   outputs/models/best_seg_model.pth")
    print("="*80 + "\n")
