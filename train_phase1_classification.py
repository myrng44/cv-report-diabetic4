"""
GIAI ĐOẠN 1: HUẤN LUYỆN PHÂN LOẠI - Tối ưu hóa Tối đa
Chạy riêng phase này trên Kaggle để tránh OOM

Cách sử dụng trong Kaggle:
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
    """Huấn luyện phân loại tối ưu hóa tối đa"""

    print("\n" + "="*80)
    print("GIAI ĐOẠN 1: HUẤN LUYỆN PHÂN LOẠI")
    print("="*80)
    print(f"Mục tiêu: >75% Accuracy")
    print(f"Chiến lược: EfficientNet + Tăng cường Mạnh + Cosine LR")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Kích thước Ảnh: {img_size}")
    print("="*80 + "\n")

    # Tạo thư mục đầu ra
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)

    # Tải dữ liệu
    print("Đang tải dữ liệu phân loại...")
    train_loader, val_loader = get_classification_loaders(
        batch_size=batch_size,
        num_workers=2,  # Giảm workers cho Kaggle
        img_size=img_size
    )

    print(f"✓ Mẫu huấn luyện: {len(train_loader.dataset)}")
    print(f"✓ Mẫu validation: {len(val_loader.dataset)}\n")

    # Tạo mô hình
    print("Đang tạo mô hình EfficientNet-B3...")
    model = create_classification_model(num_classes=5, pretrained=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Tổng tham số: {total_params:,}")
    print(f"✓ Tham số có thể huấn luyện: {trainable_params:,}\n")

    # Loss và optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    # Cosine annealing với warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Độ chính xác hỗn hợp
    device_type = 'cuda' if device == 'cuda' else 'cpu'
    scaler = GradScaler(device_type) if device == 'cuda' else None

    # Trạng thái huấn luyện
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
    print("Bắt đầu Huấn luyện...")
    print("="*80 + "\n")

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # ============ Giai đoạn Huấn luyện ============
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

        # ============ Giai đoạn Validation ============
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

        # Cập nhật scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Ghi lại metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        epoch_time = (time.time() - epoch_start) / 60

        # In kết quả
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs} - Thời gian: {epoch_time:.1f}phút - LR: {current_lr:.2e}")
        print(f"{'='*80}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"Val F1:     {val_f1:.4f}")

        # Lưu mô hình tốt nhất
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
            print(f"Best model saved! (Acc: {best_acc:.4f}, F1: {best_f1:.4f})")
            patience = 0
        else:
            patience += 1
            print(f"Patience: {patience}/{max_patience}")

        print(f"{'='*80}\n")

        # Dừng sớm
        if patience >= max_patience:
            print(f"\nDừng sớm kích hoạt sau {epoch+1} epochs")
            print(f"Độ chính xác tốt nhất: {best_acc:.4f} ({best_acc*100:.2f}%)")
            break

        # Lưu checkpoint mỗi 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f'outputs/models/classification_checkpoint_epoch{epoch+1}.pth')

    training_time = (time.time() - start_time) / 60

    # Tổng kết Cuối cùng
    print("\n" + "="*80)
    print("GIAI ĐOẠN 1: HUẤN LUYỆN PHÂN LOẠI HOÀN THÀNH!")
    print("="*80)
    print(f"Độ chính xác tốt nhất: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"Điểm F1 tốt nhất: {best_f1:.4f}")
    print(f"Thời gian Huấn luyện: {training_time:.1f} phút")
    print(f"Mô hình đã lưu: outputs/models/best_classification_model.pth")
    print("="*80 + "\n")

    # Vẽ biểu đồ huấn luyện
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Biểu đồ Loss
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Huấn luyện & Validation')
        axes[0].legend()
        axes[0].grid(True)

        # Biểu đồ Accuracy
        axes[1].plot(history['val_acc'], label='Val Accuracy', color='green')
        axes[1].plot(history['val_f1'], label='Val F1', color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Metrics Validation')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('outputs/logs/classification_training_curves.png', dpi=150, bbox_inches='tight')
        print("Biểu đồ huấn luyện đã lưu: outputs/logs/classification_training_curves.png\n")
    except Exception as e:
        print(f"Không thể lưu biểu đồ: {e}\n")

    # In ma trận nhầm lẫn
    try:
        print("Ma trận Nhầm lẫn (Tập Validation):")
        cm = confusion_matrix(val_labels, val_preds)
        print(cm)
        print()
    except:
        pass

    return best_acc, best_f1, history


if __name__ == '__main__':
    # Thiết lập thiết bị
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*80)
    print("BỆNH VõNG MẠC TIỂU ĐƯỜNG - GIAI ĐOẠN 1: PHÂN LOẠI")
    print("="*80)
    print(f"Thời gian Bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Thiết bị: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("="*80 + "\n")

    # Đặt seed cho tính tái tạo
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Huấn luyện
    best_acc, best_f1, history = train_classification_phase(
        epochs=100,
        batch_size=8,
        img_size=768,
        device=device
    )

    print("\n" + "="*80)
    print("GIAI ĐOẠN 1 HOÀN THÀNH - SẴn sàng cho Giai đoạn 2!")
    print("="*80)
    print(f"Bước tiếp theo: Chạy train_phase2_segmentation.py")
    print("="*80 + "\n")

