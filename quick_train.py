"""
QUICK START - Chạy Training Đơn Giản
Sử dụng file này để train nhanh và dễ dàng
"""

import torch
import config
from train_classification import train_classification_model
from train_segmentation_optimized import train_segmentation_model


def main():
    print("=" * 80)
    print("🚀 DIABETIC RETINOPATHY TRAINING - QUICK START")
    print("=" * 80)

    # Kiểm tra GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ Device: {device.upper()}")

    if device == 'cuda':
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Cấu hình tối ưu
    print("\n" + "=" * 80)
    print("CONFIGURATION:")
    print("=" * 80)
    print("\n📊 CLASSIFICATION:")
    print("   • Epochs: 100")
    print("   • Batch Size: 8")
    print("   • Image Size: 768")
    print("   • Target: >75% accuracy")

    print("\n🎯 SEGMENTATION:")
    print("   • Epochs: 100")
    print("   • Batch Size: 4")
    print("   • Image Size: 1024 (HIGH for tiny lesions!)")
    print("   • Target: IoU >0.40")

    print("\n" + "=" * 80)
    print("Bắt đầu training trong 5 giây...")
    print("Nhấn Ctrl+C để hủy")
    print("=" * 80)

    import time
    for i in range(5, 0, -1):
        print(f"\r⏳ {i}...", end='', flush=True)
        time.sleep(1)
    print("\r✓ Start!     ")

    # ===========================================================================
    # PHASE 1: CLASSIFICATION
    # ===========================================================================

    print("\n" + "=" * 80)
    print("PHASE 1: CLASSIFICATION TRAINING")
    print("=" * 80)

    try:
        best_acc = train_classification_model(
            num_epochs=100,
            batch_size=8,
            img_size=768,
            learning_rate=1e-4,
            device=device
        )

        print("\n" + "=" * 80)
        print(f"✅ CLASSIFICATION COMPLETED!")
        print(f"   Best Accuracy: {best_acc:.4f}")
        print(f"   Model saved: outputs/models/best_model.pth")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Classification failed: {e}")
        print("Dừng training.")
        return

    # ===========================================================================
    # PHASE 2: SEGMENTATION
    # ===========================================================================

    print("\n" + "=" * 80)
    print("PHASE 2: SEGMENTATION TRAINING")
    print("=" * 80)

    try:
        best_iou = train_segmentation_model(
            num_epochs=100,
            batch_size=4,
            img_size=1024,
            learning_rate=5e-5,
            device=device
        )

        print("\n" + "=" * 80)
        print(f"✅ SEGMENTATION COMPLETED!")
        print(f"   Best IoU: {best_iou:.4f}")
        print(f"   Model saved: outputs/models/best_seg_model.pth")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Segmentation failed: {e}")
        return

    # ===========================================================================
    # SUMMARY
    # ===========================================================================

    print("\n" + "=" * 80)
    print("🎉 TRAINING HOÀN TẤT!")
    print("=" * 80)
    print(f"\n✅ Classification: {best_acc:.4f} {'(✓ Đạt target!)' if best_acc >= 0.75 else '(Chưa đạt 75%)'}")
    print(f"✅ Segmentation IoU: {best_iou:.4f} {'(✓ Đạt target!)' if best_iou >= 0.40 else '(Chưa đạt 0.40)'}")

    print("\n📁 Models đã lưu:")
    print("   • outputs/models/best_model.pth (classification)")
    print("   • outputs/models/best_seg_model.pth (segmentation)")

    print("\n🎯 Tiếp theo:")
    print("   • Test model: python demo_inference.py --image path/to/image.jpg")
    print("   • Đánh giá: python evaluate.py")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

