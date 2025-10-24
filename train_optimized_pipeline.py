"""
Optimized Complete Training Pipeline
Target: Classification >75%, Segmentation IoU >0.40
"""

import os
import sys
import time
import argparse
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import config
from train_classification import train_classification_model
from train_segmentation_optimized import train_segmentation_model


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80 + "\n")


def print_section(text):
    """Print section header"""
    print("\n" + "-" * 80)
    print(text)
    print("-" * 80 + "\n")


def train_complete_pipeline(
    class_epochs=100,
    seg_epochs=100,
    class_img_size=768,
    seg_img_size=1024,
    class_batch_size=8,
    seg_batch_size=4,
    device='cuda'
):
    """
    Complete training pipeline with optimized parameters

    Args:
        class_epochs: Epochs for classification
        seg_epochs: Epochs for segmentation
        class_img_size: Image size for classification (768 recommended)
        seg_img_size: Image size for segmentation (1024+ for tiny lesions)
        class_batch_size: Batch size for classification
        seg_batch_size: Batch size for segmentation (lower due to larger images)
        device: 'cuda' or 'cpu'
    """

    start_time = time.time()

    print_header("🚀 OPTIMIZED TRAINING PIPELINE")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device.upper()}\n")

    print_section("CONFIGURATION:")
    print(f"📊 Classification:")
    print(f"   - Epochs: {class_epochs}")
    print(f"   - Batch Size: {class_batch_size}")
    print(f"   - Image Size: {class_img_size}")
    print(f"   - Target Accuracy: >75%")
    print(f"\n🎯 Segmentation:")
    print(f"   - Epochs: {seg_epochs}")
    print(f"   - Batch Size: {seg_batch_size}")
    print(f"   - Image Size: {seg_img_size} (HIGH for tiny lesions)")
    print(f"   - Target IoU: >0.40")
    print("=" * 80)

    results = {}

    # ============================================================================
    # PHASE 1: CLASSIFICATION TRAINING
    # ============================================================================

    print_header("PHASE 1: CLASSIFICATION TRAINING")
    print("Training DR Grade Classification (0-4)")
    print(f"Epochs: {class_epochs} | Batch: {class_batch_size} | Size: {class_img_size}")
    print("=" * 80)

    class_start = time.time()

    try:
        # Train classification with optimized parameters
        best_acc = train_classification_model(
            num_epochs=class_epochs,
            batch_size=class_batch_size,
            img_size=class_img_size,
            learning_rate=config.LEARNING_RATE,
            device=device
        )

        results['classification'] = {
            'best_accuracy': best_acc,
            'time': (time.time() - class_start) / 60
        }

        print_header(f"✅ CLASSIFICATION COMPLETED!")
        print(f"   Best Accuracy: {best_acc:.4f}")
        print(f"   Training Time: {results['classification']['time']:.1f} minutes")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Classification training failed: {str(e)}")
        results['classification'] = {'error': str(e)}
        return results

    # ============================================================================
    # PHASE 2: SEGMENTATION TRAINING
    # ============================================================================

    print_header("PHASE 2: SEGMENTATION TRAINING")
    print("Training Lesion Segmentation (Microaneurysms, Haemorrhages, Exudates)")
    print(f"Epochs: {seg_epochs} | Batch: {seg_batch_size} | Size: {seg_img_size}")
    print("=" * 80)

    seg_start = time.time()

    try:
        # Train segmentation with optimized parameters
        best_iou = train_segmentation_model(
            num_epochs=seg_epochs,
            batch_size=seg_batch_size,
            img_size=seg_img_size,
            learning_rate=config.SEG_LEARNING_RATE,
            device=device
        )

        results['segmentation'] = {
            'best_iou': best_iou,
            'time': (time.time() - seg_start) / 60
        }

        print_header(f"✅ SEGMENTATION COMPLETED!")
        print(f"   Best IoU: {best_iou:.4f}")
        print(f"   Training Time: {results['segmentation']['time']:.1f} minutes")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Segmentation training failed: {str(e)}")
        results['segmentation'] = {'error': str(e)}
        return results

    # ============================================================================
    # SUMMARY
    # ============================================================================

    total_time = (time.time() - start_time) / 60

    print_header("🎉 TRAINING PIPELINE COMPLETED!")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Time: {total_time:.1f} minutes ({total_time/60:.2f} hours)\n")

    print_section("RESULTS SUMMARY:")

    if 'classification' in results and 'best_accuracy' in results['classification']:
        acc = results['classification']['best_accuracy']
        status = "✅" if acc >= 0.75 else "⚠️"
        print(f"{status} Classification:")
        print(f"   - Best Accuracy: {acc:.4f} (Target: >0.75)")
        print(f"   - Time: {results['classification']['time']:.1f} min")
        print(f"   - Model: outputs/models/best_model.pth")

    print()

    if 'segmentation' in results and 'best_iou' in results['segmentation']:
        iou = results['segmentation']['best_iou']
        status = "✅" if iou >= 0.40 else "⚠️"
        print(f"{status} Segmentation:")
        print(f"   - Best IoU: {iou:.4f} (Target: >0.40)")
        print(f"   - Time: {results['segmentation']['time']:.1f} min")
        print(f"   - Model: outputs/models/best_seg_model.pth")

    print_section("NEXT STEPS:")
    print("1. Run inference on test images:")
    print("   python demo_inference.py --image path/to/image.jpg\n")
    print("2. Evaluate on test set:")
    print("   python evaluate.py\n")
    print("3. View training logs:")
    print("   outputs/logs/training.log")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description='Optimized Complete Training Pipeline')

    # Classification parameters
    parser.add_argument('--class_epochs', type=int, default=100, help='Classification epochs')
    parser.add_argument('--class_batch_size', type=int, default=8, help='Classification batch size')
    parser.add_argument('--class_img_size', type=int, default=768, help='Classification image size')

    # Segmentation parameters
    parser.add_argument('--seg_epochs', type=int, default=100, help='Segmentation epochs')
    parser.add_argument('--seg_batch_size', type=int, default=4, help='Segmentation batch size (lower for 1024)')
    parser.add_argument('--seg_img_size', type=int, default=1024, help='Segmentation image size (HIGH for lesions)')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)

    # Run training
    results = train_complete_pipeline(
        class_epochs=args.class_epochs,
        seg_epochs=args.seg_epochs,
        class_img_size=args.class_img_size,
        seg_img_size=args.seg_img_size,
        class_batch_size=args.class_batch_size,
        seg_batch_size=args.seg_batch_size,
        device=args.device
    )

    return results


if __name__ == "__main__":
    main()
