"""
Complete Training Pipeline - Train both Classification and Segmentation models
Với cấu hình riêng biệt cho từng model

Usage:
    python train_complete_pipeline.py
    python train_complete_pipeline.py --skip_classification
    python train_complete_pipeline.py --skip_segmentation
"""

import os
import argparse
import torch
import numpy as np
import time
from datetime import datetime

import config
from train_classification import train_classification_model
from train_segmentation import train_segmentation_model


def train_complete_pipeline(
    # Classification params
    class_epochs=70,
    class_batch_size=4,
    class_img_size=384,

    # Segmentation params
    seg_epochs=50,
    seg_batch_size=4,
    seg_img_size=512,

    # Control flags
    skip_classification=False,
    skip_segmentation=False
):
    """
    Train cả 2 models với cấu hình riêng biệt
    """

    print("\n" + "="*80)
    print("🚀 COMPLETE TRAINING PIPELINE")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("\n" + "="*80)
    print("CONFIGURATION:")
    print("="*80)
    if not skip_classification:
        print(f"📊 Classification:")
        print(f"   - Epochs: {class_epochs}")
        print(f"   - Batch Size: {class_batch_size}")
        print(f"   - Image Size: {class_img_size}")
    if not skip_segmentation:
        print(f"🎯 Segmentation:")
        print(f"   - Epochs: {seg_epochs}")
        print(f"   - Batch Size: {seg_batch_size}")
        print(f"   - Image Size: {seg_img_size}")
    print("="*80 + "\n")

    results = {}
    start_time = time.time()

    # ========================================================================
    # PHASE 1: CLASSIFICATION TRAINING
    # ========================================================================
    if not skip_classification:
        print("\n" + "="*80)
        print("PHASE 1: CLASSIFICATION TRAINING")
        print("="*80)
        print(f"Training DR Grade Classification (0-4)")
        print(f"Epochs: {class_epochs} | Batch: {class_batch_size} | Size: {class_img_size}")
        print("="*80 + "\n")

        # Set config for classification
        config.NUM_EPOCHS = class_epochs
        config.BATCH_SIZE = class_batch_size
        config.IMG_SIZE = class_img_size

        try:
            class_start = time.time()
            best_acc = train_classification_model()
            class_time = time.time() - class_start

            results['classification'] = {
                'best_accuracy': best_acc,
                'training_time': class_time,
                'status': 'SUCCESS'
            }

            print("\n" + "="*80)
            print(f"✅ CLASSIFICATION COMPLETED!")
            print(f"   Best Accuracy: {best_acc:.4f}")
            print(f"   Training Time: {class_time/60:.1f} minutes")
            print("="*80)

        except Exception as e:
            print(f"\n❌ Classification training failed: {e}")
            results['classification'] = {'status': 'FAILED', 'error': str(e)}
    else:
        print("\n⏭️  Skipping Classification Training")
        results['classification'] = {'status': 'SKIPPED'}

    # ========================================================================
    # PHASE 2: SEGMENTATION TRAINING
    # ========================================================================
    if not skip_segmentation:
        print("\n" + "="*80)
        print("PHASE 2: SEGMENTATION TRAINING")
        print("="*80)
        print(f"Training Lesion Segmentation (Microaneurysms, Haemorrhages, Exudates)")
        print(f"Epochs: {seg_epochs} | Batch: {seg_batch_size} | Size: {seg_img_size}")
        print("="*80 + "\n")

        # Set config for segmentation
        config.NUM_EPOCHS = seg_epochs
        config.BATCH_SIZE = seg_batch_size
        config.IMG_SIZE = seg_img_size

        try:
            seg_start = time.time()
            best_iou = train_segmentation_model()
            seg_time = time.time() - seg_start

            results['segmentation'] = {
                'best_iou': best_iou,
                'training_time': seg_time,
                'status': 'SUCCESS'
            }

            print("\n" + "="*80)
            print(f"✅ SEGMENTATION COMPLETED!")
            print(f"   Best IoU: {best_iou:.4f}")
            print(f"   Training Time: {seg_time/60:.1f} minutes")
            print("="*80)

        except Exception as e:
            print(f"\n❌ Segmentation training failed: {e}")
            results['segmentation'] = {'status': 'FAILED', 'error': str(e)}
    else:
        print("\n⏭️  Skipping Segmentation Training")
        results['segmentation'] = {'status': 'SKIPPED'}

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    total_time = time.time() - start_time

    print("\n\n" + "="*80)
    print("🎉 TRAINING PIPELINE COMPLETED!")
    print("="*80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print("\n" + "-"*80)
    print("RESULTS SUMMARY:")
    print("-"*80)

    if 'classification' in results and results['classification']['status'] == 'SUCCESS':
        print(f"✅ Classification:")
        print(f"   - Best Accuracy: {results['classification']['best_accuracy']:.4f}")
        print(f"   - Time: {results['classification']['training_time']/60:.1f} min")
        print(f"   - Model: outputs/models/best_model.pth")
    elif results.get('classification', {}).get('status') == 'SKIPPED':
        print(f"⏭️  Classification: Skipped")
    else:
        print(f"❌ Classification: Failed")

    print()

    if 'segmentation' in results and results['segmentation']['status'] == 'SUCCESS':
        print(f"✅ Segmentation:")
        print(f"   - Best IoU: {results['segmentation']['best_iou']:.4f}")
        print(f"   - Time: {results['segmentation']['training_time']/60:.1f} min")
        print(f"   - Model: outputs/models/best_seg_model.pth")
    elif results.get('segmentation', {}).get('status') == 'SKIPPED':
        print(f"⏭️  Segmentation: Skipped")
    else:
        print(f"❌ Segmentation: Failed")

    print("\n" + "-"*80)
    print("NEXT STEPS:")
    print("-"*80)
    print("1. Run inference on test images:")
    print("   python demo_inference.py --image path/to/image.jpg")
    print("\n2. Evaluate on test set:")
    print("   python evaluate.py")
    print("\n3. View training logs:")
    print("   outputs/logs/training.log")
    print("="*80 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description='Complete Training Pipeline')

    # Classification arguments
    parser.add_argument('--class_epochs', type=int, default=70,
                       help='Classification epochs (default: 70)')
    parser.add_argument('--class_batch_size', type=int, default=4,
                       help='Classification batch size (default: 4)')
    parser.add_argument('--class_img_size', type=int, default=384,
                       help='Classification image size (default: 384)')

    # Segmentation arguments
    parser.add_argument('--seg_epochs', type=int, default=50,
                       help='Segmentation epochs (default: 50)')
    parser.add_argument('--seg_batch_size', type=int, default=4,
                       help='Segmentation batch size (default: 4)')
    parser.add_argument('--seg_img_size', type=int, default=512,
                       help='Segmentation image size (default: 512)')

    # Control flags
    parser.add_argument('--skip_classification', action='store_true',
                       help='Skip classification training')
    parser.add_argument('--skip_segmentation', action='store_true',
                       help='Skip segmentation training')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)

    # Run pipeline
    results = train_complete_pipeline(
        class_epochs=args.class_epochs,
        class_batch_size=args.class_batch_size,
        class_img_size=args.class_img_size,
        seg_epochs=args.seg_epochs,
        seg_batch_size=args.seg_batch_size,
        seg_img_size=args.seg_img_size,
        skip_classification=args.skip_classification,
        skip_segmentation=args.skip_segmentation
    )

    return results


if __name__ == '__main__':
    main()

