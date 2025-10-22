"""
Main script to run SANGO-based DR Detection System
"""

import os
import argparse
import torch
import numpy as np

import config
from train_classification import train_classification_model
from train_segmentation import train_segmentation_model
from inference import DRInference


def main():
    parser = argparse.ArgumentParser(description='SANGO-based Diabetic Retinopathy Detection System')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train_classification', 'train_segmentation', 'train_all', 'inference'],
                       help='Mode to run: train_classification, train_segmentation, train_all, or inference')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to image for inference')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--img_size', type=int, default=None,
                       help='Image size (overrides config)')

    args = parser.parse_args()

    # Override config if specified
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.img_size:
        config.IMG_SIZE = args.img_size

    # Set random seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)

    print("="*80)
    print("SANGO-based Diabetic Retinopathy Detection System")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Image Size: {config.IMG_SIZE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print("="*80)

    if args.mode == 'train_classification':
        print("\n🚀 Starting Classification Training...")
        best_acc = train_classification_model()
        print(f"\n✓ Classification training completed!")
        print(f"Best accuracy: {best_acc:.4f}")

    elif args.mode == 'train_segmentation':
        print("\n🚀 Starting Segmentation Training...")
        best_iou = train_segmentation_model()
        print(f"\n✓ Segmentation training completed!")
        print(f"Best IoU: {best_iou:.4f}")

    elif args.mode == 'train_all':
        print("\n🚀 Starting Complete Training Pipeline...")

        # Train classification first
        print("\n" + "="*80)
        print("PHASE 1: Classification Training")
        print("="*80)
        best_acc = train_classification_model()
        print(f"\n✓ Classification training completed! Best accuracy: {best_acc:.4f}")

        # Train segmentation
        print("\n" + "="*80)
        print("PHASE 2: Segmentation Training")
        print("="*80)
        best_iou = train_segmentation_model()
        print(f"\n✓ Segmentation training completed! Best IoU: {best_iou:.4f}")

        print("\n" + "="*80)
        print("✓ Complete training pipeline finished!")
        print(f"Classification Accuracy: {best_acc:.4f}")
        print(f"Segmentation IoU: {best_iou:.4f}")
        print("="*80)

    elif args.mode == 'inference':
        if args.image_path is None:
            print("❌ Error: --image_path is required for inference mode")
            return

        if not os.path.exists(args.image_path):
            print(f"❌ Error: Image not found: {args.image_path}")
            return

        print(f"\n🔍 Running inference on: {args.image_path}")

        # Load models
        class_model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
        seg_model_path = os.path.join(config.MODEL_DIR, 'best_seg_model.pth')

        inference = DRInference(
            classification_model_path=class_model_path if os.path.exists(class_model_path) else None,
            segmentation_model_path=seg_model_path if os.path.exists(seg_model_path) else None
        )

        # Run inference
        save_path = os.path.join(config.RESULT_DIR, f'inference_{os.path.basename(args.image_path)}')
        inference.visualize_results(args.image_path, save_path)

        print(f"\n✓ Inference completed! Results saved to {save_path}")


if __name__ == '__main__':
    main()
