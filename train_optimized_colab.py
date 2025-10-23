"""
Optimized Training Script for Google Colab T4 GPU
Cấu hình tối ưu cho GPU T4 (16GB VRAM)

Usage:
    !python train_optimized_colab.py
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


def train_optimized_pipeline():
    """
    Training pipeline tối ưu cho Colab T4 GPU

    Chiến lược:
    - Classification: Image size lớn OK (768) vì model nhỏ hơn
    - Segmentation: Giảm image size (512-640) vì model nặng hơn
    - Batch size: Điều chỉnh động dựa trên VRAM
    """

    print("\n" + "="*80)
    print("🚀 OPTIMIZED TRAINING FOR COLAB T4 GPU")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check GPU
    if not torch.cuda.is_available():
        print("⚠️  WARNING: No GPU detected! Training will be very slow.")
        print("Please enable GPU in Colab: Runtime -> Change runtime type -> GPU")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {gpu_memory:.1f} GB")
    print("="*80 + "\n")

    results = {}
    start_time = time.time()

    # ========================================================================
    # PHASE 1: CLASSIFICATION TRAINING
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 1: CLASSIFICATION TRAINING")
    print("="*80)
    print("Strategy: Large image size OK (model is lightweight)")
    print("Config:")
    print("  - Image Size: 640 (optimal for T4)")
    print("  - Batch Size: 8 (good balance)")
    print("  - Epochs: 100")
    print("  - Mixed Precision: Enabled")
    print("="*80 + "\n")

    # Set config for classification
    config.IMG_SIZE = 640  # Tối ưu cho T4 (không quá lớn, không quá nhỏ)
    config.BATCH_SIZE = 8  # Batch size tốt cho T4
    config.NUM_EPOCHS = 100
    config.EARLY_STOPPING_PATIENCE = 20
    config.USE_AMP = True  # Bật mixed precision

    try:
        class_start = time.time()
        best_acc = train_classification_model()
        class_time = time.time() - class_start

        results['classification'] = {
            'best_accuracy': best_acc,
            'training_time': class_time,
            'status': 'SUCCESS',
            'config': {
                'img_size': 640,
                'batch_size': 8,
                'epochs': 100
            }
        }

        print("\n" + "="*80)
        print(f"✅ CLASSIFICATION COMPLETED!")
        print(f"   Best Accuracy: {best_acc:.4f}")
        print(f"   Training Time: {class_time/60:.1f} minutes")
        print(f"   Image Size: 640")
        print(f"   Batch Size: 8")
        print("="*80)

        # Clear GPU cache
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n❌ Classification training failed: {e}")
        results['classification'] = {'status': 'FAILED', 'error': str(e)}
        torch.cuda.empty_cache()

    # Wait a bit before next phase
    print("\n⏳ Clearing GPU cache before segmentation...")
    time.sleep(5)
    torch.cuda.empty_cache()

    # ========================================================================
    # PHASE 2: SEGMENTATION TRAINING
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2: SEGMENTATION TRAINING")
    print("="*80)
    print("Strategy: Reduce image size (segmentation model is heavier)")
    print("Config:")
    print("  - Image Size: 512 (optimal for T4 + U-Net)")
    print("  - Batch Size: 6 (safe for complex model)")
    print("  - Epochs: 150 (need more for convergence)")
    print("  - Mixed Precision: Enabled")
    print("  - Gradient Accumulation: 2 steps (effective batch=12)")
    print("="*80 + "\n")

    # Set config for segmentation - CRITICAL CHANGES!
    config.IMG_SIZE = 512  # 512 instead of 768 - KEY IMPROVEMENT!
    config.BATCH_SIZE = 6  # 6 instead of 8 - safer for U-Net
    config.NUM_EPOCHS = 150  # More epochs for better convergence
    config.EARLY_STOPPING_PATIENCE = 30
    config.USE_AMP = True
    config.GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 12

    try:
        seg_start = time.time()
        best_iou = train_segmentation_model()
        seg_time = time.time() - seg_start

        results['segmentation'] = {
            'best_iou': best_iou,
            'training_time': seg_time,
            'status': 'SUCCESS',
            'config': {
                'img_size': 512,
                'batch_size': 6,
                'epochs': 150
            }
        }

        print("\n" + "="*80)
        print(f"✅ SEGMENTATION COMPLETED!")
        print(f"   Best IoU: {best_iou:.4f}")
        print(f"   Training Time: {seg_time/60:.1f} minutes")
        print(f"   Image Size: 512")
        print(f"   Batch Size: 6")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Segmentation training failed: {e}")
        results['segmentation'] = {'status': 'FAILED', 'error': str(e)}
    finally:
        torch.cuda.empty_cache()

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
        print(f"   - Config: {results['classification']['config']}")
        print(f"   - Model: outputs/models/best_classification_model.pth")
    else:
        print(f"❌ Classification: Failed or Skipped")

    print()

    if 'segmentation' in results and results['segmentation']['status'] == 'SUCCESS':
        print(f"✅ Segmentation:")
        print(f"   - Best IoU: {results['segmentation']['best_iou']:.4f}")
        print(f"   - Time: {results['segmentation']['training_time']/60:.1f} min")
        print(f"   - Config: {results['segmentation']['config']}")
        print(f"   - Model: outputs/models/best_seg_model.pth")
    else:
        print(f"❌ Segmentation: Failed or Skipped")

    print("\n" + "-"*80)
    print("PERFORMANCE ANALYSIS:")
    print("-"*80)

    if 'classification' in results and results['classification']['status'] == 'SUCCESS':
        acc = results['classification']['best_accuracy']
        if acc >= 0.75:
            print(f"🌟 Classification: EXCELLENT ({acc:.1%})")
        elif acc >= 0.70:
            print(f"✅ Classification: GOOD ({acc:.1%})")
        elif acc >= 0.65:
            print(f"⭐ Classification: ACCEPTABLE ({acc:.1%})")
        else:
            print(f"⚠️  Classification: NEEDS IMPROVEMENT ({acc:.1%})")

    if 'segmentation' in results and results['segmentation']['status'] == 'SUCCESS':
        iou = results['segmentation']['best_iou']
        if iou >= 0.45:
            print(f"🌟 Segmentation: EXCELLENT (IoU={iou:.3f})")
        elif iou >= 0.35:
            print(f"✅ Segmentation: GOOD (IoU={iou:.3f})")
        elif iou >= 0.25:
            print(f"⭐ Segmentation: ACCEPTABLE (IoU={iou:.3f})")
        else:
            print(f"⚠️  Segmentation: NEEDS IMPROVEMENT (IoU={iou:.3f})")

    print("\n" + "-"*80)
    print("NEXT STEPS:")
    print("-"*80)
    print("1. Run inference on test images:")
    print("   !python demo_inference.py --image path/to/image.jpg")
    print("\n2. If results not satisfactory, try:")
    print("   - Increase epochs (150 -> 200)")
    print("   - Adjust learning rate")
    print("   - Try different batch sizes")
    print("\n3. View training logs:")
    print("   outputs/logs/training.log")
    print("\n4. Compare with previous results")
    print("="*80 + "\n")

    # Save results to file
    import json
    results_file = os.path.join(config.RESULT_DIR, 'training_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'gpu': gpu_name,
            'gpu_memory_gb': gpu_memory,
            'total_time_minutes': total_time/60,
            'results': results
        }, f, indent=2)
    print(f"📊 Results saved to: {results_file}\n")

    return results


if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     🚀 DIABETIC RETINOPATHY DETECTION SYSTEM 🚀              ║
    ║                                                               ║
    ║          Optimized for Google Colab T4 GPU                   ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    This script will train both Classification and Segmentation models
    with optimized settings for T4 GPU (16GB VRAM).
    
    Expected results:
    - Classification: 70-75% accuracy (improved from 65%)
    - Segmentation: 35-42% IoU (improved from 21%)
    
    Training time: ~2-2.5 hours
    
    """)

    # Set random seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)

    # Run training
    train_optimized_pipeline()

