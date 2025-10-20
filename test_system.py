"""
Test script to verify installation and data setup
"""

import os
import sys
import torch
import cv2
import numpy as np

def test_imports():
    """Test if all required libraries are installed"""
    print("Testing imports...")
    try:
        import torch
        import torchvision
        import cv2
        import numpy as np
        import pandas as pd
        import sklearn
        import albumentations
        from pytorch_grad_cam import GradCAM
        print("✓ All required libraries are installed")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("⚠ CUDA is not available. Training will use CPU (very slow)")
        return False


def test_data_structure():
    """Test if data is properly organized"""
    print("\nTesting data structure...")

    import config

    required_paths = [
        (config.CLASS_TRAIN_IMG_DIR, "Classification Training Images"),
        (config.CLASS_TRAIN_LABELS, "Classification Training Labels"),
        (config.SEG_TRAIN_IMG_DIR, "Segmentation Training Images"),
        (config.SEG_TRAIN_MASK_DIR, "Segmentation Training Masks"),
    ]

    all_exist = True
    for path, name in required_paths:
        if os.path.exists(path):
            if path.endswith('.csv'):
                print(f"✓ {name}: Found")
            else:
                files = [f for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.tif')]
                print(f"✓ {name}: {len(files)} files")
        else:
            print(f"❌ {name}: Not found at {path}")
            all_exist = False

    return all_exist


def test_preprocessing():
    """Test preprocessing pipeline"""
    print("\nTesting preprocessing...")
    try:
        from preprocessing import ImagePreprocessor

        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        preprocessor = ImagePreprocessor(target_size=256)
        processed = preprocessor.preprocess(dummy_image, apply_gabor=False)
        normalized = preprocessor.normalize(processed)

        assert processed.shape == (256, 256, 3), "Preprocessing output shape mismatch"
        assert normalized.max() <= 1.0, "Normalization failed"

        print("✓ Preprocessing pipeline works correctly")
        return True
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False


def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    try:
        from classification_model import create_classification_model
        from segmentation_model import create_segmentation_model

        # Test classification model
        class_model = create_classification_model(num_classes=5)
        print(f"✓ Classification model created")

        # Test segmentation model
        seg_model = create_segmentation_model(in_channels=3, out_channels=3)
        print(f"✓ Segmentation model created")

        # Test forward pass
        dummy_input = torch.randn(1, 3, 256, 256)

        class_output = class_model(dummy_input)
        assert class_output.shape == (1, 5), "Classification output shape mismatch"
        print(f"✓ Classification forward pass successful")

        seg_output = seg_model(dummy_input)
        assert seg_output.shape == (1, 3, 256, 256), "Segmentation output shape mismatch"
        print(f"✓ Segmentation forward pass successful")

        return True
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test dataset loading"""
    print("\nTesting dataset loading...")
    try:
        from dataset import DRClassificationDataset, get_classification_transforms
        import config

        # Test if we can load the dataset
        transform = get_classification_transforms(is_train=False, img_size=256)
        dataset = DRClassificationDataset(
            image_dir=config.CLASS_TRAIN_IMG_DIR,
            labels_csv=config.CLASS_TRAIN_LABELS,
            transform=transform,
            is_train=True
        )

        print(f"✓ Classification dataset loaded: {len(dataset)} samples")

        # Try to get one sample
        image, label = dataset[0]
        assert image.shape == (3, 256, 256), "Dataset image shape mismatch"
        print(f"✓ Dataset sample loading successful")

        return True
    except Exception as e:
        print(f"❌ Dataset loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sango_optimizer():
    """Test SANGO optimizer"""
    print("\nTesting SANGO optimizer...")
    try:
        from sango_optimizer import SANGOOptimizer

        # Simple test function (sphere function)
        def test_function(x):
            return np.sum(x**2)

        # Create optimizer
        optimizer = SANGOOptimizer(
            objective_function=test_function,
            dim=2,
            bounds=[(-5, 5), (-5, 5)],  # Fixed: Changed − to -
            population_size=10,
            max_iterations=5
        )

        # Run optimization
        best_solution, best_fitness = optimizer.optimize(verbose=False)

        print(f"✓ SANGO optimizer works correctly")
        print(f"  Best fitness: {best_fitness:.6f}")

        return True
    except Exception as e:
        print(f"❌ SANGO optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*80)
    print("SANGO System Test Suite")
    print("="*80)

    tests = [
        ("Import Test", test_imports),
        ("CUDA Test", test_cuda),
        ("Data Structure Test", test_data_structure),
        ("Preprocessing Test", test_preprocessing),
        ("Model Creation Test", test_model_creation),
        ("Dataset Loading Test", test_dataset_loading),
        ("SANGO Optimizer Test", test_sango_optimizer),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print("="*80)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Train classification model:")
        print("     python main.py --mode train_classification")
        print("  2. Train segmentation model:")
        print("     python main.py --mode train_segmentation")
        print("  3. Or train both:")
        print("     python main.py --mode train_all")
    else:
        print("\n⚠ Some tests failed. Please fix the issues before proceeding.")

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
