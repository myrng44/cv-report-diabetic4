"""
Quick script to run Grad-CAM visualization on trained model
Usage: python run_gradcam.py
"""

import torch
import os
from gradcam_visualization import visualize_predictions_with_gradcam, batch_visualize_gradcam
from classification_model import create_classification_model

def main():
    print("\n" + "="*80)
    print("🔍 GRAD-CAM VISUALIZATION FOR DR CLASSIFICATION")
    print("="*80 + "\n")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    print("Loading trained model...")
    model = create_classification_model(num_classes=5, pretrained=False)

    # Try to load ultra-optimized model first, fallback to regular
    if os.path.exists('outputs/models/best_model_ultra.pth'):
        model.load_state_dict(torch.load('outputs/models/best_model_ultra.pth', map_location=device))
        print("✓ Loaded ultra-optimized model\n")
    elif os.path.exists('outputs/models/best_model.pth'):
        model.load_state_dict(torch.load('outputs/models/best_model.pth', map_location=device))
        print("✓ Loaded regular model\n")
    else:
        print("✗ No trained model found!")
        print("Please train a model first using:")
        print("  python train_ultra_optimized.py")
        print("  or")
        print("  python main.py --mode train_classification")
        return

    model = model.to(device)
    model.eval()

    # Create output directory
    os.makedirs('outputs/results/gradcam_visualizations', exist_ok=True)

    # Option 1: Visualize specific image
    test_image = 'data/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_01.jpg'

    if os.path.exists(test_image):
        print("Generating Grad-CAM for sample image...")
        visualize_predictions_with_gradcam(
            model,
            test_image,
            device,
            save_path='outputs/results/gradcam_example.png'
        )
        print()

    # Option 2: Batch visualization (10 random samples)
    train_dir = 'data/B. Disease Grading/1. Original Images/a. Training Set'

    if os.path.exists(train_dir):
        print(f"Generating Grad-CAM for 10 random training images...")
        batch_visualize_gradcam(
            model,
            train_dir,
            device,
            'outputs/results/gradcam_visualizations',
            num_samples=10
        )

    print("\n" + "="*80)
    print("✅ GRAD-CAM VISUALIZATION COMPLETED!")
    print("="*80)
    print("\nCheck results in:")
    print("  - outputs/results/gradcam_example.png")
    print("  - outputs/results/gradcam_visualizations/")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
