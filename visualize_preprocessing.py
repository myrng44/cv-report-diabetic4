"""
Visualize CLAHE Preprocessing
Shows comparison between Original, Standard Histogram Equalization, and CLAHE
Usage: python visualize_preprocessing.py --image <path_to_image>
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from preprocessing import CLAHEEnhancer


def histogram_equalization(image):
    """Standard Histogram Equalization (HE)"""
    if len(image.shape) == 3:
        # Convert to YCrCb and apply HE to Y channel
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return cv2.equalizeHist(image)


def adaptive_histogram_equalization(image, clip_limit=40.0, tile_grid_size=(8, 8)):
    """Adaptive Histogram Equalization (AHE) - with high clip limit"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image.shape) == 3:
        # Convert to LAB and apply to L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(image)


def visualize_clahe_preprocessing(image_path, output_dir='outputs/preprocess'):
    """
    Visualize CLAHE preprocessing with different clip limits:
    1. Original Image
    2. HE (Histogram Equalization)
    3. AHE (Adaptive HE - clip limit = 40.0)
    4. CLAHE (clip limit = 2.0) - Used in training
    5. CLAHE (clip limit = 4.0) - Alternative
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read image
    print(f"Reading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Resize for consistency
    image = cv2.resize(image, (512, 512))

    print("Applying CLAHE preprocessing...")

    # Step 1: Original
    original = image.copy()

    # Step 2: HE (Standard Histogram Equalization)
    print("  - Histogram Equalization (HE)...")
    he_result = histogram_equalization(image.copy())

    # Step 3: AHE (Adaptive Histogram Equalization with high clip limit)
    print("  - Adaptive Histogram Equalization (AHE, clip=40)...")
    ahe_result = adaptive_histogram_equalization(image.copy(), clip_limit=40.0)

    # Step 4: CLAHE (clip limit = 2.0) - Default
    print("  - CLAHE (clip=2.0) - Default...")
    clahe_2 = CLAHEEnhancer(clip_limit=2.0, tile_grid_size=(8, 8))
    clahe_2_result = clahe_2.apply(image.copy())

    # Step 5: CLAHE (clip limit = 4.0) - Used in training
    print("  - CLAHE (clip=4.0) - Used in Training...")
    clahe_4 = CLAHEEnhancer(clip_limit=4.0, tile_grid_size=(8, 8))
    clahe_4_result = clahe_4.apply(image.copy())

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CLAHE Preprocessing Comparison', fontsize=20, fontweight='bold')

    images = [
        (original, 'Original Image'),
        (he_result, 'HE (Histogram Equalization)'),
        (ahe_result, 'AHE (clip=40.0)'),
        (clahe_2_result, 'CLAHE (clip=2.0)'),
        (clahe_4_result, 'CLAHE (clip=4.0) ⭐ Training'),
        (np.zeros_like(original), '')  # Empty placeholder
    ]

    for idx, (img, title) in enumerate(images):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        if idx == 5:  # Empty placeholder
            ax.axis('off')
            continue

        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()

    # Save comprehensive visualization
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f'{image_name}_clahe_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved CLAHE comparison: {output_path}")

    # Create side-by-side comparison: Original vs CLAHE variations
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle('CLAHE Enhancement Levels', fontsize=18, fontweight='bold')

    key_images = [
        (original, 'Original'),
        (clahe_2_result, 'CLAHE (clip=2.0)\\nSubtle Enhancement'),
        (clahe_4_result, 'CLAHE (clip=4.0)\\n⭐ Training Pipeline')
    ]

    for idx, (img, title) in enumerate(key_images):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes2[idx].imshow(img_rgb)
        axes2[idx].set_title(title, fontsize=14, fontweight='bold')
        axes2[idx].axis('off')

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, f'{image_name}_clahe_levels.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved CLAHE levels comparison: {comparison_path}")

    # Create histogram comparison
    fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
    fig3.suptitle('Histogram Analysis - CLAHE Effects', fontsize=18, fontweight='bold')

    for idx, (img, title) in enumerate(images[:5]):  # Skip empty placeholder
        row = idx // 3
        col = idx % 3
        ax = axes3[row, col]

        # Calculate histogram for each channel
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, alpha=0.7, linewidth=1.5)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.set_xlim([0, 256])
        ax.grid(True, alpha=0.3)

    # Remove last subplot
    fig3.delaxes(axes3[1, 2])

    plt.tight_layout()
    histogram_path = os.path.join(output_dir, f'{image_name}_clahe_histograms.png')
    plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved histogram analysis: {histogram_path}")

    print(f"\n{'='*80}")
    print(f"✅ CLAHE VISUALIZATION COMPLETED!")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Files created:")
    print(f"  1. {image_name}_clahe_comparison.png - All CLAHE variations")
    print(f"  2. {image_name}_clahe_levels.png - Key comparison")
    print(f"  3. {image_name}_clahe_histograms.png - Histogram analysis")
    print(f"{'='*80}\n")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize CLAHE preprocessing')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='outputs/preprocess',
                       help='Output directory (default: outputs/preprocess)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🔍 CLAHE PREPROCESSING VISUALIZATION")
    print("="*80 + "\n")

    visualize_clahe_preprocessing(args.image, args.output)


if __name__ == "__main__":
    main()

