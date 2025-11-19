"""
Visualize Preprocessing Steps
Shows each step of preprocessing pipeline including HE, AHE, CLAHE, and Adaptive Gabor
Usage: python visualize_preprocessing.py --image <path_to_image>
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from preprocessing import AdaptiveGaborFilter, CLAHEEnhancer


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


def visualize_all_preprocessing_steps(image_path, output_dir='outputs/preprocess'):
    """
    Visualize all preprocessing steps:
    1. Original Image
    2. HE (Histogram Equalization)
    3. AHE (Adaptive Histogram Equalization - high clip limit)
    4. CLAHE (Contrast Limited AHE - clip limit = 2.0)
    5. Adaptive Gabor Filter
    6. Gabor + CLAHE (Full Pipeline)
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

    print("Applying preprocessing steps...")

    # Step 1: Original
    original = image.copy()

    # Step 2: HE (Standard Histogram Equalization)
    print("  - Histogram Equalization (HE)...")
    he_result = histogram_equalization(image.copy())

    # Step 3: AHE (Adaptive Histogram Equalization with high clip limit)
    print("  - Adaptive Histogram Equalization (AHE)...")
    ahe_result = adaptive_histogram_equalization(image.copy(), clip_limit=40.0)

    # Step 4: CLAHE (Contrast Limited AHE with clip limit = 2.0)
    print("  - CLAHE (Contrast Limited AHE)...")
    clahe_enhancer = CLAHEEnhancer(clip_limit=2.0, tile_grid_size=(8, 8))
    clahe_result = clahe_enhancer.apply(image.copy())

    # Step 5: Adaptive Gabor Filter
    print("  - Adaptive Gabor Filter...")
    gabor_filter = AdaptiveGaborFilter(ksize=31, sigma=3.0, lambd=10.0, gamma=0.5, n_orientations=8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gabor_result = gabor_filter.apply(gray)
    gabor_result_bgr = cv2.cvtColor(gabor_result, cv2.COLOR_GRAY2BGR)

    # Step 6: Gabor + CLAHE (Full pipeline)
    print("  - Gabor + CLAHE (Full Pipeline)...")
    gabor_clahe = clahe_enhancer.apply(gabor_result_bgr)

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Preprocessing Steps Visualization', fontsize=20, fontweight='bold')

    images = [
        (original, 'Original Image'),
        (he_result, 'HE (Histogram Equalization)'),
        (ahe_result, 'AHE (Adaptive HE, clip=40)'),
        (clahe_result, 'CLAHE (clip=2.0)'),
        (gabor_result_bgr, 'Adaptive Gabor Filter'),
        (gabor_clahe, 'Gabor + CLAHE (Full Pipeline)')
    ]

    for idx, (img, title) in enumerate(images):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()

    # Save comprehensive visualization
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f'{image_name}_all_steps.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comprehensive visualization: {output_path}")

    # Save individual images
    print("\nSaving individual processed images...")
    for img, title in images:
        safe_title = title.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '')
        individual_path = os.path.join(output_dir, f'{image_name}_{safe_title}.png')
        cv2.imwrite(individual_path, img)
        print(f"  ✓ {safe_title}")

    # Create side-by-side comparison: Original vs CLAHE vs Gabor+CLAHE
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle('Key Preprocessing Comparison', fontsize=18, fontweight='bold')

    key_images = [
        (original, 'Original'),
        (clahe_result, 'CLAHE Only'),
        (gabor_clahe, 'Adaptive Gabor + CLAHE')
    ]

    for idx, (img, title) in enumerate(key_images):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes2[idx].imshow(img_rgb)
        axes2[idx].set_title(title, fontsize=14, fontweight='bold')
        axes2[idx].axis('off')

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, f'{image_name}_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved key comparison: {comparison_path}")

    # Create histogram comparison
    fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
    fig3.suptitle('Histogram Analysis', fontsize=18, fontweight='bold')

    for idx, (img, title) in enumerate(images):
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

    plt.tight_layout()
    histogram_path = os.path.join(output_dir, f'{image_name}_histograms.png')
    plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved histogram analysis: {histogram_path}")

    print(f"\n{'='*80}")
    print(f"✅ ALL VISUALIZATIONS COMPLETED!")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Files created:")
    print(f"  1. {image_name}_all_steps.png - All 6 preprocessing steps")
    print(f"  2. {image_name}_comparison.png - Key 3-way comparison")
    print(f"  3. {image_name}_histograms.png - Histogram analysis")
    print(f"  4. Individual images for each step")
    print(f"{'='*80}\n")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize preprocessing steps')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='outputs/preprocess',
                       help='Output directory (default: outputs/preprocess)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🔍 PREPROCESSING VISUALIZATION")
    print("="*80 + "\n")

    visualize_all_preprocessing_steps(args.image, args.output)


if __name__ == "__main__":
    main()

