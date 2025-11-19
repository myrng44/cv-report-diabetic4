"""
Visualize Feature Extraction Steps (LBP and SURF/ORB)
Shows visual representation of LBP patterns and SURF keypoints
Usage: python visualize_features.py --image <path_to_image>
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from skimage.feature import local_binary_pattern
from matplotlib.patches import Circle


def visualize_lbp_features(image_path, output_dir='outputs/preprocess'):
    """
    Visualize Local Binary Pattern (LBP) features:
    1. Original Image
    2. Grayscale Image
    3. LBP Pattern Image
    4. LBP Histogram
    5. LBP with different parameters
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("Applying LBP with different parameters...")

    # Different LBP configurations
    lbp_configs = [
        (8, 1, 'LBP (P=8, R=1)'),
        (8, 2, 'LBP (P=8, R=2)'),
        (16, 2, 'LBP (P=16, R=2)'),
        (24, 3, 'LBP (P=24, R=3)')
    ]

    lbp_results = []
    lbp_histograms = []

    for n_points, radius, title in lbp_configs:
        print(f"  - Computing {title}...")
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_normalized = ((lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255).astype(np.uint8)
        lbp_results.append((lbp_normalized, title))

        # Compute histogram
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        lbp_histograms.append((hist, title))

    # Create comprehensive LBP visualization
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
    fig.suptitle('Local Binary Pattern (LBP) Feature Visualization', fontsize=22, fontweight='bold', y=0.98)

    # Row 1: Original and Gray
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=13, fontweight='bold', pad=10)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(gray, cmap='gray')
    ax2.set_title('Grayscale Image', fontsize=13, fontweight='bold', pad=10)
    ax2.axis('off')

    # Add LBP explanation
    ax3 = fig.add_subplot(gs[0, 2:])
    ax3.axis('off')
    explanation = (
        "Local Binary Pattern (LBP) Analysis:\n\n"
        "• LBP encodes local texture by comparing each pixel\n"
        "   with its neighbors\n"
        "• Parameters: P (number of points), R (radius)\n"
        "• Uniform patterns capture most texture information\n"
        "• Histogram of LBP values forms the feature descriptor\n\n"
        "Different configurations capture texture at various scales:\n"
        "  - Small R: Fine details and micro-textures\n"
        "  - Large R: Coarser patterns and macro-structures"
    )
    ax3.text(0.05, 0.5, explanation, fontsize=10.5, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
             family='monospace')

    # Row 2: LBP pattern images
    for idx, (lbp_img, title) in enumerate(lbp_results):
        ax = fig.add_subplot(gs[1, idx])
        im = ax.imshow(lbp_img, cmap='viridis')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 3: LBP histograms
    for idx, (hist, title) in enumerate(lbp_histograms):
        ax = fig.add_subplot(gs[2, idx])
        ax.bar(range(len(hist)), hist, color='steelblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
        ax.set_title(f'{title}\nHistogram', fontsize=10, fontweight='bold', pad=8)
        ax.set_xlabel('LBP Pattern', fontsize=9, labelpad=5)
        ax.set_ylabel('Frequency', fontsize=9, labelpad=5)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=8)

    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.98)

    # Save LBP visualization
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f'{image_name}_LBP_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    print(f"\n✓ Saved LBP visualization: {output_path}")

    return lbp_results, lbp_histograms


def visualize_surf_features(image_path, output_dir='outputs/preprocess'):
    """
    Visualize SURF/ORB keypoint features:
    1. Original Image
    2. Keypoints overlay
    3. Keypoint density map
    4. Top strongest keypoints
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read image
    print(f"\nReading image for SURF/ORB: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Resize for consistency
    image = cv2.resize(image, (512, 512))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("Detecting ORB keypoints (SURF alternative)...")

    # Create ORB detector with different configurations
    orb_configs = [
        (100, 'ORB (100 features)'),
        (200, 'ORB (200 features)'),
        (500, 'ORB (500 features)'),
        (1000, 'ORB (1000 features)')
    ]

    orb_results = []

    for n_features, title in orb_configs:
        print(f"  - Computing {title}...")
        orb = cv2.ORB_create(nfeatures=n_features)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        # Draw keypoints
        img_with_kp = cv2.drawKeypoints(image.copy(), keypoints, None,
                                        color=(0, 255, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Create density map
        density_map = np.zeros(gray.shape, dtype=np.float32)
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size)
            cv2.circle(density_map, (x, y), size, kp.response, -1)

        orb_results.append((img_with_kp, density_map, keypoints, title, n_features))

    # Create comprehensive SURF/ORB visualization
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(5, 4, hspace=0.5, wspace=0.4,
                         top=0.94, bottom=0.04, left=0.04, right=0.96)
    fig.suptitle('SURF/ORB Keypoint Feature Visualization',
                 fontsize=24, fontweight='bold', y=0.97)

    # Row 1: Original and explanation (larger spacing)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=14, fontweight='bold', pad=15)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.axis('off')
    explanation = (
        "SURF/ORB Keypoint Detection:\n\n"
        "• SURF (Speeded-Up Robust Features) detects interest points\n"
        "• ORB (Oriented FAST and Rotated BRIEF) - free alternative\n"
        "• Each keypoint: location, scale, orientation, response strength\n"
        "• Descriptors encode local image patterns around keypoints\n"
        "• Used for matching, recognition, feature-based classification\n\n"
        "Visualization:\n"
        "  - Green circles: Keypoint locations and scales\n"
        "  - Density maps: Spatial distribution of features"
    )
    ax2.text(0.05, 0.5, explanation, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
             family='monospace', linespacing=1.5)

    # Rows 2-5: Different ORB configurations (one per row for better spacing)
    for idx, (img_kp, density, keypoints, title, n_feat) in enumerate(orb_results):
        row = 1 + idx  # Start from row 1

        # Keypoints image
        ax_kp = fig.add_subplot(gs[row, 0:2])
        ax_kp.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
        ax_kp.set_title(f'{title} - {len(keypoints)} keypoints detected',
                       fontsize=13, fontweight='bold', pad=12)
        ax_kp.axis('off')

        # Density map
        ax_dens = fig.add_subplot(gs[row, 2:4])
        im = ax_dens.imshow(density, cmap='hot', interpolation='gaussian')
        ax_dens.set_title(f'{title} - Density Map',
                         fontsize=13, fontweight='bold', pad=12)
        ax_dens.axis('off')
        cbar = plt.colorbar(im, ax=ax_dens, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

    plt.subplots_adjust(top=0.96, bottom=0.04, left=0.04, right=0.98)

    # Save SURF/ORB visualization
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f'{image_name}_SURF_ORB_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.5)
    print(f"\n✓ Saved SURF/ORB visualization: {output_path}")
    plt.close()

    return orb_results


def visualize_combined_features(image_path, output_dir='outputs/preprocess'):
    """
    Create a combined visualization showing both LBP and SURF/ORB
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (512, 512))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("\nCreating combined feature visualization...")

    # LBP
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    lbp_normalized = ((lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255).astype(np.uint8)

    # ORB
    orb = cv2.ORB_create(nfeatures=200)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    img_with_kp = cv2.drawKeypoints(image.copy(), keypoints, None,
                                    color=(0, 255, 0),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Create combined visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle('Combined Feature Extraction: LBP + SURF/ORB',
                 fontsize=20, fontweight='bold', y=0.98)

    # Original
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=15, fontweight='bold', pad=12)
    axes[0, 0].axis('off')

    # LBP
    im1 = axes[0, 1].imshow(lbp_normalized, cmap='viridis')
    axes[0, 1].set_title('LBP Texture Pattern\n(P=8, R=1)',
                        fontsize=15, fontweight='bold', pad=12)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # ORB Keypoints
    axes[1, 0].imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'ORB Keypoints\n({len(keypoints)} features detected)',
                         fontsize=15, fontweight='bold', pad=12)
    axes[1, 0].axis('off')

    # Combined overlay
    # Create colored LBP overlay
    lbp_colored = cv2.applyColorMap(lbp_normalized, cv2.COLORMAP_VIRIDIS)
    combined = cv2.addWeighted(image, 0.6, lbp_colored, 0.4, 0)
    combined_with_kp = cv2.drawKeypoints(combined, keypoints, None,
                                         color=(255, 255, 0),
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    axes[1, 1].imshow(cv2.cvtColor(combined_with_kp, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Combined: LBP Pattern + ORB Keypoints',
                         fontsize=15, fontweight='bold', pad=12)
    axes[1, 1].axis('off')

    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95,
                       hspace=0.25, wspace=0.25)

    # Save combined visualization
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f'{image_name}_Combined_Features.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    print(f"✓ Saved combined visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize LBP and SURF/ORB feature extraction')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='outputs/preprocess',
                       help='Output directory (default: outputs/preprocess)')
    parser.add_argument('--features', type=str, default='all',
                       choices=['lbp', 'surf', 'all'],
                       help='Which features to visualize (default: all)')

    args = parser.parse_args()

    print("\n" + "="*100)
    print("🔍 FEATURE EXTRACTION VISUALIZATION (LBP + SURF/ORB)")
    print("="*100 + "\n")

    if args.features in ['lbp', 'all']:
        print("=" * 100)
        print("📊 LOCAL BINARY PATTERN (LBP) FEATURES")
        print("=" * 100)
        visualize_lbp_features(args.image, args.output)

    if args.features in ['surf', 'all']:
        print("\n" + "=" * 100)
        print("🎯 SURF/ORB KEYPOINT FEATURES")
        print("=" * 100)
        visualize_surf_features(args.image, args.output)

    if args.features == 'all':
        print("\n" + "=" * 100)
        print("🔗 COMBINED FEATURE VISUALIZATION")
        print("=" * 100)
        visualize_combined_features(args.image, args.output)

    print("\n" + "="*100)
    print("✅ FEATURE VISUALIZATION COMPLETED!")
    print("="*100)
    print(f"\nOutput directory: {args.output}")
    print(f"Check the generated images for detailed feature analysis.")
    print("="*100 + "\n")

    plt.show()


if __name__ == "__main__":
    main()
