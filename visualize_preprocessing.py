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


def calculate_contrast(image):
    """
    Tính Michelson Contrast
    Formula: C = (I_max - I_min) / (I_max + I_min)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    I_max = float(gray.max())
    I_min = float(gray.min())
    
    if I_max + I_min == 0:
        return 0.0
    
    contrast = (I_max - I_min) / (I_max + I_min)
    return contrast


def calculate_rms_contrast(image):
    """
    Tính RMS (Root Mean Square) Contrast
    Formula: C_RMS = sqrt(mean((I - mean(I))^2))
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    gray = gray.astype(np.float64)
    mean_intensity = gray.mean()
    
    rms_contrast = np.sqrt(np.mean((gray - mean_intensity) ** 2))
    return rms_contrast


def calculate_snr_db(image):
    """
    Tính SNR (Signal-to-Noise Ratio) trong dB
    
    Approach:
    - Signal: Mean intensity của vùng có signal (> threshold)
    - Noise: Std của vùng background (< threshold)
    
    Formula: SNR(dB) = 20 × log₁₀(μ_signal / σ_noise)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    gray = gray.astype(np.float64)
    
    # Otsu thresholding để tách signal và background
    threshold = cv2.threshold(gray.astype(np.uint8), 0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    
    # Signal region (foreground)
    signal_mask = gray > threshold
    signal_pixels = gray[signal_mask]
    
    # Noise region (background)
    noise_mask = gray <= threshold
    noise_pixels = gray[noise_mask]
    
    if len(signal_pixels) == 0 or len(noise_pixels) == 0:
        return 0.0
    
    # Calculate signal mean and noise std
    mu_signal = signal_pixels.mean()
    sigma_noise = noise_pixels.std()
    
    if sigma_noise == 0:
        return 100.0  # Very high SNR
    
    # SNR in dB
    snr = mu_signal / sigma_noise
    snr_db = 20 * np.log10(snr) if snr > 0 else 0.0
    
    return snr_db


def calculate_edge_strength(image):
    """
    Tính Edge Strength (Gradient Magnitude) using Sobel
    
    Formula: sqrt(Gx^2 + Gy^2)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Sobel gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Mean edge strength
    edge_strength = gradient_magnitude.mean()
    
    return edge_strength


def evaluate_preprocessing_metrics(image):
    """
    Đánh giá tất cả metrics cho 1 ảnh
    
    Returns:
        dict: {
            'contrast': float,
            'rms_contrast': float,
            'snr_db': float,
            'edge_strength': float
        }
    """
    metrics = {
        'contrast': calculate_contrast(image),
        'rms_contrast': calculate_rms_contrast(image),
        'snr_db': calculate_snr_db(image),
        'edge_strength': calculate_edge_strength(image)
    }
    
    return metrics


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

    # ===== METRICS EVALUATION =====
    print(f"\n{'='*80}")
    print(f"BẢNG 5.1: SO SÁNH ĐỊNH LƯỢNG TIỀN XỬ LÝ")
    print(f"{'='*80}")
    
    # Đánh giá metrics cho từng method
    methods = [
        ('Original', original),
        ('HE', he_result),
        ('AHE (clip=40)', ahe_result),
        ('CLAHE (clip=2.0)', clahe_2_result),
        ('CLAHE (clip=4.0)', clahe_4_result)
    ]
    
    # Create table header
    print(f"{'Method':<25} {'Contrast':<12} {'SNR (dB)':<12} {'Edge Strength':<15}")
    print("-" * 80)
    
    metrics_dict = {}
    
    for method_name, img in methods:
        metrics = evaluate_preprocessing_metrics(img)
        metrics_dict[method_name] = metrics
        
        print(f"{method_name:<25} "
              f"{metrics['contrast']:<12.4f} "
              f"{metrics['snr_db']:<12.2f} "
              f"{metrics['edge_strength']:<15.2f}")
    
    print("="*80 + "\n")
    
    # ===== VISUALIZE METRICS COMPARISON =====
    fig_metrics, axes_metrics = plt.subplots(1, 3, figsize=(18, 6))
    fig_metrics.suptitle('Preprocessing Metrics Comparison', 
                         fontsize=16, fontweight='bold')
    
    method_names = [m[0] for m in methods]
    contrasts = [metrics_dict[m]['contrast'] for m in method_names]
    snrs = [metrics_dict[m]['snr_db'] for m in method_names]
    edges = [metrics_dict[m]['edge_strength'] for m in method_names]
    
    colors = ['gray', 'blue', 'orange', 'green', 'red']
    
    # Contrast bar plot
    bars1 = axes_metrics[0].bar(range(len(method_names)), contrasts, color=colors)
    axes_metrics[0].set_xticks(range(len(method_names)))
    axes_metrics[0].set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
    axes_metrics[0].set_ylabel('Contrast', fontsize=11)
    axes_metrics[0].set_title('Contrast Comparison', fontsize=12, fontweight='bold')
    axes_metrics[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, contrasts):
        height = bar.get_height()
        axes_metrics[0].text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # SNR bar plot
    bars2 = axes_metrics[1].bar(range(len(method_names)), snrs, color=colors)
    axes_metrics[1].set_xticks(range(len(method_names)))
    axes_metrics[1].set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
    axes_metrics[1].set_ylabel('SNR (dB)', fontsize=11)
    axes_metrics[1].set_title('SNR Comparison', fontsize=12, fontweight='bold')
    axes_metrics[1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, snrs):
        height = bar.get_height()
        axes_metrics[1].text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Edge Strength bar plot
    bars3 = axes_metrics[2].bar(range(len(method_names)), edges, color=colors)
    axes_metrics[2].set_xticks(range(len(method_names)))
    axes_metrics[2].set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
    axes_metrics[2].set_ylabel('Edge Strength', fontsize=11)
    axes_metrics[2].set_title('Edge Strength Comparison', fontsize=12, fontweight='bold')
    axes_metrics[2].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars3, edges):
        height = bar.get_height()
        axes_metrics[2].text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save metrics plot
    metrics_plot_path = os.path.join(output_dir, f'{image_name}_metrics_comparison.png')
    plt.savefig(metrics_plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved metrics comparison: {metrics_plot_path}")
    
    # ===== SUMMARY =====
    print(f"\n{'='*80}")
    print(f"CLAHE VISUALIZATION COMPLETED!")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Files created:")
    print(f"  1. {image_name}_clahe_comparison.png - All CLAHE variations")
    print(f"  2. {image_name}_clahe_levels.png - Key comparison")
    print(f"  3. {image_name}_clahe_histograms.png - Histogram analysis")
    print(f"  4. {image_name}_metrics_comparison.png - Metrics comparison ⭐NEW")
    print(f"{'='*80}\n")
    
    # Highlight best method
    best_method = max(metrics_dict.items(), 
                     key=lambda x: x[1]['snr_db'] + x[1]['contrast'] * 10)
    print(f"🏆 BEST METHOD: {best_method[0]}")
    print(f"   - Contrast: {best_method[1]['contrast']:.4f}")
    print(f"   - SNR: {best_method[1]['snr_db']:.2f} dB")
    print(f"   - Edge Strength: {best_method[1]['edge_strength']:.2f}")
    print(f"{'='*80}\n")

    plt.show()
    
    # Return metrics for further analysis
    return metrics_dict


def main():
    parser = argparse.ArgumentParser(description='Visualize CLAHE preprocessing')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='outputs/preprocess',
                       help='Output directory (default: outputs/preprocess)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("CLAHE PREPROCESSING VISUALIZATION")
    print("="*80 + "\n")

    visualize_clahe_preprocessing(args.image, args.output)


if __name__ == "__main__":
    main()

