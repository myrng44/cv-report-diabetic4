"""
Utility functions for data analysis and visualization
"""

import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import torch

import config


def analyze_classification_data():
    """Analyze classification dataset distribution"""
    print("="*80)
    print("Classification Dataset Analysis")
    print("="*80)

    # Load training labels
    df_train = pd.read_csv(config.CLASS_TRAIN_LABELS)

    print(f"\nTraining samples: {len(df_train)}")
    print("\nClass distribution:")
    print("-"*40)

    class_counts = df_train['Retinopathy grade'].value_counts().sort_index()
    dr_grades = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'PDR'}

    total = len(df_train)
    for grade, count in class_counts.items():
        percentage = count / total * 100
        print(f"Grade {grade} ({dr_grades[grade]:12s}): {count:4d} samples ({percentage:5.2f}%)")

    # Check for class imbalance
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count

    print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 5:
        print("⚠ Warning: High class imbalance detected!")
        print("  Recommendation: Use Focal Loss or class weights")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    grades = [dr_grades[i] for i in range(5)]
    counts = [class_counts.get(i, 0) for i in range(5)]

    axes[0].bar(grades, counts, color='steelblue', alpha=0.8)
    axes[0].set_xlabel('DR Grade', fontsize=12)
    axes[0].set_ylabel('Number of Samples', fontsize=12)
    axes[0].set_title('Classification Dataset Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # Pie chart
    axes[1].pie(counts, labels=grades, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2'))
    axes[1].set_title('Class Distribution (%)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(config.RESULT_DIR, 'classification_data_distribution.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n✓ Distribution plot saved to {save_path}")

    return class_counts


def analyze_image_properties():
    """Analyze image properties (size, quality, etc.)"""
    print("\n" + "="*80)
    print("Image Properties Analysis")
    print("="*80)

    # Sample 50 images
    image_files = [f for f in os.listdir(config.CLASS_TRAIN_IMG_DIR) if f.endswith('.jpg')][:50]

    widths = []
    heights = []
    aspect_ratios = []

    print(f"\nAnalyzing {len(image_files)} sample images...")

    for img_file in image_files:
        img_path = os.path.join(config.CLASS_TRAIN_IMG_DIR, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)

    print(f"\nImage dimensions:")
    print(f"  Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}")
    print(f"  Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")
    print(f"  Aspect ratio: mean={np.mean(aspect_ratios):.2f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(widths, bins=20, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Width (pixels)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Image Width Distribution')
    axes[0].grid(alpha=0.3)

    axes[1].hist(heights, bins=20, color='lightcoral', edgecolor='black')
    axes[1].set_xlabel('Height (pixels)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Image Height Distribution')
    axes[1].grid(alpha=0.3)

    axes[2].scatter(widths, heights, alpha=0.6, color='purple')
    axes[2].set_xlabel('Width (pixels)')
    axes[2].set_ylabel('Height (pixels)')
    axes[2].set_title('Width vs Height')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.RESULT_DIR, 'image_properties_analysis.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n✓ Image properties plot saved to {save_path}")


def visualize_sample_images():
    """Visualize sample images from each class"""
    print("\n" + "="*80)
    print("Sample Images Visualization")
    print("="*80)

    df = pd.read_csv(config.CLASS_TRAIN_LABELS)
    dr_grades = {0: 'No DR', 1: 'Mild NPDR', 2: 'Moderate NPDR', 3: 'Severe NPDR', 4: 'PDR'}

    fig, axes = plt.subplots(5, 3, figsize=(12, 18))

    for grade in range(5):
        # Get images for this grade
        grade_images = df[df['Retinopathy grade'] == grade]['Image name'].values[:3]

        for i, img_name in enumerate(grade_images):
            img_path = os.path.join(config.CLASS_TRAIN_IMG_DIR, f"{img_name}.jpg")
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (256, 256))

                axes[grade, i].imshow(img_resized)
                axes[grade, i].axis('off')

                if i == 0:
                    axes[grade, i].set_ylabel(f"Grade {grade}\n{dr_grades[grade]}",
                                             fontsize=11, fontweight='bold')

    plt.suptitle('Sample Images by DR Grade', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = os.path.join(config.RESULT_DIR, 'sample_images_by_grade.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Sample images saved to {save_path}")


def estimate_training_time():
    """Estimate training time based on hardware"""
    print("\n" + "="*80)
    print("Training Time Estimation")
    print("="*80)

    # Load dataset info
    df_train = pd.read_csv(config.CLASS_TRAIN_LABELS)
    n_samples = len(df_train)

    batch_size = config.BATCH_SIZE
    epochs = config.NUM_EPOCHS

    batches_per_epoch = n_samples // batch_size

    print(f"\nDataset size: {n_samples} samples")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {batches_per_epoch}")
    print(f"Number of epochs: {epochs}")
    print(f"Total iterations: {batches_per_epoch * epochs}")

    # Rough estimates (seconds per batch)
    if torch.cuda.is_available():
        seconds_per_batch = 0.5  # With GPU
        device_name = "GPU"
    else:
        seconds_per_batch = 5.0  # CPU only
        device_name = "CPU"

    total_seconds = batches_per_epoch * epochs * seconds_per_batch
    hours = total_seconds / 3600

    print(f"\nEstimated training time ({device_name}):")
    print(f"  Classification: ~{hours:.1f} hours")
    print(f"  Segmentation: ~{hours * 1.5:.1f} hours (more complex)")
    print(f"  Total (both models): ~{hours * 2.5:.1f} hours")

    if not torch.cuda.is_available():
        print("\n⚠ Warning: No GPU detected! Training will be very slow on CPU.")
        print("  Recommendation: Use Google Colab or a machine with GPU")


def main():
    """Run all analysis"""
    try:
        analyze_classification_data()
        analyze_image_properties()
        visualize_sample_images()
        estimate_training_time()

        print("\n" + "="*80)
        print("✓ Data analysis completed!")
        print("="*80)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
