"""
Script to visualize sample images from each DR class (0-5)
Creates a grid image showing examples from each class
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import config

def visualize_class_samples(num_samples_per_class=3, output_path=None):
    """
    Create visualization of sample images from each DR class

    Args:
        num_samples_per_class: Number of samples to show per class
        output_path: Path to save the output image
    """

    # Read training labels
    labels_csv = config.CLASS_TRAIN_LABELS
    img_dir = config.CLASS_TRAIN_IMG_DIR

    print(f"Reading labels from: {labels_csv}")
    df = pd.read_csv(labels_csv)

    # Get class distribution
    print("\nClass distribution:")
    class_counts = df['Retinopathy grade'].value_counts().sort_index()
    for grade, count in class_counts.items():
        print(f"  Grade {grade}: {count} images")

    # Prepare figure
    num_classes = 5  # Classes 0-4
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(num_classes, num_samples_per_class + 1,
                  figure=fig, wspace=0.3, hspace=0.4)

    # Grade descriptions
    grade_descriptions = {
        0: "Grade 0: No DR\n(Normal)",
        1: "Grade 1: Mild NPDR\n(Microaneurysms only)",
        2: "Grade 2: Moderate NPDR\n(More than MA)",
        3: "Grade 3: Severe NPDR\n(Multiple hemorrhages)",
        4: "Grade 4: PDR\n(Proliferative DR)"
    }

    # For each class, get sample images
    for class_idx in range(num_classes):
        # Get images for this class
        class_df = df[df['Retinopathy grade'] == class_idx]

        if len(class_df) == 0:
            print(f"\nWarning: No images found for class {class_idx}")
            continue

        # Sample random images
        sample_size = min(num_samples_per_class, len(class_df))
        sampled = class_df.sample(n=sample_size)  # Ngẫu nhiên mỗi lần chạy

        # Add class label in first column
        ax_label = fig.add_subplot(gs[class_idx, 0])
        ax_label.text(0.5, 0.5, grade_descriptions.get(class_idx, f"Grade {class_idx}"),
                     ha='center', va='center', fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax_label.axis('off')

        # Display sample images
        for img_idx, (_, row) in enumerate(sampled.iterrows()):
            img_name = row['Image name']
            img_path = os.path.join(img_dir, f"{img_name}.jpg")

            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue

            # Read and display image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Add to subplot
            ax = fig.add_subplot(gs[class_idx, img_idx + 1])
            ax.imshow(img)
            ax.set_title(f"{img_name}", fontsize=9)
            ax.axis('off')

    # Overall title
    fig.suptitle('Diabetic Retinopathy - Sample Images by Grade (IDRiD Dataset)',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    if output_path is None:
        output_path = os.path.join(config.OUTPUT_DIR, 'class_samples_visualization.png')

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Visualization saved to: {output_path}")

    # Also create a second version with preprocessing examples
    create_preprocessed_samples(df, img_dir, num_samples_per_class=2)

    plt.close()

def create_preprocessed_samples(df, img_dir, num_samples_per_class=2):
    """Create visualization showing original vs preprocessed images"""

    from preprocessing import preprocess_fundus_image

    num_classes = 5
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(num_classes, num_samples_per_class * 2 + 1,
                  figure=fig, wspace=0.2, hspace=0.4)

    grade_descriptions = {
        0: "Grade 0\nNo DR",
        1: "Grade 1\nMild",
        2: "Grade 2\nModerate",
        3: "Grade 3\nSevere",
        4: "Grade 4\nPDR"
    }

    for class_idx in range(num_classes):
        class_df = df[df['Retinopathy grade'] == class_idx]

        if len(class_df) == 0:
            continue

        sample_size = min(num_samples_per_class, len(class_df))
        sampled = class_df.sample(n=sample_size)  # Bỏ random_state để ngẫu nhiên mỗi lần

        # Class label
        ax_label = fig.add_subplot(gs[class_idx, 0])
        ax_label.text(0.5, 0.5, grade_descriptions.get(class_idx, f"Grade {class_idx}"),
                     ha='center', va='center', fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax_label.axis('off')

        # Show original and preprocessed for each sample
        for img_idx, (_, row) in enumerate(sampled.iterrows()):
            img_name = row['Image name']
            img_path = os.path.join(img_dir, f"{img_name}.jpg")

            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Original image
            ax_orig = fig.add_subplot(gs[class_idx, img_idx * 2 + 1])
            ax_orig.imshow(img_rgb)
            ax_orig.set_title(f"{img_name}\n(Original)", fontsize=8)
            ax_orig.axis('off')

            # Preprocessed image
            try:
                # Pass image path to preprocess_fundus_image
                preprocessed = preprocess_fundus_image(img_path, target_size=256, apply_gabor=False)
                ax_prep = fig.add_subplot(gs[class_idx, img_idx * 2 + 2])
                ax_prep.imshow(preprocessed)
                ax_prep.set_title(f"{img_name}\n(Preprocessed)", fontsize=8)
                ax_prep.axis('off')
            except Exception as e:
                print(f"Error preprocessing {img_name}: {e}")

    fig.suptitle('DR Classification - Original vs Preprocessed Images by Grade',
                 fontsize=16, fontweight='bold', y=0.98)

    output_path = os.path.join(config.OUTPUT_DIR, 'class_samples_with_preprocessing.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Preprocessing comparison saved to: {output_path}")
    plt.close()

def analyze_dataset_statistics():
    """Print detailed dataset statistics"""

    print("\n" + "="*70)
    print("DATASET STATISTICS - IDRiD Diabetic Retinopathy")
    print("="*70)

    # Training set
    train_df = pd.read_csv(config.CLASS_TRAIN_LABELS)
    print("\n📊 TRAINING SET:")
    print(f"Total images: {len(train_df)}")
    print("\nClass distribution:")
    for grade in sorted(train_df['Retinopathy grade'].unique()):
        count = len(train_df[train_df['Retinopathy grade'] == grade])
        percentage = (count / len(train_df)) * 100
        print(f"  Grade {grade}: {count:3d} images ({percentage:5.2f}%)")

    # Test set
    test_df = pd.read_csv(config.CLASS_TEST_LABELS)
    print("\n📊 TEST SET:")
    print(f"Total images: {len(test_df)}")
    print("\nClass distribution:")
    for grade in sorted(test_df['Retinopathy grade'].unique()):
        count = len(test_df[test_df['Retinopathy grade'] == grade])
        percentage = (count / len(test_df)) * 100
        print(f"  Grade {grade}: {count:3d} images ({percentage:5.2f}%)")

    print("\n" + "="*70)

if __name__ == "__main__":
    print("🔍 Creating visualization of DR class samples...")
    print("="*70)

    # Show dataset statistics first
    analyze_dataset_statistics()

    # Create visualizations
    print("\n📸 Generating sample images visualization...")
    visualize_class_samples(num_samples_per_class=3)

    print("\n✅ Done! Check the outputs folder for generated images.")
