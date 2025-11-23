"""
Script để trực quan hóa các mẫu ảnh từ mỗi lớp DR (0-5)
Tảo ảnh lưới hiển thị các ví dụ từ mỗi lớp
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
    Tạo trực quan hóa các mẫu ảnh từ mỗi lớp DR

    Args:
        num_samples_per_class: Số mẫu để hiển thị cho mỗi lớp
        output_path: Đường dẫn để lưu ảnh đầu ra
    """

    # Đọc nhãn huấn luyện
    labels_csv = config.CLASS_TRAIN_LABELS
    img_dir = config.CLASS_TRAIN_IMG_DIR

    print(f"Đang đọc nhãn từ: {labels_csv}")
    df = pd.read_csv(labels_csv)

    # Lấy phân phối lớp
    print("\nPhân phối lớp:")
    class_counts = df['Retinopathy grade'].value_counts().sort_index()
    for grade, count in class_counts.items():
        print(f"  Grade {grade}: {count} images")

    # Chuẩn bị hình
    num_classes = 5  # Các lớp 0-4
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(num_classes, num_samples_per_class + 1,
                  figure=fig, wspace=0.3, hspace=0.4)

    # Mô tả mức độ
    grade_descriptions = {
        0: "Grade 0: No DR\n(Normal)",
        1: "Grade 1: Mild NPDR\n(Microaneurysms only)",
        2: "Grade 2: Moderate NPDR\n(More than MA)",
        3: "Grade 3: Severe NPDR\n(Multiple hemorrhages)",
        4: "Grade 4: PDR\n(Proliferative DR)"
    }

    # Với mỗi lớp, lấy các ảnh mẫu
    for class_idx in range(num_classes):
        # Lấy ảnh cho lớp này
        class_df = df[df['Retinopathy grade'] == class_idx]

        if len(class_df) == 0:
            print(f"\nWarning: No images found for class {class_idx}")
            continue

        # Lấy mẫu ảnh ngẫu nhiên
        sample_size = min(num_samples_per_class, len(class_df))
        sampled = class_df.sample(n=sample_size)  # Ngẫu nhiên mỗi lần chạy

        # Thêm nhãn lớp ở cột đầu tiên
        ax_label = fig.add_subplot(gs[class_idx, 0])
        ax_label.text(0.5, 0.5, grade_descriptions.get(class_idx, f"Grade {class_idx}"),
                     ha='center', va='center', fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax_label.axis('off')

        # Hiển thị các ảnh mẫu
        for img_idx, (_, row) in enumerate(sampled.iterrows()):
            img_name = row['Image name']
            img_path = os.path.join(img_dir, f"{img_name}.jpg")

            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue

            # Đọc và hiển thị ảnh
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Thêm vào subplot
            ax = fig.add_subplot(gs[class_idx, img_idx + 1])
            ax.imshow(img)
            ax.set_title(f"{img_name}", fontsize=9)
            ax.axis('off')

    # Tiêu đề tổng thể
    fig.suptitle('Bệnh Võng mạc Đái tháo đường - Mẫu Ảnh theo Mức độ (Tập dữ liệu IDRiD)',
                 fontsize=16, fontweight='bold', y=0.98)

    # Lưu hình
    if output_path is None:
        output_path = os.path.join(config.OUTPUT_DIR, 'class_samples_visualization.png')

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Visualization saved to: {output_path}")

    # Also create a second version with preprocessing examples
    create_preprocessed_samples(df, img_dir, num_samples_per_class=2)

    plt.close()

def create_preprocessed_samples(df, img_dir, num_samples_per_class=2):
    """Tạo trực quan hóa hiển thị ảnh gốc so với ảnh đã tiền xử lý"""

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
    """In thống kê chi tiết của tập dữ liệu"""

    print("\n" + "="*70)
    print("THỐNG KÊ TẬP DỮ LIỆU - IDRiD Bệnh Võng mạc Đái tháo đường")
    print("="*70)

    # Tập huấn luyện
    train_df = pd.read_csv(config.CLASS_TRAIN_LABELS)
    print("\n📊 TẬP HUẤN LUYỆN:")
    print(f"Tổng số ảnh: {len(train_df)}")
    print("\nPhân phối lớp:")
    for grade in sorted(train_df['Retinopathy grade'].unique()):
        count = len(train_df[train_df['Retinopathy grade'] == grade])
        percentage = (count / len(train_df)) * 100
        print(f"  Grade {grade}: {count:3d} images ({percentage:5.2f}%)")

    # Tập kiểm tra
    test_df = pd.read_csv(config.CLASS_TEST_LABELS)
    print("\n📊 TẬP KIỂM TRA:")
    print(f"Tổng số ảnh: {len(test_df)}")
    print("\nPhân phối lớp:")
    for grade in sorted(test_df['Retinopathy grade'].unique()):
        count = len(test_df[test_df['Retinopathy grade'] == grade])
        percentage = (count / len(test_df)) * 100
        print(f"  Grade {grade}: {count:3d} images ({percentage:5.2f}%)")

    print("\n" + "="*70)

if __name__ == "__main__":
    print("🔍 Đang tạo trực quan hóa các mẫu lớp DR...")
    print("="*70)

    # Hiển thị thống kê tập dữ liệu trước
    analyze_dataset_statistics()

    # Tạo các trực quan hóa
    print("\n📸 Đang tạo trực quan hóa các ảnh mẫu...")
    visualize_class_samples(num_samples_per_class=3)

    print("\n✅ Hoàn thành! Kiểm tra thư mục outputs cho các ảnh đã tạo.")
