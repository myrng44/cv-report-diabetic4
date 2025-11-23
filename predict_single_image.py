"""
SINGLE IMAGE PREDICTION - Dự đoán DR cho 1 ảnh retina

Usage:
    python predict_single_image.py --image path/to/image.jpg
    python predict_single_image.py --image "data/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_01.jpg"
"""

import os
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from preprocessing import preprocess_fundus_image
from classification_model import create_classification_model
from advanced_segmentation_model import create_advanced_segmentation_model


def preprocess_for_segmentation(image_path, target_size=512):
    """
    ⚠️ QUAN TRỌNG: Phải khớp với tiền xử lý huấn luyện!
    Huấn luyện sử dụng chuẩn hóa ImageNet, nên suy luận CŨNG PHẢI dùng nó!
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Áp dụng cùng phép biến đổi như huấn luyện (KHÔNG tăng cường)
    transform = A.Compose([
        A.Resize(target_size, target_size),
        # ✅ QUAN TRỌNG: Chuẩn hóa giống như huấn luyện!
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    transformed = transform(image=image)
    image_tensor = transformed['image']

    return image_tensor


class DRPredictor:
    """Dự đoán DR từ 1 ảnh retina"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Sử dụng thiết bị: {self.device}")

        # Tải Mô hình Phân loại
        class_model_path = os.path.join(config.MODEL_DIR, 'best_classification_model.pth')
        if os.path.exists(class_model_path):
            print(f"Đang tải mô hình phân loại...")
            self.class_model = create_classification_model(num_classes=config.NUM_CLASSES)
            checkpoint = torch.load(class_model_path, map_location=self.device)
            self.class_model.load_state_dict(checkpoint['model_state_dict'])
            self.class_model.to(self.device)
            self.class_model.eval()
            print(f"✓ Classification model loaded")
        else:
            print(f"⚠ Classification model not found at {class_model_path}")
            self.class_model = None

        # Tải Mô hình Phân đoạn
        seg_model_path = os.path.join(config.MODEL_DIR, 'best_seg_model.pth')
        if os.path.exists(seg_model_path):
            print(f"Loading segmentation model...")
            self.seg_model = create_advanced_segmentation_model(in_channels=3, out_channels=config.SEG_CLASSES)
            checkpoint = torch.load(seg_model_path, map_location=self.device)
            self.seg_model.load_state_dict(checkpoint['model_state_dict'])
            self.seg_model.to(self.device)
            self.seg_model.eval()
            print(f"✓ Segmentation model loaded")
            print(f"  - Best IoU: {checkpoint.get('best_iou', 0):.4f}")
            print(f"  - Best Dice: {checkpoint.get('best_dice', 0):.4f}")
        else:
            print(f"⚠ Segmentation model not found at {seg_model_path}")
            self.seg_model = None

        self.dr_grades = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'PDR']
        self.lesion_types = ['Microaneurysms', 'Haemorrhages', 'Hard Exudates']

    def predict(self, image_path, save_path=None):
        """Dự đoán cho 1 ảnh"""

        print(f"\n{'='*80}")
        print(f"🔍 Analyzing: {os.path.basename(image_path)}")
        print(f"{'='*80}")

        # Tải ảnh gốc
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Cannot load image: {image_path}")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Phân loại
        pred_class = None
        confidence = None
        all_probs = None

        if self.class_model is not None:
            print("\n📊 Classification:")
            try:
                processed = preprocess_fundus_image(image_path, target_size=config.IMG_SIZE, apply_gabor=False)
                image_tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.class_model(image_tensor)
                    probs = torch.softmax(output, dim=1)
                    pred_class = torch.argmax(probs, dim=1).item()
                    confidence = probs[0, pred_class].item()
                    all_probs = probs[0].cpu().numpy()

                print(f"   Prediction: {self.dr_grades[pred_class]}")
                print(f"   Confidence: {confidence*100:.2f}%")
                print(f"\n   All probabilities:")
                for i, (grade, prob) in enumerate(zip(self.dr_grades, all_probs)):
                    print(f"      {grade:20s}: {prob*100:5.2f}%")
            except Exception as e:
                print(f"   ❌ Lỗi trong quá trình phân loại: {e}")

        # Phân đoạn
        masks = None
        if self.seg_model is not None:
            print(f"\n🎯 Phân đoạn:")
            try:
                # ✅ ĐÃ SỬA: Dùng tiền xử lý đúng với chuẩn hóa ImageNet!
                image_tensor = preprocess_for_segmentation(image_path, target_size=512)
                image_tensor = image_tensor.unsqueeze(0).to(self.device)

                print(f"   Input tensor range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")

                with torch.no_grad():
                    output = self.seg_model(image_tensor)
                    masks = torch.sigmoid(output)

                    # Thông tin debug
                    print(f"   Phạm vi logits đầu ra: [{output.min():.3f}, {output.max():.3f}]"))
                    print(f"   Phạm vi xác suất Sigmoid: [{masks.min():.6f}, {masks.max():.6f}]")

                    # ✅ Ngưỡng thích nghi dựa trên thống kê đầu ra
                    threshold = 0.15  # Bắt đầu với ngưỡng thận trọng
                    masks_binary = (masks > threshold).float()
                    masks = masks_binary[0].cpu().numpy()  # (3, H, W)

                for i, lesion in enumerate(self.lesion_types):
                    lesion_pixels = masks[i].sum()
                    print(f"   {lesion:20s}: {lesion_pixels:.0f} pixels")
            except Exception as e:
                print(f"   ❌ Error during segmentation: {e}")
                import traceback
                traceback.print_exc()
                masks = None

        # Trực quan hóa
        self.visualize_results(image_rgb, pred_class, confidence, all_probs, masks, save_path)

        return pred_class, confidence, masks

    def visualize_results(self, image_rgb, pred_class, confidence, all_probs, masks, save_path):
        """Tạo trực quan hóa đẹp"""

        fig = plt.figure(figsize=(20, 10))

        # 1. Original Image
        ax1 = plt.subplot(2, 4, 1)
        ax1.imshow(image_rgb)
        ax1.set_title('Original Retina Image', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # 2. Kết quả Phân loại
        if pred_class is not None and all_probs is not None:
            ax2 = plt.subplot(2, 4, 2)
            colors = ['green' if i == pred_class else 'gray' for i in range(len(self.dr_grades))]
            bars = ax2.barh(self.dr_grades, all_probs * 100, color=colors)
            ax2.set_xlabel('Probability (%)', fontsize=12)
            ax2.set_title(f'Classification Result\n{self.dr_grades[pred_class]} ({confidence*100:.1f}%)',
                         fontsize=14, fontweight='bold')
            ax2.set_xlim(0, 100)

            # Thêm nhãn giá trị
            for i, (bar, prob) in enumerate(zip(bars, all_probs)):
                ax2.text(prob*100 + 2, bar.get_y() + bar.get_height()/2,
                        f'{prob*100:.1f}%', va='center', fontsize=10)

        # 3-5. Masks Phân đoạn
        if masks is not None:
            lesion_colors = ['red', 'yellow', 'cyan']

            for i, (lesion, color) in enumerate(zip(self.lesion_types, lesion_colors)):
                ax = plt.subplot(2, 4, 3 + i)

                # Thay đổi kích thước mask để khớp với ảnh gốc
                mask_resized = cv2.resize(masks[i], (image_rgb.shape[1], image_rgb.shape[0]))

                # Tạo lớp phủ
                overlay = image_rgb.copy()
                colored_mask = np.zeros_like(overlay)

                if color == 'red':
                    colored_mask[:, :, 0] = mask_resized * 255
                elif color == 'yellow':
                    colored_mask[:, :, 0] = mask_resized * 255
                    colored_mask[:, :, 1] = mask_resized * 255
                elif color == 'cyan':
                    colored_mask[:, :, 1] = mask_resized * 255
                    colored_mask[:, :, 2] = mask_resized * 255

                overlay = cv2.addWeighted(overlay, 0.7, colored_mask.astype(np.uint8), 0.3, 0)

                ax.imshow(overlay)
                ax.set_title(f'{lesion}\n({mask_resized.sum():.0f} pixels)',
                           fontsize=12, fontweight='bold')
                ax.axis('off')

            # 6. Phân đoạn Kết hợp
            ax6 = plt.subplot(2, 4, 6)
            combined_overlay = image_rgb.copy()

            for i, color in enumerate(lesion_colors):
                mask_resized = cv2.resize(masks[i], (image_rgb.shape[1], image_rgb.shape[0]))
                colored_mask = np.zeros_like(combined_overlay)

                if color == 'red':
                    colored_mask[:, :, 0] = mask_resized * 255
                elif color == 'yellow':
                    colored_mask[:, :, 0] = mask_resized * 255
                    colored_mask[:, :, 1] = mask_resized * 255
                elif color == 'cyan':
                    colored_mask[:, :, 1] = mask_resized * 255
                    colored_mask[:, :, 2] = mask_resized * 255

                combined_overlay = cv2.addWeighted(combined_overlay, 1.0, colored_mask.astype(np.uint8), 0.3, 0)

            ax6.imshow(combined_overlay)
            ax6.set_title('All Lesions Combined', fontsize=14, fontweight='bold')
            ax6.axis('off')

        plt.tight_layout()

        # Lưu
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Visualization saved to: {save_path}")
        else:
            default_path = os.path.join(config.RESULT_DIR, 'prediction_result.png')
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Visualization saved to: {default_path}")

        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Predict DR from retina image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default=None, help='Path to save result')
    args = parser.parse_args()

    # Kiểm tra xem ảnh có tồn tại không
    if not os.path.exists(args.image):
        print(f"❌ Không tìm thấy ảnh: {args.image}")
        return

    # Tạo bộ dự đoán
    predictor = DRPredictor()

    # Dự đoán
    if args.output is None:
        filename = os.path.splitext(os.path.basename(args.image))[0]
        args.output = os.path.join(config.RESULT_DIR, f'prediction_{filename}.png')

    predictor.predict(args.image, args.output)

    print(f"\n{'='*80}")
    print("✅ DỰ ĐOÁN HOÀN THÀNH!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
