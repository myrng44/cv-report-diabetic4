"""
Demo Inference Script - Chạy dự đoán trên 1 ảnh retina

Sử dụng:
    python demo_inference.py --image path/to/image.jpg
    python demo_inference.py --image path/to/image.jpg --output results/
"""

import os
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import config
from preprocessing import preprocess_fundus_image

# Import models
try:
    from segmentation_model_improved import ModifiedUNet
    USE_IMPROVED = True
except:
    try:
        from segmentation_model import ModifiedUNet
        USE_IMPROVED = False
    except:
        ModifiedUNet = None

try:
    from classification_model import create_classification_model
    HAS_CLASSIFICATION = True
except:
    HAS_CLASSIFICATION = False


class DRPredictor:
    """Dự đoán DR từ 1 ảnh retina"""

    def __init__(self, seg_model_path=None, class_model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load Segmentation Model
        self.seg_model = None
        if seg_model_path and os.path.exists(seg_model_path):
            print(f"Loading segmentation model from {seg_model_path}...")
            if ModifiedUNet is not None:
                self.seg_model = ModifiedUNet(in_channels=3, out_channels=3)

                checkpoint = torch.load(seg_model_path, map_location=self.device, weights_only=False)
                self.seg_model.load_state_dict(checkpoint['model_state_dict'])
                self.seg_model.to(self.device)
                self.seg_model.eval()
                print("✓ Segmentation model loaded!")
            else:
                print("⚠ ModifiedUNet class not found")
        else:
            print("⚠ No segmentation model found")

        # Load Classification Model
        self.class_model = None
        if HAS_CLASSIFICATION and class_model_path and os.path.exists(class_model_path):
            print(f"Loading classification model from {class_model_path}...")
            self.class_model = create_classification_model(num_classes=5)
            checkpoint = torch.load(class_model_path, map_location=self.device, weights_only=False)
            self.class_model.load_state_dict(checkpoint['model_state_dict'])
            self.class_model.to(self.device)
            self.class_model.eval()
            print("✓ Classification model loaded!")
        else:
            print("⚠ No classification model found")

        # DR grades
        self.dr_grades = {
            0: 'No DR (Không bệnh)',
            1: 'Mild NPDR (Nhẹ)',
            2: 'Moderate NPDR (Trung bình)',
            3: 'Severe NPDR (Nặng)',
            4: 'PDR (Tăng sinh)'
        }

        self.lesion_names = ['Microaneurysms', 'Haemorrhages', 'Hard Exudates']
        self.lesion_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue

    def predict_dr_grade(self, image_path):
        """Dự đoán mức độ DR (0-4)"""
        if self.class_model is None:
            return None, None

        # Preprocess
        processed = preprocess_fundus_image(image_path, target_size=config.IMG_SIZE, apply_gabor=False)
        image_tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            output = self.class_model(image_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_class = torch.argmax(probs).item()
            confidence = probs[pred_class].item()

        return pred_class, confidence

    def predict_lesions(self, image_path):
        """Dự đoán các lesions (segmentation)"""
        if self.seg_model is None:
            return None

        # Preprocess
        processed = preprocess_fundus_image(image_path, target_size=config.IMG_SIZE, apply_gabor=False)
        image_tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            output = self.seg_model(image_tensor)
            masks = torch.sigmoid(output)[0]  # (3, H, W)

        return masks.cpu().numpy()

    def visualize_results(self, image_path, output_path=None):
        """Tạo visualization đầy đủ"""

        # Load original image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create figure
        fig = plt.figure(figsize=(24, 14))

        # 1. Original Image
        ax1 = plt.subplot(2, 5, 1)
        ax1.imshow(image_rgb)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # 2. Classification Result
        dr_grade, confidence = self.predict_dr_grade(image_path)
        ax2 = plt.subplot(2, 5, 2)
        if dr_grade is not None:
            result_text = f"DR Grade: {dr_grade}\n{self.dr_grades[dr_grade]}\nConfidence: {confidence:.1%}"
            ax2.text(0.5, 0.5, result_text,
                    ha='center', va='center', fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
            ax2.set_title('Classification Result', fontsize=14, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Model not available', ha='center', va='center')
            ax2.axis('off')

        # 3. Preprocessed Image
        ax3 = plt.subplot(2, 5, 3)
        processed = preprocess_fundus_image(image_path, target_size=config.IMG_SIZE, apply_gabor=False)
        ax3.imshow(processed)
        ax3.set_title('Preprocessed', fontsize=14, fontweight='bold')
        ax3.axis('off')

        # 4-7. Segmentation Results - Marked Lesions
        masks = self.predict_lesions(image_path)
        if masks is not None:
            img_resized = cv2.resize(image_rgb, (config.IMG_SIZE, config.IMG_SIZE))

            # Lower threshold
            threshold = 0.2

            # Panel 4: Chỉ masks (không có ảnh nền)
            ax4 = plt.subplot(2, 5, 4)
            mask_only = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
            for i, color in enumerate(self.lesion_colors):
                mask = masks[i]
                mask_binary = (mask > threshold).astype(np.uint8)
                mask_only[mask_binary > 0] = color
            ax4.imshow(mask_only)
            ax4.set_title('Masks Only\n(No background)', fontsize=12, fontweight='bold')
            ax4.axis('off')

            # Panel 5: All lesions với contour rõ ràng
            ax5 = plt.subplot(2, 5, 5)
            combined_marked = img_resized.copy()

            # Vẽ từng loại lesion với màu đậm và contour
            for i, (lesion_name, color) in enumerate(zip(self.lesion_names, self.lesion_colors)):
                mask = masks[i]
                mask_binary = (mask > threshold).astype(np.uint8) * 255

                # Tô màu vùng lesions
                mask_3channel = np.stack([mask_binary, mask_binary, mask_binary], axis=-1)
                colored_mask = np.zeros_like(img_resized)
                colored_mask[mask_binary > 0] = color

                # Blend chỉ vùng lesions (đậm hơn)
                alpha = 0.7
                combined_marked = np.where(mask_3channel > 0,
                                          cv2.addWeighted(combined_marked, 1-alpha, colored_mask, alpha, 0),
                                          combined_marked)

                # Vẽ contour (viền) quanh lesions
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(combined_marked, contours, -1, color, 2)

            ax5.imshow(combined_marked)
            ax5.set_title('All Lesions Marked\n(With Contours)', fontsize=12, fontweight='bold')
            ax5.axis('off')

            # Panel 6-8: Individual lesions with contours
            for i, (lesion_name, color) in enumerate(zip(self.lesion_names, self.lesion_colors)):
                ax = plt.subplot(2, 5, 6 + i)

                mask = masks[i]
                mask_binary = (mask > threshold).astype(np.uint8) * 255

                # Tạo ảnh đánh dấu
                marked_img = img_resized.copy()

                # Tô màu vùng lesions
                colored_mask = np.zeros_like(img_resized)
                colored_mask[mask_binary > 0] = color

                # Blend với alpha cao
                alpha = 0.7
                marked_img = np.where(mask_binary[:, :, None] > 0,
                                     cv2.addWeighted(marked_img, 1-alpha, colored_mask, alpha, 0),
                                     marked_img)

                # Vẽ contour
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(marked_img, contours, -1, color, 2)

                ax.imshow(marked_img)

                # Show statistics
                coverage = (mask > threshold).sum() / (mask.shape[0] * mask.shape[1]) * 100
                num_contours = len(contours)
                ax.set_title(f'{lesion_name}\n{coverage:.2f}% | {num_contours} regions',
                           fontsize=11, fontweight='bold')
                ax.axis('off')

            # Panel 9: Mask overlay với transparency
            ax9 = plt.subplot(2, 5, 9)
            overlay = img_resized.copy().astype(np.float32)
            for i, color in enumerate(self.lesion_colors):
                mask = masks[i]
                mask_binary = (mask > threshold).astype(np.float32)
                for c in range(3):
                    overlay[:, :, c] = np.where(mask_binary > 0,
                                                overlay[:, :, c] * 0.3 + color[c] * 0.7,
                                                overlay[:, :, c])
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            ax9.imshow(overlay)
            ax9.set_title('Heavy Overlay\n(70% opacity)', fontsize=12, fontweight='bold')
            ax9.axis('off')

            # Panel 10: Pure heatmap
            ax10 = plt.subplot(2, 5, 10)
            heatmap = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.float32)
            for i, color in enumerate(self.lesion_colors):
                mask = masks[i]
                # Use continuous values for heatmap
                mask_normalized = mask / (mask.max() + 1e-7)
                for c in range(3):
                    heatmap[:, :, c] += mask_normalized * color[c]
            heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
            ax10.imshow(heatmap)
            ax10.set_title('Probability Heatmap\n(Continuous)', fontsize=12, fontweight='bold')
            ax10.axis('off')

        plt.tight_layout()

        # Save or show
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Result saved to: {output_path}")

        plt.show()

        return {
            'dr_grade': dr_grade,
            'confidence': confidence,
            'dr_label': self.dr_grades.get(dr_grade, 'Unknown') if dr_grade is not None else None,
            'has_segmentation': masks is not None
        }

    def predict_single_image(self, image_path, save_dir=None):
        """Dự đoán 1 ảnh và in kết quả"""

        print(f"\n{'='*80}")
        print(f"Analyzing: {os.path.basename(image_path)}")
        print(f"{'='*80}\n")

        # Classification
        dr_grade, confidence = self.predict_dr_grade(image_path)
        if dr_grade is not None:
            print(f"🔍 CLASSIFICATION RESULT:")
            print(f"   DR Grade: {dr_grade}")
            print(f"   Diagnosis: {self.dr_grades[dr_grade]}")
            print(f"   Confidence: {confidence:.2%}\n")
        else:
            print(f"⚠ Classification model not available\n")

        # Segmentation
        masks = self.predict_lesions(image_path)
        if masks is not None:
            print(f"🎯 LESION DETECTION:")
            for i, lesion_name in enumerate(self.lesion_names):
                mask_area = (masks[i] > 0.5).sum() / (masks[i].shape[0] * masks[i].shape[1])
                print(f"   {lesion_name}: {mask_area:.2%} of image")
        else:
            print(f"⚠ Segmentation model not available")

        # Create visualization
        if save_dir:
            output_path = os.path.join(save_dir, f"result_{os.path.basename(image_path)}")
        else:
            output_path = f"result_{os.path.basename(image_path)}"

        result = self.visualize_results(image_path, output_path)

        print(f"\n{'='*80}\n")

        return result


def main():
    parser = argparse.ArgumentParser(description='DR Inference Demo')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='outputs/results/', help='Output directory')
    parser.add_argument('--seg_model', type=str, default='outputs/models/best_seg_model.pth',
                       help='Path to segmentation model')
    parser.add_argument('--class_model', type=str, default='outputs/models/best_model.pth',
                       help='Path to classification model')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Create predictor
    predictor = DRPredictor(
        seg_model_path=args.seg_model,
        class_model_path=args.class_model
    )

    # Run prediction
    predictor.predict_single_image(args.image, args.output)


if __name__ == '__main__':
    # If no arguments, run demo mode
    import sys
    if len(sys.argv) == 1:
        print("\n" + "="*80)
        print("DEMO MODE - Testing with sample images")
        print("="*80 + "\n")

        # Find a sample image
        sample_dir = config.CLASS_TRAIN_IMG_DIR
        if os.path.exists(sample_dir):
            images = [f for f in os.listdir(sample_dir) if f.endswith('.jpg')]
            if images:
                sample_image = os.path.join(sample_dir, images[0])

                predictor = DRPredictor(
                    seg_model_path='outputs/models/best_seg_model.pth',
                    class_model_path='outputs/models/best_model.pth'
                )

                predictor.predict_single_image(sample_image, 'outputs/results/')
            else:
                print("No sample images found!")
        else:
            print(f"Sample directory not found: {sample_dir}")
            print("\nUsage: python demo_inference.py --image path/to/image.jpg")
    else:
        main()
