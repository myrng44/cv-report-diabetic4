"""
Inference and Evaluation with Grad-CAM visualization
"""

import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import config
from classification_model import create_classification_model
from segmentation_model import create_segmentation_model
from preprocessing import preprocess_fundus_image


class DRInference:
    """Inference class for DR classification and segmentation"""

    def __init__(self, classification_model_path=None, segmentation_model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load classification model
        self.class_model = None
        if classification_model_path and os.path.exists(classification_model_path):
            self.class_model = create_classification_model(num_classes=config.NUM_CLASSES)
            checkpoint = torch.load(classification_model_path, map_location=self.device)
            self.class_model.load_state_dict(checkpoint['model_state_dict'])
            self.class_model.to(self.device)
            self.class_model.eval()
            print(f"✓ Classification model loaded from {classification_model_path}")

        # Load segmentation model
        self.seg_model = None
        if segmentation_model_path and os.path.exists(segmentation_model_path):
            self.seg_model = create_segmentation_model(
                in_channels=3,
                out_channels=config.SEG_CLASSES
            )
            checkpoint = torch.load(segmentation_model_path, map_location=self.device)
            self.seg_model.load_state_dict(checkpoint['model_state_dict'])
            self.seg_model.to(self.device)
            self.seg_model.eval()
            print(f"✓ Segmentation model loaded from {segmentation_model_path}")

        # DR grade labels
        self.dr_grades = {
            0: 'No DR',
            1: 'Mild NPDR',
            2: 'Moderate NPDR',
            3: 'Severe NPDR',
            4: 'PDR'
        }

        # Lesion types
        self.lesion_types = ['Microaneurysms', 'Haemorrhages', 'Hard Exudates']

    def predict_classification(self, image_path):
        """Predict DR grade"""
        if self.class_model is None:
            raise ValueError("Classification model not loaded")

        # Load and preprocess image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        processed = preprocess_fundus_image(image_path, target_size=config.IMG_SIZE, apply_gabor=False)

        # Convert to tensor
        image_tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            output = self.class_model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, pred_class].item()

        return pred_class, confidence, probabilities[0].cpu().numpy()

    def predict_segmentation(self, image_path):
        """Predict lesion segmentation"""
        if self.seg_model is None:
            raise ValueError("Segmentation model not loaded")

        # Load and preprocess image
        processed = preprocess_fundus_image(image_path, target_size=config.IMG_SIZE, apply_gabor=False)

        # Convert to tensor
        image_tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            output = self.seg_model(image_tensor)
            masks = torch.sigmoid(output)

        # Convert to numpy
        masks = masks[0].cpu().numpy()  # (3, H, W)

        return masks

    def generate_gradcam(self, image_path, target_class=None):
        """Generate Grad-CAM visualization"""
        if self.class_model is None:
            raise ValueError("Classification model not loaded")

        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (config.IMG_SIZE, config.IMG_SIZE))
        image_normalized = image_resized.astype(np.float32) / 255.0

        # Preprocess for model
        processed = preprocess_fundus_image(image_path, target_size=config.IMG_SIZE, apply_gabor=False)
        image_tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Setup Grad-CAM
        target_layers = [self.class_model.features[-1]]

        # Create CAM
        cam = GradCAM(model=self.class_model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

        # Generate CAM
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Overlay on image
        visualization = show_cam_on_image(image_normalized, grayscale_cam, use_rgb=True)

        return visualization, grayscale_cam

    def visualize_results(self, image_path, save_path=None):
        """Complete visualization of classification and segmentation results"""

        # Load original image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig = plt.figure(figsize=(20, 10))

        # Original image
        ax1 = plt.subplot(2, 4, 1)
        ax1.imshow(image_rgb)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # Classification prediction
        if self.class_model is not None:
            pred_class, confidence, probs = self.predict_classification(image_path)

            ax2 = plt.subplot(2, 4, 2)
            ax2.bar(range(5), probs, color='steelblue')
            ax2.set_xlabel('DR Grade')
            ax2.set_ylabel('Probability')
            ax2.set_title(f'Classification\nPredicted: {self.dr_grades[pred_class]} ({confidence:.2%})',
                         fontsize=14, fontweight='bold')
            ax2.set_xticks(range(5))
            ax2.set_xticklabels(['0', '1', '2', '3', '4'])

            # Grad-CAM
            gradcam_vis, _ = self.generate_gradcam(image_path, target_class=pred_class)
            ax3 = plt.subplot(2, 4, 3)
            ax3.imshow(gradcam_vis)
            ax3.set_title('Grad-CAM Visualization', fontsize=14, fontweight='bold')
            ax3.axis('off')

        # Segmentation prediction
        if self.seg_model is not None:
            masks = self.predict_segmentation(image_path)

            # Resize original image for overlay
            img_resized = cv2.resize(image_rgb, (config.IMG_SIZE, config.IMG_SIZE))

            for i, lesion_name in enumerate(self.lesion_types):
                ax = plt.subplot(2, 4, 5 + i)

                # Create colored overlay
                mask = masks[i]
                mask_binary = (mask > 0.5).astype(np.uint8)

                # Overlay on image
                overlay = img_resized.copy()
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # R, G, B for each lesion type
                overlay[mask_binary > 0] = colors[i]

                # Blend
                result = cv2.addWeighted(img_resized, 0.7, overlay, 0.3, 0)

                ax.imshow(result)
                ax.set_title(f'{lesion_name}\nSegmentation', fontsize=12, fontweight='bold')
                ax.axis('off')

            # Combined segmentation
            ax_combined = plt.subplot(2, 4, 8)
            combined = img_resized.copy().astype(np.float32)
            for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
                mask = (masks[i] > 0.5).astype(np.float32)
                for c in range(3):
                    combined[:, :, c] += mask * color[c] * 0.3
            combined = np.clip(combined, 0, 255).astype(np.uint8)

            ax_combined.imshow(combined)
            ax_combined.set_title('All Lesions Combined', fontsize=12, fontweight='bold')
            ax_combined.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")

        plt.show()

        return fig


def demo_inference():
    """Demo inference on sample images"""

    # Paths to models
    class_model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    seg_model_path = os.path.join(config.MODEL_DIR, 'best_seg_model.pth')

    # Create inference object
    inference = DRInference(
        classification_model_path=class_model_path if os.path.exists(class_model_path) else None,
        segmentation_model_path=seg_model_path if os.path.exists(seg_model_path) else None
    )

    # Get sample images
    sample_dir = config.CLASS_TRAIN_IMG_DIR
    sample_images = [f for f in os.listdir(sample_dir) if f.endswith('.jpg')][:3]

    # Run inference on samples
    for img_name in sample_images:
        img_path = os.path.join(sample_dir, img_name)
        save_path = os.path.join(config.RESULT_DIR, f'inference_{img_name}')

        print(f"\nProcessing {img_name}...")
        inference.visualize_results(img_path, save_path)


if __name__ == '__main__':
    demo_inference()

