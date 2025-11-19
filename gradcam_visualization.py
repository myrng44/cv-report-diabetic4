"""
Grad-CAM Visualization for DR Classification Model
Helps explain which regions the model focuses on
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple
import os


class GradCAM:
    """Gradient-weighted Class Activation Mapping"""

    def __init__(self, model, target_layer):
        """
        Args:
            model: trained classification model
            target_layer: layer to compute gradients (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Hook to save forward activations"""
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradients"""
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        """
        Generate CAM for input image

        Args:
            input_image: preprocessed image tensor [1, 3, H, W]
            target_class: class to generate CAM for (None = predicted class)

        Returns:
            cam: Class activation map [H, W]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]

        # Weighted sum of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy(), target_class

    def visualize(self, original_image, cam, alpha=0.4):
        """
        Overlay CAM on original image

        Args:
            original_image: original image [H, W, 3] in RGB, range [0, 255]
            cam: class activation map [H, W]
            alpha: overlay transparency

        Returns:
            overlayed image
        """
        # Resize CAM to match original image
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay
        overlayed = heatmap * alpha + original_image * (1 - alpha)
        overlayed = np.uint8(overlayed)

        return overlayed, heatmap


def visualize_predictions_with_gradcam(model, image_path, device,
                                       class_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'],
                                       save_path=None):
    """
    Complete visualization with GradCAM

    Args:
        model: trained model
        image_path: path to input image
        device: torch device
        class_names: list of class names
        save_path: path to save visualization
    """
    from dataset import get_classification_transforms
    from PIL import Image

    # Load and preprocess image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    transform = get_classification_transforms(is_train=False, img_size=384)
    image_pil = Image.fromarray(original_image)
    input_tensor = transform(image=np.array(image_pil))['image'].unsqueeze(0).to(device)

    # Get model's last conv layer
    # For DRClassificationModel, the features are stored in model.features
    # Access the last denseblock (denseblock4) from the features sequential
    target_layer = None
    for name, module in model.features.named_children():
        if 'denseblock4' in name or name == 'denseblock4':
            target_layer = module
            break

    # If denseblock4 not found by name, get the last DenseBlock-like layer
    if target_layer is None:
        # Try to find it by checking module types
        for module in reversed(list(model.features.children())):
            if hasattr(module, 'denselayer1'):  # DenseBlock has denselayers
                target_layer = module
                break

    # Fallback: use the entire features module
    if target_layer is None:
        target_layer = model.features

    # Create GradCAM
    gradcam = GradCAM(model, target_layer)

    # Generate CAM
    cam, predicted_class = gradcam.generate_cam(input_tensor)

    # Get prediction probabilities
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]

    # Visualize
    overlayed, heatmap = gradcam.visualize(original_image, cam, alpha=0.4)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Heatmap
    axes[0, 1].imshow(heatmap)
    axes[0, 1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Overlayed
    axes[1, 0].imshow(overlayed)
    axes[1, 0].set_title(f'Prediction: {class_names[predicted_class]} (Confidence: {probs[predicted_class]:.2%})',
                        fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Probability bar chart
    axes[1, 1].barh(class_names, probs.cpu().numpy(), color='steelblue')
    axes[1, 1].set_xlabel('Probability', fontsize=12)
    axes[1, 1].set_title('Class Probabilities', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlim([0, 1])

    # Add grid for better readability
    axes[1, 1].grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")

    plt.show()

    return predicted_class, probs.cpu().numpy()


def batch_visualize_gradcam(model, image_dir, device, output_dir, num_samples=10):
    """
    Generate GradCAM visualizations for multiple images

    Args:
        model: trained model
        image_dir: directory containing images
        device: torch device
        output_dir: directory to save visualizations
        num_samples: number of images to visualize
    """
    import glob
    import random

    os.makedirs(output_dir, exist_ok=True)

    # Get random sample of images
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    sampled_paths = random.sample(image_paths, min(num_samples, len(image_paths)))

    print(f"\n{'='*80}")
    print(f"Generating Grad-CAM visualizations for {len(sampled_paths)} images...")
    print(f"{'='*80}\n")

    for i, img_path in enumerate(sampled_paths, 1):
        img_name = os.path.basename(img_path)
        save_path = os.path.join(output_dir, f"gradcam_{img_name}")

        print(f"[{i}/{len(sampled_paths)}] Processing {img_name}...")

        try:
            pred_class, probs = visualize_predictions_with_gradcam(
                model, img_path, device, save_path=save_path
            )
            print(f"   ✓ Predicted class: {pred_class}, Max prob: {probs.max():.2%}")
        except Exception as e:
            print(f"   ✗ Failed: {e}")

    print(f"\n{'='*80}")
    print(f"✓ All visualizations saved to {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    """Example usage"""
    import torch
    from classification_model import create_classification_model

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_classification_model(num_classes=5, pretrained=False)
    model.load_state_dict(torch.load('outputs/models/best_model.pth', map_location=device))
    model = model.to(device)
    model.eval()

    # Single image visualization
    visualize_predictions_with_gradcam(
        model,
        'data/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_01.jpg',
        device,
        save_path='outputs/results/gradcam_example.png'
    )

    # Batch visualization
    batch_visualize_gradcam(
        model,
        'data/B. Disease Grading/1. Original Images/a. Training Set',
        device,
        'outputs/results/gradcam_visualizations',
        num_samples=10
    )
