"""
EVALUATION SCRIPT - Đánh giá model đã train trên test set
Chạy sau khi train xong để có metrics cuối cùng

Usage:
    python evaluate_models.py
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                            recall_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

import config
from dataset import get_classification_loaders, get_segmentation_loaders
from classification_model import create_classification_model
from advanced_segmentation_model import create_advanced_segmentation_model


def evaluate_classification():
    """Đánh giá Classification Model trên test set"""

    print("\n" + "="*80)
    print("📊 EVALUATING CLASSIFICATION MODEL")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model_path = os.path.join(config.MODEL_DIR, 'best_classification_model.pth')
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    model = create_classification_model(num_classes=config.NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded from {model_path}")

    # Load test data
    _, test_loader = get_classification_loaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        img_size=config.IMG_SIZE
    )

    print(f"✓ Test samples: {len(test_loader.dataset)}")

    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

    print("\n" + "="*80)
    print("📈 CLASSIFICATION TEST RESULTS:")
    print("="*80)
    print(f"✓ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✓ F1 Score:  {f1:.4f}")
    print(f"✓ Precision: {precision:.4f}")
    print(f"✓ Recall:    {recall:.4f}")
    print("="*80)

    # Detailed classification report
    print("\n📋 Detailed Classification Report:")
    print("-"*80)
    dr_grades = ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR']
    print(classification_report(all_labels, all_preds, target_names=dr_grades, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=dr_grades, yticklabels=dr_grades)
    plt.title('Confusion Matrix - DR Classification (Test Set)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(config.RESULT_DIR, 'confusion_matrix_test.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to {save_path}")
    plt.close()

    # Per-class accuracy
    print("\n📊 Per-class Accuracy:")
    print("-"*80)
    for i, grade in enumerate(dr_grades):
        mask = np.array(all_labels) == i
        if mask.sum() > 0:
            class_acc = accuracy_score(np.array(all_labels)[mask], np.array(all_preds)[mask])
            print(f"{grade:15s}: {class_acc:.4f} ({class_acc*100:.2f}%) - {mask.sum()} samples")

    # Save results to file
    results_file = os.path.join(config.RESULT_DIR, 'classification_test_results.txt')
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CLASSIFICATION TEST RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n\n")
        f.write("-"*80 + "\n")
        f.write(classification_report(all_labels, all_preds, target_names=dr_grades, zero_division=0))
        f.write("\n" + "-"*80 + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")

    print(f"\n✓ Results saved to {results_file}")

    return accuracy, f1, precision, recall


def evaluate_segmentation():
    """Đánh giá Segmentation Model trên test set"""

    print("\n" + "="*80)
    print("📊 EVALUATING SEGMENTATION MODEL")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model_path = os.path.join(config.MODEL_DIR, 'best_seg_model.pth')
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    model = create_advanced_segmentation_model(in_channels=3, out_channels=config.SEG_CLASSES)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded from {model_path}")

    # Load test data
    _, test_loader = get_segmentation_loaders(
        batch_size=4,
        num_workers=config.NUM_WORKERS,
        img_size=512
    )

    print(f"✓ Test samples: {len(test_loader.dataset)}")

    # Evaluate
    ious = []
    dices = []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc='Testing Segmentation'):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)
            # ✅ HẠ THRESHOLD từ 0.5 → 0.15 cho tiny lesions
            preds = (preds > 0.15).float()

            # Calculate IoU and Dice per sample
            for i in range(images.size(0)):
                pred_mask = preds[i]
                true_mask = masks[i]

                intersection = (pred_mask * true_mask).sum()
                union = pred_mask.sum() + true_mask.sum() - intersection

                iou = (intersection / (union + 1e-7)).item()
                dice = (2 * intersection / (pred_mask.sum() + true_mask.sum() + 1e-7)).item()

                ious.append(iou)
                dices.append(dice)

    mean_iou = np.mean(ious)
    mean_dice = np.mean(dices)

    print("\n" + "="*80)
    print("📈 SEGMENTATION TEST RESULTS:")
    print("="*80)
    print(f"✓ Mean IoU:  {mean_iou:.4f}")
    print(f"✓ Mean Dice: {mean_dice:.4f}")
    print("="*80)

    # Save results
    results_file = os.path.join(config.RESULT_DIR, 'segmentation_test_results.txt')
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SEGMENTATION TEST RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Mean IoU:  {mean_iou:.4f}\n")
        f.write(f"Mean Dice: {mean_dice:.4f}\n")

    print(f"\n✓ Results saved to {results_file}")

    return mean_iou, mean_dice


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🎯 MODEL EVALUATION - TEST SET")
    print("="*80)

    # Create results directory
    os.makedirs(config.RESULT_DIR, exist_ok=True)

    # Evaluate both models
    print("\n" + "🔍 Starting Evaluation...\n")

    # 1. Classification
    try:
        class_metrics = evaluate_classification()
    except Exception as e:
        print(f"❌ Error evaluating classification: {e}")
        import traceback
        traceback.print_exc()

    # 2. Segmentation
    try:
        seg_metrics = evaluate_segmentation()
    except Exception as e:
        print(f"❌ Error evaluating segmentation: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETED!")
    print("="*80)
    print(f"📁 Results saved in: {config.RESULT_DIR}")
    print("="*80 + "\n")
