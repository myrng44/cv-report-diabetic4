"""
Evaluation script for trained models
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
from segmentation_model import create_segmentation_model


def evaluate_classification_model(model_path):
    """Evaluate classification model"""

    print("="*80)
    print("Evaluating Classification Model")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
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
        for images, labels in tqdm(test_loader, desc='Evaluating'):
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
    print("Classification Results:")
    print("="*80)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("="*80)

    # Detailed classification report
    print("\nDetailed Classification Report:")
    print("-"*80)
    dr_grades = ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR']
    print(classification_report(all_labels, all_preds, target_names=dr_grades, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=dr_grades, yticklabels=dr_grades)
    plt.title('Confusion Matrix - DR Classification', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(config.RESULT_DIR, 'confusion_matrix_classification.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n✓ Confusion matrix saved to {save_path}")

    # Per-class accuracy
    print("\nPer-class Accuracy:")
    print("-"*80)
    for i, grade in enumerate(dr_grades):
        mask = np.array(all_labels) == i
        if mask.sum() > 0:
            class_acc = (np.array(all_preds)[mask] == i).sum() / mask.sum()
            print(f"{grade:12s}: {class_acc:.4f} ({class_acc*100:.2f}%) - {mask.sum()} samples")

    return accuracy, f1, precision, recall


def evaluate_segmentation_model(model_path):
    """Evaluate segmentation model"""

    print("\n" + "="*80)
    print("Evaluating Segmentation Model")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = create_segmentation_model(
        in_channels=3,
        out_channels=config.SEG_CLASSES,
        base_filters=32
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded from {model_path}")

    # Load test data
    _, test_loader = get_segmentation_loaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        img_size=config.IMG_SIZE
    )

    print(f"✓ Test samples: {len(test_loader.dataset)}")

    # Evaluate
    all_ious = {i: [] for i in range(config.SEG_CLASSES)}
    all_dices = {i: [] for i in range(config.SEG_CLASSES)}

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)

            # Calculate metrics for each lesion type
            for i in range(config.SEG_CLASSES):
                pred_binary = (preds[:, i] > 0.5).float()
                target_binary = masks[:, i]

                # IoU
                intersection = (pred_binary * target_binary).sum()
                union = pred_binary.sum() + target_binary.sum() - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                all_ious[i].append(iou.item())

                # Dice
                dice = (2.0 * intersection + 1e-6) / (pred_binary.sum() + target_binary.sum() + 1e-6)
                all_dices[i].append(dice.item())

    # Calculate mean metrics
    lesion_types = ['Microaneurysms', 'Haemorrhages', 'Hard Exudates']

    print("\n" + "="*80)
    print("Segmentation Results:")
    print("="*80)

    mean_iou = np.mean([np.mean(all_ious[i]) for i in range(config.SEG_CLASSES)])
    mean_dice = np.mean([np.mean(all_dices[i]) for i in range(config.SEG_CLASSES)])

    print(f"Mean IoU:  {mean_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print("="*80)

    print("\nPer-lesion Metrics:")
    print("-"*80)
    for i, lesion in enumerate(lesion_types):
        iou = np.mean(all_ious[i])
        dice = np.mean(all_dices[i])
        print(f"{lesion:20s}: IoU={iou:.4f}, Dice={dice:.4f}")

    # Plot metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # IoU plot
    ious = [np.mean(all_ious[i]) for i in range(config.SEG_CLASSES)]
    axes[0].bar(lesion_types, ious, color='steelblue')
    axes[0].set_ylabel('IoU Score', fontsize=12)
    axes[0].set_title('IoU by Lesion Type', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)

    # Dice plot
    dices = [np.mean(all_dices[i]) for i in range(config.SEG_CLASSES)]
    axes[1].bar(lesion_types, dices, color='coral')
    axes[1].set_ylabel('Dice Score', fontsize=12)
    axes[1].set_title('Dice Score by Lesion Type', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.RESULT_DIR, 'segmentation_evaluation.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n✓ Evaluation plot saved to {save_path}")

    return mean_iou, mean_dice


def main():
    """Main evaluation function"""

    # Check if models exist
    class_model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    seg_model_path = os.path.join(config.MODEL_DIR, 'best_seg_model.pth')

    results = {}

    # Evaluate classification
    if os.path.exists(class_model_path):
        acc, f1, prec, rec = evaluate_classification_model(class_model_path)
        results['classification'] = {
            'accuracy': acc,
            'f1_score': f1,
            'precision': prec,
            'recall': rec
        }
    else:
        print(f"⚠ Classification model not found: {class_model_path}")

    # Evaluate segmentation
    if os.path.exists(seg_model_path):
        iou, dice = evaluate_segmentation_model(seg_model_path)
        results['segmentation'] = {
            'mean_iou': iou,
            'mean_dice': dice
        }
    else:
        print(f"⚠ Segmentation model not found: {seg_model_path}")

    # Summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    if 'classification' in results:
        print("\nClassification:")
        for metric, value in results['classification'].items():
            print(f"  {metric:12s}: {value:.4f}")

    if 'segmentation' in results:
        print("\nSegmentation:")
        for metric, value in results['segmentation'].items():
            print(f"  {metric:12s}: {value:.4f}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()

