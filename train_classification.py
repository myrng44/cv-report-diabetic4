"""
Training script for DR Classification with SANGO optimization
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import config
from dataset import get_classification_loaders
from classification_model import create_classification_model, FocalLoss
from advanced_strategies import (
    get_class_weights,
    create_weighted_loss,
    get_cosine_schedule_with_warmup,
    mixup_data,
    mixup_criterion,
    LabelSmoothingCrossEntropy,
    clip_gradients
)


class ClassificationTrainer:
    """Trainer for DR classification"""

    def __init__(self, model, device, use_amp=True):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None

        # Loss function - Use advanced strategies
        if config.USE_CLASS_WEIGHTS:
            print("✓ Using weighted loss for class imbalance")
            self.criterion = create_weighted_loss(device)
        elif config.USE_LABEL_SMOOTHING:
            print("✓ Using label smoothing")
            self.criterion = LabelSmoothingCrossEntropy(smoothing=config.LABEL_SMOOTHING)
        else:
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Learning rate scheduler - Use cosine schedule with warmup
        self.use_cosine_schedule = config.USE_COSINE_SCHEDULE
        self.scheduler = None  # Will be set in train() method

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with MixUp augmentation"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Apply MixUp augmentation
            if config.USE_MIXUP and np.random.rand() < 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=config.MIXUP_ALPHA)
                use_mixup = True
            else:
                use_mixup = False

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    if use_mixup:
                        loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                    else:
                        loss = self.criterion(outputs, labels)

                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if config.GRADIENT_CLIP_NORM > 0:
                    self.scaler.unscale_(self.optimizer)
                    clip_gradients(self.model, max_norm=config.GRADIENT_CLIP_NORM)

                # Gradient accumulation
                if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                if use_mixup:
                    loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if config.GRADIENT_CLIP_NORM > 0:
                    clip_gradients(self.model, max_norm=config.GRADIENT_CLIP_NORM)

                self.optimizer.step()

            # Track metrics (use original labels for accuracy)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            if use_mixup:
                # For mixup, use the dominant label
                true_labels = labels_a.cpu().numpy()
            else:
                true_labels = labels.cpu().numpy()
            all_labels.extend(true_labels)

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

            # Step scheduler if using cosine schedule
            if self.use_cosine_schedule and self.scheduler is not None:
                self.scheduler.step()

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        val_loss = running_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

        return val_loss, val_acc, val_f1, val_precision, val_recall, all_preds, all_labels

    def train(self, train_loader, val_loader, num_epochs):
        """Complete training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")

        # Initialize scheduler
        if self.use_cosine_schedule:
            total_steps = len(train_loader) * num_epochs
            warmup_steps = len(train_loader) * config.WARMUP_EPOCHS
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, warmup_steps, total_steps
            )
            print(f"✓ Using Cosine Schedule with {config.WARMUP_EPOCHS} warmup epochs")
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )

        if config.USE_MIXUP:
            print(f"✓ Using MixUp augmentation (alpha={config.MIXUP_ALPHA})")
        if config.GRADIENT_CLIP_NORM > 0:
            print(f"✓ Using gradient clipping (max_norm={config.GRADIENT_CLIP_NORM})")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc, val_f1, val_precision, val_recall, val_preds, val_labels = self.validate(val_loader)

            # Update learning rate (only for ReduceLROnPlateau)
            if not self.use_cosine_schedule:
                self.scheduler.step(val_loss)

            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.val_f1_scores.append(val_f1)

            # Print epoch results
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # Save model
                model_path = os.path.join(config.MODEL_DIR, 'best_classification_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                }, model_path)
                print(f"✓ Best model saved! (Acc: {val_acc:.4f})")

                # Save confusion matrix for best model
                self.save_confusion_matrix(val_labels, val_preds, epoch)
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

        print(f"\n✓ Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")

        # Plot training curves
        self.plot_training_curves()

        return self.best_val_acc

    def save_confusion_matrix(self, labels, preds, epoch):
        """Save confusion matrix"""
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'Grade {i}' for i in range(config.NUM_CLASSES)],
                    yticklabels=[f'Grade {i}' for i in range(config.NUM_CLASSES)])
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULT_DIR, f'confusion_matrix_epoch_{epoch}.png'))
        plt.close()

    def plot_training_curves(self):
        """Plot training curves"""
        epochs = range(1, len(self.train_losses) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Loss
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss')
        axes[0, 0].set_title('Loss over Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy
        axes[0, 1].plot(epochs, self.val_accuracies, 'g-', label='Val Accuracy')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # F1 Score
        axes[1, 0].plot(epochs, self.val_f1_scores, 'm-', label='Val F1 Score')
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Summary
        axes[1, 1].text(0.1, 0.8, f'Best Val Acc: {self.best_val_acc:.4f}', fontsize=14)
        axes[1, 1].text(0.1, 0.6, f'Best Val Loss: {self.best_val_loss:.4f}', fontsize=14)
        axes[1, 1].text(0.1, 0.4, f'Final Val F1: {self.val_f1_scores[-1]:.4f}', fontsize=14)
        axes[1, 1].text(0.1, 0.2, f'Total Epochs: {len(epochs)}', fontsize=14)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Training Summary')

        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULT_DIR, 'training_curves.png'))
        plt.close()
        print(f"Metrics plot saved to {config.RESULT_DIR}")


def train_classification_model():
    """Main function to train classification model"""

    # Set random seed
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loaders
    print("Loading datasets...")
    train_loader, val_loader = get_classification_loaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        img_size=config.IMG_SIZE
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Create model
    print("\nCreating classification model...")
    model = create_classification_model(
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        gru_hidden_size=config.GRU_HIDDEN_SIZE,
        gru_num_layers=config.GRU_NUM_LAYERS
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = ClassificationTrainer(model, device, use_amp=config.USE_AMP)

    # Train
    best_acc = trainer.train(train_loader, val_loader, config.NUM_EPOCHS)

    # Plot metrics
    trainer.plot_metrics()

    return best_acc


if __name__ == '__main__':
    train_classification_model()
