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


class ClassificationTrainer:
    """Trainer for DR classification"""

    def __init__(self, model, device, use_amp=True):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None

        # Loss function (Focal Loss for imbalanced classes)
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Track metrics
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

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

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc, val_f1, val_prec, val_rec, _, _ = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.val_f1_scores.append(val_f1)

            # Print metrics
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            print(f"Val Precision: {val_prec:.4f}, Val Recall: {val_rec:.4f}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, 'best_model.pth')
                print(f"✓ Best model saved! (Acc: {val_acc:.4f})")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pth')

        print("\n✓ Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")

        return self.best_val_acc

    def save_checkpoint(self, epoch, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
        }

        path = os.path.join(config.MODEL_DIR, filename)
        torch.save(checkpoint, path)

    def plot_metrics(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy plot
        axes[0, 1].plot(self.val_accuracies, label='Val Accuracy', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # F1 Score plot
        axes[1, 0].plot(self.val_f1_scores, label='Val F1 Score', color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULT_DIR, 'classification_metrics.png'), dpi=300)
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

