#!/usr/bin/env python3
"""
Comprehensive example script demonstrating CNN Transfer Learning for Visual Emotion Recognition.

This script shows step-by-step implementation of transfer learning with CNN networks,
comparing it to baseline CNN approaches.

Author: AI Assistant
Date: 2024
"""

import os
import sys
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

from sklearn.metrics import classification_report, confusion_matrix

# Add source directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from models.cnn_transfer_learning import CNNTransferLearning, create_cnn_transfer_model
    from models.cnn_baseline import ImprovedCNN
    from data.dataset_emotion import EmotionDataset
    from utils.seed import set_seed
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Running in standalone mode...")


class DummyDataset(Dataset):
    """Dummy dataset for demonstration when real data is not available."""
    
    def __init__(self, size=1000, num_classes=7, transform=None, rgb=True):
        self.size = size
        self.num_classes = num_classes
        self.transform = transform
        self.rgb = rgb
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Create random image
        if self.rgb:
            image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        else:
            image = Image.fromarray(np.random.randint(0, 255, (48, 48), dtype=np.uint8))
        
        # Random label
        label = np.random.randint(0, self.num_classes)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def create_transforms():
    """Create data transforms for both baseline and transfer learning models."""
    
    # ImageNet normalization for transfer learning
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Transforms for baseline CNN (grayscale, 48x48)
    baseline_train_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    baseline_val_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Transforms for transfer learning (RGB, 224x224)
    transfer_train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    transfer_val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return {
        'baseline_train': baseline_train_transform,
        'baseline_val': baseline_val_transform,
        'transfer_train': transfer_train_transform,
        'transfer_val': transfer_val_transform
    }


def create_data_loaders(transforms_dict, batch_size=32, dataset_size=1000):
    """Create data loaders for both model types."""
    
    # Baseline CNN datasets (grayscale)
    train_dataset_baseline = DummyDataset(
        size=dataset_size, 
        transform=transforms_dict['baseline_train'], 
        rgb=False
    )
    
    val_dataset_baseline = DummyDataset(
        size=dataset_size//5, 
        transform=transforms_dict['baseline_val'], 
        rgb=False
    )
    
    # Transfer learning datasets (RGB)
    train_dataset_transfer = DummyDataset(
        size=dataset_size, 
        transform=transforms_dict['transfer_train'], 
        rgb=True
    )
    
    val_dataset_transfer = DummyDataset(
        size=dataset_size//5, 
        transform=transforms_dict['transfer_val'], 
        rgb=True
    )
    
    # Create data loaders
    train_loader_baseline = DataLoader(train_dataset_baseline, batch_size=batch_size, shuffle=True)
    val_loader_baseline = DataLoader(val_dataset_baseline, batch_size=batch_size, shuffle=False)
    
    train_loader_transfer = DataLoader(train_dataset_transfer, batch_size=batch_size, shuffle=True)
    val_loader_transfer = DataLoader(val_dataset_transfer, batch_size=batch_size, shuffle=False)
    
    return {
        'baseline': (train_loader_baseline, val_loader_baseline),
        'transfer': (train_loader_transfer, val_loader_transfer)
    }


def train_model_epoch_with_detailed_logging(model, training_data_loader, loss_criterion, optimizer, computing_device):
    """
    🏋️ **Enhanced Training Function for One Epoch**
    
    Trains the model for one complete epoch with detailed progress monitoring
    and clear variable naming for better code readability.
    
    Args:
        model: Neural network model to train
        training_data_loader: DataLoader containing training samples
        loss_criterion: Loss function for optimization
        optimizer: Optimizer for parameter updates
        computing_device: Device (GPU/CPU) for computation
        
    Returns:
        tuple: (average_epoch_loss, epoch_accuracy_percentage)
    """
    print(f"    🚀 Starting training epoch with {len(training_data_loader)} batches...")
    
    # Set model to training mode
    model.train()
    
    # Initialize epoch metrics
    cumulative_epoch_loss = 0.0
    total_correct_predictions = 0
    total_samples_processed = 0
    
    # Process each batch in the training dataset
    for batch_index, (input_images, target_labels) in enumerate(training_data_loader):
        # Transfer data to the computing device (GPU/CPU)
        input_images = input_images.to(computing_device)
        target_labels = target_labels.to(computing_device)
        
        # Reset gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass: compute model predictions
        model_predictions = model(input_images)
        
        # Calculate loss between predictions and true labels
        batch_loss = loss_criterion(model_predictions, target_labels)
        
        # Backward pass: compute gradients
        batch_loss.backward()
        
        # Update model parameters
        optimizer.step()
        
        # Accumulate loss for epoch average
        cumulative_epoch_loss += batch_loss.item()
        
        # Calculate accuracy for this batch
        _, predicted_class_indices = torch.max(model_predictions.data, 1)
        total_samples_processed += target_labels.size(0)
        total_correct_predictions += (predicted_class_indices == target_labels).sum().item()
        
        # Print detailed progress every 10 batches
        if batch_index % 10 == 0:
            current_accuracy_percentage = 100.0 * total_correct_predictions / total_samples_processed
            print(f'    📊 Batch {batch_index:3d}/{len(training_data_loader)} | '
                  f'Loss: {batch_loss.item():.4f} | '
                  f'Accuracy: {current_accuracy_percentage:6.2f}% | '
                  f'Samples: {total_samples_processed:5d}')
        
        # Early termination for demo purposes (remove in production)
        if batch_index >= 20:  # Only train on first 20 batches for demo
            print(f"    ⚡ Early termination at batch {batch_index} for demo purposes")
            break
    
    # Calculate final epoch metrics
    average_epoch_loss = cumulative_epoch_loss / min(len(training_data_loader), 21)
    epoch_accuracy_percentage = 100.0 * total_correct_predictions / total_samples_processed
    
    print(f"    ✅ Training epoch completed: Loss={average_epoch_loss:.4f}, Accuracy={epoch_accuracy_percentage:.2f}%")
    
    return average_epoch_loss, epoch_accuracy_percentage


def validate_model_with_comprehensive_metrics(model, validation_data_loader, loss_criterion, computing_device):
    """
    🔍 **Enhanced Validation Function with Detailed Metrics**
    
    Evaluates the model on validation data with comprehensive logging
    and clear variable naming for better understanding.
    
    Args:
        model: Neural network model to validate
        validation_data_loader: DataLoader containing validation samples
        loss_criterion: Loss function for evaluation
        computing_device: Device (GPU/CPU) for computation
        
    Returns:
        tuple: (average_validation_loss, validation_accuracy_percentage)
    """
    print(f"    🔍 Starting validation on {len(validation_data_loader)} batches...")
    
    # Set model to evaluation mode (disables dropout, batch norm training mode)
    model.eval()
    
    # Initialize validation metrics
    cumulative_validation_loss = 0.0
    total_correct_predictions = 0
    total_samples_evaluated = 0
    
    # Disable gradient computation for efficiency during validation
    with torch.no_grad():
        for batch_index, (input_images, target_labels) in enumerate(validation_data_loader):
            # Transfer data to the computing device
            input_images = input_images.to(computing_device)
            target_labels = target_labels.to(computing_device)
            
            # Forward pass: compute model predictions
            model_predictions = model(input_images)
            
            # Calculate validation loss
            batch_loss = loss_criterion(model_predictions, target_labels)
            
            # Accumulate metrics
            cumulative_validation_loss += batch_loss.item()
            _, predicted_class_indices = torch.max(model_predictions.data, 1)
            total_samples_evaluated += target_labels.size(0)
            total_correct_predictions += (predicted_class_indices == target_labels).sum().item()
            
            # Print progress every 5 batches during validation
            if batch_index % 5 == 0 and batch_index > 0:
                current_accuracy = 100.0 * total_correct_predictions / total_samples_evaluated
                print(f'    📈 Validation Batch {batch_index:3d}/{len(validation_data_loader)} | '
                      f'Accuracy: {current_accuracy:6.2f}% | '
                      f'Samples: {total_samples_evaluated:5d}')
            
            # Early termination for demo purposes (remove in production)
            if batch_index >= 10:  # Only validate on first 10 batches for demo
                print(f"    ⚡ Early termination at batch {batch_index} for demo purposes")
                break
    
    # Calculate final validation metrics
    average_validation_loss = cumulative_validation_loss / min(len(validation_data_loader), 11)
    validation_accuracy_percentage = 100.0 * total_correct_predictions / total_samples_evaluated
    
    print(f"    ✅ Validation completed: Loss={average_validation_loss:.4f}, Accuracy={validation_accuracy_percentage:.2f}%")
    
    return average_validation_loss, validation_accuracy_percentage


def compare_models():
    """Main function to compare baseline CNN vs CNN Transfer Learning."""
    
    print("="*80)
    print("CNN BASELINE vs CNN TRANSFER LEARNING COMPARISON")
    print("="*80)
    
    # Set device and seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    set_seed(42)
    
    # Model parameters
    num_classes = 7
    batch_size = 16  # Smaller batch size for demo
    num_epochs = 3   # Fewer epochs for demo
    
    # Create transforms and data loaders
    print("\n1. Setting up data transforms and loaders...")
    transforms_dict = create_transforms()
    data_loaders = create_data_loaders(transforms_dict, batch_size, dataset_size=300)
    
    # Extract data loaders
    train_loader_baseline, val_loader_baseline = data_loaders['baseline']
    train_loader_transfer, val_loader_transfer = data_loaders['transfer']
    
    print(f"✓ Data loaders created (batch size: {batch_size})")
    
    # Create models
    print("\n2. Creating models...")
    
    # Baseline CNN model
    baseline_model = ImprovedCNN(num_classes=num_classes, input_channels=1).to(device)
    baseline_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
    
    # Transfer learning model
    transfer_model = CNNTransferLearning(
        num_classes=num_classes,
        backbone='vgg16',
        pretrained=True,
        freeze_backbone=False  # Fine-tuning
    ).to(device)
    transfer_params = transfer_model.get_num_params()
    
    print(f"\n✓ Models created:")
    print(f"  - Baseline CNN: {baseline_params:,} parameters")
    print(f"  - Transfer Learning CNN: {transfer_params:,} parameters")
    
    # Setup optimizers and loss
    criterion = nn.CrossEntropyLoss()
    
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Different learning rates for transfer learning
    transfer_optimizer = optim.Adam([
        {'params': transfer_model.features.parameters(), 'lr': 1e-5},      # Very small LR for backbone
        {'params': transfer_model.classifier.parameters(), 'lr': 1e-3}     # Normal LR for classifier
    ], weight_decay=1e-4)
    
    print(f"\n3. Training setup:")
    print(f"  - Loss function: CrossEntropyLoss")
    print(f"  - Baseline optimizer: Adam (lr=1e-3)")
    print(f"  - Transfer optimizer: Adam (backbone_lr=1e-5, classifier_lr=1e-3)")
    print(f"  - Training epochs: {num_epochs}")
    
    # Training results storage
    results = {
        'baseline': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []},
        'transfer': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    }
    
    # Training loop comparison
    print(f"\n4. Training comparison:")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("=" * 40)
        
        # Train baseline model
        print("Training Baseline CNN...")
        start_time = time.time()
        train_loss, train_acc = train_model_epoch_with_detailed_logging(
            baseline_model, train_loader_baseline, criterion, baseline_optimizer, device
        )
        val_loss, val_acc = validate_model_with_comprehensive_metrics(
            baseline_model, val_loader_baseline, criterion, device
        )
        baseline_time = time.time() - start_time
        
        results['baseline']['train_loss'].append(train_loss)
        results['baseline']['train_acc'].append(train_acc)
        results['baseline']['val_loss'].append(val_loss)
        results['baseline']['val_acc'].append(val_acc)
        
        print(f"Baseline - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Baseline - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Baseline - Time: {baseline_time:.2f}s")
        
        # Train transfer learning model
        print("\nTraining Transfer Learning CNN...")
        start_time = time.time()
        train_loss, train_acc = train_model_epoch_with_detailed_logging(
            transfer_model, train_loader_transfer, criterion, transfer_optimizer, device
        )
        val_loss, val_acc = validate_model_with_comprehensive_metrics(
            transfer_model, val_loader_transfer, criterion, device
        )
        transfer_time = time.time() - start_time
        
        results['transfer']['train_loss'].append(train_loss)
        results['transfer']['train_acc'].append(train_acc)
        results['transfer']['val_loss'].append(val_loss)
        results['transfer']['val_acc'].append(val_acc)
        
        print(f"Transfer - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Transfer - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Transfer - Time: {transfer_time:.2f}s")
    
    # Final comparison
    print(f"\n" + "="*60)
    print("FINAL COMPARISON RESULTS")
    print("="*60)
    
    baseline_final_acc = results['baseline']['val_acc'][-1]
    transfer_final_acc = results['transfer']['val_acc'][-1]
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"{'Metric':<25} {'Baseline CNN':<15} {'Transfer CNN':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Final Val Accuracy':<25} {baseline_final_acc:<15.2f} {transfer_final_acc:<15.2f} {transfer_final_acc-baseline_final_acc:+.2f}")
    print(f"{'Parameters':<25} {baseline_params:<15,} {transfer_params:<15,} {'+' if transfer_params > baseline_params else ''}{transfer_params-baseline_params:,}")
    
    print(f"\n🎯 KEY DIFFERENCES:")
    print(f"{'Aspect':<20} {'Baseline CNN':<25} {'Transfer Learning CNN':<25}")
    print("-" * 70)
    print(f"{'Input':<20} {'Grayscale (48x48)':<25} {'RGB (224x224)':<25}")
    print(f"{'Architecture':<20} {'Custom CNN from scratch':<25} {'Pre-trained VGG16 + Custom':<25}")
    print(f"{'Initialization':<20} {'Random weights':<25} {'ImageNet pre-trained':<25}")
    print(f"{'Training Strategy':<20} {'Train all layers':<25} {'Fine-tune (different LRs)':<25}")
    print(f"{'Normalization':<20} {'Simple [-1,1]':<25} {'ImageNet statistics':<25}")
    
    print(f"\n✅ TRANSFER LEARNING BENEFITS:")
    print("  1. 🎯 Pre-trained Features: Leverages robust features learned on ImageNet")
    print("  2. 🚀 Faster Convergence: Starts with meaningful weights, not random")
    print("  3. 🎨 Better Generalization: Less prone to overfitting on small datasets") 
    print("  4. 🔧 Flexible Architecture: Can easily swap different pre-trained backbones")
    print("  5. 📈 State-of-the-art: Uses proven CNN architectures (VGG, ResNet, etc.)")
    
    print(f"\n🛠️ WHEN TO USE EACH APPROACH:")
    print("  Baseline CNN:")
    print("    - Small datasets with domain-specific features")
    print("    - Limited computational resources")
    print("    - Educational/research purposes")
    print("    - When interpretability is crucial")
    
    print("  Transfer Learning CNN:")
    print("    - Most practical applications")
    print("    - Limited training data")
    print("    - Need fast development cycle")
    print("    - Want state-of-the-art performance")
    
    # Plot comparison if matplotlib is available
    try:
        plot_comparison(results, num_epochs)
    except ImportError:
        print("\n(Matplotlib not available for plotting)")
    
    print(f"\n✅ Comparison completed successfully!")
    return results


def plot_comparison(results, num_epochs):
    """Plot training comparison results."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, num_epochs + 1)
    
    # Training Loss
    ax1.plot(epochs, results['baseline']['train_loss'], 'b-o', label='Baseline CNN', linewidth=2)
    ax1.plot(epochs, results['transfer']['train_loss'], 'r-s', label='Transfer Learning CNN', linewidth=2)
    ax1.set_title('Training Loss Comparison', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training Accuracy
    ax2.plot(epochs, results['baseline']['train_acc'], 'b-o', label='Baseline CNN', linewidth=2)
    ax2.plot(epochs, results['transfer']['train_acc'], 'r-s', label='Transfer Learning CNN', linewidth=2)
    ax2.set_title('Training Accuracy Comparison', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Validation Loss
    ax3.plot(epochs, results['baseline']['val_loss'], 'b-o', label='Baseline CNN', linewidth=2)
    ax3.plot(epochs, results['transfer']['val_loss'], 'r-s', label='Transfer Learning CNN', linewidth=2)
    ax3.set_title('Validation Loss Comparison', fontsize=14)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Validation Accuracy
    ax4.plot(epochs, results['baseline']['val_acc'], 'b-o', label='Baseline CNN', linewidth=2)
    ax4.plot(epochs, results['transfer']['val_acc'], 'r-s', label='Transfer Learning CNN', linewidth=2)
    ax4.set_title('Validation Accuracy Comparison', fontsize=14)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cnn_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Comparison plots saved as 'cnn_comparison.png'")


def demonstrate_transfer_learning_strategies():
    """Demonstrate different transfer learning strategies."""
    
    print("\n" + "="*80)
    print("TRANSFER LEARNING STRATEGIES DEMONSTRATION")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 7
    
    print("\n1. Feature Extraction (Frozen Backbone)")
    print("-" * 50)
    
    model_frozen = CNNTransferLearning(
        num_classes=num_classes,
        backbone='vgg16', 
        pretrained=True,
        freeze_backbone=True
    ).to(device)
    
    frozen_trainable = model_frozen.get_num_params(trainable_only=True)
    frozen_total = model_frozen.get_num_params(trainable_only=False)
    
    print(f"✓ Frozen model created:")
    print(f"  - Total parameters: {frozen_total:,}")
    print(f"  - Trainable parameters: {frozen_trainable:,}")
    print(f"  - Frozen parameters: {frozen_total - frozen_trainable:,}")
    print(f"  - Strategy: Only train classifier layers")
    
    print("\n2. Fine-tuning (All Layers Trainable)")
    print("-" * 50)
    
    model_finetuned = CNNTransferLearning(
        num_classes=num_classes,
        backbone='vgg16',
        pretrained=True,
        freeze_backbone=False
    ).to(device)
    
    finetuned_trainable = model_finetuned.get_num_params(trainable_only=True)
    
    print(f"✓ Fine-tuned model created:")
    print(f"  - Total parameters: {finetuned_trainable:,}")
    print(f"  - Trainable parameters: {finetuned_trainable:,}")
    print(f"  - Strategy: Train all layers with different learning rates")
    
    print("\n3. Gradual Unfreezing (Progressive Training)")
    print("-" * 50)
    
    model_gradual = CNNTransferLearning(
        num_classes=num_classes,
        backbone='vgg16',
        pretrained=True,
        freeze_backbone=True  # Start frozen
    ).to(device)
    
    print(f"✓ Gradual unfreezing model created:")
    print(f"  - Phase 1: Train classifier only (backbone frozen)")
    print(f"  - Phase 2: Unfreeze and fine-tune all layers")
    
    # Demonstrate unfreezing
    print("  - Unfreezing backbone...")
    model_gradual.unfreeze_backbone()
    
    print("\n📋 STRATEGY COMPARISON:")
    print(f"{'Strategy':<20} {'Trainable Params':<15} {'Best For':<30}")
    print("-" * 65)
    print(f"{'Feature Extraction':<20} {frozen_trainable:<15,} {'Small datasets, fast training':<30}")
    print(f"{'Fine-tuning':<20} {finetuned_trainable:<15,} {'Best performance, more data':<30}")
    print(f"{'Gradual Unfreezing':<20} {'Progressive':<15} {'Careful training, stability':<30}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("  🎯 Start with Feature Extraction for quick prototyping")
    print("  🚀 Use Fine-tuning for best performance when you have sufficient data")
    print("  🔄 Try Gradual Unfreezing when fine-tuning is unstable")
    print("  📊 Always validate on held-out data to choose the best strategy")


if __name__ == "__main__":
    print("🎉 CNN Transfer Learning Demonstration")
    print("="*50)
    
    try:
        # Main comparison
        results = compare_models()
        
        # Demonstrate different strategies
        demonstrate_transfer_learning_strategies()
        
        print(f"\n🎯 SUMMARY:")
        print("This demonstration showed how CNN Transfer Learning:")
        print("✅ Leverages pre-trained features from large datasets (ImageNet)")
        print("✅ Often achieves better performance than training from scratch") 
        print("✅ Provides multiple training strategies for different scenarios")
        print("✅ Uses proven architectures and established best practices")
        print("\n🚀 Next steps: Replace dummy data with real emotion dataset for actual results!")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print("This might be due to missing dependencies or insufficient resources.")
        import traceback
        traceback.print_exc()