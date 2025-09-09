# Training script for CNN Transfer Learning
import os
import sys
import json
from pathlib import Path
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.cnn_transfer_learning import CNNTransferLearning, create_cnn_transfer_model
from data.dataset_emotion import EmotionDataset
from utils.seed import set_seed


def setup_data_transforms(input_size=224):
    """
    Setup data transforms for transfer learning.
    
    Args:
        input_size (int): Input image size (default 224 for VGG/AlexNet)
        
    Returns:
        tuple: (train_transform, val_transform)
    """
    # ImageNet normalization values (required for transfer learning)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Validation/test transforms without augmentation
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return train_transform, val_transform


def create_data_loaders(data_config, train_transform, val_transform, batch_size=32):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_config (dict): Data configuration with paths
        train_transform: Training transforms
        val_transform: Validation transforms
        batch_size (int): Batch size for data loaders
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes, label_map)
    """
    # Load data paths
    train_csv = data_config['train_csv']
    val_csv = data_config['val_csv']
    test_csv = data_config['test_csv']
    data_dir = data_config['data_dir']
    label_map_path = data_config['label_map_path']
    
    # Load label map
    if Path(label_map_path).exists():
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
    else:
        # Create default label map for common emotions
        label_map = {
            'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3,
            'sadness': 4, 'surprise': 5, 'neutral': 6
        }
    
    num_classes = len(label_map)
    
    # Load datasets
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    # Create datasets (RGB for transfer learning)
    train_dataset = EmotionDataset(train_df, data_dir, transform=train_transform, 
                                  label_map=label_map, rgb=True)
    val_dataset = EmotionDataset(val_df, data_dir, transform=val_transform, 
                                label_map=label_map, rgb=True)
    test_dataset = EmotionDataset(test_df, data_dir, transform=val_transform, 
                                 label_map=label_map, rgb=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    print(f"Data loaded successfully:")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Validation samples: {len(val_dataset)}")
    print(f"- Test samples: {len(test_dataset)}")
    print(f"- Number of classes: {num_classes}")
    print(f"- Classes: {list(label_map.keys())}")
    
    return train_loader, val_loader, test_loader, num_classes, label_map


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, print_freq=100):
    """
    Train the model for one epoch.
    
    Args:
        model: The CNN transfer learning model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        print_freq: How often to print progress
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if batch_idx % print_freq == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print(f'    Epoch {epoch} [{batch_idx:4d}/{len(train_loader)}] '
                  f'Loss: {running_loss/(batch_idx+1):.4f} '
                  f'Acc: {100.*correct/total:6.2f}% '
                  f'Time: {elapsed:.1f}s')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: The CNN transfer learning model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        tuple: (average_loss, accuracy, predictions, targets)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_predictions, all_targets


def train_cnn_transfer_learning(config):
    """
    Main training function for CNN transfer learning.
    
    Args:
        config (dict): Training configuration
    """
    print("="*80)
    print("CNN TRANSFER LEARNING TRAINING")
    print("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Setup data transforms
    train_transform, val_transform = setup_data_transforms(config['input_size'])
    
    # Create data loaders
    train_loader, val_loader, test_loader, num_classes, label_map = create_data_loaders(
        config['data'], train_transform, val_transform, config['batch_size']
    )
    
    # Create model
    model = create_cnn_transfer_model(
        num_classes=num_classes,
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        freeze_backbone=config['model']['freeze_backbone'],
        device=device
    )
    
    # Setup loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if config['model']['freeze_backbone']:
        # Only train classifier layers
        optimizer = optim.Adam(model.classifier.parameters(), 
                              lr=config['training']['classifier_lr'],
                              weight_decay=config['training']['weight_decay'])
        print(f"Training classifier only with LR: {config['training']['classifier_lr']}")
    else:
        # Train with different learning rates for backbone and classifier
        optimizer = optim.Adam([
            {'params': model.features.parameters(), 'lr': config['training']['backbone_lr']},
            {'params': model.classifier.parameters(), 'lr': config['training']['classifier_lr']}
        ], weight_decay=config['training']['weight_decay'])
        print(f"Fine-tuning: Backbone LR: {config['training']['backbone_lr']}, "
              f"Classifier LR: {config['training']['classifier_lr']}")
    
    # Setup scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                         step_size=config['training']['scheduler_step_size'], 
                                         gamma=config['training']['scheduler_gamma'])
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    print(f"Early stopping patience: {config['training']['patience']}")
    print("-" * 80)
    
    for epoch in range(config['training']['epochs']):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, 
                                          device, epoch+1)
        
        # Validate
        val_loss, val_acc, _, _ = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Epoch time: {epoch_time:.1f}s")
        
        if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 1:
            print(f"LR - Backbone: {optimizer.param_groups[0]['lr']:.2e}, "
                  f"Classifier: {optimizer.param_groups[1]['lr']:.2e}")
        else:
            print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            save_path = config['output']['model_path']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config,
                'label_map': label_map,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies
            }, save_path)
            print(f"✓ New best model saved! Val Acc: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config['training']['patience']}")
        
        # Early stopping
        if patience_counter >= config['training']['patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Final evaluation on test set
    print(f"\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    # Load best model
    if Path(config['output']['model_path']).exists():
        checkpoint = torch.load(config['output']['model_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Test set evaluation
    test_loss, test_acc, test_predictions, test_targets = validate_epoch(
        model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"- Test Loss: {test_loss:.4f}")
    print(f"- Test Accuracy: {test_acc:.2f}%")
    print(f"- Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Detailed classification report
    emotion_names = list(label_map.keys())
    if len(set(test_targets)) > 1:
        print(f"\nClassification Report:")
        report = classification_report(test_targets, test_predictions, 
                                     target_names=emotion_names, zero_division=0)
        print(report)
        
        # Save classification report
        report_path = config['output']['results_path']
        with open(report_path, 'w') as f:
            f.write(f"CNN Transfer Learning Results\n")
            f.write(f"============================\n\n")
            f.write(f"Model Configuration:\n")
            f.write(f"- Backbone: {config['model']['backbone']}\n")
            f.write(f"- Pretrained: {config['model']['pretrained']}\n")
            f.write(f"- Frozen: {config['model']['freeze_backbone']}\n")
            f.write(f"- Classes: {num_classes}\n\n")
            f.write(f"Results:\n")
            f.write(f"- Test Accuracy: {test_acc:.2f}%\n")
            f.write(f"- Best Val Accuracy: {best_val_acc:.2f}%\n\n")
            f.write(f"Classification Report:\n")
            f.write(report)
        
        print(f"✓ Results saved to: {report_path}")
    
    print(f"\n✓ Training completed successfully!")
    print(f"✓ Best model saved to: {config['output']['model_path']}")


def main():
    """Main function to run CNN transfer learning training."""
    parser = argparse.ArgumentParser(description='Train CNN Transfer Learning for Emotion Recognition')
    parser.add_argument('--config', type=str, default='config_cnn_transfer.json',
                       help='Path to configuration file')
    parser.add_argument('--backbone', type=str, default='vgg16',
                       choices=['vgg16', 'vgg19', 'alexnet'],
                       help='Backbone architecture to use')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--freeze', action='store_true',
                       help='Freeze backbone weights (feature extraction mode)')
    
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'seed': 42,
        'input_size': 224,
        'batch_size': args.batch_size,
        'data': {
            'train_csv': 'data/processed/EmoSet_splits/train.csv',
            'val_csv': 'data/processed/EmoSet_splits/val.csv',
            'test_csv': 'data/processed/EmoSet_splits/test.csv',
            'data_dir': 'data/raw/EmoSet',
            'label_map_path': 'data/processed/EmoSet_splits/label_map.json'
        },
        'model': {
            'backbone': args.backbone,
            'pretrained': True,
            'freeze_backbone': args.freeze
        },
        'training': {
            'epochs': args.epochs,
            'backbone_lr': 1e-5,
            'classifier_lr': 1e-3,
            'weight_decay': 1e-4,
            'scheduler_step_size': 10,
            'scheduler_gamma': 0.5,
            'patience': 5
        },
        'output': {
            'model_path': f'models/cnn_transfer_{args.backbone}.pth',
            'results_path': f'results/cnn_transfer_{args.backbone}_results.txt'
        }
    }
    
    # Load config file if exists
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)
        print(f"Loaded configuration from: {args.config}")
    
    # Create output directories
    os.makedirs(os.path.dirname(config['output']['model_path']), exist_ok=True)
    os.makedirs(os.path.dirname(config['output']['results_path']), exist_ok=True)
    
    # Start training
    train_cnn_transfer_learning(config)


if __name__ == "__main__":
    main()