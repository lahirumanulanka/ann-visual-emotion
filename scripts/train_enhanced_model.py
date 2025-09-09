#!/usr/bin/env python3
"""
Enhanced CNN Transfer Learning Training Pipeline
Trains emotion recognition models using AI-enhanced 224x224 images
"""

import sys
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.cnn_transfer_learning import create_cnn_transfer_model
from data.enhanced_dataset import EnhancedEmotionDataset, create_enhanced_dataloader
from utils.metrics import calculate_metrics
import mlflow
import mlflow.pytorch


def create_transforms(image_size=224, mode='train'):
    """Create transforms for enhanced images."""
    
    # ImageNet normalization values
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    if mode == 'train':
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # Validation/test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    return transform


def load_data(data_config, enhanced_data_dir, original_data_dir=None):
    """Load and prepare data loaders."""
    
    # Load label mapping
    with open(data_config['label_map_path'], 'r') as f:
        label_map = json.load(f)
    
    # Create transforms
    train_transform = create_transforms(mode='train')
    val_transform = create_transforms(mode='val')
    
    # Create data loaders
    train_loader = create_enhanced_dataloader(
        csv_path=data_config['train_csv'],
        enhanced_data_dir=enhanced_data_dir,
        original_data_dir=original_data_dir,
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        shuffle=True,
        transform=train_transform,
        label_map=label_map
    )
    
    val_loader = create_enhanced_dataloader(
        csv_path=data_config['val_csv'],
        enhanced_data_dir=enhanced_data_dir,
        original_data_dir=original_data_dir,
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        shuffle=False,
        transform=val_transform,
        label_map=label_map
    )
    
    test_loader = create_enhanced_dataloader(
        csv_path=data_config['test_csv'],
        enhanced_data_dir=enhanced_data_dir,
        original_data_dir=original_data_dir,
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        shuffle=False,
        transform=val_transform,
        label_map=label_map
    ) if data_config.get('test_csv') else None
    
    return train_loader, val_loader, test_loader, label_map


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if batch_idx % 50 == 0:
            logging.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, '
                        f'Acc: {100*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy, all_predictions, all_labels


def train_enhanced_model(config):
    """Main training function for enhanced emotion recognition model."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Load data
    logging.info("Loading enhanced dataset...")
    train_loader, val_loader, test_loader, label_map = load_data(
        config['data'], 
        config['enhanced_data_dir'],
        config.get('original_data_dir')
    )
    
    num_classes = len(label_map)
    logging.info(f"Number of classes: {num_classes}")
    logging.info(f"Label mapping: {label_map}")
    
    # Create model
    logging.info("Creating enhanced CNN transfer learning model...")
    model = create_cnn_transfer_model(
        num_classes=num_classes,
        backbone=config['model']['backbone'],
        pretrained=True,
        freeze_backbone=config['model']['freeze_backbone'],
        device=device
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if config['model']['freeze_backbone']:
        # Only train classifier
        optimizer = optim.Adam(model.classifier.parameters(), 
                             lr=config['training']['learning_rate'])
    else:
        # Different learning rates for backbone and classifier
        optimizer = optim.Adam([
            {'params': model.features.parameters(), 'lr': config['training']['learning_rate'] * 0.1},
            {'params': model.classifier.parameters(), 'lr': config['training']['learning_rate']}
        ])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    logging.info("Starting training...")
    
    for epoch in range(1, config['training']['epochs'] + 1):
        logging.info(f"Epoch {epoch}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Step scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config['output_dir'] / 'best_enhanced_model.pth')
            logging.info(f"New best validation accuracy: {best_val_acc:.2f}%")
        
        # Log to MLflow
        if config.get('use_mlflow', False):
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, step=epoch)
    
    # Test on best model if test set available
    if test_loader:
        logging.info("Evaluating on test set...")
        model.load_state_dict(torch.load(config['output_dir'] / 'best_enhanced_model.pth'))
        test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
        logging.info(f"Test Accuracy: {test_acc:.2f}%")
        
        # Generate classification report
        class_names = list(label_map.keys())
        report = classification_report(test_labels, test_preds, target_names=class_names)
        logging.info(f"Classification Report:\n{report}")
        
        # Save results
        results = {
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'classification_report': report
        }
        
        with open(config['output_dir'] / 'enhanced_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    logging.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced CNN Transfer Learning Model")
    
    parser.add_argument('--enhanced-data-dir', type=str, required=True,
                       help='Path to enhanced 224x224 dataset')
    parser.add_argument('--original-data-dir', type=str, 
                       help='Path to original dataset (fallback)')
    parser.add_argument('--train-csv', type=str, required=True,
                       help='Path to training CSV')
    parser.add_argument('--val-csv', type=str, required=True,
                       help='Path to validation CSV')
    parser.add_argument('--test-csv', type=str, 
                       help='Path to test CSV')
    parser.add_argument('--label-map', type=str, required=True,
                       help='Path to label mapping JSON')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--backbone', type=str, default='vgg16',
                       choices=['vgg16', 'vgg19', 'alexnet'],
                       help='CNN backbone to use')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Freeze backbone weights')
    
    args = parser.parse_args()
    
    # Create config
    config = {
        'enhanced_data_dir': args.enhanced_data_dir,
        'original_data_dir': args.original_data_dir,
        'output_dir': Path(args.output_dir),
        'data': {
            'train_csv': args.train_csv,
            'val_csv': args.val_csv,
            'test_csv': args.test_csv,
            'label_map_path': args.label_map,
            'batch_size': args.batch_size,
            'num_workers': 4
        },
        'model': {
            'backbone': args.backbone,
            'freeze_backbone': args.freeze_backbone
        },
        'training': {
            'epochs': args.epochs,
            'learning_rate': args.lr
        }
    }
    
    # Create output directory
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    
    # Train model
    model, best_acc = train_enhanced_model(config)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {config['output_dir'] / 'best_enhanced_model.pth'}")


if __name__ == "__main__":
    main()