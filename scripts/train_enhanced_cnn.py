#!/usr/bin/env python3
"""
Enhanced CNN Transfer Learning Training Script

This script trains CNN transfer learning models on enhanced emotion datasets,
comparing performance between original 48x48 images and generative AI enhanced 224x224 images.

Usage:
    python scripts/train_enhanced_cnn.py --enhanced_data data/enhanced --original_data data/raw
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.cnn_transfer_learning import CNNTransferLearning, create_cnn_transfer_model
from data.dataset_emotion import EmotionDataset
from utils.seed import set_seed


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CNN Transfer Learning with Enhanced Emotion Dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--enhanced_data', type=str,
                       default='data/enhanced',
                       help='Directory containing enhanced images')
    
    parser.add_argument('--original_data', type=str,
                       default='data/raw/EmoSet',
                       help='Directory containing original images')
    
    parser.add_argument('--enhanced_splits', type=str,
                       default='data/enhanced/enhanced_splits',
                       help='Directory containing enhanced dataset splits')
    
    parser.add_argument('--original_splits', type=str,
                       default='data/processed/EmoSet_splits',
                       help='Directory containing original dataset splits')
    
    parser.add_argument('--use_enhanced', action='store_true',
                       help='Use enhanced dataset instead of original')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='vgg16',
                       choices=['vgg16', 'vgg19', 'alexnet'],
                       help='Backbone architecture to use')
    
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone weights (feature extraction mode)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    
    parser.add_argument('--backbone_lr', type=float, default=1e-5,
                       help='Learning rate for backbone layers')
    
    parser.add_argument('--classifier_lr', type=float, default=1e-3,
                       help='Learning rate for classifier layers')
    
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimization')
    
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    
    parser.add_argument('--model_name', type=str,
                       help='Custom model name (auto-generated if not provided)')
    
    parser.add_argument('--compare_both', action='store_true',
                       help='Train and compare both original and enhanced datasets')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup and return the appropriate device."""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f} GB")
    
    return device


def setup_data_transforms(enhanced_data=True, input_size=224):
    """
    Setup data transforms for training.
    
    Args:
        enhanced_data (bool): Whether using enhanced data (already 224x224) or original (48x48)
        input_size (int): Target input size for the model
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    # ImageNet normalization values (required for transfer learning)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    if enhanced_data:
        # Enhanced data is already 224x224 and high quality
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),  # Less rotation since quality is better
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # Original data needs more aggressive augmentation due to upscaling
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05)),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size), 
                         interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return train_transform, val_transform


def load_data_and_create_loaders(data_config, train_transform, val_transform, batch_size, num_workers):
    """Load data and create data loaders."""
    # Load label map
    label_map_path = Path(data_config['splits_dir']) / 'label_map.json'
    if label_map_path.exists():
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
    else:
        # Create default label map
        label_map = {
            'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
            'neutral': 4, 'sad': 5, 'surprise': 6
        }
    
    num_classes = len(label_map)
    
    # Load split dataframes
    splits_dir = Path(data_config['splits_dir'])
    train_df = pd.read_csv(splits_dir / 'train.csv')
    val_df = pd.read_csv(splits_dir / 'val.csv')
    test_df = pd.read_csv(splits_dir / 'test.csv')
    
    # Create datasets
    train_dataset = EmotionDataset(
        train_df, data_config['data_dir'], 
        transform=train_transform, label_map=label_map, rgb=True
    )
    
    val_dataset = EmotionDataset(
        val_df, data_config['data_dir'], 
        transform=val_transform, label_map=label_map, rgb=True
    )
    
    test_dataset = EmotionDataset(
        test_df, data_config['data_dir'], 
        transform=val_transform, label_map=label_map, rgb=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Data loaded:")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Validation samples: {len(val_dataset)}")
    print(f"- Test samples: {len(test_dataset)}")
    print(f"- Number of classes: {num_classes}")
    print(f"- Classes: {list(label_map.keys())}")
    
    return train_loader, val_loader, test_loader, num_classes, label_map


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
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
        
        if batch_idx % 100 == 0 and batch_idx > 0:
            print(f'    Batch {batch_idx}/{len(train_loader)} - '
                  f'Loss: {running_loss/(batch_idx+1):.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate the model."""
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


def train_model(config, device):
    """Train a single model with the given configuration."""
    print(f"\n{'='*80}")
    print(f"TRAINING: {config['name']}")
    print(f"{'='*80}")
    print(f"Dataset: {config['data_type']} ({config['data_config']['data_dir']})")
    print(f"Backbone: {config['model']['backbone']}")
    print(f"Frozen backbone: {config['model']['freeze_backbone']}")
    
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Setup data transforms
    train_transform, val_transform = setup_data_transforms(
        enhanced_data=(config['data_type'] == 'enhanced'),
        input_size=224
    )
    
    # Load data and create loaders
    train_loader, val_loader, test_loader, num_classes, label_map = load_data_and_create_loaders(
        config['data_config'], train_transform, val_transform, 
        config['batch_size'], config['num_workers']
    )
    
    # Create model
    model = create_cnn_transfer_model(
        num_classes=num_classes,
        backbone=config['model']['backbone'],
        pretrained=True,
        freeze_backbone=config['model']['freeze_backbone'],
        device=device
    )
    
    # Setup loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if config['model']['freeze_backbone']:
        # Only train classifier layers
        optimizer = optim.Adam(model.classifier.parameters(), 
                              lr=config['classifier_lr'],
                              weight_decay=config['weight_decay'])
        print(f"Training classifier only with LR: {config['classifier_lr']}")
    else:
        # Train with different learning rates for backbone and classifier
        optimizer = optim.Adam([
            {'params': model.features.parameters(), 'lr': config['backbone_lr']},
            {'params': model.classifier.parameters(), 'lr': config['classifier_lr']}
        ], weight_decay=config['weight_decay'])
        print(f"Fine-tuning: Backbone LR: {config['backbone_lr']}, "
              f"Classifier LR: {config['classifier_lr']}")
    
    # Setup scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        
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
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            model_path = config['output']['model_path']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config,
                'label_map': label_map
            }, model_path)
            print(f"✓ New best model saved! Val Acc: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config['patience']}")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    training_time = time.time() - start_time
    
    # Final evaluation on test set
    print(f"\n{'='*50}")
    print(f"FINAL EVALUATION")
    print(f"{'='*50}")
    
    # Load best model
    checkpoint = torch.load(config['output']['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test set evaluation
    test_loss, test_acc, test_predictions, test_targets = validate_epoch(
        model, test_loader, criterion, device)
    
    print(f"Test Results:")
    print(f"- Test Loss: {test_loss:.4f}")
    print(f"- Test Accuracy: {test_acc:.2f}%")
    print(f"- Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"- Training Time: {training_time:.1f}s ({training_time/60:.1f} minutes)")
    
    # Classification report
    emotion_names = list(label_map.keys())
    report = classification_report(test_targets, test_predictions, 
                                 target_names=emotion_names, zero_division=0,
                                 output_dict=True)
    
    print(f"\nClassification Report:")
    print(classification_report(test_targets, test_predictions, 
                              target_names=emotion_names, zero_division=0))
    
    # Save results
    results = {
        'config': config,
        'training_time_seconds': training_time,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'classification_report': report,
        'test_predictions': test_predictions,
        'test_targets': test_targets
    }
    
    # Save detailed results
    results_path = config['output']['results_path']
    with open(results_path, 'w') as f:
        # Remove non-serializable items for JSON
        json_safe_results = results.copy()
        json_safe_results.pop('test_predictions', None)
        json_safe_results.pop('test_targets', None)
        json.dump(json_safe_results, f, indent=2, default=str)
    
    print(f"✓ Results saved to: {results_path}")
    
    return results


def plot_training_comparison(results_list, output_dir):
    """Plot comparison of training metrics between different models."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot training loss
    ax = axes[0, 0]
    for results in results_list:
        ax.plot(results['train_losses'], label=f"{results['config']['name']} (Train)")
        ax.plot(results['val_losses'], label=f"{results['config']['name']} (Val)", linestyle='--')
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax = axes[0, 1]
    for results in results_list:
        ax.plot(results['train_accuracies'], label=f"{results['config']['name']} (Train)")
        ax.plot(results['val_accuracies'], label=f"{results['config']['name']} (Val)", linestyle='--')
    ax.set_title('Training and Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot final test accuracy comparison
    ax = axes[1, 0]
    names = [r['config']['name'] for r in results_list]
    test_accs = [r['test_acc'] for r in results_list]
    bars = ax.bar(names, test_accs, color=['skyblue', 'lightcoral'])
    ax.set_title('Final Test Accuracy Comparison')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, test_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom')
    
    # Plot training time comparison
    ax = axes[1, 1]
    training_times = [r['training_time_seconds']/60 for r in results_list]  # Convert to minutes
    bars = ax.bar(names, training_times, color=['lightgreen', 'lightsalmon'])
    ax.set_title('Training Time Comparison')
    ax.set_ylabel('Training Time (minutes)')
    
    # Add value labels on bars
    for bar, time_min in zip(bars, training_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{time_min:.1f}m', ha='center', va='bottom')
    
    plt.tight_layout()
    plot_path = Path(output_dir) / 'training_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training comparison plot saved to: {plot_path}")


def create_summary_report(results_list, output_dir):
    """Create a comprehensive summary report."""
    report_path = Path(output_dir) / 'summary_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Enhanced CNN Transfer Learning Results\n\n")
        f.write("## Summary\n\n")
        f.write("This report compares the performance of CNN Transfer Learning models trained on:\n")
        f.write("- **Original Dataset**: 48x48 grayscale images upsampled using bicubic interpolation\n")
        f.write("- **Enhanced Dataset**: 48x48 images enhanced to 224x224 using generative AI (Enhanced SRCNN)\n\n")
        
        f.write("## Results Comparison\n\n")
        f.write("| Model | Test Accuracy | Training Time | Best Val Accuracy | Data Type |\n")
        f.write("|-------|---------------|---------------|------------------|----|\n")
        
        for results in results_list:
            config = results['config']
            f.write(f"| {config['name']} | {results['test_acc']:.2f}% | "
                   f"{results['training_time_seconds']/60:.1f}m | "
                   f"{results['best_val_acc']:.2f}% | {config['data_type']} |\n")
        
        # Calculate improvement if we have both original and enhanced results
        if len(results_list) == 2:
            enhanced_result = next((r for r in results_list if r['config']['data_type'] == 'enhanced'), None)
            original_result = next((r for r in results_list if r['config']['data_type'] == 'original'), None)
            
            if enhanced_result and original_result:
                acc_improvement = enhanced_result['test_acc'] - original_result['test_acc']
                f.write(f"\n### Performance Improvement\n\n")
                f.write(f"- **Accuracy Improvement**: {acc_improvement:+.2f}%\n")
                f.write(f"- **Relative Improvement**: {(acc_improvement/original_result['test_acc'])*100:+.1f}%\n\n")
        
        f.write("\n## Model Configurations\n\n")
        for results in results_list:
            config = results['config']
            f.write(f"### {config['name']}\n\n")
            f.write(f"- **Backbone**: {config['model']['backbone']}\n")
            f.write(f"- **Frozen Backbone**: {config['model']['freeze_backbone']}\n")
            f.write(f"- **Data Directory**: {config['data_config']['data_dir']}\n")
            f.write(f"- **Batch Size**: {config['batch_size']}\n")
            f.write(f"- **Backbone LR**: {config['backbone_lr']}\n")
            f.write(f"- **Classifier LR**: {config['classifier_lr']}\n\n")
        
        f.write("## Conclusion\n\n")
        if len(results_list) == 2 and enhanced_result and original_result:
            if acc_improvement > 0:
                f.write(f"The generative AI enhanced dataset shows a **{acc_improvement:.2f}% improvement** "
                       f"in test accuracy compared to the original dataset. This demonstrates the effectiveness "
                       f"of using generative AI techniques for image enhancement in emotion recognition tasks.\n\n")
            else:
                f.write(f"The results show that the generative AI enhancement may not have provided "
                       f"significant improvement for this specific task. Further investigation into "
                       f"enhancement parameters or alternative techniques may be beneficial.\n\n")
        
        f.write("The enhanced dataset preprocessing pipeline successfully converts 48x48 grayscale images "
               "to 224x224 high-quality images suitable for transfer learning with pre-trained CNN models.\n")
    
    print(f"✓ Summary report saved to: {report_path}")


def main():
    """Main function to train and compare CNN models."""
    args = parse_arguments()
    
    print("Enhanced CNN Transfer Learning Training")
    print("=" * 60)
    
    # Setup
    device = setup_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_list = []
    
    # Determine which datasets to train on
    datasets_to_train = []
    
    if args.compare_both:
        # Train on both original and enhanced datasets
        datasets_to_train = [
            {
                'name': f'Original_{args.backbone}',
                'data_type': 'original',
                'data_dir': args.original_data,
                'splits_dir': args.original_splits
            },
            {
                'name': f'Enhanced_{args.backbone}',
                'data_type': 'enhanced',
                'data_dir': args.enhanced_data,
                'splits_dir': args.enhanced_splits
            }
        ]
    elif args.use_enhanced:
        # Train only on enhanced dataset
        datasets_to_train = [{
            'name': f'Enhanced_{args.backbone}',
            'data_type': 'enhanced',
            'data_dir': args.enhanced_data,
            'splits_dir': args.enhanced_splits
        }]
    else:
        # Train only on original dataset
        datasets_to_train = [{
            'name': f'Original_{args.backbone}',
            'data_type': 'original',
            'data_dir': args.original_data,
            'splits_dir': args.original_splits
        }]
    
    # Train models
    for dataset_config in datasets_to_train:
        model_name = args.model_name or dataset_config['name']
        
        # Create training configuration
        config = {
            'name': model_name,
            'data_type': dataset_config['data_type'],
            'seed': args.seed,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'epochs': args.epochs,
            'backbone_lr': args.backbone_lr,
            'classifier_lr': args.classifier_lr,
            'weight_decay': args.weight_decay,
            'patience': args.patience,
            'data_config': {
                'data_dir': dataset_config['data_dir'],
                'splits_dir': dataset_config['splits_dir']
            },
            'model': {
                'backbone': args.backbone,
                'freeze_backbone': args.freeze_backbone
            },
            'output': {
                'model_path': output_dir / f'{model_name}.pth',
                'results_path': output_dir / f'{model_name}_results.json'
            }
        }
        
        # Train model
        results = train_model(config, device)
        results_list.append(results)
    
    # Create comparison plots and reports if multiple models were trained
    if len(results_list) > 1:
        print(f"\n{'='*60}")
        print("CREATING COMPARISON REPORTS")
        print(f"{'='*60}")
        
        plot_training_comparison(results_list, output_dir)
        create_summary_report(results_list, output_dir)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    
    for results in results_list:
        config = results['config']
        print(f"✓ {config['name']}: {results['test_acc']:.2f}% test accuracy")
        print(f"  Model saved: {config['output']['model_path']}")
        print(f"  Results saved: {config['output']['results_path']}")
    
    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()