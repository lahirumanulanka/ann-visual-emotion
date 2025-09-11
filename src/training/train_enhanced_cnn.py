# Enhanced training script for improved CNN Transfer Learning
import os
import sys
import json
from pathlib import Path
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.improved_cnn_transfer import (
    ImprovedCNNTransferLearning, 
    create_improved_model, 
    LabelSmoothingCrossEntropy, 
    FocalLoss
)
from data.dataset_emotion import EmotionDataset
from utils.seed import set_seed


class AlbumentationsDataset(EmotionDataset):
    """
    Enhanced dataset class using Albumentations for better data augmentation.
    """
    
    def __init__(self, dataframe, root_dir, transform=None, label_map=None, rgb=True):
        super().__init__(dataframe, root_dir, transform=None, label_map=label_map, rgb=rgb)
        self.albumentations_transform = transform
        self.force_rgb = rgb
    
    def __getitem__(self, idx):
        from PIL import Image
        import numpy as np
        
        row = self.df.iloc[idx]
        rel_path = row[self.path_col]
        label = row[self.label_col]
        
        if self.label_map and isinstance(label, str):
            label_idx = self.label_map[label]
        else:
            label_idx = int(label)
        
        img_path = self.root_dir / rel_path
        try:
            if self.force_rgb:
                image = Image.open(img_path).convert('RGB')
            else:
                image = Image.open(img_path).convert('L')
                image = image.convert('RGB')  # Convert grayscale to RGB for consistency
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Convert to numpy array for Albumentations
        image = np.array(image)
        
        # Apply Albumentations transforms
        if self.albumentations_transform:
            transformed = self.albumentations_transform(image=image)
            image = transformed['image']
        
        return image, label_idx


def create_enhanced_transforms(input_size=224, is_training=True):
    """
    Create enhanced data transforms using Albumentations.
    
    Args:
        input_size (int): Target image size
        is_training (bool): Whether to apply training augmentations
        
    Returns:
        Albumentations transform pipeline
    """
    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    if is_training:
        return A.Compose([
            A.Resize(input_size, input_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Rotate(limit=20, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=2, p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.GaussianBlur(blur_limit=(3, 3), p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=input_size//8, max_width=input_size//8, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])


def create_enhanced_data_loaders(data_config, input_size=224, batch_size=32, use_balanced=True):
    """
    Create enhanced data loaders with class balancing and advanced augmentation.
    """
    # Load data paths
    if use_balanced:
        train_csv = data_config['train_csv'].replace('.csv', '_balanced.csv')
        test_csv = data_config['test_csv'].replace('.csv', '_balanced.csv')
    else:
        train_csv = data_config['train_csv']
        test_csv = data_config['test_csv']
    
    val_csv = data_config['val_csv']
    data_dir = data_config['data_dir']
    label_map_path = data_config['label_map_path']
    
    # Load label map
    if Path(label_map_path).exists():
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
    else:
        label_map = {
            'angry': 0, 'fearful': 1, 'happy': 2, 'neutral': 3, 'sad': 4, 'surprised': 5
        }
    
    num_classes = len(label_map)
    
    # Load datasets
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    # Create transforms
    train_transform = create_enhanced_transforms(input_size, is_training=True)
    val_transform = create_enhanced_transforms(input_size, is_training=False)
    
    # Create datasets using Albumentations
    train_dataset = AlbumentationsDataset(train_df, data_dir, transform=train_transform, 
                                         label_map=label_map, rgb=True)
    val_dataset = AlbumentationsDataset(val_df, data_dir, transform=val_transform, 
                                       label_map=label_map, rgb=True)
    test_dataset = AlbumentationsDataset(test_df, data_dir, transform=val_transform, 
                                        label_map=label_map, rgb=True)
    
    # Calculate class weights for balanced training
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.FloatTensor(class_weights)
    
    # Create weighted sampler for balanced training
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    print(f"Enhanced data loaded successfully:")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Validation samples: {len(val_dataset)}")
    print(f"- Test samples: {len(test_dataset)}")
    print(f"- Number of classes: {num_classes}")
    print(f"- Classes: {list(label_map.keys())}")
    print(f"- Class weights: {class_weights.tolist()}")
    print(f"- Using balanced sampling: True")
    
    return train_loader, val_loader, test_loader, num_classes, label_map, class_weights


def train_epoch_enhanced(model, train_loader, criterion, optimizer, device, epoch, 
                        gradient_accumulation_steps=1, max_grad_norm=1.0, print_freq=50):
    """
    Enhanced training epoch with gradient accumulation and clipping.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    optimizer.zero_grad()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        output = model(data)
        loss = criterion(output, target)
        
        # Gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * gradient_accumulation_steps
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if batch_idx % print_freq == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print(f'    Epoch {epoch} [{batch_idx:4d}/{len(train_loader)}] '
                  f'Loss: {running_loss/(batch_idx+1):.4f} '
                  f'Acc: {100.*correct/total:6.2f}% '
                  f'Time: {elapsed:.1f}s')
    
    # Handle remaining gradients
    if len(train_loader) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch_enhanced(model, val_loader, criterion, device):
    """
    Enhanced validation with detailed metrics.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            loss = criterion(output, target)
            
            # Get probabilities
            probabilities = torch.softmax(output, dim=1)
            all_probabilities.extend(probabilities.cpu().numpy())
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_predictions, all_targets, all_probabilities


def train_enhanced_cnn_transfer_learning(config):
    """
    Main enhanced training function with all improvements.
    """
    print("="*80)
    print("ENHANCED CNN TRANSFER LEARNING TRAINING")
    print("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    
    # Set seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Create enhanced data loaders
    train_loader, val_loader, test_loader, num_classes, label_map, class_weights = create_enhanced_data_loaders(
        config['data'], config['input_size'], config['batch_size'], config.get('use_balanced', True)
    )
    
    # Create improved model
    model = create_improved_model(
        num_classes=num_classes,
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        freeze_backbone=config['model']['freeze_backbone'],
        dropout_rate=config['model']['dropout_rate'],
        use_attention=config['model']['use_attention'],
        device=device
    )
    
    # Enhanced loss function selection
    if config['training']['loss_type'] == 'focal':
        criterion = FocalLoss(alpha=1, gamma=2)
        print("Using Focal Loss for class imbalance")
    elif config['training']['loss_type'] == 'label_smoothing':
        criterion = LabelSmoothingCrossEntropy(smoothing=config['training']['label_smoothing'])
        print(f"Using Label Smoothing CrossEntropy (smoothing={config['training']['label_smoothing']})")
    elif config['training']['loss_type'] == 'weighted':
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Using Weighted CrossEntropy Loss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropy Loss")
    
    # Enhanced optimizer setup
    if config['model']['freeze_backbone']:
        optimizer = optim.AdamW(model.classifier.parameters(), 
                              lr=config['training']['classifier_lr'],
                              weight_decay=config['training']['weight_decay'],
                              betas=(0.9, 0.999))
        print(f"Training classifier only with AdamW optimizer")
    else:
        # Different learning rates for different parts
        param_groups = [
            {'params': model.backbone.parameters(), 'lr': config['training']['backbone_lr']},
            {'params': model.classifier.parameters(), 'lr': config['training']['classifier_lr']}
        ]
        if hasattr(model, 'attention'):
            param_groups.append({'params': model.attention.parameters(), 'lr': config['training']['classifier_lr']})
        
        optimizer = optim.AdamW(param_groups, 
                              weight_decay=config['training']['weight_decay'],
                              betas=(0.9, 0.999))
        print(f"Fine-tuning with AdamW: Backbone LR: {config['training']['backbone_lr']}, "
              f"Classifier LR: {config['training']['classifier_lr']}")
    
    # Enhanced scheduler (Cosine Annealing with Warm Restarts)
    if config['training']['scheduler_type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config['training']['T_0'], T_mult=config['training']['T_mult'])
        print("Using Cosine Annealing with Warm Restarts scheduler")
    elif config['training']['scheduler_type'] == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        print("Using ReduceLROnPlateau scheduler")
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                             step_size=config['training']['scheduler_step_size'], 
                                             gamma=config['training']['scheduler_gamma'])
        print("Using StepLR scheduler")
    
    # Training loop with enhancements
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    print(f"\nStarting enhanced training for {config['training']['epochs']} epochs...")
    print(f"Gradient accumulation steps: {config['training']['gradient_accumulation_steps']}")
    print(f"Max grad norm: {config['training']['max_grad_norm']}")
    print(f"Early stopping patience: {config['training']['patience']}")
    print("-" * 80)
    
    for epoch in range(config['training']['epochs']):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        print("-" * 40)
        
        # Train with enhancements
        train_loss, train_acc = train_epoch_enhanced(
            model, train_loader, criterion, optimizer, device, epoch+1,
            config['training']['gradient_accumulation_steps'],
            config['training']['max_grad_norm']
        )
        
        # Validate
        val_loss, val_acc, val_predictions, val_targets, val_probabilities = validate_epoch_enhanced(
            model, val_loader, criterion, device)
        
        # Update scheduler
        if config['training']['scheduler_type'] == 'reduce_on_plateau':
            scheduler.step(val_acc)
        else:
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
        
        # Print learning rates
        if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 1:
            lrs = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
            print(f"Learning rates: {lrs}")
        else:
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
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
                'class_weights': class_weights.tolist(),
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
    
    # Final evaluation
    print(f"\n" + "="*60)
    print("ENHANCED MODEL EVALUATION")
    print("="*60)
    
    # Load best model
    if Path(config['output']['model_path']).exists():
        checkpoint = torch.load(config['output']['model_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Test set evaluation
    test_loss, test_acc, test_predictions, test_targets, test_probabilities = validate_epoch_enhanced(
        model, test_loader, criterion, device)
    
    print(f"\nFinal Results:")
    print(f"- Test Loss: {test_loss:.4f}")
    print(f"- Test Accuracy: {test_acc:.2f}%")
    print(f"- Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Check if target accuracy achieved
    target_accuracy = config.get('target_accuracy', 80.0)
    if test_acc >= target_accuracy:
        print(f"🎉 TARGET ACCURACY ACHIEVED! {test_acc:.2f}% >= {target_accuracy}%")
    else:
        print(f"⚠️  Target accuracy not reached: {test_acc:.2f}% < {target_accuracy}%")
    
    # Detailed classification report
    emotion_names = list(label_map.keys())
    if len(set(test_targets)) > 1:
        print(f"\nDetailed Classification Report:")
        report = classification_report(test_targets, test_predictions, 
                                     target_names=emotion_names, zero_division=0)
        print(report)
        
        # Per-class accuracy
        cm = confusion_matrix(test_targets, test_predictions)
        print(f"\nPer-class Accuracy:")
        for i, emotion in enumerate(emotion_names):
            class_acc = 100 * cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            print(f"- {emotion}: {class_acc:.2f}%")
        
        # Save detailed results
        results_info = {
            'model_config': config,
            'final_results': {
                'test_accuracy': float(test_acc),
                'best_val_accuracy': float(best_val_acc),
                'test_loss': float(test_loss),
                'target_achieved': test_acc >= target_accuracy
            },
            'per_class_accuracy': {
                emotion_names[i]: float(100 * cm[i, i] / cm[i, :].sum()) if cm[i, :].sum() > 0 else 0.0
                for i in range(len(emotion_names))
            },
            'classification_report': report,
            'training_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies
            }
        }
        
        results_path = config['output']['results_path'].replace('.txt', '.json')
        with open(results_path, 'w') as f:
            json.dump(results_info, f, indent=2)
        
        print(f"✓ Detailed results saved to: {results_path}")
    
    print(f"\n✓ Enhanced training completed successfully!")
    print(f"✓ Best model saved to: {config['output']['model_path']}")
    
    return model, test_acc >= target_accuracy


def main():
    """Main function to run enhanced CNN transfer learning training."""
    parser = argparse.ArgumentParser(description='Enhanced CNN Transfer Learning Training')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet50', 'resnet101', 'efficientnet_b4', 'densenet121', 'vgg16'],
                       help='Backbone architecture to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--input_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--target_accuracy', type=float, default=80.0,
                       help='Target accuracy to achieve')
    
    args = parser.parse_args()
    
    # Enhanced configuration - optimized for faster training
    config = {
        'seed': 42,
        'input_size': min(args.input_size, 224),  # Limit input size for speed
        'batch_size': min(args.batch_size, 16),   # Smaller batch size for CPU
        'target_accuracy': args.target_accuracy,
        'use_balanced': True,
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
            'freeze_backbone': args.epochs <= 10,  # Freeze backbone for short training
            'dropout_rate': 0.3,
            'use_attention': False  # Disable attention for speed
        },
        'training': {
            'epochs': args.epochs,
            'backbone_lr': 1e-5,
            'classifier_lr': 1e-3,
            'weight_decay': 1e-4,
            'gradient_accumulation_steps': 1,  # No gradient accumulation for speed
            'max_grad_norm': 1.0,
            'loss_type': 'weighted',  # Use weighted loss for class imbalance
            'label_smoothing': 0.1,
            'scheduler_type': 'step',  # Simple scheduler for speed
            'T_0': 10,
            'T_mult': 2,
            'scheduler_step_size': max(args.epochs // 3, 5),
            'scheduler_gamma': 0.7,
            'patience': max(args.epochs // 2, 3)
        },
        'output': {
            'model_path': f'models/enhanced_cnn_transfer_{args.backbone}.pth',
            'results_path': f'results/enhanced_cnn_transfer_{args.backbone}_results.json'
        }
    }
    
    # Create output directories
    os.makedirs(os.path.dirname(config['output']['model_path']), exist_ok=True)
    os.makedirs(os.path.dirname(config['output']['results_path']), exist_ok=True)
    
    # Start enhanced training
    model, target_achieved = train_enhanced_cnn_transfer_learning(config)
    
    if target_achieved:
        print(f"\n🎉 SUCCESS: Target accuracy of {args.target_accuracy}% achieved!")
    else:
        print(f"\n⚠️  Target accuracy of {args.target_accuracy}% not achieved. Consider:")
        print("   - Increasing number of epochs")
        print("   - Trying different backbone (resnet101, efficientnet_b4)")
        print("   - Adjusting hyperparameters")
        print("   - Using ensemble methods")


if __name__ == "__main__":
    main()