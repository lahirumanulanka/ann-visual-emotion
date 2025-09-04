#!/usr/bin/env python3
"""
Simple Enhanced Emotion Recognition Model Training Script

This script demonstrates the key improvements needed to achieve 80%+ accuracy
in emotion recognition. It's designed to work reliably on both CPU and GPU.
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import accuracy_score, f1_score, classification_report
from PIL import Image
from collections import Counter

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SimpleEnhancedDataset(Dataset):
    """Simplified dataset with key enhancements"""
    def __init__(self, csv_path, images_root, str2idx, img_size=224, train=True):
        self.df = pd.read_csv(csv_path)
        self.images_root = Path(images_root)
        self.str2idx = str2idx
        self.train = train
        
        # Handle column names
        self.path_col = "image_path" if "image_path" in self.df.columns else "image"
        
        # Convert labels
        if self.df["label"].dtype == object:
            self.df["label_idx"] = self.df["label"].map(self.str2idx).astype(int)
        else:
            self.df["label_idx"] = self.df["label"].astype(int)
        
        # Enhanced transforms
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img_path = row[self.path_col]
            
            # Handle path
            if not img_path.startswith('/'):
                if img_path.startswith('data/raw/EmoSet/'):
                    img_path = img_path.replace('data/raw/EmoSet/', '')
                img_path = self.images_root / img_path
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            label = int(row["label_idx"])
            
            return image, label
        except Exception as e:
            print(f"Error loading image {idx}: {e}")
            # Return dummy data
            return torch.zeros(3, 224, 224), 0

class SimpleFocalLoss(nn.Module):
    """Simplified Focal Loss implementation"""
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class EnhancedEmotionModel(nn.Module):
    """Enhanced model with ResNet50 backbone"""
    def __init__(self, num_classes, pretrained=True, dropout_rate=0.4):
        super().__init__()
        
        # Load ResNet50
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = resnet50(weights=None)
        
        # Get feature dimension
        feature_dim = self.backbone.fc.in_features
        
        # Replace classifier with enhanced version
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim // 2),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(feature_dim // 2, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def create_weighted_sampler(dataset):
    """Create weighted sampler for class imbalance"""
    labels = []
    
    # Sample subset for speed
    sample_size = min(len(dataset), 2000)
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for idx in indices:
        try:
            _, label = dataset[idx]
            labels.append(label)
        except:
            continue
    
    if not labels:
        return None
    
    # Compute weights
    class_counts = Counter(labels)
    total = len(labels)
    
    class_weights = {}
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = total / (len(class_counts) * count)
    
    # Create weights for full dataset
    full_labels = []
    for idx in range(len(dataset)):
        try:
            _, label = dataset[idx]
            full_labels.append(label)
        except:
            full_labels.append(0)  # Default
    
    sample_weights = [class_weights.get(label, 1.0) for label in full_labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    """Training loop for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, targets) in enumerate(loader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    return total_loss / len(loader), 100.0 * correct / total

def evaluate_model(model, loader, criterion, device):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    return total_loss / len(loader), accuracy, f1_macro, all_targets, all_preds

def main():
    """Main training function"""
    print("Enhanced Emotion Recognition Training")
    print("=" * 50)
    
    # Configuration
    PROJECT_ROOT = "/home/runner/work/ann-visual-emotion/ann-visual-emotion"
    TRAIN_CSV = f"{PROJECT_ROOT}/data/processed/EmoSet_splits/train.csv"
    VAL_CSV = f"{PROJECT_ROOT}/data/processed/EmoSet_splits/val.csv"
    TEST_CSV = f"{PROJECT_ROOT}/data/processed/EmoSet_splits/test.csv"
    IMAGES_ROOT = f"{PROJECT_ROOT}/data/raw/EmoSet"
    OUT_DIR = f"{PROJECT_ROOT}/outputs/simple_enhanced_model"
    
    # Training parameters
    BATCH_SIZE = 16  # Balanced for CPU/GPU
    EPOCHS = 20      # Reasonable for testing
    BASE_LR = 1e-3
    MAX_LR = 3e-3
    
    # Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_CSV)
    unique_labels = sorted(train_df['label'].unique())
    str2idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx2str = {idx: label for label, idx in str2idx.items()}
    num_classes = len(str2idx)
    
    print(f"Found {num_classes} classes: {list(idx2str.values())}")
    
    # Create datasets
    train_dataset = SimpleEnhancedDataset(TRAIN_CSV, IMAGES_ROOT, str2idx, train=True)
    val_dataset = SimpleEnhancedDataset(VAL_CSV, IMAGES_ROOT, str2idx, train=False)
    test_dataset = SimpleEnhancedDataset(TEST_CSV, IMAGES_ROOT, str2idx, train=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create samplers and loaders
    print("Creating data loaders...")
    train_sampler = create_weighted_sampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=1, pin_memory=False, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE*2, shuffle=False,
        num_workers=1, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE*2, shuffle=False,
        num_workers=1, pin_memory=False
    )
    
    # Create model
    print("Creating enhanced model...")
    model = EnhancedEmotionModel(num_classes=num_classes, pretrained=True, dropout_rate=0.4)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = SimpleFocalLoss(alpha=0.25, gamma=2.0)
    
    optimizer = AdamW(
        model.parameters(),
        lr=BASE_LR,
        weight_decay=1e-4
    )
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    # Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("Target: 80%+ accuracy")
    print("-" * 50)
    
    best_val_acc = 0
    best_model_path = Path(OUT_DIR) / "best_model.pth"
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        
        # Validation
        val_loss, val_acc, val_f1, y_true, y_pred = evaluate_model(
            model, val_loader, criterion, device
        )
        
        epoch_time = time.time() - start_time
        
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1
            }, best_model_path)
            print(f"  ✓ New best model saved! Acc: {val_acc:.4f}")
        
        # Check target
        if val_acc >= 0.80:
            print(f"  🎯 TARGET ACHIEVED! Validation accuracy: {val_acc:.4f}")
            break
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)
    
    # Load best model
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    test_loss, test_acc, test_f1, y_true_test, y_pred_test = evaluate_model(
        model, test_loader, criterion, device
    )
    
    print(f"\nFINAL RESULTS:")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"Target (80%) Achieved: {'✓ YES' if test_acc >= 0.80 else '✗ NO'}")
    
    # Classification report
    class_names = [idx2str[i] for i in range(num_classes)]
    report = classification_report(
        y_true_test, y_pred_test,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print(f"\nDetailed Classification Report:")
    print(report)
    
    # Save results
    with open(Path(OUT_DIR) / "results.txt", "w") as f:
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1-Score: {test_f1:.4f}\n")
        f.write(f"Target Achieved: {'YES' if test_acc >= 0.80 else 'NO'}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"\nResults saved to: {OUT_DIR}")
    
    # Summary
    improvements = [
        "✓ ResNet50 backbone (vs ResNet18)",
        "✓ Enhanced data augmentation",
        "✓ Focal Loss for class imbalance", 
        "✓ Weighted sampling",
        "✓ OneCycleLR scheduler",
        "✓ Progressive dropout",
        "✓ Batch normalization",
        "✓ Gradient clipping"
    ]
    
    print(f"\nKey Improvements Applied:")
    for imp in improvements:
        print(f"  {imp}")
    
    if test_acc >= 0.80:
        print(f"\n🎉 SUCCESS! Achieved {test_acc*100:.2f}% accuracy (≥80% target)")
    else:
        print(f"\n⚠️  Close! Achieved {test_acc*100:.2f}% accuracy")
        print("   Consider: longer training, larger model, or ensemble methods")
    
    return test_acc >= 0.80

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)