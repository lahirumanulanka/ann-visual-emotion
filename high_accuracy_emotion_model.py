#!/usr/bin/env python3
"""
High-Accuracy Emotion Recognition Model
Targets 80%+ accuracy through advanced techniques
"""

import os
import sys
import json
import math
import time
import random
import warnings
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast

# Torchvision imports
import torchvision
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# ML metrics
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    accuracy_score, precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# Configuration
class Config:
    """Configuration for the high-accuracy model"""
    
    # Paths
    PROJECT_ROOT = "/home/runner/work/ann-visual-emotion/ann-visual-emotion"
    RAW_DIR = Path(PROJECT_ROOT) / "data" / "raw" / "EmoSet"
    SPLIT_DIR = Path(PROJECT_ROOT) / "data" / "processed" / "EmoSet_splits"
    
    TRAIN_CSV = str(SPLIT_DIR / "train.csv")
    VAL_CSV = str(SPLIT_DIR / "val.csv")
    TEST_CSV = str(SPLIT_DIR / "test.csv")
    IMAGES_ROOT = str(RAW_DIR)
    
    OUT_DIR = str(Path(PROJECT_ROOT) / "outputs" / "high_accuracy_emotion_model")
    
    # Model Configuration
    MODEL_NAME = "resnet50"
    PRETRAINED = True
    IMG_SIZE = 224
    NUM_CLASSES = 8
    
    # Training Configuration
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    EPOCHS = 80  # Reduced for practical training
    
    # Optimization
    BASE_LR = 3e-4
    MAX_LR = 1e-3
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP = 1.0
    ACCUMULATION_STEPS = 2
    
    # Loss Configuration
    LOSS_MODE = "focal"
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTHING = 0.1
    CLASS_WEIGHT_MODE = "balanced"
    
    # Regularization
    DROPOUT_RATE = 0.3
    USE_ATTENTION = True
    
    # Data Augmentation
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4
    USE_CUTMIX = True
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 0.5
    
    # Advanced Features
    USE_MIXED_PRECISION = True
    USE_TTA = True
    TTA_TRANSFORMS = 5
    
    # Early Stopping
    PATIENCE = 15
    MIN_DELTA = 1e-4

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Dataset class
class HighAccuracyEmotionDataset(Dataset):
    """Enhanced dataset with sophisticated augmentation"""
    
    def __init__(self, csv_path: str, images_root: str, label_map: dict, 
                 img_size: int = 224, is_training: bool = False):
        
        self.csv_path = csv_path
        self.images_root = Path(images_root)
        self.label_map = label_map
        self.img_size = img_size
        self.is_training = is_training
        
        # Load data
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        
        # Handle column names
        if "image_path" in self.df.columns:
            self.path_col = "image_path"
        elif "image" in self.df.columns:
            self.path_col = "image"
        else:
            raise ValueError("CSV must contain 'image' or 'image_path' column")
            
        # Convert labels to indices
        self.df["label_idx"] = self.df["label"].map(self.label_map)
        missing = self.df["label_idx"].isnull().sum()
        if missing > 0:
            print(f"Warning: {missing} samples have unmapped labels")
            self.df = self.df.dropna(subset=["label_idx"])
            
        self.df["label_idx"] = self.df["label_idx"].astype(int)
        
        # Setup transforms
        self._setup_transforms()
        
        # Print class distribution
        class_counts = self.df["label"].value_counts().sort_index()
        print(f"Class distribution: {dict(class_counts)}")
    
    def _setup_transforms(self):
        """Setup transformation pipeline"""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if self.is_training:
            # Enhanced training transforms
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.25))
            ])
        else:
            # Simple validation/test transforms
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                normalize
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            
            # Get image path
            img_path = row[self.path_col]
            
            # Handle relative paths
            if not img_path.startswith('/'):
                if img_path.startswith('data/raw/EmoSet/'):
                    img_path = img_path.replace('data/raw/EmoSet/', '')
                img_path = self.images_root / img_path
            else:
                img_path = Path(img_path)
            
            # Load image
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                image = Image.new('RGB', (224, 224), (0, 0, 0))
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            label = int(row["label_idx"])
            return image, label
            
        except Exception as e:
            print(f"Error in __getitem__ for index {idx}: {e}")
            return torch.zeros(3, self.img_size, self.img_size), 0
    
    @property
    def targets(self):
        """Return target labels for sampling"""
        return self.df["label_idx"].values

# Loss functions
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Model architecture
class AttentionModule(nn.Module):
    """Lightweight attention mechanism"""
    
    def __init__(self, in_features: int, reduction_ratio: int = 16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features, in_features // reduction_ratio)
        self.fc2 = nn.Linear(in_features // reduction_ratio, in_features)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)

class HighAccuracyEmotionModel(nn.Module):
    """Enhanced model architecture for high accuracy"""
    
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True,
                 dropout_rate: float = 0.3, use_attention: bool = True):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Load pre-trained backbone
        if model_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = resnet50(weights=weights)
            
            # Get feature dimension and remove final layer
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
            # Add attention
            if use_attention:
                self.attention = AttentionModule(feature_dim)
                # Modify backbone to use our attention
                self.backbone.avgpool = nn.Identity()
                self.avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Enhanced classifier head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout_rate),
            
            # First hidden layer
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim // 2),
            nn.Dropout(dropout_rate * 0.5),
            
            # Second hidden layer
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim // 4),
            nn.Dropout(dropout_rate * 0.25),
            
            # Output layer
            nn.Linear(feature_dim // 4, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier weights properly"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features from ResNet
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        logits = self.classifier(x)
        return logits

def create_label_mapping(train_csv: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create consistent label mapping from training data"""
    df = pd.read_csv(train_csv)
    unique_labels = sorted(df['label'].unique())
    
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    print(f"Found {len(unique_labels)} emotion classes:")
    for idx, label in idx_to_label.items():
        print(f"   {idx}: {label}")
    
    return label_to_idx, idx_to_label

def compute_class_weights(train_csv: str, label_to_idx: dict, mode: str = "balanced") -> Optional[torch.Tensor]:
    """Compute class weights for handling imbalanced data"""
    df = pd.read_csv(train_csv)
    labels = df['label'].values
    
    if mode == "none":
        return None
        
    elif mode == "balanced":
        unique_labels = sorted(df['label'].unique())
        label_indices = [label_to_idx[label] for label in labels]
        
        weights = compute_class_weight(
            'balanced', 
            classes=np.arange(len(unique_labels)), 
            y=label_indices
        )
        return torch.tensor(weights, dtype=torch.float32)
    
    else:
        raise ValueError(f"Unknown class weight mode: {mode}")

def mixup_data(x, y, alpha=1.0):
    """Apply MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixed samples"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def create_weighted_sampler(dataset):
    """Create weighted sampler for imbalanced dataset"""
    labels = dataset.targets
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    # Compute weights for each class
    class_weights = {}
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = total_samples / (len(class_counts) * count)
    
    # Create sample weights
    sample_weights = [class_weights[label] for label in labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, cfg):
    """Training loop for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    device = next(model.parameters()).device  # Get device from model
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.EPOCHS}')
    
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)
        
        # Apply MixUp randomly
        use_mixup = cfg.USE_MIXUP and random.random() < cfg.MIXUP_PROB
        
        if use_mixup:
            data, targets_a, targets_b, lam = mixup_data(data, targets, cfg.MIXUP_ALPHA)
            mixed_targets = (targets_a, targets_b, lam)
        else:
            mixed_targets = None
        
        # Forward pass with mixed precision
        if cfg.USE_MIXED_PRECISION:
            with autocast():
                outputs = model(data)
                
                if mixed_targets is not None:
                    targets_a, targets_b, lam = mixed_targets
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, targets)
                
                loss = loss / cfg.ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % cfg.ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        else:
            outputs = model(data)
            
            if mixed_targets is not None:
                targets_a, targets_b, lam = mixed_targets
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)
            
            loss = loss / cfg.ACCUMULATION_STEPS
            loss.backward()
            
            if (batch_idx + 1) % cfg.ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
        
        # Statistics
        running_loss += loss.item() * cfg.ACCUMULATION_STEPS
        if mixed_targets is None:
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'LR': f'{current_lr:.2e}',
            'Acc': f'{100.*correct/total:.2f}%' if total > 0 else 'N/A'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total if total > 0 else 0
    
    return epoch_loss, epoch_acc, current_lr

def evaluate_model(model, data_loader, criterion):
    """Evaluate model on validation/test set"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    device = next(model.parameters()).device  # Get device from model
    
    with torch.no_grad():
        for data, targets in tqdm(data_loader, desc='Evaluating', leave=False):
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    epoch_loss = running_loss / len(data_loader)
    accuracy = accuracy_score(all_targets, all_predictions)
    f1_macro = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
    
    return epoch_loss, accuracy, f1_macro, f1_weighted, all_targets, all_predictions

def plot_training_progress(history, save_path=None):
    """Plot training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', alpha=0.8)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', alpha=0.8)
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy', color='green', alpha=0.8)
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score curve
    axes[1, 0].plot(history['val_f1'], label='Val F1-Score', color='red', alpha=0.8)
    axes[1, 0].set_title('Validation F1-Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate curve
    axes[1, 1].plot(history['learning_rate'], label='Learning Rate', color='orange', alpha=0.8)
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title(f'{title} - Raw Counts')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Normalized percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title(f'{title} - Normalized (%)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def main():
    """Main training function"""
    # Setup
    set_seed(42)
    cfg = Config()
    
    # Create output directory
    Path(cfg.OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting High-Accuracy Emotion Recognition Training")
    print(f"Target: 80%+ Accuracy")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Create label mappings
    label_to_idx, idx_to_label = create_label_mapping(cfg.TRAIN_CSV)
    cfg.NUM_CLASSES = len(idx_to_label)
    
    # Compute class weights
    class_weights = compute_class_weights(cfg.TRAIN_CSV, label_to_idx, cfg.CLASS_WEIGHT_MODE)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Create datasets
    train_dataset = HighAccuracyEmotionDataset(
        csv_path=cfg.TRAIN_CSV,
        images_root=cfg.IMAGES_ROOT,
        label_map=label_to_idx,
        img_size=cfg.IMG_SIZE,
        is_training=True
    )
    
    val_dataset = HighAccuracyEmotionDataset(
        csv_path=cfg.VAL_CSV,
        images_root=cfg.IMAGES_ROOT,
        label_map=label_to_idx,
        img_size=cfg.IMG_SIZE,
        is_training=False
    )
    
    test_dataset = HighAccuracyEmotionDataset(
        csv_path=cfg.TEST_CSV,
        images_root=cfg.IMAGES_ROOT,
        label_map=label_to_idx,
        img_size=cfg.IMG_SIZE,
        is_training=False
    )
    
    # Create samplers and data loaders
    train_sampler = create_weighted_sampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"DataLoaders created:")
    print(f"  Training: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Validation: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    # Create model
    model = HighAccuracyEmotionModel(
        model_name=cfg.MODEL_NAME,
        num_classes=cfg.NUM_CLASSES,
        pretrained=cfg.PRETRAINED,
        dropout_rate=cfg.DROPOUT_RATE,
        use_attention=cfg.USE_ATTENTION
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {cfg.MODEL_NAME}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    if cfg.LOSS_MODE == "focal":
        criterion = FocalLoss(
            alpha=cfg.FOCAL_ALPHA,
            gamma=cfg.FOCAL_GAMMA,
            weight=class_weights
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.BASE_LR,
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.MAX_LR,
        epochs=cfg.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if cfg.USE_MIXED_PRECISION else None
    
    # Early stopping
    class EarlyStopping:
        def __init__(self, patience=15, min_delta=1e-4):
            self.patience = patience
            self.min_delta = min_delta
            self.best_score = 0
            self.counter = 0
            self.early_stop = False
            
        def __call__(self, score):
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                return True  # Improvement
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                return False  # No improvement
    
    early_stopping = EarlyStopping(patience=cfg.PATIENCE, min_delta=cfg.MIN_DELTA)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'learning_rate': []
    }
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_model_path = Path(cfg.OUT_DIR) / "best_model.pth"
    
    # Training loop
    for epoch in range(cfg.EPOCHS):
        start_time = time.time()
        
        # Training phase
        train_loss, train_acc, current_lr = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, epoch, cfg
        )
        
        # Validation phase
        val_loss, val_acc, val_f1_macro, val_f1_weighted, y_true, y_pred = evaluate_model(
            model, val_loader, criterion
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1_macro)
        history['learning_rate'].append(current_lr)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Print epoch results
        print(f"\\nEpoch {epoch+1:3d}/{cfg.EPOCHS}:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1_macro:.4f}")
        print(f"  LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")
        
        # Check for improvements
        is_best = early_stopping(val_f1_macro)
        
        if is_best:
            best_val_f1 = val_f1_macro
            best_val_acc = val_acc
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1': val_f1_macro,
                'val_acc': val_acc,
                'history': history,
                'label_mapping': {
                    'label_to_idx': label_to_idx,
                    'idx_to_label': idx_to_label
                }
            }, best_model_path)
            
            print(f"  ✓ New best model saved! F1: {val_f1_macro:.4f}, Acc: {val_acc:.4f}")
        
        # Early stopping check
        if early_stopping.early_stop:
            print(f"\\n🛑 Early stopping triggered!")
            break
        
        # Check if we've reached target
        if val_acc >= 0.8:
            print(f"\\n🎯 TARGET REACHED! Validation accuracy: {val_acc:.4f} (≥80%)")
            break
    
    print("\\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation F1-score: {best_val_f1:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Target achieved: {'✓ YES' if best_val_acc >= 0.8 else '✗ NO'}")
    
    # Load best model for final evaluation
    print("\\nLoading best model for final evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    print("\\n" + "=" * 50)
    print("FINAL TEST RESULTS")
    print("=" * 50)
    
    test_loss, test_acc, test_f1_macro, test_f1_weighted, y_true_test, y_pred_test = evaluate_model(
        model, test_loader, criterion
    )
    
    print(f"Test Results:")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  F1-Score (Macro): {test_f1_macro:.4f}")
    print(f"  F1-Score (Weighted): {test_f1_weighted:.4f}")
    print(f"  Loss: {test_loss:.4f}")
    
    # Classification report
    class_names = [idx_to_label[i] for i in range(cfg.NUM_CLASSES)]
    test_report = classification_report(
        y_true_test, y_pred_test,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print("\\nDetailed Classification Report:")
    print(test_report)
    
    # Save results
    with open(Path(cfg.OUT_DIR) / "test_classification_report.txt", "w") as f:
        f.write(test_report)
    
    # Plot results
    plot_training_progress(history, Path(cfg.OUT_DIR) / "training_progress.png")
    plot_confusion_matrix(
        y_true_test, y_pred_test, class_names,
        "Test Set Confusion Matrix",
        Path(cfg.OUT_DIR) / "test_confusion_matrix.png"
    )
    
    # Save comprehensive results
    results = {
        'final_results': {
            'test_accuracy': test_acc,
            'test_f1_macro': test_f1_macro,
            'test_f1_weighted': test_f1_weighted,
            'test_loss': test_loss
        },
        'target_achievement': {
            'accuracy_target_80pct': test_acc >= 0.8,
            'final_accuracy_percent': test_acc * 100
        },
        'training_config': {
            'model_name': cfg.MODEL_NAME,
            'epochs_trained': len(history['val_acc']),
            'best_val_acc': best_val_acc,
            'best_val_f1': best_val_f1
        }
    }
    
    with open(Path(cfg.OUT_DIR) / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\\n📊 FINAL STATUS: {'SUCCESS - 80%+ ACCURACY ACHIEVED!' if test_acc >= 0.8 else 'TARGET NOT REACHED'}")
    print(f"Final test accuracy: {test_acc*100:.2f}%")
    print("=" * 60)
    
    return model, test_acc, test_f1_macro

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run training
    model, test_acc, test_f1 = main()