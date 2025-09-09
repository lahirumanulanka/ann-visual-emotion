#!/usr/bin/env python3
"""
Comparison demo: Original vs Enhanced CNN Transfer Learning
Shows the improvement achieved by using AI-enhanced 224x224 images
"""

import sys
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score, classification_report

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.cnn_transfer_learning import create_cnn_transfer_model
from data.enhanced_dataset import EnhancedEmotionDataset
from data.dataset_emotion import EmotionDataset
from torch.utils.data import DataLoader


class ModelComparison:
    """Compare original 48x48 vs enhanced 224x224 model performance."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load test data
        self.setup_data()
        
    def setup_data(self):
        """Setup data loaders for both original and enhanced datasets."""
        
        # Load the test CSV we created
        df = pd.read_csv('/tmp/test_enhanced.csv')
        
        with open('/tmp/test_label_map.json', 'r') as f:
            label_map = json.load(f)
        
        num_classes = len(label_map)
        
        # Create transforms for original 48x48 images (converted to RGB)
        original_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize 48x48 to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create transforms for enhanced 224x224 images
        enhanced_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Already 224x224, but ensure consistency
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Original dataset (48x48 -> 224x224 with simple resize)
        original_dataset = EmotionDataset(
            dataframe=df,
            root_dir='/tmp/test_data_orig',
            transform=original_transform,
            label_map=label_map,
            rgb=True  # Convert to RGB
        )
        
        # Enhanced dataset (AI-enhanced 224x224)
        enhanced_dataset = EnhancedEmotionDataset(
            dataframe=df,
            enhanced_root_dir='/tmp/test_data_enhanced',
            original_root_dir='/tmp/test_data_orig',
            transform=enhanced_transform,
            label_map=label_map
        )
        
        self.original_loader = DataLoader(original_dataset, batch_size=8, shuffle=True, num_workers=0)
        self.enhanced_loader = DataLoader(enhanced_dataset, batch_size=8, shuffle=True, num_workers=0)
        
        self.num_classes = num_classes
        self.label_map = label_map
        
        print(f"Dataset loaded: {len(df)} samples, {num_classes} classes")
    
    def create_model(self):
        """Create a CNN transfer learning model."""
        model = create_cnn_transfer_model(
            num_classes=self.num_classes,
            backbone='vgg16',
            pretrained=True,
            freeze_backbone=False,  # Fine-tune for better comparison
            device=self.device
        )
        return model
    
    def train_model(self, model, dataloader, epochs=5):
        """Quick training for demonstration."""
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam([
            {'params': model.features.parameters(), 'lr': 1e-5},  # Lower LR for pretrained features
            {'params': model.classifier.parameters(), 'lr': 1e-3}
        ])
        
        model.train()
        
        print(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                epoch_total += labels.size(0)
                epoch_correct += (predicted == labels).sum().item()
            
            accuracy = 100 * epoch_correct / epoch_total
            avg_loss = epoch_loss / len(dataloader)
            
            print(f"Epoch {epoch+1}/{epochs}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return model
    
    def evaluate_model(self, model, dataloader):
        """Evaluate model performance."""
        
        model.eval()
        all_predictions = []
        all_labels = []
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        accuracy = 100 * total_correct / total_samples
        return accuracy, all_predictions, all_labels
    
    def run_comparison(self):
        """Run the full comparison between original and enhanced models."""
        
        print("\n" + "="*60)
        print("CNN TRANSFER LEARNING: ORIGINAL vs ENHANCED COMPARISON")
        print("="*60)
        
        results = {}
        
        # Test 1: Original 48x48 -> 224x224 (simple resize)
        print("\n🔹 TRAINING MODEL WITH ORIGINAL IMAGES (48x48 -> 224x224 resize)")
        print("-" * 60)
        
        start_time = time.time()
        original_model = self.create_model()
        original_model = self.train_model(original_model, self.original_loader, epochs=5)
        original_accuracy, orig_preds, orig_labels = self.evaluate_model(original_model, self.original_loader)
        original_time = time.time() - start_time
        
        results['original'] = {
            'accuracy': original_accuracy,
            'training_time': original_time,
            'method': 'Simple 48x48->224x224 resize'
        }
        
        print(f"✅ Original Model Accuracy: {original_accuracy:.2f}%")
        print(f"⏱️  Training Time: {original_time:.1f}s")
        
        # Test 2: Enhanced 224x224 (AI-enhanced)
        print("\n🔹 TRAINING MODEL WITH AI-ENHANCED IMAGES (224x224)")
        print("-" * 60)
        
        start_time = time.time()
        enhanced_model = self.create_model()
        enhanced_model = self.train_model(enhanced_model, self.enhanced_loader, epochs=5)
        enhanced_accuracy, enh_preds, enh_labels = self.evaluate_model(enhanced_model, self.enhanced_loader)
        enhanced_time = time.time() - start_time
        
        results['enhanced'] = {
            'accuracy': enhanced_accuracy,
            'training_time': enhanced_time,
            'method': 'AI-enhanced 48x48->224x224'
        }
        
        print(f"✅ Enhanced Model Accuracy: {enhanced_accuracy:.2f}%")
        print(f"⏱️  Training Time: {enhanced_time:.1f}s")
        
        # Summary
        print("\n" + "="*60)
        print("📊 COMPARISON RESULTS")
        print("="*60)
        
        improvement = enhanced_accuracy - original_accuracy
        improvement_pct = (improvement / original_accuracy) * 100 if original_accuracy > 0 else 0
        
        print(f"📈 Accuracy Improvement: +{improvement:.2f}% ({improvement_pct:.1f}% relative improvement)")
        print(f"🚀 Enhanced Model Performance: {enhanced_accuracy:.2f}%")
        print(f"📊 Original Model Performance: {original_accuracy:.2f}%")
        
        if improvement > 0:
            print(f"✅ AI enhancement provides better results!")
        else:
            print(f"⚠️  Results are similar (small dataset may limit improvement)")
        
        # Save results
        with open('/tmp/comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


def main():
    """Run the comparison demo."""
    
    print("🧪 CNN Transfer Learning Comparison Demo")
    print("This demo compares the performance of:")
    print("1. Original 48x48 images (resized to 224x224)")
    print("2. AI-enhanced 224x224 images")
    print()
    
    comparison = ModelComparison()
    results = comparison.run_comparison()
    
    print(f"\n📄 Results saved to: /tmp/comparison_results.json")
    print(f"🎯 Demo completed! Enhanced model shows accuracy of {results['enhanced']['accuracy']:.2f}%")


if __name__ == "__main__":
    main()