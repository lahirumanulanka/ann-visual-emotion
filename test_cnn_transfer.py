#!/usr/bin/env python3
"""
Simple test script to verify CNN Transfer Learning implementation.
This script tests the core functionality without requiring actual data.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image

def test_model_creation():
    """Test that we can create the transfer learning model."""
    print("1. Testing Model Creation...")
    
    try:
        from models.cnn_transfer_learning import CNNTransferLearning
        
        # Test different backbones
        backbones = ['vgg16', 'vgg19', 'alexnet']
        
        for backbone in backbones:
            model = CNNTransferLearning(
                num_classes=7,
                backbone=backbone,
                pretrained=True,
                freeze_backbone=False
            )
            
            params = model.get_num_params()
            print(f"   ✓ {backbone}: {params:,} parameters")
        
        print("   ✅ Model creation successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False


def test_forward_pass():
    """Test forward pass with dummy data."""
    print("\n2. Testing Forward Pass...")
    
    try:
        from models.cnn_transfer_learning import CNNTransferLearning
        
        model = CNNTransferLearning(num_classes=7, backbone='vgg16')
        model.eval()
        
        # Test different batch sizes
        batch_sizes = [1, 4]
        
        for batch_size in batch_sizes:
            dummy_input = torch.randn(batch_size, 3, 224, 224)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            expected_shape = (batch_size, 7)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
            print(f"   ✓ Batch size {batch_size}: {output.shape}")
        
        print("   ✅ Forward pass successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False


def test_training_setup():
    """Test training setup with optimizers and loss."""
    print("\n3. Testing Training Setup...")
    
    try:
        from models.cnn_transfer_learning import CNNTransferLearning
        import torch.optim as optim
        
        model = CNNTransferLearning(num_classes=7, backbone='vgg16')
        
        # Test loss function
        criterion = nn.CrossEntropyLoss()
        
        # Test optimizer setup for fine-tuning
        optimizer = optim.Adam([
            {'params': model.features.parameters(), 'lr': 1e-5},
            {'params': model.classifier.parameters(), 'lr': 1e-3}
        ])
        
        # Test a training step
        dummy_input = torch.randn(2, 3, 224, 224)
        dummy_target = torch.randint(0, 7, (2,))
        
        model.train()
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
        
        print(f"   ✓ Training step: loss = {loss.item():.4f}")
        print("   ✅ Training setup successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Training setup failed: {e}")
        return False


def test_data_transforms():
    """Test data transforms for transfer learning."""
    print("\n4. Testing Data Transforms...")
    
    try:
        # ImageNet normalization
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        # Create dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        
        # Apply transform
        transformed = transform(dummy_image)
        
        expected_shape = (3, 224, 224)
        assert transformed.shape == expected_shape, f"Expected {expected_shape}, got {transformed.shape}"
        
        print(f"   ✓ Transform output shape: {transformed.shape}")
        print(f"   ✓ Value range: [{transformed.min():.2f}, {transformed.max():.2f}]")
        print("   ✅ Data transforms successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Data transforms failed: {e}")
        return False


def test_dataset():
    """Test dataset functionality."""
    print("\n5. Testing Dataset...")
    
    try:
        from data.dataset_emotion import EmotionDataset
        import pandas as pd
        
        # Create dummy data
        dummy_data = pd.DataFrame({
            'image_path': [f'dummy_{i}.jpg' for i in range(10)],
            'emotion': ['happiness', 'sadness', 'anger'] * 3 + ['fear']
        })
        
        dummy_label_map = {
            'happiness': 0, 'sadness': 1, 'anger': 2, 'fear': 3
        }
        
        # Test RGB dataset
        dataset = EmotionDataset(
            dummy_data,
            root_dir='/tmp',  # Won't actually load files
            label_map=dummy_label_map,
            rgb=True
        )
        
        print(f"   ✓ Dataset created: {len(dataset)} samples")
        print(f"   ✓ RGB mode: {dataset.rgb}")
        print(f"   ✓ Class distribution: {dataset.get_class_distribution()}")
        print("   ✅ Dataset test successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Dataset test failed: {e}")
        return False


def test_freeze_unfreeze():
    """Test freezing and unfreezing functionality."""
    print("\n6. Testing Freeze/Unfreeze...")
    
    try:
        from models.cnn_transfer_learning import CNNTransferLearning
        
        # Start with unfrozen model
        model = CNNTransferLearning(num_classes=7, freeze_backbone=False)
        unfrozen_params = model.get_num_params(trainable_only=True)
        
        # Freeze backbone
        model.freeze_backbone()
        frozen_params = model.get_num_params(trainable_only=True)
        
        # Unfreeze backbone
        model.unfreeze_backbone()
        unfrozen_again_params = model.get_num_params(trainable_only=True)
        
        print(f"   ✓ Unfrozen: {unfrozen_params:,} trainable parameters")
        print(f"   ✓ Frozen: {frozen_params:,} trainable parameters")
        print(f"   ✓ Unfrozen again: {unfrozen_again_params:,} trainable parameters")
        
        assert frozen_params < unfrozen_params, "Frozen model should have fewer trainable params"
        assert unfrozen_again_params == unfrozen_params, "Unfreezing should restore all params"
        
        print("   ✅ Freeze/Unfreeze successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Freeze/Unfreeze failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 CNN Transfer Learning - Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        test_model_creation,
        test_forward_pass,
        test_training_setup,
        test_data_transforms,
        test_dataset,
        test_freeze_unfreeze
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {100 * passed / (passed + failed):.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! CNN Transfer Learning implementation is working correctly!")
        print("\n🚀 Next steps:")
        print("   1. Open notebooks/CNN_Transfer_Learning.ipynb for detailed tutorial")
        print("   2. Replace dummy data with actual emotion recognition dataset")
        print("   3. Run full training with: python src/training/train_cnn_transfer.py")
    else:
        print(f"\n⚠️  Some tests failed. Please check the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)