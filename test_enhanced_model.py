#!/usr/bin/env python3
"""
🧪 Test Script for Enhanced CNN Architecture
Tests the enhanced 5-layer dense classifier without running full training.
"""

import torch
import torch.nn as nn
from torchvision import models
import sys

def test_enhanced_resnet50_classifier(num_emotion_classes=7, use_pretrained_weights=False):
    """
    Test the enhanced ResNet50 model with 5-layer dense classifier.
    
    Args:
        num_emotion_classes (int): Number of emotion classes
        use_pretrained_weights (bool): Whether to use pretrained weights (False for testing)
    
    Returns:
        torch.nn.Module: Enhanced emotion classification model
    """
    print(f"🧪 Testing Enhanced ResNet50 Classifier...")
    print(f"   📊 Number of emotion classes: {num_emotion_classes}")
    print(f"   🏗️  Using pretrained weights: {use_pretrained_weights}")
    
    # Load ResNet50 backbone
    resnet_backbone = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT if use_pretrained_weights else None
    )
    
    # Extract feature dimensions from the original fully connected layer
    backbone_feature_dimensions = resnet_backbone.fc.in_features  # 2048 for ResNet50
    print(f"   📐 Backbone feature dimensions: {backbone_feature_dimensions}")
    
    # Define enhanced classifier architecture dimensions
    dense_layer_1_features = 1024  # First dense layer: reduce from 2048 to 1024
    dense_layer_2_features = 512   # Second dense layer: reduce to 512
    dense_layer_3_features = 256   # Third dense layer: reduce to 256
    dense_layer_4_features = 128   # Fourth dense layer: reduce to 128
    
    # Dropout probabilities for regularization (progressive decrease)
    dropout_rate_layer_1 = 0.5     # Higher dropout for first layer
    dropout_rate_layer_2 = 0.4     # Moderate dropout for second layer
    dropout_rate_layer_3 = 0.3     # Lower dropout for third layer
    dropout_rate_layer_4 = 0.2     # Minimal dropout for fourth layer
    
    print(f"   🏗️  Building enhanced 5-layer dense classifier:")
    print(f"      Layer 1: {backbone_feature_dimensions} → {dense_layer_1_features} (dropout: {dropout_rate_layer_1})")
    print(f"      Layer 2: {dense_layer_1_features} → {dense_layer_2_features} (dropout: {dropout_rate_layer_2})")
    print(f"      Layer 3: {dense_layer_2_features} → {dense_layer_3_features} (dropout: {dropout_rate_layer_3})")
    print(f"      Layer 4: {dense_layer_3_features} → {dense_layer_4_features} (dropout: {dropout_rate_layer_4})")
    print(f"      Output:  {dense_layer_4_features} → {num_emotion_classes} (final classification)")
    
    # Create the enhanced 5-layer classifier (no loops!)
    enhanced_emotion_classifier = nn.Sequential(
        # 🔥 Dense Layer 1: Feature compression and initial emotion feature extraction
        nn.Linear(backbone_feature_dimensions, dense_layer_1_features, bias=True),
        nn.BatchNorm1d(dense_layer_1_features),  # Normalize features for stable training
        nn.ReLU(inplace=True),                   # Non-linear activation
        nn.Dropout(dropout_rate_layer_1),       # Regularization to prevent overfitting
        
        # 🔥 Dense Layer 2: Intermediate emotion feature refinement
        nn.Linear(dense_layer_1_features, dense_layer_2_features, bias=True),
        nn.BatchNorm1d(dense_layer_2_features),  # Batch normalization for training stability
        nn.ReLU(inplace=True),                   # ReLU activation for non-linearity
        nn.Dropout(dropout_rate_layer_2),       # Moderate regularization
        
        # 🔥 Dense Layer 3: Advanced emotion pattern recognition
        nn.Linear(dense_layer_2_features, dense_layer_3_features, bias=True),
        nn.BatchNorm1d(dense_layer_3_features),  # Continue normalization
        nn.ReLU(inplace=True),                   # Maintain non-linearity
        nn.Dropout(dropout_rate_layer_3),       # Reduced regularization
        
        # 🔥 Dense Layer 4: Fine-grained emotion feature extraction
        nn.Linear(dense_layer_3_features, dense_layer_4_features, bias=True),
        nn.BatchNorm1d(dense_layer_4_features),  # Final feature normalization
        nn.ReLU(inplace=True),                   # Last non-linear transformation
        nn.Dropout(dropout_rate_layer_4),       # Minimal regularization
        
        # 🎯 Final Classification Layer: Emotion class prediction
        nn.Linear(dense_layer_4_features, num_emotion_classes, bias=True)
        # Note: No activation here - CrossEntropyLoss expects raw logits
    )
    
    # Replace the original classifier with our enhanced version
    resnet_backbone.fc = enhanced_emotion_classifier
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in enhanced_emotion_classifier.parameters())
    trainable_params = sum(p.numel() for p in enhanced_emotion_classifier.parameters() if p.requires_grad)
    
    print(f"   ✅ Enhanced emotion classification model created successfully!")
    print(f"   📊 Total parameters in enhanced classifier: {total_params:,}")
    print(f"   🎯 Trainable parameters: {trainable_params:,}")
    
    return resnet_backbone

def test_model_forward_pass():
    """Test that the model can perform a forward pass successfully."""
    print(f"\n🔬 Testing Model Forward Pass...")
    
    # Create the enhanced model
    model = test_enhanced_resnet50_classifier(num_emotion_classes=7, use_pretrained_weights=False)
    model.eval()  # Set to evaluation mode
    
    # Create a dummy input batch (batch_size=4, channels=3, height=224, width=224)
    dummy_input = torch.randn(4, 3, 224, 224)
    print(f"   📥 Input shape: {dummy_input.shape}")
    
    # Perform forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"   📤 Output shape: {output.shape}")
    print(f"   🎯 Expected output shape: torch.Size([4, 7]) for 7 emotion classes")
    
    # Verify output shape is correct
    expected_shape = torch.Size([4, 7])
    if output.shape == expected_shape:
        print(f"   ✅ Forward pass successful! Output shape matches expected.")
        return True
    else:
        print(f"   ❌ Forward pass failed! Output shape {output.shape} != {expected_shape}")
        return False

def test_discriminative_parameter_groups():
    """Test parameter group creation for discriminative learning rates."""
    print(f"\n⚙️  Testing Discriminative Parameter Groups...")
    
    model = test_enhanced_resnet50_classifier(num_emotion_classes=7, use_pretrained_weights=False)
    
    # Simple parameter group creation (simplified version)
    backbone_parameters = []
    classifier_parameters = []
    
    # Get classifier module
    classifier_module = model.fc
    classifier_param_ids = set(id(p) for p in classifier_module.parameters())
    
    # Separate parameters
    for param in model.parameters():
        if id(param) in classifier_param_ids:
            classifier_parameters.append(param)
        else:
            backbone_parameters.append(param)
    
    print(f"   🏗️  Backbone parameters: {len(backbone_parameters):,}")
    print(f"   🎯 Classifier parameters: {len(classifier_parameters):,}")
    
    # Verify we have parameters in both groups
    if len(backbone_parameters) > 0 and len(classifier_parameters) > 0:
        print(f"   ✅ Parameter group separation successful!")
        return True
    else:
        print(f"   ❌ Parameter group separation failed!")
        return False

def main():
    """Run all tests for the enhanced model."""
    print(f"{'='*80}")
    print(f"🧪 ENHANCED CNN ARCHITECTURE TESTING")
    print(f"{'='*80}")
    
    tests_passed = 0
    total_tests = 3
    
    try:
        # Test 1: Model Creation
        print(f"\n🧪 Test 1/3: Model Creation")
        model = test_enhanced_resnet50_classifier()
        if model is not None:
            print(f"   ✅ Test 1 PASSED")
            tests_passed += 1
        else:
            print(f"   ❌ Test 1 FAILED")
    except Exception as e:
        print(f"   ❌ Test 1 FAILED with error: {e}")
    
    try:
        # Test 2: Forward Pass
        print(f"\n🧪 Test 2/3: Forward Pass")
        if test_model_forward_pass():
            print(f"   ✅ Test 2 PASSED")
            tests_passed += 1
        else:
            print(f"   ❌ Test 2 FAILED")
    except Exception as e:
        print(f"   ❌ Test 2 FAILED with error: {e}")
    
    try:
        # Test 3: Parameter Groups
        print(f"\n🧪 Test 3/3: Parameter Groups")
        if test_discriminative_parameter_groups():
            print(f"   ✅ Test 3 PASSED")
            tests_passed += 1
        else:
            print(f"   ❌ Test 3 FAILED")
    except Exception as e:
        print(f"   ❌ Test 3 FAILED with error: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Tests passed: {tests_passed}/{total_tests}")
    print(f"📈 Success rate: {tests_passed/total_tests*100:.1f}%")
    
    if tests_passed == total_tests:
        print(f"🎉 ALL TESTS PASSED! Enhanced architecture is ready for training.")
        return 0
    else:
        print(f"❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())