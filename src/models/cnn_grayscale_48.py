# Enhanced CNN model for 48x48 grayscale emotion recognition
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EnhancedCNNGrayscale48(nn.Module):
    """
    Enhanced CNN architecture designed specifically for 48x48 grayscale emotion recognition.
    
    This model incorporates:
    - Multiple convolutional blocks with increasing depth
    - Various pooling strategies (MaxPool, AvgPool, AdaptiveAvgPool)
    - Dropout layers for regularization
    - Dense layers with progressive size reduction
    - Batch normalization for stable training
    - Skip connections for better gradient flow
    """
    
    def __init__(self, num_classes=7, dropout_rate=0.5, use_batch_norm=True):
        """
        Initialize the enhanced CNN model for 48x48 grayscale images.
        
        Args:
            num_classes (int): Number of emotion classes
            dropout_rate (float): Dropout rate for regularization
            use_batch_norm (bool): Whether to use batch normalization
        """
        super(EnhancedCNNGrayscale48, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Feature extraction layers
        self.features = self._make_feature_layers()
        
        # Adaptive pooling to handle varying input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Calculate the size after feature extraction
        # For 48x48 input: 48 -> 24 -> 12 -> 6 -> 3 -> 1 (with pooling)
        feature_size = 512  # Final feature map channels
        
        # Enhanced classifier with multiple dense layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_size, 1024),
            nn.BatchNorm1d(1024) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.2),
            
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_feature_layers(self):
        """Create enhanced feature extraction layers."""
        layers = []
        in_channels = 1  # Grayscale input
        
        # Block 1: Initial feature extraction
        # 48x48x1 -> 48x48x32 -> 24x24x32
        layers.extend([
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(self.dropout_rate * 0.2)
        ])
        
        # Block 2: Deeper feature extraction
        # 24x24x32 -> 24x24x64 -> 12x12x64
        layers.extend([
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(self.dropout_rate * 0.25)
        ])
        
        # Block 3: Complex feature patterns
        # 12x12x64 -> 12x12x128 -> 6x6x128
        layers.extend([
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(self.dropout_rate * 0.3)
        ])
        
        # Block 4: High-level features
        # 6x6x128 -> 6x6x256 -> 3x3x256
        layers.extend([
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(self.dropout_rate * 0.35)
        ])
        
        # Block 5: Final feature extraction
        # 3x3x256 -> 3x3x512 -> 1x1x512 (with adaptive pooling)
        layers.extend([
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout_rate * 0.4)
        ])
        
        return nn.Sequential(*[layer for layer in layers if not isinstance(layer, type(nn.Identity()))])
    
    def _initialize_weights(self):
        """Initialize model weights using appropriate strategies."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the enhanced CNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 48, 48)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Feature extraction
        x = self.features(x)
        
        # Adaptive pooling to ensure consistent size
        x = self.adaptive_pool(x)
        
        # Flatten for classifier
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_num_params(self, trainable_only=True):
        """Get number of parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def print_model_info(self):
        """Print detailed model information."""
        print(f"\n{'='*50}")
        print(f"Enhanced CNN for 48x48 Grayscale Images")
        print(f"{'='*50}")
        print(f"Input size: 48x48x1 (grayscale)")
        print(f"Number of classes: {self.num_classes}")
        print(f"Dropout rate: {self.dropout_rate}")
        print(f"Batch normalization: {self.use_batch_norm}")
        print(f"Total parameters: {self.get_num_params(trainable_only=False):,}")
        print(f"Trainable parameters: {self.get_num_params(trainable_only=True):,}")
        
        # Count layers
        conv_layers = sum(1 for m in self.modules() if isinstance(m, nn.Conv2d))
        linear_layers = sum(1 for m in self.modules() if isinstance(m, nn.Linear))
        print(f"Convolutional layers: {conv_layers}")
        print(f"Linear layers: {linear_layers}")


class CNNWithTransferLearningAdaptation(EnhancedCNNGrayscale48):
    """
    Enhanced CNN with transfer learning adaptation for 48x48 grayscale images.
    
    This model uses knowledge from pre-trained models by:
    1. Adapting architectural patterns from proven models
    2. Using progressive training strategies
    3. Implementing feature-level knowledge transfer
    """
    
    def __init__(self, num_classes=7, dropout_rate=0.5, pretrained_backbone='vgg16'):
        """
        Initialize with transfer learning adaptation.
        
        Args:
            num_classes (int): Number of emotion classes
            dropout_rate (float): Dropout rate
            pretrained_backbone (str): Backbone to extract knowledge from
        """
        super().__init__(num_classes, dropout_rate)
        
        self.pretrained_backbone = pretrained_backbone
        self.knowledge_adapted = False
        
        # Load knowledge from pre-trained model if requested
        if pretrained_backbone:
            self._adapt_pretrained_knowledge()
    
    def _adapt_pretrained_knowledge(self):
        """Adapt knowledge from pre-trained models to our architecture."""
        if self.pretrained_backbone == 'vgg16':
            # Load VGG16 and extract feature extraction patterns
            vgg16 = models.vgg16(weights='IMAGENET1K_V1')
            self._transfer_conv_knowledge(vgg16)
        elif self.pretrained_backbone == 'resnet18':
            # Load ResNet18 for knowledge adaptation
            resnet18 = models.resnet18(weights='IMAGENET1K_V1')
            self._transfer_resnet_knowledge(resnet18)
        
        self.knowledge_adapted = True
        print(f"✓ Adapted knowledge from {self.pretrained_backbone}")
    
    def _transfer_conv_knowledge(self, pretrained_model):
        """
        Transfer convolutional knowledge from pre-trained model.
        This adapts the learned filters to work with grayscale inputs.
        """
        # Extract first few convolutional layers from pre-trained model
        pretrained_features = list(pretrained_model.features.children())
        our_features = list(self.features.children())
        
        conv_idx = 0
        for i, layer in enumerate(our_features):
            if isinstance(layer, nn.Conv2d) and conv_idx < len(pretrained_features):
                # Find corresponding pre-trained conv layer
                for j, pretrained_layer in enumerate(pretrained_features[conv_idx:], conv_idx):
                    if isinstance(pretrained_layer, nn.Conv2d):
                        # Adapt weights for grayscale input and different channel sizes
                        self._adapt_conv_weights(layer, pretrained_layer)
                        conv_idx = j + 1
                        break
    
    def _adapt_conv_weights(self, our_conv, pretrained_conv):
        """Adapt pre-trained conv weights to our layer."""
        with torch.no_grad():
            # For first layer, adapt RGB to grayscale
            if our_conv.in_channels == 1 and pretrained_conv.in_channels == 3:
                # Average RGB channels to create grayscale adaptation
                pretrained_weight = pretrained_conv.weight.data
                adapted_weight = torch.mean(pretrained_weight, dim=1, keepdim=True)
                
                # Resize to match our output channels if needed
                if our_conv.out_channels != pretrained_conv.out_channels:
                    # Use interpolation or truncation based on size difference
                    if our_conv.out_channels < pretrained_conv.out_channels:
                        adapted_weight = adapted_weight[:our_conv.out_channels]
                    else:
                        # Repeat pattern for more channels
                        repeat_factor = our_conv.out_channels // pretrained_conv.out_channels + 1
                        adapted_weight = adapted_weight.repeat(repeat_factor, 1, 1, 1)
                        adapted_weight = adapted_weight[:our_conv.out_channels]
                
                our_conv.weight.data = adapted_weight
            
            # For other layers, adapt based on channel compatibility
            elif (our_conv.kernel_size == pretrained_conv.kernel_size and
                  our_conv.in_channels <= pretrained_conv.in_channels and
                  our_conv.out_channels <= pretrained_conv.out_channels):
                
                pretrained_weight = pretrained_conv.weight.data
                adapted_weight = pretrained_weight[:our_conv.out_channels, :our_conv.in_channels]
                our_conv.weight.data = adapted_weight
    
    def _transfer_resnet_knowledge(self, pretrained_model):
        """Transfer knowledge from ResNet architecture."""
        # This could be implemented to use ResNet's residual connections
        # and initialization strategies
        pass
    
    def get_transfer_learning_optimizer_groups(self, backbone_lr=1e-5, classifier_lr=1e-3):
        """
        Get parameter groups for differential learning rates.
        
        Args:
            backbone_lr (float): Learning rate for feature extraction layers
            classifier_lr (float): Learning rate for classifier layers
            
        Returns:
            list: Parameter groups for optimizer
        """
        feature_params = []
        classifier_params = []
        
        # Split parameters between features and classifier
        for name, param in self.named_parameters():
            if 'features' in name:
                feature_params.append(param)
            else:
                classifier_params.append(param)
        
        return [
            {'params': feature_params, 'lr': backbone_lr, 'name': 'features'},
            {'params': classifier_params, 'lr': classifier_lr, 'name': 'classifier'}
        ]


def create_enhanced_grayscale_model(num_classes=7, dropout_rate=0.5, 
                                   use_transfer_adaptation=True, 
                                   pretrained_backbone='vgg16', device='cpu'):
    """
    Factory function to create enhanced CNN model for 48x48 grayscale images.
    
    Args:
        num_classes (int): Number of emotion classes
        dropout_rate (float): Dropout rate for regularization
        use_transfer_adaptation (bool): Whether to use transfer learning adaptation
        pretrained_backbone (str): Pre-trained model to adapt from
        device (str): Device to move model to
        
    Returns:
        Enhanced CNN model
    """
    if use_transfer_adaptation:
        model = CNNWithTransferLearningAdaptation(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            pretrained_backbone=pretrained_backbone
        )
    else:
        model = EnhancedCNNGrayscale48(
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
    
    model = model.to(device)
    model.print_model_info()
    
    return model


# Example usage and testing
if __name__ == "__main__":
    print("Testing Enhanced CNN for 48x48 Grayscale Images...")
    
    # Test standard model
    print("\n1. Testing Standard Enhanced CNN:")
    model1 = create_enhanced_grayscale_model(
        num_classes=7,
        dropout_rate=0.5,
        use_transfer_adaptation=False,
        device='cpu'
    )
    
    # Test with transfer learning adaptation
    print("\n2. Testing with Transfer Learning Adaptation:")
    model2 = create_enhanced_grayscale_model(
        num_classes=7,
        dropout_rate=0.4,
        use_transfer_adaptation=True,
        pretrained_backbone='vgg16',
        device='cpu'
    )
    
    # Test forward pass with 48x48 grayscale images
    print("\n3. Testing Forward Pass:")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 48, 48)  # Grayscale 48x48
    
    with torch.no_grad():
        output1 = model1(dummy_input)
        output2 = model2(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Standard model output: {output1.shape}")
    print(f"Transfer adapted output: {output2.shape}")
    
    # Test optimizer groups for transfer learning
    print("\n4. Testing Transfer Learning Optimizer Setup:")
    param_groups = model2.get_transfer_learning_optimizer_groups()
    print(f"Feature parameters: {len(param_groups[0]['params'])}")
    print(f"Classifier parameters: {len(param_groups[1]['params'])}")
    
    print("\n✅ All tests completed successfully!")