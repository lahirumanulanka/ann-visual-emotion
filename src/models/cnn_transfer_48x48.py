# CNN Transfer Learning model optimized for 48x48 input
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class CNNTransferLearning48x48(nn.Module):
    """
    CNN Transfer Learning model optimized for 48x48 input images.
    
    This model adapts pre-trained CNN architectures to work directly with 48x48 input
    without requiring upscaling to 224x224. It uses transfer learning principles while
    being computationally efficient for small images.
    """
    
    def __init__(self, num_classes=7, backbone='vgg16', pretrained=True, freeze_backbone=False):
        """
        Initialize the 48x48 transfer learning model.
        
        Args:
            num_classes (int): Number of emotion classes
            backbone (str): Pre-trained model to use ('vgg16', 'vgg19', 'resnet18')
            pretrained (bool): Whether to use pre-trained weights
            freeze_backbone (bool): Whether to freeze backbone weights during training
        """
        super(CNNTransferLearning48x48, self).__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.frozen = freeze_backbone
        
        # Create adapted backbone for 48x48 input
        if backbone == 'vgg16':
            self._create_vgg16_48x48(pretrained)
        elif backbone == 'vgg19':
            self._create_vgg19_48x48(pretrained)
        elif backbone == 'resnet18':
            self._create_resnet18_48x48(pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. "
                           f"Supported: ['vgg16', 'vgg19', 'resnet18']")
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
    
    def _create_vgg16_48x48(self, pretrained):
        """Create VGG16 adapted for 48x48 input."""
        # Load pre-trained VGG16
        vgg = models.vgg16(pretrained=pretrained)
        
        # Adapt the features for 48x48 input
        # We'll use the first several layers but with modifications
        features = []
        
        # First block: 48x48 -> 24x24 (keep original structure)
        features.extend([
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Modified for RGB input
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        
        # Second block: 24x24 -> 12x12
        features.extend([
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        
        # Third block: 12x12 -> 6x6
        features.extend([
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        
        # Fourth block: 6x6 -> 3x3
        features.extend([
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        
        self.features = nn.Sequential(*features)
        
        # Copy pre-trained weights where possible
        if pretrained:
            self._copy_vgg_weights(vgg)
        
        # Adaptive pooling to handle variable input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Custom classifier for emotion recognition (smaller than original)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier()
    
    def _create_vgg19_48x48(self, pretrained):
        """Create VGG19 adapted for 48x48 input."""
        # Similar to VGG16 but with more layers
        vgg = models.vgg19(pretrained=pretrained)
        
        features = []
        
        # First block: 48x48 -> 24x24
        features.extend([
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        
        # Second block: 24x24 -> 12x12
        features.extend([
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        
        # Third block: 12x12 -> 6x6 (more layers than VGG16)
        features.extend([
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Extra layer for VGG19
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        
        # Fourth block: 6x6 -> 3x3
        features.extend([
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Extra layer for VGG19
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        
        self.features = nn.Sequential(*features)
        
        if pretrained:
            self._copy_vgg_weights(vgg)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Same classifier as VGG16
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )
        
        self._initialize_classifier()
    
    def _create_resnet18_48x48(self, pretrained):
        """Create ResNet18 adapted for 48x48 input."""
        resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer for smaller input
        # Original: 7x7 conv with stride 2
        # Modified: 3x3 conv with stride 1 for 48x48 input
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip the maxpool layer since our input is small
        # Use ResNet's remaining layers but adapted
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels  
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        # Copy pre-trained weights for layers 1-4
        if pretrained:
            self.layer1.load_state_dict(resnet.layer1.state_dict())
            self.layer2.load_state_dict(resnet.layer2.state_dict())
            self.layer3.load_state_dict(resnet.layer3.state_dict())
            self.layer4.load_state_dict(resnet.layer4.state_dict())
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )
        
        self._initialize_classifier()
    
    def _copy_vgg_weights(self, pretrained_vgg):
        """Copy compatible weights from pre-trained VGG to our adapted model."""
        pretrained_features = pretrained_vgg.features
        our_features = self.features
        
        # Map layers that have compatible shapes
        layer_mapping = []
        pretrained_idx = 0
        our_idx = 0
        
        while pretrained_idx < len(pretrained_features) and our_idx < len(our_features):
            pretrained_layer = pretrained_features[pretrained_idx]
            our_layer = our_features[our_idx]
            
            if isinstance(pretrained_layer, nn.Conv2d) and isinstance(our_layer, nn.Conv2d):
                # Check if shapes are compatible (skip first layer since it might be different)
                if (pretrained_layer.weight.shape == our_layer.weight.shape and 
                    our_idx > 0):  # Skip first layer adaptation
                    layer_mapping.append((pretrained_idx, our_idx))
            
            pretrained_idx += 1
            our_idx += 1
        
        # Copy compatible weights
        for p_idx, o_idx in layer_mapping:
            our_features[o_idx].weight.data = pretrained_features[p_idx].weight.data.clone()
            if pretrained_features[p_idx].bias is not None:
                our_features[o_idx].bias.data = pretrained_features[p_idx].bias.data.clone()
        
        print(f"Copied {len(layer_mapping)} compatible layers from pre-trained {self.backbone_name}")
    
    def _freeze_backbone(self):
        """Freeze all parameters in the backbone."""
        for param in self.features.parameters():
            param.requires_grad = False
        if hasattr(self, 'layer1'):  # ResNet case
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"Backbone ({self.backbone_name}) weights frozen")
    
    def _unfreeze_backbone(self):
        """Unfreeze all parameters in the backbone."""
        for param in self.features.parameters():
            param.requires_grad = True
        if hasattr(self, 'layer1'):  # ResNet case
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for param in layer.parameters():
                    param.requires_grad = True
        print(f"Backbone ({self.backbone_name}) weights unfrozen")
        self.frozen = False
    
    def _initialize_classifier(self):
        """Initialize classifier weights using Xavier initialization."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 48, 48)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        if self.backbone_name == 'resnet18':
            # ResNet forward pass
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            # Skip maxpool for 48x48 input
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        else:
            # VGG forward pass
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        
        return x
    
    def unfreeze_backbone(self):
        """Public method to unfreeze backbone for fine-tuning."""
        self._unfreeze_backbone()
    
    def freeze_backbone(self):
        """Public method to freeze backbone layers."""
        self._freeze_backbone()
        self.frozen = True
    
    def get_num_params(self, trainable_only=True):
        """Get number of parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def get_backbone_params(self):
        """Get number of parameters in the backbone."""
        if self.backbone_name == 'resnet18':
            backbone_modules = [self.conv1, self.layer1, self.layer2, self.layer3, self.layer4]
            return sum(sum(p.numel() for p in module.parameters()) for module in backbone_modules)
        else:
            return sum(p.numel() for p in self.features.parameters())
    
    def get_classifier_params(self):
        """Get number of parameters in the classifier."""
        return sum(p.numel() for p in self.classifier.parameters())
    
    def print_model_info(self):
        """Print detailed information about the model."""
        print(f"\n{'='*60}")
        print(f"CNN Transfer Learning Model for 48x48 Input")
        print(f"{'='*60}")
        print(f"Backbone: {self.backbone_name}")
        print(f"Input size: 48x48x3 (RGB)")
        print(f"Pretrained: Yes (adapted)")
        print(f"Frozen: {self.frozen}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Total parameters: {self.get_num_params(trainable_only=False):,}")
        print(f"Trainable parameters: {self.get_num_params(trainable_only=True):,}")
        print(f"Backbone parameters: {self.get_backbone_params():,}")
        print(f"Classifier parameters: {self.get_classifier_params():,}")
        
        if self.frozen:
            print(f"Training strategy: Feature extraction (backbone frozen)")
        else:
            print(f"Training strategy: Fine-tuning (all layers trainable)")


def create_cnn_transfer_48x48_model(num_classes=7, backbone='vgg16', pretrained=True, 
                                   freeze_backbone=False, device='cpu'):
    """
    Factory function to create and initialize a CNN transfer learning model for 48x48 input.
    
    Args:
        num_classes (int): Number of emotion classes
        backbone (str): Pre-trained backbone to use
        pretrained (bool): Whether to use pre-trained weights  
        freeze_backbone (bool): Whether to freeze backbone weights
        device (str): Device to move model to
        
    Returns:
        CNNTransferLearning48x48: Initialized model
    """
    model = CNNTransferLearning48x48(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    
    model = model.to(device)
    model.print_model_info()
    
    return model


# Example usage and testing
if __name__ == "__main__":
    print("Testing CNN Transfer Learning Model for 48x48 Input...")
    
    # Test VGG16 model
    model = create_cnn_transfer_48x48_model(
        num_classes=7,
        backbone='vgg16',
        pretrained=True,
        freeze_backbone=False,
        device='cpu'
    )
    
    # Test forward pass with 48x48 input
    dummy_input = torch.randn(4, 3, 48, 48)  # Batch of 4 RGB 48x48 images
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test ResNet18 model
    print(f"\nTesting ResNet18 for 48x48...")
    resnet_model = create_cnn_transfer_48x48_model(
        num_classes=7,
        backbone='resnet18',
        pretrained=True,
        freeze_backbone=False,
        device='cpu'
    )
    
    resnet_output = resnet_model(dummy_input)
    print(f"ResNet18 output shape: {resnet_output.shape}")