# CNN Transfer Learning model for emotion recognition
import torch
import torch.nn as nn
from torchvision import models


class CNNTransferLearning(nn.Module):
    """
    CNN Transfer Learning model using pre-trained backbones for emotion recognition.
    
    This model uses pre-trained CNN architectures (VGG, AlexNet, etc.) as feature 
    extractors and adds custom classifier layers for emotion classification.
    """
    
    def __init__(self, num_classes=7, backbone='vgg16', pretrained=True, freeze_backbone=False):
        """
        Initialize the transfer learning model.
        
        Args:
            num_classes (int): Number of emotion classes
            backbone (str): Pre-trained model to use ('vgg16', 'vgg19', 'alexnet')
            pretrained (bool): Whether to use pre-trained weights
            freeze_backbone (bool): Whether to freeze backbone weights during training
        """
        super(CNNTransferLearning, self).__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.frozen = freeze_backbone
        
        # Load pre-trained backbone
        if backbone == 'vgg16':
            self.backbone = models.vgg16(pretrained=pretrained)
            backbone_out_features = 25088  # VGG16 feature output size
        elif backbone == 'vgg19':
            self.backbone = models.vgg19(pretrained=pretrained)
            backbone_out_features = 25088  # VGG19 feature output size
        elif backbone == 'alexnet':
            self.backbone = models.alexnet(pretrained=pretrained)
            backbone_out_features = 9216   # AlexNet feature output size
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. "
                           f"Supported: ['vgg16', 'vgg19', 'alexnet']")
        
        # Extract features and adaptive pooling from backbone
        if backbone in ['vgg16', 'vgg19']:
            self.features = self.backbone.features
            self.avgpool = self.backbone.avgpool
        elif backbone == 'alexnet':
            self.features = self.backbone.features
            self.avgpool = self.backbone.avgpool
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        
        # Create custom classifier for emotion recognition
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(backbone_out_features, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier()
    
    def _freeze_backbone(self):
        """Freeze all parameters in the backbone."""
        for param in self.features.parameters():
            param.requires_grad = False
        print(f"Backbone ({self.backbone_name}) weights frozen")
    
    def _unfreeze_backbone(self):
        """Unfreeze all parameters in the backbone."""
        for param in self.features.parameters():
            param.requires_grad = True
        print(f"Backbone ({self.backbone_name}) weights unfrozen")
        self.frozen = False
    
    def _initialize_classifier(self):
        """Initialize classifier weights using Xavier initialization."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Extract features using pre-trained backbone
        x = self.features(x)
        x = self.avgpool(x)
        
        # Flatten features
        x = torch.flatten(x, 1)
        
        # Classify emotions
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
        """
        Get number of parameters in the model.
        
        Args:
            trainable_only (bool): If True, count only trainable parameters
            
        Returns:
            int: Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def get_backbone_params(self):
        """Get number of parameters in the backbone."""
        return sum(p.numel() for p in self.features.parameters())
    
    def get_classifier_params(self):
        """Get number of parameters in the classifier."""
        return sum(p.numel() for p in self.classifier.parameters())
    
    def print_model_info(self):
        """Print detailed information about the model."""
        print(f"\n{'='*50}")
        print(f"CNN Transfer Learning Model Information")
        print(f"{'='*50}")
        print(f"Backbone: {self.backbone_name}")
        print(f"Pretrained: Yes")
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


class CNNTransferLearningEnsemble(nn.Module):
    """
    Ensemble of multiple CNN Transfer Learning models with different backbones.
    """
    
    def __init__(self, num_classes=7, backbones=['vgg16', 'vgg19'], pretrained=True):
        """
        Initialize ensemble model.
        
        Args:
            num_classes (int): Number of emotion classes
            backbones (list): List of backbone names to use
            pretrained (bool): Whether to use pre-trained weights
        """
        super(CNNTransferLearningEnsemble, self).__init__()
        
        self.models = nn.ModuleList()
        self.backbone_names = backbones
        
        for backbone in backbones:
            model = CNNTransferLearning(
                num_classes=num_classes,
                backbone=backbone,
                pretrained=pretrained,
                freeze_backbone=False
            )
            self.models.append(model)
        
        print(f"Ensemble created with {len(backbones)} models: {backbones}")
    
    def forward(self, x):
        """
        Forward pass through ensemble (averages predictions).
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Averaged output logits
        """
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Average the outputs
        ensemble_output = torch.stack(outputs).mean(dim=0)
        return ensemble_output
    
    def get_num_params(self):
        """Get total number of parameters in the ensemble."""
        return sum(model.get_num_params() for model in self.models)


def create_cnn_transfer_model(num_classes=7, backbone='vgg16', pretrained=True, 
                             freeze_backbone=False, device='cpu'):
    """
    Factory function to create and initialize a CNN transfer learning model.
    
    Args:
        num_classes (int): Number of emotion classes
        backbone (str): Pre-trained backbone to use
        pretrained (bool): Whether to use pre-trained weights
        freeze_backbone (bool): Whether to freeze backbone weights
        device (str): Device to move model to
        
    Returns:
        CNNTransferLearning: Initialized model
    """
    model = CNNTransferLearning(
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
    # Test the model creation
    print("Testing CNN Transfer Learning Model...")
    
    # Create model
    model = create_cnn_transfer_model(
        num_classes=7,
        backbone='vgg16',
        pretrained=True,
        freeze_backbone=False,
        device='cpu'
    )
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch of 4 RGB images
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test ensemble
    print(f"\nTesting Ensemble Model...")
    ensemble = CNNTransferLearningEnsemble(
        num_classes=7,
        backbones=['vgg16', 'alexnet'],
        pretrained=True
    )
    
    ensemble_output = ensemble(dummy_input)
    print(f"Ensemble output shape: {ensemble_output.shape}")
    print(f"Total ensemble parameters: {ensemble.get_num_params():,}")