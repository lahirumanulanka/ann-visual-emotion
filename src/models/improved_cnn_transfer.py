# Improved CNN Transfer Learning model for emotion recognition
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ImprovedCNNTransferLearning(nn.Module):
    """
    Improved CNN Transfer Learning model with enhanced architecture and features.
    
    This model implements several improvements over the basic transfer learning:
    - ResNet50 backbone for better feature extraction
    - Enhanced classifier with dropout and batch normalization
    - Support for multiple backbone architectures
    - Advanced regularization techniques
    """
    
    def __init__(self, num_classes=7, backbone='resnet50', pretrained=True, 
                 freeze_backbone=False, dropout_rate=0.5, use_attention=False):
        """
        Initialize the improved transfer learning model.
        
        Args:
            num_classes (int): Number of emotion classes
            backbone (str): Pre-trained model to use ('resnet50', 'resnet101', 'efficientnet_b4', etc.)
            pretrained (bool): Whether to use pre-trained weights
            freeze_backbone (bool): Whether to freeze backbone weights during training
            dropout_rate (float): Dropout rate for regularization
            use_attention (bool): Whether to add attention mechanism
        """
        super(ImprovedCNNTransferLearning, self).__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.frozen = freeze_backbone
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        
        # Load pre-trained backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_out_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final FC layer
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            backbone_out_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'efficientnet_b4':
            self.backbone = models.efficientnet_b4(pretrained=pretrained)
            backbone_out_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            backbone_out_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'vgg16':
            self.backbone = models.vgg16(pretrained=pretrained)
            self.features = self.backbone.features
            self.avgpool = self.backbone.avgpool
            backbone_out_features = 25088
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        
        # Add attention mechanism if requested
        if use_attention:
            self.attention = SpatialAttention(backbone_out_features)
        
        # Create enhanced classifier
        if backbone == 'vgg16':
            self.classifier = self._create_vgg_classifier(backbone_out_features)
        else:
            self.classifier = self._create_resnet_classifier(backbone_out_features)
        
        # Initialize classifier weights
        self._initialize_classifier()
    
    def _create_resnet_classifier(self, in_features):
        """Create classifier for ResNet-like architectures."""
        return nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Dropout(self.dropout_rate * 0.8),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Dropout(self.dropout_rate * 0.6),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Dropout(self.dropout_rate * 0.4),
            nn.Linear(256, self.num_classes)
        )
    
    def _create_vgg_classifier(self, in_features):
        """Create classifier for VGG architecture."""
        return nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features, 4096),
            nn.ReLU(True),
            nn.BatchNorm1d(4096),
            nn.Dropout(self.dropout_rate * 0.8),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.BatchNorm1d(2048),
            nn.Dropout(self.dropout_rate * 0.6),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Dropout(self.dropout_rate * 0.4),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Dropout(self.dropout_rate * 0.2),
            nn.Linear(512, self.num_classes)
        )
    
    def _freeze_backbone(self):
        """Freeze all parameters in the backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"Backbone ({self.backbone_name}) weights frozen")
    
    def _unfreeze_backbone(self):
        """Unfreeze all parameters in the backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f"Backbone ({self.backbone_name}) weights unfrozen")
        self.frozen = False
    
    def _initialize_classifier(self):
        """Initialize classifier weights using Xavier initialization."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
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
        if self.backbone_name == 'vgg16':
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        else:
            x = self.backbone(x)
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)
        
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
        """Get number of parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def print_model_info(self):
        """Print detailed information about the model."""
        print(f"\n{'='*60}")
        print(f"Improved CNN Transfer Learning Model Information")
        print(f"{'='*60}")
        print(f"Backbone: {self.backbone_name}")
        print(f"Pretrained: Yes")
        print(f"Frozen: {self.frozen}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Dropout rate: {self.dropout_rate}")
        print(f"Attention: {self.use_attention}")
        print(f"Total parameters: {self.get_num_params(trainable_only=False):,}")
        print(f"Trainable parameters: {self.get_num_params(trainable_only=True):,}")
        
        if self.frozen:
            print(f"Training strategy: Feature extraction (backbone frozen)")
        else:
            print(f"Training strategy: Fine-tuning (all layers trainable)")


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism for feature enhancement.
    """
    
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if len(x.shape) == 2:
            return x  # Skip if already flattened
        
        attention_weights = self.attention(x)
        return x * attention_weights


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing cross entropy loss for better generalization.
    """
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        num_classes = pred.size(1)
        target = target.long()
        
        # Create smoothed labels
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=1), dim=1))


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_improved_model(num_classes=7, backbone='resnet50', pretrained=True, 
                         freeze_backbone=False, dropout_rate=0.5, use_attention=False, device='cpu'):
    """
    Factory function to create and initialize an improved CNN transfer learning model.
    
    Args:
        num_classes (int): Number of emotion classes
        backbone (str): Pre-trained backbone to use
        pretrained (bool): Whether to use pre-trained weights
        freeze_backbone (bool): Whether to freeze backbone weights
        dropout_rate (float): Dropout rate for regularization
        use_attention (bool): Whether to add attention mechanism
        device (str): Device to move model to
        
    Returns:
        ImprovedCNNTransferLearning: Initialized model
    """
    model = ImprovedCNNTransferLearning(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate,
        use_attention=use_attention
    )
    
    model = model.to(device)
    model.print_model_info()
    
    return model


# Example usage and testing
if __name__ == "__main__":
    print("Testing Improved CNN Transfer Learning Model...")
    
    # Test ResNet50 model
    model = create_improved_model(
        num_classes=6,
        backbone='resnet50',
        pretrained=True,
        freeze_backbone=False,
        dropout_rate=0.5,
        use_attention=False,
        device='cpu'
    )
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test loss functions
    print(f"\nTesting loss functions...")
    target = torch.randint(0, 6, (4,))
    
    # Standard CrossEntropy
    ce_loss = F.cross_entropy(output, target)
    print(f"CrossEntropy Loss: {ce_loss.item():.4f}")
    
    # Label Smoothing
    ls_loss = LabelSmoothingCrossEntropy(smoothing=0.1)(output, target)
    print(f"Label Smoothing Loss: {ls_loss.item():.4f}")
    
    # Focal Loss
    focal_loss = FocalLoss(alpha=1, gamma=2)(output, target)
    print(f"Focal Loss: {focal_loss.item():.4f}")