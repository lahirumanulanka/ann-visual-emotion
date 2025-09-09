# Generative AI-based Image Enhancement Pipeline for Emotion Recognition
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class SRCNN(nn.Module):
    """
    Super-Resolution Convolutional Neural Network (SRCNN)
    A lightweight CNN for image super-resolution that works well for facial emotion images.
    """
    def __init__(self, num_channels=1, upscale_factor=4):
        super(SRCNN, self).__init__()
        self.upscale_factor = upscale_factor
        
        # Feature extraction layer
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        
        # Non-linear mapping layers
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=5, padding=2)
        
        # Reconstruction layer
        self.conv4 = nn.Conv2d(16, num_channels, kernel_size=5, padding=2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass through SRCNN."""
        # Bicubic upsampling first
        x = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        
        # Feature extraction
        x = F.relu(self.conv1(x))
        
        # Non-linear mapping
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Reconstruction
        x = self.conv4(x)
        
        return torch.clamp(x, 0, 1)


class EnhancedSRCNN(nn.Module):
    """
    Enhanced SRCNN with additional layers and skip connections for better emotion preservation.
    """
    def __init__(self, num_channels=1, upscale_factor=4):
        super(EnhancedSRCNN, self).__init__()
        self.upscale_factor = upscale_factor
        
        # Initial feature extraction
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Enhanced feature mapping with residual blocks
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Attention mechanism for facial features
        self.attention = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final reconstruction
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16, num_channels, kernel_size=3, padding=1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Store low-resolution input for skip connection
        lr_input = x
        
        # Bicubic upsampling
        x = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        lr_upsampled = x.clone()
        
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Enhanced feature mapping
        residual = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = x + residual  # Residual connection
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Apply attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Reconstruction with skip connection
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        
        # Add skip connection from upsampled input
        x = x + lr_upsampled
        
        return torch.clamp(x, 0, 1)


class EmotionImageEnhancer:
    """
    Main class for enhancing emotion images using generative AI techniques.
    Combines multiple enhancement methods for optimal emotion preservation.
    """
    
    def __init__(self, model_type='enhanced_srcnn', device='cpu'):
        """
        Initialize the emotion image enhancer.
        
        Args:
            model_type (str): Type of enhancement model ('srcnn', 'enhanced_srcnn')
            device (str): Device to run models on
        """
        self.device = torch.device(device)
        self.model_type = model_type
        self.model = None
        
        # Create model
        if model_type == 'srcnn':
            self.model = SRCNN(num_channels=1, upscale_factor=4)
        elif model_type == 'enhanced_srcnn':
            self.model = EnhancedSRCNN(num_channels=1, upscale_factor=4)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Emotion Image Enhancer initialized with {model_type}")
        print(f"✓ Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def preprocess_image(self, image_path, target_size=48):
        """
        Preprocess image for enhancement.
        
        Args:
            image_path (str/Path): Path to the image file
            target_size (int): Target size for preprocessing
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Load image as grayscale
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('L')
            else:
                image = image_path.convert('L')
            
            # Resize to target size if needed
            if image.size != (target_size, target_size):
                image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
            
            # Convert to tensor and normalize
            transform = transforms.Compose([
                transforms.ToTensor(),
                # Normalize to [0, 1] range (already done by ToTensor for PIL images)
            ])
            
            tensor = transform(image).unsqueeze(0)  # Add batch dimension
            tensor = tensor.float()  # Ensure float32 type
            return tensor.to(self.device)
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def enhance_image(self, image_tensor, apply_postprocessing=True):
        """
        Enhance a single image tensor using the loaded model.
        
        Args:
            image_tensor (torch.Tensor): Input image tensor (1, 1, H, W)
            apply_postprocessing (bool): Whether to apply additional post-processing
            
        Returns:
            torch.Tensor: Enhanced image tensor
        """
        # Enhanced SRCNN upsampling
        with torch.no_grad():
            # Run through enhancement model
            enhanced = self.model(image_tensor)
            
            if apply_postprocessing:
                enhanced = self._apply_postprocessing(enhanced)
            
            # Ensure correct data type
            enhanced = enhanced.float()
            
            return enhanced
    
    def _apply_postprocessing(self, enhanced_tensor):
        """
        Apply additional post-processing to enhance emotion features.
        
        Args:
            enhanced_tensor (torch.Tensor): Enhanced image tensor
            
        Returns:
            torch.Tensor: Post-processed tensor
        """
        # Convert to PIL for traditional image processing
        enhanced_np = enhanced_tensor.squeeze().cpu().numpy()
        enhanced_pil = Image.fromarray((enhanced_np * 255).astype(np.uint8), mode='L')
        
        # Apply subtle sharpening to enhance facial features
        enhanced_pil = enhanced_pil.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        
        # Enhance contrast slightly
        enhancer = ImageEnhance.Contrast(enhanced_pil)
        enhanced_pil = enhancer.enhance(1.1)
        
        # Convert back to tensor
        enhanced_array = np.array(enhanced_pil) / 255.0
        enhanced_tensor = torch.from_numpy(enhanced_array).unsqueeze(0).unsqueeze(0).to(self.device)
        
        return torch.clamp(enhanced_tensor, 0, 1)
    
    def enhance_image_file(self, input_path, output_path=None, target_size=(224, 224)):
        """
        Enhance a single image file and save the result.
        
        Args:
            input_path (str/Path): Path to input image
            output_path (str/Path): Path to save enhanced image
            target_size (tuple): Final target size for the enhanced image
            
        Returns:
            bool: Success status
        """
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(input_path)
            if image_tensor is None:
                return False
            
            # Enhance image
            enhanced_tensor = self.enhance_image(image_tensor)
            
            # Convert to PIL and resize to target size
            enhanced_np = enhanced_tensor.squeeze().cpu().numpy()
            enhanced_pil = Image.fromarray((enhanced_np * 255).astype(np.uint8), mode='L')
            
            # Resize to final target size with high quality
            if enhanced_pil.size != target_size:
                enhanced_pil = enhanced_pil.resize(target_size, Image.Resampling.LANCZOS)
            
            # Save enhanced image
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                enhanced_pil.save(output_path)
            
            return True
            
        except Exception as e:
            print(f"Error enhancing image {input_path}: {e}")
            return False
    
    def enhance_dataset(self, dataset_df, input_dir, output_dir, 
                       path_column='image_path', batch_size=16):
        """
        Enhance an entire dataset of images.
        
        Args:
            dataset_df (pd.DataFrame): Dataset dataframe with image paths
            input_dir (str/Path): Input directory containing original images
            output_dir (str/Path): Output directory for enhanced images
            path_column (str): Column name containing image paths
            batch_size (int): Batch size for processing
            
        Returns:
            dict: Enhancement statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {'total': len(dataset_df), 'success': 0, 'failed': 0}
        
        print(f"Enhancing {stats['total']} images using {self.model_type}...")
        print(f"Input dir: {input_dir}")
        print(f"Output dir: {output_dir}")
        
        # Process images
        for idx, row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
            input_path = input_dir / row[path_column]
            output_path = output_dir / row[path_column]
            
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Enhance image
            success = self.enhance_image_file(input_path, output_path, target_size=(224, 224))
            
            if success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        print(f"\n✓ Enhancement completed!")
        print(f"✓ Success: {stats['success']}/{stats['total']}")
        print(f"✗ Failed: {stats['failed']}/{stats['total']}")
        
        return stats
    
    def train_enhancement_model(self, train_dataset, val_dataset, epochs=50, lr=1e-3):
        """
        Train the enhancement model on emotion-specific data.
        This is optional and can be used to fine-tune the model for emotion recognition.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs (int): Number of training epochs
            lr (float): Learning rate
        """
        print(f"Training {self.model_type} for emotion image enhancement...")
        
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            self.model.train()
            
            for batch_idx, (lr_images, hr_images) in enumerate(train_loader):
                lr_images, hr_images = lr_images.to(self.device), hr_images.to(self.device)
                
                optimizer.zero_grad()
                sr_images = self.model(lr_images)
                loss = criterion(sr_images, hr_images)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            val_loss = 0.0
            self.model.eval()
            
            with torch.no_grad():
                for lr_images, hr_images in val_loader:
                    lr_images, hr_images = lr_images.to(self.device), hr_images.to(self.device)
                    sr_images = self.model(lr_images)
                    loss = criterion(sr_images, hr_images)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f'models/emotion_enhancer_{self.model_type}.pth')
                print(f"✓ New best model saved!")
        
        self.model.eval()
    
    def load_pretrained_weights(self, weights_path):
        """
        Load pre-trained weights for the enhancement model.
        
        Args:
            weights_path (str/Path): Path to the weights file
        """
        if Path(weights_path).exists():
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"✓ Loaded pre-trained weights from {weights_path}")
        else:
            print(f"⚠ Weights file not found: {weights_path}")


def create_enhancement_transforms():
    """Create transforms for the enhancement pipeline."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])


class EmotionEnhancementDataset(Dataset):
    """Dataset for training emotion enhancement models."""
    
    def __init__(self, dataframe, data_dir, lr_size=48, hr_size=192):
        self.df = dataframe
        self.data_dir = Path(data_dir)
        self.lr_size = lr_size
        self.hr_size = hr_size
        
        self.lr_transform = transforms.Compose([
            transforms.Resize((lr_size, lr_size)),
            transforms.ToTensor()
        ])
        
        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_size, hr_size)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Assume first column is image path
        img_path = self.data_dir / row.iloc[0]
        
        try:
            # Load as grayscale
            image = Image.open(img_path).convert('L')
            
            # Create low-res and high-res versions
            lr_image = self.lr_transform(image)
            hr_image = self.hr_transform(image)
            
            return lr_image, hr_image
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy tensors
            return torch.zeros(1, self.lr_size, self.lr_size), torch.zeros(1, self.hr_size, self.hr_size)


# Utility functions
def enhance_emotion_dataset(csv_path, input_dir, output_dir, model_type='enhanced_srcnn'):
    """
    Convenience function to enhance an entire emotion dataset.
    
    Args:
        csv_path (str): Path to CSV file with image paths
        input_dir (str): Input directory with original images
        output_dir (str): Output directory for enhanced images
        model_type (str): Type of enhancement model to use
    
    Returns:
        dict: Enhancement statistics
    """
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Initialize enhancer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enhancer = EmotionImageEnhancer(model_type=model_type, device=device)
    
    # Auto-detect image path column
    possible_cols = [col for col in df.columns if 'path' in col.lower() or 'file' in col.lower() or 'image' in col.lower()]
    path_col = possible_cols[0] if possible_cols else df.columns[0]
    
    # Enhance dataset
    stats = enhancer.enhance_dataset(df, input_dir, output_dir, path_column=path_col)
    
    return stats


def compare_enhancement_methods(image_path, output_dir='comparison_results'):
    """
    Compare different enhancement methods on a single image.
    
    Args:
        image_path (str): Path to test image
        output_dir (str): Directory to save comparison results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Original image
    original = Image.open(image_path).convert('L')
    original.save(output_dir / 'original_48x48.png')
    
    # Simple bicubic upsampling
    bicubic = original.resize((224, 224), Image.Resampling.BICUBIC)
    bicubic.save(output_dir / 'bicubic_224x224.png')
    
    # SRCNN enhancement
    srcnn_enhancer = EmotionImageEnhancer('srcnn', device)
    srcnn_enhancer.enhance_image_file(image_path, output_dir / 'srcnn_224x224.png')
    
    # Enhanced SRCNN
    enhanced_enhancer = EmotionImageEnhancer('enhanced_srcnn', device)
    enhanced_enhancer.enhance_image_file(image_path, output_dir / 'enhanced_srcnn_224x224.png')
    
    print(f"✓ Comparison results saved to: {output_dir}")


if __name__ == "__main__":
    # Example usage
    print("Emotion Image Enhancement Pipeline")
    print("=" * 50)
    
    # Test enhancement
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize enhancer
    enhancer = EmotionImageEnhancer('enhanced_srcnn', device)
    
    print("✓ Emotion Image Enhancer ready for use!")
    print("\nUsage examples:")
    print("1. Enhance single image: enhancer.enhance_image_file('path/to/image.jpg', 'enhanced.jpg')")
    print("2. Enhance dataset: enhance_emotion_dataset('train.csv', 'input_dir', 'output_dir')")
    print("3. Compare methods: compare_enhancement_methods('test_image.jpg')")
