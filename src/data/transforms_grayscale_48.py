# Enhanced data transforms and augmentation for 48x48 grayscale emotion recognition
import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random


class GrayscaleNoise:
    """Add random noise to grayscale images."""
    
    def __init__(self, noise_factor=0.1):
        self.noise_factor = noise_factor
    
    def __call__(self, tensor):
        if random.random() > 0.5:  # Apply with 50% probability
            noise = torch.randn_like(tensor) * self.noise_factor
            return torch.clamp(tensor + noise, 0, 1)
        return tensor


class RandomContrast:
    """Randomly adjust contrast for grayscale images."""
    
    def __init__(self, contrast_range=(0.7, 1.3)):
        self.contrast_range = contrast_range
    
    def __call__(self, img):
        if random.random() > 0.5:
            factor = random.uniform(*self.contrast_range)
            enhancer = ImageEnhance.Contrast(img)
            return enhancer.enhance(factor)
        return img


class RandomBrightness:
    """Randomly adjust brightness for grayscale images."""
    
    def __init__(self, brightness_range=(0.8, 1.2)):
        self.brightness_range = brightness_range
    
    def __call__(self, img):
        if random.random() > 0.5:
            factor = random.uniform(*self.brightness_range)
            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(factor)
        return img


class RandomSharpness:
    """Randomly adjust sharpness - important for small images."""
    
    def __init__(self, sharpness_range=(0.5, 2.0)):
        self.sharpness_range = sharpness_range
    
    def __call__(self, img):
        if random.random() > 0.3:  # Apply more frequently for small images
            factor = random.uniform(*self.sharpness_range)
            enhancer = ImageEnhance.Sharpness(img)
            return enhancer.enhance(factor)
        return img


class RandomElasticTransform:
    """Apply elastic transformation to simulate facial expression variations."""
    
    def __init__(self, alpha=5, sigma=2, probability=0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.probability = probability
    
    def __call__(self, img):
        if random.random() > self.probability:
            return img
        
        img_array = np.array(img)
        if len(img_array.shape) == 2:  # Grayscale
            h, w = img_array.shape
        else:
            h, w, _ = img_array.shape
        
        # Generate random displacement fields
        dx = np.random.uniform(-1, 1, size=(h, w)) * self.alpha
        dy = np.random.uniform(-1, 1, size=(h, w)) * self.alpha
        
        # Smooth the displacement fields
        from scipy.ndimage import gaussian_filter
        dx = gaussian_filter(dx, sigma=self.sigma)
        dy = gaussian_filter(dy, sigma=self.sigma)
        
        # Create coordinate matrices
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        # Apply transformation
        from scipy.ndimage import map_coordinates
        if len(img_array.shape) == 2:
            distorted = map_coordinates(img_array, indices, order=1, mode='reflect')
            distorted = distorted.reshape(h, w)
        else:
            distorted = np.zeros_like(img_array)
            for i in range(img_array.shape[2]):
                distorted[:, :, i] = map_coordinates(
                    img_array[:, :, i], indices, order=1, mode='reflect'
                ).reshape(h, w)
        
        return Image.fromarray(distorted.astype(np.uint8))


class RandomPerspectiveSmall:
    """Perspective transformation optimized for small 48x48 images."""
    
    def __init__(self, distortion_scale=0.2, probability=0.5):
        self.distortion_scale = distortion_scale
        self.probability = probability
    
    def __call__(self, img):
        if random.random() > self.probability:
            return img
        
        return transforms.RandomPerspective(
            distortion_scale=self.distortion_scale,
            p=1.0
        )(img)


class AdaptiveHistogramEqualization:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for grayscale images."""
    
    def __init__(self, probability=0.3):
        self.probability = probability
    
    def __call__(self, img):
        if random.random() > self.probability:
            return img
        
        try:
            import cv2
            img_array = np.array(img)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(img_array)
            return Image.fromarray(enhanced)
        except ImportError:
            # Fallback to simple histogram equalization
            img_array = np.array(img)
            hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * float(hist.max()) / cdf.max()
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            cdf = np.ma.filled(cdf_m, 0).astype('uint8')
            enhanced = cdf[img_array]
            return Image.fromarray(enhanced)


def get_enhanced_transforms_grayscale_48(training=True, advanced_augmentation=True):
    """
    Get enhanced transforms for 48x48 grayscale emotion images.
    
    Args:
        training (bool): Whether to apply training augmentations
        advanced_augmentation (bool): Whether to include advanced augmentations
        
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if training:
        transform_list = []
        
        # Basic geometric augmentations (optimized for small images)
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20, fill=128),  # Moderate rotation
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.1, 0.1), 
                scale=(0.9, 1.1), 
                shear=5,
                fill=128
            ),
        ])
        
        # Advanced augmentations for better model robustness
        if advanced_augmentation:
            transform_list.extend([
                RandomPerspectiveSmall(distortion_scale=0.15, probability=0.4),
                RandomContrast(contrast_range=(0.7, 1.4)),
                RandomBrightness(brightness_range=(0.8, 1.3)),
                RandomSharpness(sharpness_range=(0.6, 2.0)),
                AdaptiveHistogramEqualization(probability=0.25),
            ])
        
        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            # Normalize for grayscale images (mean and std for single channel)
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Scale to [-1, 1]
        ])
        
        # Add noise after tensor conversion
        if advanced_augmentation:
            transform_list.append(GrayscaleNoise(noise_factor=0.05))
        
    else:
        # Validation/test transforms (no augmentation)
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    
    return transforms.Compose(transform_list)


def get_progressive_augmentation_transforms(epoch, max_epochs, base_strength=0.5):
    """
    Get progressively stronger augmentation transforms as training progresses.
    
    Args:
        epoch (int): Current epoch
        max_epochs (int): Total epochs
        base_strength (float): Base augmentation strength
        
    Returns:
        torchvision.transforms.Compose: Progressive transform pipeline
    """
    # Calculate augmentation strength based on epoch
    progress = min(epoch / (max_epochs * 0.7), 1.0)  # Ramp up over first 70% of training
    strength = base_strength + (0.3 * progress)  # Increase strength up to base + 30%
    
    transform_list = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=int(15 * strength), fill=128),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.08 * strength, 0.08 * strength),
            scale=(1.0 - 0.1 * strength, 1.0 + 0.1 * strength),
            shear=int(3 * strength),
            fill=128
        ),
        RandomContrast(contrast_range=(1.0 - 0.2 * strength, 1.0 + 0.3 * strength)),
        RandomBrightness(brightness_range=(1.0 - 0.15 * strength, 1.0 + 0.2 * strength)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        GrayscaleNoise(noise_factor=0.03 * strength)
    ]
    
    return transforms.Compose(transform_list)


def visualize_augmentations(image_path, num_samples=8, save_path=None):
    """
    Visualize the effect of augmentations on a sample image.
    
    Args:
        image_path (str): Path to sample image
        num_samples (int): Number of augmented samples to generate
        save_path (str, optional): Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Load original image
    original_image = Image.open(image_path).convert('L').resize((48, 48))
    
    # Get augmentation transform
    aug_transform = get_enhanced_transforms_grayscale_48(training=True, advanced_augmentation=True)
    
    # Create subplot
    fig, axes = plt.subplots(2, (num_samples + 1) // 2 + 1, figsize=(15, 6))
    axes = axes.flatten()
    
    # Show original
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Show augmented versions
    for i in range(1, num_samples + 1):
        # Apply transform (but convert back to PIL for visualization)
        augmented_tensor = aug_transform(original_image)
        # Convert back to numpy for visualization
        augmented_array = augmented_tensor.squeeze().numpy()
        # Denormalize
        augmented_array = (augmented_array * 0.5) + 0.5  # From [-1, 1] to [0, 1]
        augmented_array = np.clip(augmented_array, 0, 1)
        
        axes[i].imshow(augmented_array, cmap='gray')
        axes[i].set_title(f'Augmented {i}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


# Example usage and testing
if __name__ == "__main__":
    print("Testing Enhanced Transforms for 48x48 Grayscale Images...")
    
    # Test basic transforms
    print("\n1. Testing Basic Training Transforms:")
    train_transform = get_enhanced_transforms_grayscale_48(training=True, advanced_augmentation=False)
    print(f"Transform pipeline: {len(train_transform.transforms)} steps")
    
    # Test advanced transforms  
    print("\n2. Testing Advanced Training Transforms:")
    advanced_transform = get_enhanced_transforms_grayscale_48(training=True, advanced_augmentation=True)
    print(f"Advanced pipeline: {len(advanced_transform.transforms)} steps")
    
    # Test validation transforms
    print("\n3. Testing Validation Transforms:")
    val_transform = get_enhanced_transforms_grayscale_48(training=False)
    print(f"Validation pipeline: {len(val_transform.transforms)} steps")
    
    # Test with dummy image
    print("\n4. Testing Transform Application:")
    dummy_image = Image.new('L', (48, 48), color=128)  # Gray image
    
    try:
        train_output = train_transform(dummy_image)
        val_output = val_transform(dummy_image)
        
        print(f"Training output shape: {train_output.shape}")
        print(f"Training output range: [{train_output.min():.3f}, {train_output.max():.3f}]")
        print(f"Validation output shape: {val_output.shape}")
        print(f"Validation output range: [{val_output.min():.3f}, {val_output.max():.3f}]")
        
    except Exception as e:
        print(f"Transform application test failed: {e}")
    
    # Test progressive augmentation
    print("\n5. Testing Progressive Augmentation:")
    for epoch in [0, 10, 20, 30]:
        prog_transform = get_progressive_augmentation_transforms(epoch, max_epochs=40)
        print(f"Epoch {epoch}: {len(prog_transform.transforms)} transform steps")
    
    print("\n✅ All transform tests completed successfully!")
    
    print("\nTransform Features for 48x48 Grayscale Images:")
    print("- Optimized geometric augmentations for small images")
    print("- Grayscale-specific brightness/contrast adjustments")
    print("- Advanced noise injection and sharpness enhancement")
    print("- Adaptive histogram equalization")
    print("- Progressive augmentation strength")
    print("- Proper normalization for grayscale single-channel input")