# Advanced Features Implementation for CNN Transfer Learning
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import json

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.
    """
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        # Find the target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                self.hooks.append(module.register_backward_hook(backward_hook))
                self.hooks.append(module.register_forward_hook(forward_hook))
                break
    
    def generate_cam(self, input_tensor, class_idx=None):
        # Forward pass
        model_output = self.model(input_tensor)
        if class_idx is None:
            class_idx = model_output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = model_output[:, class_idx]
        class_score.backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension
        
        weights = torch.mean(gradients, dim=(1, 2))  # Global average pooling
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        
        for k, w in enumerate(weights):
            cam += w * activations[k]
        
        cam = F.relu(cam)  # Apply ReLU
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                           size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().numpy()
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

class EnsembleMethods:
    """
    Ensemble methods for combining multiple models.
    """
    
    @staticmethod
    def voting_ensemble(models, data_loader, device, method='soft'):
        """
        Voting ensemble: combine predictions from multiple models.
        Args:
            models: List of trained models
            data_loader: DataLoader for predictions
            method: 'soft' or 'hard' voting
        """
        all_predictions = []
        all_labels = []
        
        for model in models:
            model.eval()
            model_predictions = []
            
            with torch.no_grad():
                for inputs, labels in data_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    
                    if method == 'soft':
                        probs = F.softmax(outputs, dim=1)
                        model_predictions.append(probs.cpu().numpy())
                    else:  # hard voting
                        preds = torch.argmax(outputs, dim=1)
                        model_predictions.append(preds.cpu().numpy())
                    
                    if len(all_labels) == 0:  # Only collect labels once
                        all_labels.extend(labels.numpy())
            
            all_predictions.append(np.vstack(model_predictions))
        
        # Combine predictions
        if method == 'soft':
            ensemble_preds = np.mean(all_predictions, axis=0)
            final_preds = np.argmax(ensemble_preds, axis=1)
        else:  # hard voting
            ensemble_preds = np.array(all_predictions).T  # Shape: (n_samples, n_models)
            final_preds = []
            for sample_preds in ensemble_preds:
                # Get majority vote
                unique, counts = np.unique(sample_preds, return_counts=True)
                final_preds.append(unique[np.argmax(counts)])
            final_preds = np.array(final_preds)
        
        return final_preds, np.array(all_labels)
    
    @staticmethod
    def weighted_ensemble(models, weights, data_loader, device):
        """
        Weighted ensemble: combine model predictions with specified weights.
        """
        assert len(models) == len(weights), "Number of models must match number of weights"
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
        
        all_predictions = []
        all_labels = []
        
        for i, model in enumerate(models):
            model.eval()
            model_predictions = []
            
            with torch.no_grad():
                for inputs, labels in data_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    probs = F.softmax(outputs, dim=1)
                    model_predictions.append(probs.cpu().numpy())
                    
                    if i == 0:  # Only collect labels once
                        all_labels.extend(labels.numpy())
            
            all_predictions.append(np.vstack(model_predictions))
        
        # Weight the predictions
        weighted_preds = sum(w * pred for w, pred in zip(weights, all_predictions))
        final_preds = np.argmax(weighted_preds, axis=1)
        
        return final_preds, np.array(all_labels)

class AdvancedAugmentation:
    """
    Advanced data augmentation using albumentations and synthetic data generation.
    """
    
    @staticmethod
    def get_advanced_augmentation_pipeline():
        """
        Create an advanced augmentation pipeline using albumentations.
        """
        return A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
            ], p=0.8),
            
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.2),
            
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.OpticalDistortion(distort_limit=0.05, p=1.0),
            ], p=0.3),
            
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
            
            A.RandomRotate90(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Transpose(p=0.2),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    @staticmethod
    def mixup_data(x, y, alpha=1.0, device='cpu'):
        """
        Apply MixUp augmentation.
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    @staticmethod
    def cutmix_data(x, y, alpha=1.0, device='cpu'):
        """
        Apply CutMix augmentation.
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(device)
        
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam

def rand_bbox(size, lam):
    """
    Generate random bounding box for CutMix.
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class ModelOptimization:
    """
    Model optimization for deployment.
    """
    
    @staticmethod
    def quantize_model(model, data_loader, device='cpu'):
        """
        Apply dynamic quantization to the model.
        """
        model.eval()
        model = model.to('cpu')  # Quantization works on CPU
        
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear, torch.nn.Conv2d}, 
            dtype=torch.qint8
        )
        
        return quantized_model
    
    @staticmethod
    def export_to_onnx(model, input_shape, output_path, device='cpu'):
        """
        Export model to ONNX format.
        """
        model.eval()
        model = model.to(device)
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape).to(device)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        return output_path
    
    @staticmethod
    def compare_model_sizes(original_model, optimized_model):
        """
        Compare sizes of original and optimized models.
        """
        def get_model_size(model):
            param_size = 0
            buffer_size = 0
            
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            return (param_size + buffer_size) / (1024 ** 2)  # Size in MB
        
        original_size = get_model_size(original_model)
        optimized_size = get_model_size(optimized_model)
        compression_ratio = original_size / optimized_size
        
        return {
            'original_size_mb': original_size,
            'optimized_size_mb': optimized_size,
            'compression_ratio': compression_ratio,
            'size_reduction_percent': (1 - optimized_size / original_size) * 100
        }

def visualize_gradcam(original_image, cam_mask, alpha=0.4):
    """
    Visualize Grad-CAM overlay on original image.
    """
    # Convert original image to numpy if it's a tensor
    if isinstance(original_image, torch.Tensor):
        if original_image.dim() == 4:  # Remove batch dimension
            original_image = original_image.squeeze(0)
        if original_image.dim() == 3 and original_image.shape[0] == 3:  # CHW to HWC
            original_image = original_image.permute(1, 2, 0)
        original_image = original_image.detach().cpu().numpy()
    
    # Normalize original image to [0, 1]
    if original_image.max() > 1.0:
        original_image = original_image / 255.0
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap / 255.0
    
    # Overlay heatmap on original image
    overlayed_img = (1 - alpha) * original_image + alpha * heatmap
    
    return overlayed_img