from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default labels as provided by user
DEFAULT_LABELS = [
    'angry',
    'fearful',
    'happy',
    'neutral',
    'sad',
    'surprised',
]

_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def load_image_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert('RGB')

def b64_to_image(data_url: str) -> Image.Image:
    # Expect format: "data:image/jpeg;base64,<base64>" or raw base64
    if ',' in data_url:
        _, b64data = data_url.split(',', 1)
    else:
        b64data = data_url
    image_bytes = base64.b64decode(b64data)
    return load_image_bytes(image_bytes)

class EmotionModel:
    def __init__(
        self,
        model_path: Path | str,
        labels: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.labels = labels or DEFAULT_LABELS
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model(num_classes=len(self.labels))
        self._load_weights()
        self.model.to(self.device).eval()

    def _build_model(self, num_classes: int):
        # Try to guess architecture: default to resnet50 (keys in checkpoint looked like ResNet)
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        return backbone

    def _load_weights(self):
        # Try strict safe loading first, then relax if needed
        try:
            state = torch.load(self.model_path, map_location='cpu')
        except Exception:
            # Retry disabling weights_only for older checkpoints (trusted local file)
            state = torch.load(self.model_path, map_location='cpu', weights_only=False)  # type: ignore[arg-type]
        # Support either full model or state_dict
        if isinstance(state, dict) and 'state_dict' in state:
            self.model.load_state_dict(state['state_dict'])
            return
        if isinstance(state, dict) and 'model_state_dict' in state:
            # Training script stored model_state_dict separately
            msd = state['model_state_dict']
            # Adjust classifier head shape if needed
            if 'fc.weight' in msd and msd['fc.weight'].shape[0] != len(self.labels):
                # Reinitialize fc already sized to len(labels)
                pass
            self.model.load_state_dict(msd, strict=False)
            return
        if isinstance(state, dict) and all(
            k.startswith('features') or k.startswith('classifier') for k in state.keys()
        ):
            self.model.load_state_dict(state)
            return
        # Fallback attempt: maybe entire model object was saved
        try:
            if hasattr(state, 'state_dict'):
                # Replace classifier layer weights if mismatch in size
                self.model.load_state_dict(state.state_dict(), strict=False)
            else:
                raise TypeError('Unsupported checkpoint object type')
        except Exception as exc:  # noqa: BLE001
            raise ValueError('Unsupported model checkpoint format.') from exc

    @torch.inference_mode()
    def predict(self, img: Image.Image) -> Tuple[str, List[Tuple[str, float]]]:
        tensor = _preprocess(img).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        top_idx = int(np.argmax(probs))
        top_label = self.labels[top_idx] if top_idx < len(self.labels) else str(top_idx)
        ranked = sorted(
            [
                (self.labels[i] if i < len(self.labels) else str(i), float(p))
                for i, p in enumerate(probs)
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        return top_label, ranked

# Singleton pattern for reuse
_model_instance: Optional[EmotionModel] = None


def get_model(
    model_path: Path | str = 'models/best_model.pth',
    labels: Optional[List[str]] = None,
) -> EmotionModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = EmotionModel(model_path=model_path, labels=labels)
    return _model_instance
