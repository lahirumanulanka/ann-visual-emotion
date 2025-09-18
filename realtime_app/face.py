from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Simple Haar cascade face detector (fallback). If unavailable, use OpenCV default.
# You can place a haarcascade_frontalface_default.xml in realtime_app or models.

CASCADE_CANDIDATES = [
    Path(__file__).parent / 'haarcascade_frontalface_default.xml',
    Path('scripts/haarcascades/haarcascade_frontalface_default.xml'),
]

_face_cascade = None

def get_face_cascade():
    global _face_cascade
    if _face_cascade is not None:
        return _face_cascade
    for p in CASCADE_CANDIDATES:
        if p.exists():
            _face_cascade = cv2.CascadeClassifier(str(p))
            break
    if _face_cascade is None or _face_cascade.empty():
        # Fallback to built-in if custom file not found
        built_in = Path(cv2.__file__).parent / 'data' / 'haarcascade_frontalface_default.xml'
        _face_cascade = cv2.CascadeClassifier(str(built_in))
    return _face_cascade

def detect_faces(
    image_rgb: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    cascade = get_face_cascade()
    faces = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

def crop_face(
    image_rgb: np.ndarray,
    box: Tuple[int, int, int, int],
    expand: float = 0.15,
) -> np.ndarray:
    h, w, _ = image_rgb.shape
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2
    side = int(max(bw, bh) * (1 + expand))
    nx1 = int(max(0, cx - side / 2))
    ny1 = int(max(0, cy - side / 2))
    nx2 = int(min(w, cx + side / 2))
    ny2 = int(min(h, cy + side / 2))
    return image_rgb[ny1:ny2, nx1:nx2]
