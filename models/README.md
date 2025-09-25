---
language:
- en
license: mit
tags:
- computer-vision
- image-classification
- emotions
- pytorch
pipeline_tag: image-classification
library_name: pytorch
labels:
- 0
- 1
- 2
- 3
- 4
- 5
model-index:
- name: Emotion Face Detector
  results: []
---

# Emotion Face Detector

This repository contains the best model artifacts for an image-based facial emotion classifier.

Artifacts included:
- `best_model.pth`: PyTorch weights
- `model.onnx`: Exported ONNX model
- `label_map.json`: Class id to label mapping

## Usage (ONNX Runtime)
```python
import onnxruntime as ort
import numpy as np
from PIL import Image
import json

sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
with open("label_map.json", "r") as f:
    label_map = json.load(f)
    labels = list(label_map.values()) if isinstance(label_map, dict) else label_map

# Preprocess your image to NCHW float32 in [0,1]
img = Image.open("face.jpg").convert("RGB").resize((224, 224))
x = np.asarray(img).astype("float32") / 255.0
x = np.transpose(x, (2, 0, 1))[None, ...]  # (1,3,224,224)

outputs = sess.run(None, {sess.get_inputs()[0].name: x})
probs = outputs[0][0]
pred = int(np.argmax(probs))
print(labels[pred], float(probs[pred]))
```

## License
This model is released under the MIT License.
