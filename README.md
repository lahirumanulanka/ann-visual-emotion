---
title: Emotion ONNX API (FastAPI + Docker)
emoji: 😄
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "0.0.0"
app_file: app.py
pinned: false
---

# Hugging Face Space: Emotion ONNX API

This Space serves your ONNX emotion classifier via a simple HTTP API.

- POST /predict with multipart form-data field `file` (image).
- Returns JSON: `{ "emotion": str, "confidence": float, "all_predictions": {label: score} }`

Deploy this as a Space (Docker) in your HF org, and point the Flutter app to:
`https://<org>-<space-name>.hf.space/predict`
