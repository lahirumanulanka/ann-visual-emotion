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

## Dockerfile notes
- Ensure your Dockerfile copies the model and label map into the image:
  
	```Dockerfile
	COPY model.onnx /app/model.onnx
	COPY label_map.json /app/label_map.json
	ENV MODEL_PATH=/app/model.onnx
	ENV LABEL_MAP_JSON=/app/label_map.json
	```

## Flutter app configuration
Set your app to call the Space endpoint via dart-define flags:

```bash
flutter run \
	--dart-define=HF_API_URL=https://<org>-<space-name>.hf.space/predict \
	--dart-define=HF_API_TOKEN=<optional_bearer_token_if_private>
```
