# Realtime Emotion Detection App

This folder contains a minimal FastAPI + plain JS web application for realtime emotion detection using your webcam and the trained PyTorch model at `models/best_model.pth`.

## Features
- Webcam capture in browser (no external JS frameworks)
- Optional face detection (OpenCV Haar cascade) with bounding box overlay
- Transfer learning model (assumed VGG16 backbone) auto-reconstructed then weights loaded
- Top label + probability bars for all classes

## Endpoints
- `GET /health` - health check
- `POST /predict` - JSON body `{image_base64: str, detect_face: bool}` returns prediction
- `GET /` - serves the demo page

## Running
```bash
uvicorn realtime_app.main:app --reload --port 8000
```
Open: http://localhost:8000

If you have a GPU and CUDA installed the model will automatically use it.

## Adjusting Labels
If your model class ordering differs, edit `DEFAULT_LABELS` in `realtime_app/model.py` or implement logic to load from a JSON label map (currently the provided label map files are empty).

## Notes
- If your checkpoint is not a plain state_dict you may need to adapt `_load_weights`.
- Face detection uses Haar cascades; for higher accuracy consider moving to a DNN (e.g. retinalface or mediapipe) later.
- For higher FPS switch to a WebSocket stream and reduce image size / quality.

## Next Steps / Ideas
See Optional Enhancements section in main project README.
