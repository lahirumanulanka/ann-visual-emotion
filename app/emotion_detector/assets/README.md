# Setup Instructions for Model Files

## Required Model Files

This app requires the emotion detection model to function. Due to file size constraints, the model files are not included in the repository.

### Setting up the model:

1. Copy the `model.onnx` file from the `models/` directory in the repository root to:
   ```
   app/emotion_detector/assets/models/model.onnx
   ```

2. The label mapping file is already included at:
   ```
   app/emotion_detector/assets/labels/label_map.json
   ```

### Model Details:
- **Input Size**: 48x48 grayscale images
- **Output**: 6 emotion classes (angry, fearful, happy, neutral, sad, surprised)
- **Format**: ONNX (Open Neural Network Exchange)

### Directory Structure:
```
app/emotion_detector/assets/
├── models/
│   └── model.onnx        # <- Copy this file here
└── labels/
    └── label_map.json    # <- Already included
```

## Running the App

After setting up the model files:

1. Install dependencies:
   ```bash
   flutter pub get
   ```

2. Run the app:
   ```bash
   flutter run
   ```

For iOS deployment, ensure you have:
- Xcode installed
- iOS Simulator or physical device connected
- Camera permissions granted