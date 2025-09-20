# Real-time Emotion Detector

A full-featured Flutter iOS app that detects emotions in real-time using a camera feed and an ONNX neural network model.

## Features

- **Real-time Emotion Detection**: Analyzes facial expressions from live camera feed
- **6 Emotion Classes**: Detects angry, fearful, happy, neutral, sad, and surprised emotions
- **Live Performance Metrics**: Shows FPS and confidence scores
- **Beautiful UI**: Modern interface with emotion icons and color-coded feedback
- **iOS Optimized**: Native iOS app built with Flutter

## Supported Emotions

1. **Happy** 😊 - Green
2. **Sad** 😢 - Blue  
3. **Angry** 😠 - Red
4. **Surprised** 😮 - Orange
5. **Fearful** 😨 - Purple
6. **Neutral** 😐 - Gray

## Setup Instructions

### Prerequisites
- Flutter SDK (3.8.1 or higher)
- Xcode (for iOS development)
- iOS device or simulator

### Installation

1. **Clone the repository and navigate to the app:**
   ```bash
   cd app/emotion_detector
   ```

2. **Install dependencies:**
   ```bash
   flutter pub get
   ```

3. **Setup model files:**
   - Copy `models/model.onnx` to `assets/models/model.onnx`
   - See `assets/README.md` for detailed instructions

4. **Run the app:**
   ```bash
   flutter run
   ```

## Architecture

- **main.dart**: App entry point and initialization
- **services/emotion_detection_service.dart**: ONNX model inference and image processing
- **screens/camera_screen.dart**: Real-time camera view and emotion display UI

## Model Details

- **Input**: 48x48 grayscale images
- **Output**: Probability scores for 6 emotion classes
- **Framework**: ONNX Runtime for Flutter
- **Performance**: Real-time inference on mobile devices

## Permissions

The app requires camera access for real-time emotion detection. Permissions are automatically requested on first launch.

## Technical Implementation

- **Camera Stream**: Continuous image capture from device camera
- **Image Preprocessing**: Resize, grayscale conversion, and normalization
- **Model Inference**: ONNX Runtime for neural network execution
- **UI Updates**: Real-time emotion display with confidence scores

Built with ❤️ using Flutter and ONNX Runtime
