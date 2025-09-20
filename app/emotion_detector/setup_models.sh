#!/bin/bash

# Setup script for emotion detector Flutter app
# This script copies the model files from the repository root to the Flutter assets

echo "Setting up emotion detector model files..."

# Check if we're in the correct directory
if [ ! -f "pubspec.yaml" ]; then
    echo "Error: Please run this script from the app/emotion_detector directory"
    exit 1
fi

# Check if model file exists in repository root
MODEL_SOURCE="../../models/model.onnx"
LABEL_SOURCE="../../models/label_map.json"

if [ ! -f "$MODEL_SOURCE" ]; then
    echo "Error: Model file not found at $MODEL_SOURCE"
    echo "Please ensure the model.onnx file exists in the models/ directory"
    exit 1
fi

# Create assets directories if they don't exist
mkdir -p assets/models
mkdir -p assets/labels

# Copy model files
echo "Copying model.onnx..."
cp "$MODEL_SOURCE" assets/models/model.onnx

echo "Copying label_map.json..."
cp "$LABEL_SOURCE" assets/labels/label_map.json

# Verify files were copied
if [ -f "assets/models/model.onnx" ] && [ -f "assets/labels/label_map.json" ]; then
    echo "✅ Model files successfully set up!"
    echo ""
    echo "Next steps:"
    echo "1. Run 'flutter pub get' to install dependencies"
    echo "2. Run 'flutter run' to start the app"
    echo ""
    echo "Assets ready:"
    echo "  - assets/models/model.onnx ($(ls -lh assets/models/model.onnx | awk '{print $5}'))"
    echo "  - assets/labels/label_map.json"
else
    echo "❌ Error: Failed to copy model files"
    exit 1
fi