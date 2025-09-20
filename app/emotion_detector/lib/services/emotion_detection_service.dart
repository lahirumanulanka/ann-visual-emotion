import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:math' as math;
import 'dart:ui' as ui;
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'package:camera/camera.dart';

class EmotionDetectionService {
  late OrtSession _session;
  late Map<String, int> _labelMap;
  late List<String> _emotionLabels;
  bool _isInitialized = false;

  // Model input size (assuming 48x48 grayscale based on common emotion detection models)
  static const int INPUT_SIZE = 48;

  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      // Load label map
      final labelMapData = await rootBundle.loadString('assets/labels/label_map.json');
      _labelMap = Map<String, int>.from(json.decode(labelMapData));
      
      // Create ordered list of emotion labels
      _emotionLabels = List.filled(_labelMap.length, '');
      _labelMap.forEach((emotion, index) {
        _emotionLabels[index] = emotion;
      });

      // Load ONNX model
      final modelBytes = await rootBundle.load('assets/models/model.onnx');
      final sessionOptions = OrtSessionOptions();
      _session = OrtSession.fromBuffer(modelBytes.buffer.asUint8List(), sessionOptions);

      _isInitialized = true;
      print('Emotion detection service initialized successfully');
    } catch (e) {
      print('Error initializing emotion detection service: $e');
      rethrow;
    }
  }

  Future<EmotionResult> detectEmotion(CameraImage image) async {
    if (!_isInitialized) {
      throw Exception('Service not initialized');
    }

    try {
      // Convert CameraImage to Uint8List
      final Uint8List imageBytes = _convertYUV420ToRGB(image);
      
      // Create image from bytes
      final img.Image? originalImage = img.decodeImage(imageBytes);
      if (originalImage == null) {
        throw Exception('Failed to decode image');
      }

      // Preprocess image
      final processedImage = _preprocessImage(originalImage);

      // Run inference
      final result = await _runInference(processedImage);

      return result;
    } catch (e) {
      print('Error in emotion detection: $e');
      return EmotionResult(
        emotion: 'neutral',
        confidence: 0.0,
        allScores: {},
      );
    }
  }

  Uint8List _convertYUV420ToRGB(CameraImage image) {
    // This is a simplified conversion for YUV420 to RGB
    // In a production app, you might want to use a more optimized conversion
    final int width = image.width;
    final int height = image.height;
    final int uvRowStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel!;
    final Uint8List rgbBytes = Uint8List(width * height * 3);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int yIndex = y * width + x;
        final int uvIndex = uvPixelStride * (x / 2).floor() + uvRowStride * (y / 2).floor();
        
        final int yValue = image.planes[0].bytes[yIndex];
        final int uValue = image.planes[1].bytes[uvIndex];
        final int vValue = image.planes[2].bytes[uvIndex];
        
        // Convert YUV to RGB
        final int r = (yValue + 1.402 * (vValue - 128)).round().clamp(0, 255);
        final int g = (yValue - 0.34414 * (uValue - 128) - 0.71414 * (vValue - 128)).round().clamp(0, 255);
        final int b = (yValue + 1.772 * (uValue - 128)).round().clamp(0, 255);
        
        final int rgbIndex = yIndex * 3;
        rgbBytes[rgbIndex] = r;
        rgbBytes[rgbIndex + 1] = g;
        rgbBytes[rgbIndex + 2] = b;
      }
    }
    
    return rgbBytes;
  }

  Float32List _preprocessImage(img.Image image) {
    // Resize to model input size and convert to grayscale
    final resized = img.copyResize(image, width: INPUT_SIZE, height: INPUT_SIZE);
    final grayscale = img.grayscale(resized);
    
    // Convert to Float32List and normalize to [0, 1]
    final Float32List input = Float32List(INPUT_SIZE * INPUT_SIZE);
    final pixels = grayscale.getBytes();
    
    for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
      // Assuming grayscale, so we can use any channel (R, G, B are the same)
      input[i] = pixels[i * 4] / 255.0; // Divide by 255 to normalize to [0, 1]
    }
    
    return input;
  }

  Future<EmotionResult> _runInference(Float32List input) async {
    try {
      // Create input tensor
      final inputOrt = OrtValueTensor.createTensorWithDataList(
        input,
        [1, 1, INPUT_SIZE, INPUT_SIZE], // [batch_size, channels, height, width]
      );
      
      // Run inference
      final inputs = {'input': inputOrt}; // Adjust input name based on your model
      final outputs = await _session.runAsync(OrtRunOptions(), inputs);
      
      // Get output tensor
      final outputTensor = outputs[0] as OrtValueTensor;
      final outputData = outputTensor.value as List<List<double>>;
      
      // Apply softmax and get predictions
      final scores = _applySoftmax(outputData[0]);
      
      // Find the emotion with highest confidence
      int maxIndex = 0;
      double maxScore = scores[0];
      for (int i = 1; i < scores.length; i++) {
        if (scores[i] > maxScore) {
          maxScore = scores[i];
          maxIndex = i;
        }
      }
      
      // Create result map
      final Map<String, double> allScores = {};
      for (int i = 0; i < _emotionLabels.length && i < scores.length; i++) {
        allScores[_emotionLabels[i]] = scores[i];
      }
      
      return EmotionResult(
        emotion: _emotionLabels[maxIndex],
        confidence: maxScore,
        allScores: allScores,
      );
    } catch (e) {
      print('Error in inference: $e');
      rethrow;
    }
  }

  List<double> _applySoftmax(List<double> logits) {
    final double maxLogit = logits.reduce((a, b) => a > b ? a : b);
    final List<double> expValues = logits.map((x) => math.exp(x - maxLogit)).toList();
    final double sum = expValues.reduce((a, b) => a + b);
    return expValues.map((x) => x / sum).toList();
  }

  void dispose() {
    if (_isInitialized) {
      _session.release();
      _isInitialized = false;
    }
  }
}

class EmotionResult {
  final String emotion;
  final double confidence;
  final Map<String, double> allScores;

  EmotionResult({
    required this.emotion,
    required this.confidence,
    required this.allScores,
  });
}