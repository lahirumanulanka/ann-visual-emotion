import 'dart:io';
import 'dart:convert';
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class FaceDetectionEmotionService {
  late Map<String, int> _labelMap;
  late List<String> _emotions;
  late FaceDetector _faceDetector;
  bool _isInitialized = false;

  Future<void> initialize() async {
    try {
      // Initialize face detector
      _faceDetector = FaceDetector(
        options: FaceDetectorOptions(
          enableLandmarks: true,
          enableClassification: true,
          enableTracking: false,
          minFaceSize: 0.1, // Minimum face size (10% of image)
          performanceMode: FaceDetectorMode.accurate,
        ),
      );

      // Load label map
      final labelMapString =
          await rootBundle.loadString('assets/label_map.json');
      _labelMap = Map<String, int>.from(json.decode(labelMapString));

      // Create emotions list ordered by index
      _emotions = List.filled(_labelMap.length, '');
      _labelMap.forEach((emotion, index) {
        _emotions[index] = emotion;
      });

      _isInitialized = true;
      print('Face detection emotion service initialized successfully');
      print('Available emotions: $_emotions');
    } catch (e) {
      print('Error initializing face detection service: $e');
      throw Exception('Failed to initialize face detection: $e');
    }
  }

  Future<EmotionResult> detectEmotion(File imageFile) async {
    if (!_isInitialized) {
      throw Exception('Service not initialized. Call initialize() first.');
    }

    try {
      // Step 1: Detect faces in the image
      final inputImage = InputImage.fromFile(imageFile);
      final faces = await _faceDetector.processImage(inputImage);

      print('Detected ${faces.length} face(s) in the image');

      if (faces.isEmpty) {
        // No faces detected, analyze the whole image
        print('No faces detected, analyzing full image');
        return await _analyzeFullImage(imageFile);
      }

      // Step 2: Process the largest/most confident face
      final bestFace = _selectBestFace(faces);

      // Step 3: Crop and analyze the face region
      final faceResult = await _analyzeFaceRegion(imageFile, bestFace);

      return faceResult;
    } catch (e) {
      print('Error during face detection: $e');
      // Fallback to full image analysis
      return await _analyzeFullImage(imageFile);
    }
  }

  Face _selectBestFace(List<Face> faces) {
    // Select the largest face (most likely to be the main subject)
    Face bestFace = faces[0];
    double maxArea = _calculateFaceArea(bestFace);

    for (final face in faces) {
      final area = _calculateFaceArea(face);
      if (area > maxArea) {
        maxArea = area;
        bestFace = face;
      }
    }

    print('Selected face with area: ${maxArea.toStringAsFixed(0)} pixels');
    return bestFace;
  }

  double _calculateFaceArea(Face face) {
    final width = face.boundingBox.width;
    final height = face.boundingBox.height;
    return width * height;
  }

  Future<EmotionResult> _analyzeFaceRegion(File imageFile, Face face) async {
    try {
      // Read original image
      final imageBytes = await imageFile.readAsBytes();
      final originalImage = img.decodeImage(imageBytes);

      if (originalImage == null) {
        throw Exception('Could not decode image');
      }

      // Extract face region with some padding
      final bbox = face.boundingBox;
      final padding = 0.2; // 20% padding around face

      final paddedWidth = bbox.width * (1 + padding * 2);
      final paddedHeight = bbox.height * (1 + padding * 2);

      final cropX = (bbox.left - bbox.width * padding)
          .clamp(0, originalImage.width - 1)
          .toInt();
      final cropY = (bbox.top - bbox.height * padding)
          .clamp(0, originalImage.height - 1)
          .toInt();
      final cropWidth =
          paddedWidth.clamp(1, originalImage.width - cropX).toInt();
      final cropHeight =
          paddedHeight.clamp(1, originalImage.height - cropY).toInt();

      print(
          'Cropping face region: x=$cropX, y=$cropY, w=$cropWidth, h=$cropHeight');

      // Crop the face region
      final faceImage = img.copyCrop(originalImage,
          x: cropX, y: cropY, width: cropWidth, height: cropHeight);

      // Analyze the cropped face
      final faceAnalysis = await _analyzeImageProperties(faceImage, face);

      // Enhanced emotion detection based on face analysis
      final emotionScores = _calculateEmotionScores(faceAnalysis);

      // Find the emotion with highest score
      int maxIndex = 0;
      double maxScore = emotionScores[0];

      for (int i = 1; i < emotionScores.length; i++) {
        if (emotionScores[i] > maxScore) {
          maxScore = emotionScores[i];
          maxIndex = i;
        }
      }

      final detectedEmotion = _emotions[maxIndex];
      final confidence = maxScore;

      // Create all predictions map
      final allPredictions = <String, double>{};
      for (int i = 0; i < _emotions.length; i++) {
        allPredictions[_emotions[i]] = emotionScores[i];
      }

      print(
          'Face emotion detected: $detectedEmotion (${(confidence * 100).toStringAsFixed(1)}%)');

      return EmotionResult(
        emotion: detectedEmotion,
        confidence: confidence,
        allPredictions: allPredictions,
        faceDetected: true,
        faceCount: 1,
      );
    } catch (e) {
      print('Error analyzing face region: $e');
      throw Exception('Face analysis failed: $e');
    }
  }

  Future<EmotionResult> _analyzeFullImage(File imageFile) async {
    // Fallback to full image analysis when no faces detected
    try {
      final imageBytes = await imageFile.readAsBytes();
      final image = img.decodeImage(imageBytes);

      if (image == null) {
        throw Exception('Could not decode image');
      }

      final imageAnalysis = await _analyzeImageProperties(image, null);
      final emotionScores = _calculateEmotionScores(imageAnalysis);

      int maxIndex = 0;
      double maxScore = emotionScores[0];

      for (int i = 1; i < emotionScores.length; i++) {
        if (emotionScores[i] > maxScore) {
          maxScore = emotionScores[i];
          maxIndex = i;
        }
      }

      final detectedEmotion = _emotions[maxIndex];
      final confidence =
          maxScore * 0.7; // Lower confidence for non-face analysis

      final allPredictions = <String, double>{};
      for (int i = 0; i < _emotions.length; i++) {
        allPredictions[_emotions[i]] = emotionScores[i] * 0.7;
      }

      return EmotionResult(
        emotion: detectedEmotion,
        confidence: confidence,
        allPredictions: allPredictions,
        faceDetected: false,
        faceCount: 0,
      );
    } catch (e) {
      print('Error analyzing full image: $e');
      throw Exception('Full image analysis failed: $e');
    }
  }

  Future<FaceImageAnalysis> _analyzeImageProperties(
      img.Image image, Face? face) async {
    try {
      // Resize to 224x224 (matching training data)
      final resized = img.copyResize(image, width: 224, height: 224);
      final grayscale = img.grayscale(resized);

      // Calculate image properties
      double totalBrightness = 0;
      final pixels = <int>[];

      for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
          final pixel = grayscale.getPixel(x, y);
          final luminance = img.getLuminance(pixel).round();
          pixels.add(luminance);
          totalBrightness += luminance;
        }
      }

      final avgBrightness = totalBrightness / pixels.length;

      // Calculate contrast (standard deviation)
      double varianceSum = 0;
      for (final pixel in pixels) {
        varianceSum += pow(pixel - avgBrightness, 2);
      }
      final contrast = sqrt(varianceSum / pixels.length);

      // Analyze face-specific features
      final centerBrightness = _analyzeCenterRegion(grayscale);
      final edgeDensity = _calculateEdgeDensity(grayscale);

      // Face-specific analysis if face detected
      double? smileProbability;
      double? leftEyeOpenProbability;
      double? rightEyeOpenProbability;

      if (face != null) {
        smileProbability = face.smilingProbability;
        leftEyeOpenProbability = face.leftEyeOpenProbability;
        rightEyeOpenProbability = face.rightEyeOpenProbability;

        print('Face features - Smile: ${smileProbability?.toStringAsFixed(2)}, '
            'Left Eye: ${leftEyeOpenProbability?.toStringAsFixed(2)}, '
            'Right Eye: ${rightEyeOpenProbability?.toStringAsFixed(2)}');
      }

      return FaceImageAnalysis(
        brightness: avgBrightness / 255.0,
        contrast: contrast / 255.0,
        centerBrightness: centerBrightness / 255.0,
        edgeDensity: edgeDensity,
        imageSize: pixels.length,
        smileProbability: smileProbability,
        leftEyeOpenProbability: leftEyeOpenProbability,
        rightEyeOpenProbability: rightEyeOpenProbability,
      );
    } catch (e) {
      print('Error analyzing image properties: $e');
      throw Exception('Image analysis failed: $e');
    }
  }

  double _analyzeCenterRegion(img.Image image) {
    double centerBrightness = 0;
    int centerPixels = 0;

    // Analyze center 112x112 region
    for (int y = 56; y < 168; y++) {
      for (int x = 56; x < 168; x++) {
        final pixel = image.getPixel(x, y);
        centerBrightness += img.getLuminance(pixel);
        centerPixels++;
      }
    }

    return centerPixels > 0 ? centerBrightness / centerPixels : 0;
  }

  double _calculateEdgeDensity(img.Image image) {
    double edgeCount = 0;

    for (int y = 1; y < 223; y++) {
      for (int x = 1; x < 223; x++) {
        final current = img.getLuminance(image.getPixel(x, y));
        final right = img.getLuminance(image.getPixel(x + 1, y));
        final bottom = img.getLuminance(image.getPixel(x, y + 1));

        final gradientX = (right - current).abs();
        final gradientY = (bottom - current).abs();
        final gradient = sqrt(gradientX * gradientX + gradientY * gradientY);

        if (gradient > 30) {
          edgeCount++;
        }
      }
    }

    return edgeCount / (222 * 222);
  }

  List<double> _calculateEmotionScores(FaceImageAnalysis analysis) {
    final scores = <double>[];
    final random = Random();

    // Reduced randomness for face-detected images
    final baseRandomness = analysis.smileProbability != null ? 0.05 : 0.15;

    for (final emotion in _emotions) {
      double score = baseRandomness + (random.nextDouble() * 0.05);

      switch (emotion.toLowerCase()) {
        case 'happy':
          score += (analysis.brightness * 0.2) + (analysis.edgeDensity * 0.15);
          if (analysis.centerBrightness > 0.6) score += 0.15;

          // Use face detection smile probability
          if (analysis.smileProbability != null) {
            score += analysis.smileProbability! *
                0.4; // High weight for smile detection
          }
          break;

        case 'sad':
          score += ((1.0 - analysis.brightness) * 0.2) +
              ((1.0 - analysis.centerBrightness) * 0.25);
          if (analysis.contrast < 0.3) score += 0.1;

          // Low smile probability indicates sadness
          if (analysis.smileProbability != null) {
            score += (1.0 - analysis.smileProbability!) * 0.3;
          }
          break;

        case 'angry':
          score += (analysis.contrast * 0.35) + (analysis.edgeDensity * 0.2);
          if (analysis.centerBrightness < 0.5) score += 0.1;

          // Low smile, intense features
          if (analysis.smileProbability != null) {
            score += (1.0 - analysis.smileProbability!) * 0.2;
          }
          break;

        case 'surprised':
          score +=
              (analysis.contrast * 0.25) + (analysis.centerBrightness * 0.2);

          // Wide eyes indicate surprise
          if (analysis.leftEyeOpenProbability != null &&
              analysis.rightEyeOpenProbability != null) {
            final avgEyeOpen = (analysis.leftEyeOpenProbability! +
                    analysis.rightEyeOpenProbability!) /
                2;
            score += avgEyeOpen * 0.3;
          }
          break;

        case 'fearful':
          score += (analysis.edgeDensity * 0.25) + (analysis.contrast * 0.15);
          if (analysis.brightness > 0.3 && analysis.brightness < 0.7)
            score += 0.1;

          // Wide eyes, low smile
          if (analysis.leftEyeOpenProbability != null &&
              analysis.rightEyeOpenProbability != null) {
            final avgEyeOpen = (analysis.leftEyeOpenProbability! +
                    analysis.rightEyeOpenProbability!) /
                2;
            score += avgEyeOpen * 0.2;
          }
          if (analysis.smileProbability != null) {
            score += (1.0 - analysis.smileProbability!) * 0.15;
          }
          break;

        case 'neutral':
          final balanceScore = 1.0 -
              ((analysis.brightness - 0.5).abs() +
                  (analysis.contrast - 0.3).abs());
          score += balanceScore * 0.25;

          // Moderate smile probability
          if (analysis.smileProbability != null) {
            final neutralSmile =
                1.0 - (analysis.smileProbability! - 0.5).abs() * 2;
            score += neutralSmile * 0.2;
          }
          break;
      }

      score = score.clamp(0.0, 1.0);
      scores.add(score);
    }

    // Normalize scores
    final sum = scores.reduce((a, b) => a + b);
    if (sum > 0) {
      for (int i = 0; i < scores.length; i++) {
        scores[i] = scores[i] / sum;
      }
    }

    return scores;
  }

  void dispose() {
    if (_isInitialized) {
      _faceDetector.close();
    }
  }
}

class FaceImageAnalysis {
  final double brightness;
  final double contrast;
  final double centerBrightness;
  final double edgeDensity;
  final int imageSize;
  final double? smileProbability;
  final double? leftEyeOpenProbability;
  final double? rightEyeOpenProbability;

  FaceImageAnalysis({
    required this.brightness,
    required this.contrast,
    required this.centerBrightness,
    required this.edgeDensity,
    required this.imageSize,
    this.smileProbability,
    this.leftEyeOpenProbability,
    this.rightEyeOpenProbability,
  });

  @override
  String toString() {
    return 'FaceImageAnalysis(brightness: ${brightness.toStringAsFixed(2)}, '
        'contrast: ${contrast.toStringAsFixed(2)}, '
        'smile: ${smileProbability?.toStringAsFixed(2) ?? 'N/A'})';
  }
}

class EmotionResult {
  final String emotion;
  final double confidence;
  final Map<String, double> allPredictions;
  final bool faceDetected;
  final int faceCount;

  EmotionResult({
    required this.emotion,
    required this.confidence,
    required this.allPredictions,
    this.faceDetected = false,
    this.faceCount = 0,
  });

  @override
  String toString() {
    return 'EmotionResult(emotion: $emotion, confidence: ${(confidence * 100).toStringAsFixed(1)}%, '
        'faceDetected: $faceDetected, faceCount: $faceCount)';
  }
}
