import 'dart:io';
import 'dart:math';
import 'package:image/image.dart' as img;
import '../models/emotion_result.dart';

class EmotionDetectionService {
  late Map<String, int> _labelMap;
  late List<String> _emotionLabels;
  final Random _random = Random();

  // Initialize the emotion detection service
  Future<void> initialize() async {
    try {
      // Initialize label map
      _labelMap = {
        "angry": 0,
        "fearful": 1,
        "happy": 2,
        "neutral": 3,
        "sad": 4,
        "surprised": 5
      };

      _emotionLabels = _labelMap.keys.toList();
    } catch (e) {
      throw Exception('Failed to initialize emotion detection service: $e');
    }
  }

  // Detect emotion from an image file
  Future<EmotionResult> detectEmotion(File imageFile) async {
    try {
      // Add a small delay to simulate processing
      await Future.delayed(const Duration(milliseconds: 1500));

      // Load and analyze the image
      final imageBytes = await imageFile.readAsBytes();
      final image = img.decodeImage(imageBytes);
      if (image == null) {
        throw Exception('Failed to decode image');
      }

      // Simulate emotion detection based on image characteristics
      final emotionResult = _simulateEmotionDetection(image);

      return emotionResult;
    } catch (e) {
      throw Exception('Failed to detect emotion: $e');
    }
  }

  // Simulate emotion detection based on image characteristics
  EmotionResult _simulateEmotionDetection(img.Image image) {
    // Analyze image characteristics
    final brightness = _calculateBrightness(image);
    final contrast = _calculateContrast(image);

    // Create more realistic probabilities based on image analysis
    final Map<String, double> probabilities = {};

    // Base probabilities with some randomness
    for (String emotion in _emotionLabels) {
      probabilities[emotion] =
          _random.nextDouble() * 0.3 + 0.05; // 0.05 to 0.35
    }

    // Adjust probabilities based on image characteristics
    if (brightness > 0.6) {
      // Brighter images tend to be happier
      probabilities['happy'] = probabilities['happy']! + 0.4;
      probabilities['neutral'] = probabilities['neutral']! + 0.2;
    } else if (brightness < 0.3) {
      // Darker images tend to be sadder or more fearful
      probabilities['sad'] = probabilities['sad']! + 0.3;
      probabilities['fearful'] = probabilities['fearful']! + 0.2;
    }

    if (contrast > 0.5) {
      // High contrast might indicate surprise or anger
      probabilities['surprised'] = probabilities['surprised']! + 0.2;
      probabilities['angry'] = probabilities['angry']! + 0.15;
    }

    // Normalize probabilities to sum to 1
    final total = probabilities.values.reduce((a, b) => a + b);
    probabilities.updateAll((key, value) => value / total);

    // Find the emotion with highest probability
    String topEmotion = probabilities.keys.first;
    double maxProb = probabilities.values.first;

    for (final entry in probabilities.entries) {
      if (entry.value > maxProb) {
        maxProb = entry.value;
        topEmotion = entry.key;
      }
    }

    return EmotionResult(
      emotion: topEmotion,
      confidence: maxProb,
      allPredictions: probabilities,
    );
  }

  // Calculate average brightness of the image
  double _calculateBrightness(img.Image image) {
    int totalBrightness = 0;
    final int totalPixels = image.width * image.height;

    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        // Calculate luminance using standard formula
        final brightness =
            (0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b);
        totalBrightness += brightness.round();
      }
    }

    return totalBrightness / (totalPixels * 255);
  }

  // Calculate contrast of the image
  double _calculateContrast(img.Image image) {
    final brightness = _calculateBrightness(image);
    double variance = 0.0;
    final int totalPixels = image.width * image.height;

    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        final pixelBrightness =
            (0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b) / 255;
        variance += pow(pixelBrightness - brightness, 2);
      }
    }

    return sqrt(variance / totalPixels);
  }
}
