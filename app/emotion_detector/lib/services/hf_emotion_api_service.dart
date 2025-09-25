import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:cross_file/cross_file.dart';

class HFEmotionApiService {
  final String apiUrl;
  final String? apiToken; // Optional: for private models

  HFEmotionApiService({required this.apiUrl, this.apiToken});

  Future<EmotionResult> detectEmotion(XFile imageFile) async {
    final bytes = await imageFile.readAsBytes();
    final request = http.MultipartRequest('POST', Uri.parse(apiUrl));
    request.files.add(
        http.MultipartFile.fromBytes('file', bytes, filename: 'image.jpg'));
    if (apiToken != null && apiToken!.isNotEmpty) {
      request.headers['Authorization'] = 'Bearer $apiToken';
    }
    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);
    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      // Support either a custom JSON with keys {emotion, confidence, all_predictions}
      // or the standard Hugging Face image-classification output: [{"label": ..., "score": ...}, ...]
      if (data is Map<String, dynamic> && data.containsKey('emotion')) {
        return EmotionResult.fromJson(data);
      }
      if (data is List) {
        // Some endpoints return [[{label, score}, ...]]; flatten if needed
        final List<dynamic> flat = (data.isNotEmpty && data.first is List)
            ? List<dynamic>.from(data.first as List)
            : List<dynamic>.from(data);
        final preds = <String, double>{};
        for (final item in flat) {
          if (item is Map<String, dynamic> &&
              item.containsKey('label') &&
              item.containsKey('score')) {
            final label = item['label'] as String;
            final score = (item['score'] as num).toDouble();
            preds[label] = score;
          }
        }
        if (preds.isEmpty) {
          throw Exception('Unexpected API response shape: $data');
        }
        // pick top
        String top = preds.keys.first;
        double best = preds[top]!;
        preds.forEach((k, v) {
          if (v > best) {
            best = v;
            top = k;
          }
        });
        return EmotionResult(
          emotion: top,
          confidence: best,
          allPredictions: preds,
        );
      }
      throw Exception('Unknown API response format');
    } else {
      throw Exception('API error: ${response.statusCode} ${response.body}');
    }
  }
}

class EmotionResult {
  final String emotion;
  final double confidence;
  final Map<String, double> allPredictions;

  EmotionResult(
      {required this.emotion,
      required this.confidence,
      required this.allPredictions});

  factory EmotionResult.fromJson(Map<String, dynamic> json) {
    return EmotionResult(
      emotion: json['emotion'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      allPredictions: Map<String, double>.from(json['all_predictions'] as Map),
    );
  }
}
