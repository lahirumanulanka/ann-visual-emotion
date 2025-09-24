class EmotionResult {
  final String emotion;
  final double confidence;
  final Map<String, double> allPredictions;

  EmotionResult({
    required this.emotion,
    required this.confidence,
    required this.allPredictions,
  });

  @override
  String toString() {
    return 'EmotionResult(emotion: $emotion, confidence: $confidence, allPredictions: $allPredictions)';
  }
}
