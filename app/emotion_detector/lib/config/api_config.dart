/// Centralized configuration for the Emotion API.
///
/// Configure via --dart-define at build/run time:
///   --dart-define=HF_API_URL=https://<your-space>.hf.space/predict
///   --dart-define=HF_API_TOKEN= (optional if your Space is public)
class ApiConfig {
  static const String apiUrl = String.fromEnvironment(
    'HF_API_URL',
    defaultValue:
        'https://hirumunasinghe-human-face-emotion-detector.hf.space/predict',
  );

  static const String _apiToken = String.fromEnvironment(
    'HF_API_TOKEN',
    defaultValue: '',
  );

  static String? get apiToken => _apiToken.isEmpty ? null : _apiToken;
}
