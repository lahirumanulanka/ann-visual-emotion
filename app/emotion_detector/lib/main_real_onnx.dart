import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'services/real_emotion_detection_service.dart';

void main() {
  runApp(const EmotionDetectorApp());
}

class EmotionDetectorApp extends StatelessWidget {
  const EmotionDetectorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Emotion Detector',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const EmotionDetectorHome(),
    );
  }
}

class EmotionDetectorHome extends StatefulWidget {
  const EmotionDetectorHome({super.key});

  @override
  State<EmotionDetectorHome> createState() => _EmotionDetectorHomeState();
}

class _EmotionDetectorHomeState extends State<EmotionDetectorHome> {
  File? _selectedImage;
  String _emotion = 'No emotion detected yet';
  double _confidence = 0.0;
  Map<String, double> _allPredictions = {};
  bool _isLoading = false;
  bool _isInitializing = true;
  String _statusMessage = 'Initializing AI model...';

  final ImagePicker _picker = ImagePicker();
  final EmotionDetectionService _emotionService = EmotionDetectionService();

  @override
  void initState() {
    super.initState();
    _initializeService();
  }

  Future<void> _initializeService() async {
    try {
      setState(() {
        _isInitializing = true;
        _statusMessage = 'Loading ONNX model...';
      });

      await _emotionService.initialize();

      setState(() {
        _isInitializing = false;
        _statusMessage = 'AI model ready!';
      });

      // Hide status message after 2 seconds
      Future.delayed(const Duration(seconds: 2), () {
        if (mounted) {
          setState(() {
            _statusMessage = '';
          });
        }
      });
    } catch (e) {
      setState(() {
        _isInitializing = false;
        _statusMessage = 'Error loading model: $e';
      });
      _showError('Failed to initialize AI model: $e');
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      setState(() {
        _isLoading = true;
      });

      final XFile? image = await _picker.pickImage(source: source);

      if (image != null) {
        setState(() {
          _selectedImage = File(image.path);
        });

        // Automatically detect emotion when image is selected
        await _detectEmotion();
      }
    } catch (e) {
      _showError('Error picking image: $e');
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> _detectEmotion() async {
    if (_selectedImage == null || _isInitializing) return;

    try {
      setState(() {
        _isLoading = true;
        _statusMessage = 'Analyzing emotion...';
      });

      final result = await _emotionService.detectEmotion(_selectedImage!);

      setState(() {
        _emotion = result.emotion;
        _confidence = result.confidence;
        _allPredictions = result.allPredictions;
        _statusMessage = '';
      });
    } catch (e) {
      _showError('Error detecting emotion: $e');
      setState(() {
        _statusMessage = 'Detection failed';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red,
        duration: const Duration(seconds: 5),
      ),
    );
  }

  void _showImageSourceDialog() {
    if (_isInitializing) {
      _showError('Please wait for AI model to finish loading');
      return;
    }

    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Select Image Source'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                leading: const Icon(Icons.camera_alt),
                title: const Text('Camera'),
                onTap: () {
                  Navigator.of(context).pop();
                  _pickImage(ImageSource.camera);
                },
              ),
              ListTile(
                leading: const Icon(Icons.photo_library),
                title: const Text('Gallery'),
                onTap: () {
                  Navigator.of(context).pop();
                  _pickImage(ImageSource.gallery);
                },
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildEmotionIcon(String emotion) {
    IconData icon;
    Color color;

    switch (emotion.toLowerCase()) {
      case 'happy':
        icon = Icons.sentiment_very_satisfied;
        color = Colors.green;
        break;
      case 'sad':
        icon = Icons.sentiment_very_dissatisfied;
        color = Colors.blue;
        break;
      case 'angry':
        icon = Icons.sentiment_very_dissatisfied;
        color = Colors.red;
        break;
      case 'surprised':
        icon = Icons.sentiment_satisfied;
        color = Colors.orange;
        break;
      case 'fearful':
        icon = Icons.sentiment_dissatisfied;
        color = Colors.purple;
        break;
      case 'neutral':
        icon = Icons.sentiment_neutral;
        color = Colors.grey;
        break;
      default:
        icon = Icons.psychology;
        color = Colors.deepPurple;
    }

    return Icon(icon, color: color, size: 32);
  }

  @override
  void dispose() {
    _emotionService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('AI Emotion Detector'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            // Status message
            if (_statusMessage.isNotEmpty) ...[
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12.0),
                decoration: BoxDecoration(
                  color: _isInitializing ? Colors.blue[50] : Colors.green[50],
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: _isInitializing ? Colors.blue : Colors.green,
                  ),
                ),
                child: Row(
                  children: [
                    _isInitializing
                        ? const SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : Icon(
                            Icons.check_circle,
                            color: Colors.green,
                            size: 16,
                          ),
                    const SizedBox(width: 8),
                    Expanded(child: Text(_statusMessage)),
                  ],
                ),
              ),
              const SizedBox(height: 16),
            ],

            // Image display area
            Container(
              width: double.infinity,
              height: 300,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
                borderRadius: BorderRadius.circular(8),
              ),
              child: _selectedImage != null
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: Image.file(
                        _selectedImage!,
                        fit: BoxFit.cover,
                      ),
                    )
                  : const Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.image_outlined,
                            size: 64,
                            color: Colors.grey,
                          ),
                          SizedBox(height: 16),
                          Text(
                            'Select an image to detect emotion',
                            style: TextStyle(
                              color: Colors.grey,
                              fontSize: 16,
                            ),
                          ),
                        ],
                      ),
                    ),
            ),

            const SizedBox(height: 24),

            // Results area
            if (_emotion != 'No emotion detected yet') ...[
              Card(
                elevation: 4,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          _buildEmotionIcon(_emotion),
                          const SizedBox(width: 12),
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Detected Emotion:',
                                style: Theme.of(context).textTheme.titleMedium,
                              ),
                              Text(
                                _emotion.toUpperCase(),
                                style: Theme.of(context)
                                    .textTheme
                                    .headlineSmall
                                    ?.copyWith(
                                      color: Colors.deepPurple,
                                      fontWeight: FontWeight.bold,
                                    ),
                              ),
                            ],
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      Text(
                        'Confidence: ${(_confidence * 100).toStringAsFixed(1)}%',
                        style: Theme.of(context).textTheme.bodyLarge,
                      ),
                      const SizedBox(height: 8),
                      LinearProgressIndicator(
                        value: _confidence,
                        backgroundColor: Colors.grey[300],
                        valueColor: const AlwaysStoppedAnimation<Color>(
                            Colors.deepPurple),
                      ),

                      // Show all predictions if available
                      if (_allPredictions.isNotEmpty) ...[
                        const SizedBox(height: 16),
                        const Text(
                          'All Predictions:',
                          style: TextStyle(fontWeight: FontWeight.bold),
                        ),
                        const SizedBox(height: 8),
                        ..._allPredictions.entries
                            .map((entry) => Padding(
                                  padding:
                                      const EdgeInsets.symmetric(vertical: 2),
                                  child: Row(
                                    mainAxisAlignment:
                                        MainAxisAlignment.spaceBetween,
                                    children: [
                                      Row(
                                        children: [
                                          _buildEmotionIcon(entry.key),
                                          const SizedBox(width: 8),
                                          Text(entry.key),
                                        ],
                                      ),
                                      Text(
                                          '${(entry.value * 100).toStringAsFixed(1)}%'),
                                    ],
                                  ),
                                ))
                            .toList(),
                      ],
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 24),
            ],

            // Loading indicator
            if (_isLoading) ...[
              const CircularProgressIndicator(),
              const SizedBox(height: 16),
              Text(_statusMessage.isNotEmpty
                  ? _statusMessage
                  : 'Processing image...'),
              const SizedBox(height: 24),
            ],

            // Action buttons
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: (_isLoading || _isInitializing)
                      ? null
                      : _showImageSourceDialog,
                  icon: const Icon(Icons.add_a_photo),
                  label: const Text('Select Image'),
                ),
                if (_selectedImage != null)
                  ElevatedButton.icon(
                    onPressed:
                        (_isLoading || _isInitializing) ? null : _detectEmotion,
                    icon: const Icon(Icons.analytics),
                    label: const Text('Analyze Again'),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
