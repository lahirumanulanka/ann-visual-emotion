import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'services/enhanced_emotion_service.dart';

void main() {
  runApp(const EmotionDetectorApp());
}

class EmotionDetectorApp extends StatelessWidget {
  const EmotionDetectorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Emotion Detector',
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
  final EnhancedEmotionDetectionService _emotionService =
      EnhancedEmotionDetectionService();

  @override
  void initState() {
    super.initState();
    _initializeService();
  }

  Future<void> _initializeService() async {
    try {
      setState(() {
        _isInitializing = true;
        _statusMessage = 'Loading emotion detection model...';
      });

      await _emotionService.initialize();

      setState(() {
        _isInitializing = false;
        _statusMessage =
            '✅ AI model ready! Select an image to analyze emotions.';
      });

      // Hide status message after 3 seconds
      Future.delayed(const Duration(seconds: 3), () {
        if (mounted) {
          setState(() {
            _statusMessage = '';
          });
        }
      });
    } catch (e) {
      setState(() {
        _isInitializing = false;
        _statusMessage = '❌ Error loading model: $e';
      });
      _showError('Failed to initialize AI model: $e');
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      setState(() {
        _isLoading = true;
      });

      final XFile? image = await _picker.pickImage(
        source: source,
        maxWidth: 800,
        maxHeight: 800,
        imageQuality: 85,
      );

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
        _statusMessage = '🔍 Analyzing facial expression...';
      });

      final result = await _emotionService.detectEmotion(_selectedImage!);

      setState(() {
        _emotion = result.emotion;
        _confidence = result.confidence;
        _allPredictions = result.allPredictions;
        _statusMessage = '✨ Analysis complete!';
      });

      // Clear status message after 2 seconds
      Future.delayed(const Duration(seconds: 2), () {
        if (mounted) {
          setState(() {
            _statusMessage = '';
          });
        }
      });
    } catch (e) {
      _showError('Error detecting emotion: $e');
      setState(() {
        _statusMessage = '❌ Detection failed';
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
        behavior: SnackBarBehavior.floating,
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
                leading: const Icon(Icons.camera_alt, color: Colors.blue),
                title: const Text('Camera'),
                subtitle: const Text('Take a new photo'),
                onTap: () {
                  Navigator.of(context).pop();
                  _pickImage(ImageSource.camera);
                },
              ),
              ListTile(
                leading: const Icon(Icons.photo_library, color: Colors.green),
                title: const Text('Gallery'),
                subtitle: const Text('Choose from photos'),
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

    return Icon(icon, color: color, size: 28);
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
        title: const Text('🤖 AI Emotion Detector'),
        elevation: 2,
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
                padding: const EdgeInsets.all(16.0),
                decoration: BoxDecoration(
                  color: _isInitializing ? Colors.blue[50] : Colors.green[50],
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: _isInitializing ? Colors.blue : Colors.green,
                    width: 1,
                  ),
                ),
                child: Row(
                  children: [
                    _isInitializing
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(
                            Icons.info_outline,
                            color: Colors.blue,
                            size: 20,
                          ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        _statusMessage,
                        style: const TextStyle(fontSize: 14),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),
            ],

            // Image display area
            Container(
              width: double.infinity,
              height: 300,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey[300]!, width: 2),
                borderRadius: BorderRadius.circular(12),
                color: Colors.grey[50],
              ),
              child: _selectedImage != null
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(10),
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
                            Icons.add_photo_alternate_outlined,
                            size: 64,
                            color: Colors.grey,
                          ),
                          SizedBox(height: 16),
                          Text(
                            'Select an image to detect emotions',
                            style: TextStyle(
                              color: Colors.grey,
                              fontSize: 16,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                          SizedBox(height: 8),
                          Text(
                            'AI will analyze facial expressions',
                            style: TextStyle(
                              color: Colors.grey,
                              fontSize: 14,
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
                elevation: 6,
                shadowColor: Colors.deepPurple.withOpacity(0.3),
                child: Padding(
                  padding: const EdgeInsets.all(20.0),
                  child: Column(
                    children: [
                      // Main emotion result
                      Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: Colors.deepPurple[50],
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            _buildEmotionIcon(_emotion),
                            const SizedBox(width: 16),
                            Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'Detected Emotion:',
                                  style: Theme.of(context)
                                      .textTheme
                                      .titleMedium
                                      ?.copyWith(
                                        color: Colors.grey[600],
                                      ),
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
                      ),

                      const SizedBox(height: 16),

                      // Confidence display
                      Text(
                        'Confidence: ${(_confidence * 100).toStringAsFixed(1)}%',
                        style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                              fontWeight: FontWeight.w600,
                            ),
                      ),
                      const SizedBox(height: 8),
                      LinearProgressIndicator(
                        value: _confidence,
                        backgroundColor: Colors.grey[300],
                        valueColor:
                            AlwaysStoppedAnimation<Color>(Colors.deepPurple),
                        minHeight: 8,
                      ),

                      // Show all predictions if available
                      if (_allPredictions.isNotEmpty) ...[
                        const SizedBox(height: 20),
                        const Divider(),
                        const SizedBox(height: 12),
                        Text(
                          'All Emotion Predictions:',
                          style:
                              Theme.of(context).textTheme.titleMedium?.copyWith(
                                    fontWeight: FontWeight.bold,
                                  ),
                        ),
                        const SizedBox(height: 12),
                        ..._allPredictions.entries
                            .map((entry) => Container(
                                  margin:
                                      const EdgeInsets.symmetric(vertical: 4),
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 12, vertical: 8),
                                  decoration: BoxDecoration(
                                    color: entry.key == _emotion
                                        ? Colors.deepPurple[50]
                                        : Colors.grey[50],
                                    borderRadius: BorderRadius.circular(8),
                                    border: Border.all(
                                      color: entry.key == _emotion
                                          ? Colors.deepPurple
                                          : Colors.grey[300]!,
                                    ),
                                  ),
                                  child: Row(
                                    mainAxisAlignment:
                                        MainAxisAlignment.spaceBetween,
                                    children: [
                                      Row(
                                        children: [
                                          _buildEmotionIcon(entry.key),
                                          const SizedBox(width: 8),
                                          Text(
                                            entry.key.toUpperCase(),
                                            style: TextStyle(
                                              fontWeight: entry.key == _emotion
                                                  ? FontWeight.bold
                                                  : FontWeight.normal,
                                            ),
                                          ),
                                        ],
                                      ),
                                      Text(
                                        '${(entry.value * 100).toStringAsFixed(1)}%',
                                        style: TextStyle(
                                          fontWeight: FontWeight.w600,
                                          color: entry.key == _emotion
                                              ? Colors.deepPurple
                                              : Colors.grey[600],
                                        ),
                                      ),
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
              const CircularProgressIndicator(
                color: Colors.deepPurple,
              ),
              const SizedBox(height: 16),
              Text(
                _statusMessage.isNotEmpty
                    ? _statusMessage
                    : 'Processing image...',
                style: const TextStyle(
                  fontSize: 16,
                  color: Colors.deepPurple,
                ),
              ),
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
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 20, vertical: 12),
                    backgroundColor: Colors.deepPurple,
                    foregroundColor: Colors.white,
                  ),
                ),
                if (_selectedImage != null)
                  ElevatedButton.icon(
                    onPressed:
                        (_isLoading || _isInitializing) ? null : _detectEmotion,
                    icon: const Icon(Icons.analytics),
                    label: const Text('Analyze Again'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 20, vertical: 12),
                      backgroundColor: Colors.orange,
                      foregroundColor: Colors.white,
                    ),
                  ),
              ],
            ),

            const SizedBox(height: 20),

            // Info text
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.blue[50],
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                '💡 This AI analyzes facial expressions and features to detect emotions with enhanced accuracy.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Colors.blue[700],
                  fontSize: 12,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
