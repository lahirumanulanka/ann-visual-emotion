import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'dart:math';

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
  bool _isLoading = false;
  final ImagePicker _picker = ImagePicker();

  // Emotion labels matching the model
  final Map<String, int> emotionLabels = {
    'angry': 0,
    'fearful': 1,
    'happy': 2,
    'neutral': 3,
    'sad': 4,
    'surprised': 5,
  };

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

        // Simulate emotion detection processing
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
    if (_selectedImage == null) return;

    try {
      setState(() {
        _isLoading = true;
      });

      // Simulate processing time
      await Future.delayed(const Duration(seconds: 1));

      // Simple mock emotion detection based on current time
      final random = Random();
      final emotions = emotionLabels.keys.toList();
      final detectedEmotion = emotions[random.nextInt(emotions.length)];
      final confidence =
          0.60 + (random.nextDouble() * 0.35); // 60-95% confidence

      setState(() {
        _emotion = detectedEmotion;
        _confidence = confidence;
      });
    } catch (e) {
      _showError('Error detecting emotion: $e');
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
      ),
    );
  }

  void _showImageSourceDialog() {
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Emotion Detector'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
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
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      Row(
                        children: [
                          const Icon(Icons.psychology, color: Colors.blue),
                          const SizedBox(width: 8),
                          Text(
                            'Detected Emotion:',
                            style: Theme.of(context).textTheme.titleMedium,
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Text(
                        _emotion.toUpperCase(),
                        style: Theme.of(context)
                            .textTheme
                            .headlineMedium
                            ?.copyWith(
                              color: Colors.deepPurple,
                              fontWeight: FontWeight.bold,
                            ),
                      ),
                      const SizedBox(height: 8),
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
              const Text('Processing image...'),
              const SizedBox(height: 24),
            ],

            // Action buttons
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: _isLoading ? null : _showImageSourceDialog,
                  icon: const Icon(Icons.add_a_photo),
                  label: const Text('Select Image'),
                ),
                if (_selectedImage != null)
                  ElevatedButton.icon(
                    onPressed: _isLoading ? null : _detectEmotion,
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
