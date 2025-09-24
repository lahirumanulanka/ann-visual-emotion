import 'dart:typed_data';
import 'dart:convert';
import 'dart:math' as math;
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import '../services/face_detection_emotion_service.dart';

/// Simple emotion image generator using the bundled classification ONNX model
/// as an entropy source. Since the provided `model.onnx` is a classifier (not a
/// generative diffusion model), we simulate generation by:
/// 1. Loading the raw model bytes + selected emotion label to build a hash
/// 2. Using the hash to seed procedural gradient + noise + emoji-like glyph
/// 3. Returning the produced PNG bytes for display.
///
/// This is a placeholder architecture that can later be replaced by a real
/// generative pipeline (e.g., calling an API or a local diffusion model) while
/// keeping the UI stable.
class EmotionImageGeneratorService {
  bool _loaded = false;
  late List<String> _emotions;
  late Uint8List _modelBytes;

  Future<void> initialize() async {
    if (_loaded) return;
    // Load label map
    final labelMapString = await rootBundle.loadString('assets/label_map.json');
    final map = Map<String, dynamic>.from(json.decode(labelMapString));
    final ordered = List<String>.filled(map.length, '');
    map.forEach((k, v) => ordered[v as int] = k);
    _emotions = ordered;

    // Load model bytes (for entropy / deterministic seed)
    final modelData = await rootBundle.load('assets/model.onnx');
    _modelBytes = modelData.buffer.asUint8List();
    _loaded = true;
  }

  List<String> get emotions => _emotions;

  /// Generate an image for an emotion. Returns PNG bytes.
  Future<Uint8List> generate(String emotion) async {
    if (!_loaded) throw Exception('Service not initialized');
    if (!_emotions.contains(emotion)) {
      throw Exception('Unknown emotion: $emotion');
    }

    // Derive a pseudo seed from model bytes + emotion text
    int seed = 0;
    for (final b in _modelBytes.take(512)) {
      seed = (seed * 31 + b) & 0x7fffffff;
    }
    for (final codeUnit in emotion.codeUnits) {
      seed = (seed * 131 + codeUnit) & 0x7fffffff;
    }

    final rng = _PseudoRandom(seed);

    // Canvas setup
    const size = 512;
    final canvas = img.Image(width: size, height: size); // RGBA 8-bit

    // Choose base + accent colors heuristically per emotion
    final palette = _emotionPalette(emotion);

    // Radial gradient background
    final centerX = size / 2;
    final centerY = size / 2;
    final maxDist = (size * 0.72);
    for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++) {
        final dx = x - centerX;
        final dy = y - centerY;
        final d = math.sqrt(dx * dx + dy * dy);
        final t = (d / maxDist).clamp(0.0, 1.0);
        final r =
            (palette.bgStart.r + (palette.bgEnd.r - palette.bgStart.r) * t)
                .round();
        final g =
            (palette.bgStart.g + (palette.bgEnd.g - palette.bgStart.g) * t)
                .round();
        final b =
            (palette.bgStart.b + (palette.bgEnd.b - palette.bgStart.b) * t)
                .round();
        canvas.setPixelRgba(x, y, r, g, b, 255);
      }
    }

    // Add procedural noise / sparkles
    final sparkleCount = 800 + (rng.nextDouble() * 400).round();
    for (int i = 0; i < sparkleCount; i++) {
      final x = rng.nextInt(size);
      final y = rng.nextInt(size);
      final alpha = 120 + rng.nextInt(100);
      final c = palette.accent;
      canvas.setPixelRgba(x, y, c.r, c.g, c.b, alpha);
    }

    // Draw a simple stylized face / glyph representing the emotion
    _drawEmotionGlyph(canvas, emotion, palette, rng);

    // Encode PNG
    final png = img.encodePng(canvas);
    return Uint8List.fromList(png);
  }

  void _drawEmotionGlyph(
      img.Image canvas, String emotion, _Palette p, _PseudoRandom rng) {
    final faceSize = 260;
    final centerX = canvas.width ~/ 2;
    final centerY = canvas.height ~/ 2 - 30;
    final radius = faceSize ~/ 2;

    // Face circle
    img.fillCircle(
      canvas,
      x: centerX,
      y: centerY,
      radius: radius,
      color: img.ColorUint8.rgb(p.face.r, p.face.g, p.face.b),
    );

    // Eyes
    final eyeOffsetX = (radius * 0.45).round();
    final eyeOffsetY = (radius * 0.3).round();
    final eyeRadius = (radius * 0.12).round();
    img.fillCircle(
      canvas,
      x: centerX - eyeOffsetX,
      y: centerY - eyeOffsetY,
      radius: eyeRadius,
      color: img.ColorUint8.rgb(p.eye.r, p.eye.g, p.eye.b),
    );
    img.fillCircle(
      canvas,
      x: centerX + eyeOffsetX,
      y: centerY - eyeOffsetY,
      radius: eyeRadius,
      color: img.ColorUint8.rgb(p.eye.r, p.eye.g, p.eye.b),
    );

    // Mouth shape varies by emotion
    final mouthWidth = (radius * 0.9).round();
    final mouthHeight = (radius * 0.55).round();
    final mouthY = centerY + (radius * 0.25).round();

    switch (emotion.toLowerCase()) {
      case 'happy':
        // Smile: upper arc (20° to 160°) slightly below center
        _drawArc(
          canvas,
          centerX,
          mouthY,
          mouthWidth,
          (radius * 0.8).round(),
          20,
          160,
          p.mouth,
        );
        break;
      case 'sad':
        // Frown: lower arc (200° to 340°) a bit lower to emphasize frown
        _drawArc(
          canvas,
          centerX,
          mouthY + (radius * 0.15).round(),
          mouthWidth,
          mouthHeight,
          200,
          340,
          p.mouth,
        );
        break;
      case 'angry':
        _drawArc(
            canvas, centerX, mouthY, mouthWidth, mouthHeight, 200, 340, p.mouth,
            thickness: 10);
        // Eyebrows
        _drawLine(canvas, centerX - eyeOffsetX - 20, centerY - eyeOffsetY - 30,
            centerX - eyeOffsetX + 20, centerY - eyeOffsetY - 10, p.eye, 6);
        _drawLine(canvas, centerX + eyeOffsetX + 20, centerY - eyeOffsetY - 30,
            centerX + eyeOffsetX - 20, centerY - eyeOffsetY - 10, p.eye, 6);
        break;
      case 'surprised':
        img.fillCircle(
          canvas,
          x: centerX,
          y: mouthY,
          radius: (radius * 0.22).round(),
          color: img.ColorUint8.rgb(p.mouth.r, p.mouth.g, p.mouth.b),
        );
        break;
      case 'fearful':
        img.fillCircle(
          canvas,
          x: centerX,
          y: mouthY,
          radius: (radius * 0.25).round(),
          color: img.ColorUint8.rgb(p.mouth.r, p.mouth.g, p.mouth.b),
        );
        // jitter lines
        for (int i = 0; i < 12; i++) {
          final angle = (i / 12) * math.pi * 2;
          final rRand = radius + 10 + (rng.nextDouble() * 10);
          final x1 = centerX + (radius - 10) * math.cos(angle);
          final y1 = centerY + (radius - 10) * math.sin(angle);
          final x2 = centerX + rRand * math.cos(angle);
          final y2 = centerY + rRand * math.sin(angle);
          _drawLine(canvas, x1.round(), y1.round(), x2.round(), y2.round(),
              p.accent, 2);
        }
        break;
      case 'neutral':
      default:
        _drawLine(canvas, centerX - mouthWidth ~/ 2, mouthY,
            centerX + mouthWidth ~/ 2, mouthY, p.mouth, 12);
    }
  }

  void _drawArc(img.Image canvas, int cx, int cy, int w, int h, int startDeg,
      int endDeg, _Color color,
      {int thickness = 16}) {
    final startRad = startDeg * math.pi / 180;
    final endRad = endDeg * math.pi / 180;
    final steps = 120;
    int? prevX;
    int? prevY;
    for (int i = 0; i <= steps; i++) {
      final t = startRad + (endRad - startRad) * (i / steps);
      final x = cx + (w / 2) * math.cos(t);
      final y = cy + (h / 2) * math.sin(t);
      if (prevX != null) {
        _drawLine(
            canvas, prevX, prevY!, x.round(), y.round(), color, thickness);
      }
      prevX = x.round();
      prevY = y.round();
    }
  }

  void _drawLine(img.Image canvas, int x1, int y1, int x2, int y2, _Color color,
      int thickness) {
    img.drawLine(
      canvas,
      x1: x1,
      y1: y1,
      x2: x2,
      y2: y2,
      color: img.ColorUint8.rgb(color.r, color.g, color.b),
      thickness: thickness,
    );
  }

  _Palette _emotionPalette(String emotion) {
    switch (emotion.toLowerCase()) {
      case 'happy':
        return _Palette(
          bgStart: _Color(255, 220, 120),
          bgEnd: _Color(255, 140, 0),
          accent: _Color(255, 255, 255),
          face: _Color(255, 235, 59),
          eye: _Color(60, 60, 60),
          mouth: _Color(200, 60, 40),
          text: _Color(40, 40, 40),
        );
      case 'sad':
        return _Palette(
          bgStart: _Color(70, 90, 140),
          bgEnd: _Color(20, 30, 60),
          accent: _Color(180, 200, 255),
          face: _Color(120, 150, 200),
          eye: _Color(30, 40, 70),
          mouth: _Color(30, 40, 80),
          text: _Color(220, 230, 250),
        );
      case 'angry':
        return _Palette(
          bgStart: _Color(255, 90, 70),
          bgEnd: _Color(120, 0, 0),
          accent: _Color(255, 180, 160),
          face: _Color(255, 120, 90),
          eye: _Color(30, 10, 10),
          mouth: _Color(80, 0, 0),
          text: _Color(255, 230, 230),
        );
      case 'surprised':
        return _Palette(
          bgStart: _Color(255, 250, 210),
          bgEnd: _Color(255, 200, 60),
          accent: _Color(255, 255, 255),
          face: _Color(255, 230, 120),
          eye: _Color(40, 40, 40),
          mouth: _Color(200, 100, 50),
          text: _Color(80, 50, 10),
        );
      case 'fearful':
        return _Palette(
          bgStart: _Color(100, 80, 140),
          bgEnd: _Color(30, 10, 60),
          accent: _Color(200, 180, 255),
          face: _Color(170, 150, 210),
          eye: _Color(30, 20, 50),
          mouth: _Color(50, 30, 80),
          text: _Color(230, 220, 255),
        );
      case 'neutral':
      default:
        return _Palette(
          bgStart: _Color(200, 200, 200),
          bgEnd: _Color(120, 120, 120),
          accent: _Color(255, 255, 255),
          face: _Color(210, 210, 210),
          eye: _Color(40, 40, 40),
          mouth: _Color(60, 60, 60),
          text: _Color(30, 30, 30),
        );
    }
  }
}

class EmotionImageGeneratorScreen extends StatefulWidget {
  const EmotionImageGeneratorScreen({super.key});

  @override
  State<EmotionImageGeneratorScreen> createState() => _EmotionImageGeneratorScreenState();
}

class _EmotionImageGeneratorScreenState extends State<EmotionImageGeneratorScreen> {
  // Existing synthetic generator (kept for fallback / entropy & emotions list)
  final EmotionImageGeneratorService _service = EmotionImageGeneratorService();
  // Face detection for reading original emotion (pseudo) and face bbox
  final FaceDetectionEmotionService _faceService = FaceDetectionEmotionService();
  final ImagePicker _picker = ImagePicker();

  bool _loading = true; // initialization
  String? _error;
  String? _selectedEmotion; // target emotion
  String? _originalDetectedEmotion; // detected from uploaded image
  double? _originalConfidence;

  File? _inputImageFile; // original picked file
  Uint8List? _transformedImageBytes; // result of transformation
  bool _processing = false; // busy flag
  bool _faceFound = false;
  int _faceCount = 0;

  @override
  void initState() {
    super.initState();
    _initializeAll();
  }

  Future<void> _initializeAll() async {
    try {
      await Future.wait([
        _service.initialize(),
        _faceService.initialize(),
      ]);
      setState(() { _loading = false; });
    } catch (e) {
      setState(() { _error = 'Init failed: $e'; _loading = false; });
    }
  }

  Future<void> _pickImage() async {
    try {
      final picked = await _picker.pickImage(source: ImageSource.gallery);
      if (picked == null) return;
      setState(() {
        _inputImageFile = File(picked.path);
        _originalDetectedEmotion = null;
        _originalConfidence = null;
        _transformedImageBytes = null;
        _faceFound = false;
        _faceCount = 0;
      });
      await _detectOriginalEmotion();
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Pick image failed: $e')));
    }
  }

  Future<void> _detectOriginalEmotion() async {
    if (_inputImageFile == null) return;
    try {
      setState(() { _processing = true; });
      final r = await _faceService.detectEmotion(_inputImageFile!);
      setState(() {
        _originalDetectedEmotion = r.emotion;
        _originalConfidence = r.confidence;
        _faceFound = r.faceDetected;
        _faceCount = r.faceCount;
      });
    } catch (e) {
      // ignore but show toast
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Detection failed: $e')));
    } finally {
      setState(() { _processing = false; });
    }
  }

  Future<void> _transformToTarget() async {
    if (_inputImageFile == null || _selectedEmotion == null) return;
    try {
      setState(() { _processing = true; });
      final originalBytes = await _inputImageFile!.readAsBytes();
      final original = img.decodeImage(originalBytes);
      if (original == null) throw 'Cannot decode image';

      // For simplicity we re-run detection for a face bbox if any (largest) using same logic as in face service (can't directly access bbox list, so approximate by reusing its cropping heuristic) --> We'll duplicate minimal code for cropping entire image if no face.
      // Instead of internal face list (private), we just apply a global transformation + localized overlays heuristically centered.

      final transformed = img.copyResize(original, width: original.width, height: original.height); // clone

      // Apply global adjustments & overlays based on target emotion.
      _applyEmotionFilter(transformed, _selectedEmotion!);

      final outPng = img.encodePng(transformed);
      setState(() { _transformedImageBytes = Uint8List.fromList(outPng); });
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Transform failed: $e')));
    } finally {
      setState(() { _processing = false; });
    }
  }

  void _applyEmotionFilter(img.Image image, String emotion) {
    switch (emotion.toLowerCase()) {
      case 'happy':
        _adjustBrightness(image, 1.15);
        _tint(image, 255, 220, 100, strength: 0.12);
        _overlaySmile(image, smile: true);
        break;
      case 'sad':
        _adjustBrightness(image, 0.9);
        _adjustContrast(image, 0.85);
        _tint(image, 60, 90, 150, strength: 0.18);
        _overlaySmile(image, smile: false);
        break;
      case 'angry':
        _adjustContrast(image, 1.25);
        _tint(image, 255, 60, 40, strength: 0.20);
        _overlayEyebrows(image, angle: -0.35);
        _overlaySmile(image, smile: false, arcStart: 205, arcEnd: 335);
        break;
      case 'surprised':
        _adjustBrightness(image, 1.12);
        _adjustContrast(image, 1.05);
        _tint(image, 255, 230, 120, strength: 0.10);
        _overlayMouthCircle(image);
        break;
      case 'fearful':
        _adjustContrast(image, 1.10);
        _tint(image, 140, 120, 200, strength: 0.15);
        _overlayMouthCircle(image, jitter: true);
        break;
      case 'neutral':
      default:
        _adjustContrast(image, 0.98);
        _tint(image, 200, 200, 200, strength: 0.05);
    }
  }

  void _adjustBrightness(img.Image imgIn, double factor) {
    for (int y = 0; y < imgIn.height; y++) {
      for (int x = 0; x < imgIn.width; x++) {
        final p = imgIn.getPixel(x, y);
        int r = (p.r * factor).clamp(0, 255).toInt();
        int g = (p.g * factor).clamp(0, 255).toInt();
        int b = (p.b * factor).clamp(0, 255).toInt();
        imgIn.setPixelRgba(x, y, r, g, b, p.a);
      }
    }
  }

  void _adjustContrast(img.Image imgIn, double factor) {
    // Contrast adjustment about mid-point 128
    for (int y = 0; y < imgIn.height; y++) {
      for (int x = 0; x < imgIn.width; x++) {
        final p = imgIn.getPixel(x, y);
        int r = ((p.r - 128) * factor + 128).clamp(0, 255).toInt();
        int g = ((p.g - 128) * factor + 128).clamp(0, 255).toInt();
        int b = ((p.b - 128) * factor + 128).clamp(0, 255).toInt();
        imgIn.setPixelRgba(x, y, r, g, b, p.a);
      }
    }
  }

  void _tint(img.Image imgIn, int tr, int tg, int tb, {double strength = 0.1}) {
    for (int y = 0; y < imgIn.height; y++) {
      for (int x = 0; x < imgIn.width; x++) {
        final p = imgIn.getPixel(x, y);
        final r = (p.r * (1 - strength) + tr * strength).clamp(0, 255).toInt();
        final g = (p.g * (1 - strength) + tg * strength).clamp(0, 255).toInt();
        final b = (p.b * (1 - strength) + tb * strength).clamp(0, 255).toInt();
        imgIn.setPixelRgba(x, y, r, g, b, p.a);
      }
    }
  }

  void _overlaySmile(img.Image imgIn, {required bool smile, int arcStart = 20, int arcEnd = 160}) {
    final cx = imgIn.width ~/ 2;
    final cy = (imgIn.height * 0.60).round();
    final w = (imgIn.width * 0.35).round();
    final h = (imgIn.height * 0.25).round();
    final start = smile ? arcStart : 200;
    final end = smile ? arcEnd : 340;
    final color = img.ColorUint8.rgb(255, 255, 255);
    int? prevX; int? prevY;
    for (int i = 0; i <= 120; i++) {
      final t = start + (end - start) * (i / 120);
      final rad = t * math.pi / 180;
      final x = cx + (w / 2) * math.cos(rad);
      final y = cy + (h / 2) * math.sin(rad);
      if (prevX != null) {
        img.drawLine(imgIn, x1: prevX, y1: prevY!, x2: x.round(), y2: y.round(), color: color, thickness: 6);
      }
      prevX = x.round(); prevY = y.round();
    }
  }

  void _overlayEyebrows(img.Image imgIn, {double angle = -0.3}) {
    final cx = imgIn.width ~/ 2;
    final cy = (imgIn.height * 0.42).round();
    final half = (imgIn.width * 0.18).round();
    final offsetY = (imgIn.height * 0.05).round();
    final color = img.ColorUint8.rgb(255, 255, 255);
    // left brow
    img.drawLine(imgIn,
        x1: cx - half - 40,
        y1: cy - offsetY,
        x2: cx - 40,
        y2: (cy - offsetY + half * math.tan(angle)).round(),
        color: color,
        thickness: 5);
    // right brow
    img.drawLine(imgIn,
        x1: cx + half + 40,
        y1: cy - offsetY,
        x2: cx + 40,
        y2: (cy - offsetY - half * math.tan(angle)).round(),
        color: color,
        thickness: 5);
  }

  void _overlayMouthCircle(img.Image imgIn, {bool jitter = false}) {
    final cx = imgIn.width ~/ 2;
    final cy = (imgIn.height * 0.60).round();
    final baseR = (math.min(imgIn.width, imgIn.height) * 0.09).round();
    final color = img.ColorUint8.rgb(255, 255, 255);
    // start point
    double ang0 = 0;
    int prevX = (cx + baseR * math.cos(ang0)).round();
    int prevY = (cy + baseR * math.sin(ang0)).round();
    for (int i = 1; i <= 120; i++) {
      final ang = (i / 120) * math.pi * 2;
      final r = jitter ? (baseR + (math.sin(i.toDouble()) * 3).round()) : baseR;
      final x = (cx + r * math.cos(ang)).round();
      final y = (cy + r * math.sin(ang)).round();
      img.drawLine(imgIn, x1: prevX, y1: prevY, x2: x, y2: y, color: color, thickness: 5);
      prevX = x; prevY = y;
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_error != null) {
      return Center(child: Text('Error: $_error'));
    }
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Emotion Image Transformer', style: Theme.of(context).textTheme.headlineSmall),
          const SizedBox(height: 8),
          Text('1. Pick an image  2. Detect original emotion  3. Choose a target emotion  4. Apply transformation.'),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            children: [
              ElevatedButton.icon(
                onPressed: _processing ? null : _pickImage,
                icon: const Icon(Icons.photo_library),
                label: const Text('Pick Image'),
              ),
              if (_inputImageFile != null)
                TextButton.icon(
                  onPressed: _processing ? null : () {
                    setState(() { _inputImageFile = null; _transformedImageBytes = null; _originalDetectedEmotion = null;});
                  },
                  icon: const Icon(Icons.delete_outline),
                  label: const Text('Remove'),
                ),
            ],
          ),
          const SizedBox(height: 12),
          if (_inputImageFile != null) ...[
            Row(
              children: [
                Expanded(
                  child: AspectRatio(
                    aspectRatio: 1,
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: Image.file(_inputImageFile!, fit: BoxFit.cover),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: AspectRatio(
                    aspectRatio: 1,
                    child: _transformedImageBytes == null
                        ? Container(
                            decoration: BoxDecoration(
                              border: Border.all(color: Colors.grey.shade300),
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: const Center(child: Text('No transform yet')),
                          )
                        : ClipRRect(
                            borderRadius: BorderRadius.circular(12),
                            child: Image.memory(_transformedImageBytes!, fit: BoxFit.cover),
                          ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Icon(_faceFound ? Icons.face_retouching_natural : Icons.info, color: _faceFound ? Colors.green : Colors.blue),
                const SizedBox(width: 6),
                Expanded(child: Text(_originalDetectedEmotion == null ? (_processing ? 'Analyzing...' : 'Tap Apply after selecting target emotion.') : 'Detected: ${_originalDetectedEmotion!.toUpperCase()} ${( (_originalConfidence ?? 0)*100).toStringAsFixed(1)}%  Faces: $_faceCount')),
              ],
            ),
            const Divider(height: 24),
          ],
          Text('Target Emotion:', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 6),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: _service.emotions.map((e) {
              final selected = e == _selectedEmotion;
              return ChoiceChip(
                label: Text(e),
                selected: selected,
                onSelected: (v) { if (_processing) return; setState(() { _selectedEmotion = v ? e : null; }); },
              );
            }).toList(),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              ElevatedButton.icon(
                onPressed: _processing || _inputImageFile == null || _selectedEmotion == null ? null : _transformToTarget,
                icon: _processing ? const SizedBox(width:16,height:16,child:CircularProgressIndicator(strokeWidth:2)) : const Icon(Icons.auto_fix_high),
                label: Text(_processing ? 'Processing...' : 'Apply Emotion'),
              ),
              const SizedBox(width: 12),
              if (_transformedImageBytes != null)
                TextButton.icon(
                  onPressed: _processing ? null : () { setState(() { _transformedImageBytes = null; }); },
                  icon: const Icon(Icons.refresh),
                  label: const Text('Reset Result'),
                ),
            ],
          ),
          const SizedBox(height: 16),
          if (_inputImageFile == null)
            Expanded(
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Icon(Icons.photo_library_outlined, size: 72, color: Colors.grey),
                    const SizedBox(height: 12),
                    Text('Pick an image to start transforming emotions.', style: Theme.of(context).textTheme.bodyLarge),
                  ],
                ),
              ),
            ) else const SizedBox.shrink(),
        ],
      ),
    );
  }
}

// Utility structures
class _Color {
  final int r, g, b;
  const _Color(this.r, this.g, this.b);
}

class _Palette {
  final _Color bgStart;
  final _Color bgEnd;
  final _Color accent;
  final _Color face;
  final _Color eye;
  final _Color mouth;
  final _Color text;
  const _Palette(
      {required this.bgStart,
      required this.bgEnd,
      required this.accent,
      required this.face,
      required this.eye,
      required this.mouth,
      required this.text});
}

class _PseudoRandom {
  int _state;
  _PseudoRandom(this._state);
  double nextDouble() {
    _state = (_state * 48271) % 0x7fffffff;
    return _state / 0x7fffffff;
  }

  int nextInt(int max) => (nextDouble() * max).floor();
}

// (Removed custom math approximations; using dart:math for clarity & reliability.)
