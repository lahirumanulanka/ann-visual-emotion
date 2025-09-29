import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

/// Placeholder explanation service that simulates Grad-CAM / Grad-CAM++ heatmaps
/// and SHAP / LIME style region attributions. In a production setup these
/// methods would call a backend API (e.g., FastAPI/Flask server) that performs
/// model introspection on the true CNN. The API would return raw saliency map
/// tensors which we would post-process to RGBA overlays.
class ExplanationService {
  static final ExplanationService _singleton = ExplanationService._internal();
  factory ExplanationService() => _singleton;
  ExplanationService._internal();

  bool _initialized = false;

  Future<void> initialize() async {
    // Potentially load model metadata, color maps, etc.
    _initialized = true;
  }

  bool get isInitialized => _initialized;

  /// Simulate a Grad-CAM / Grad-CAM++ heatmap of size (h,w) returned as rgba bytes.
  /// Values concentrate near random Gaussian blobs for visual variety.
  Future<Uint8List> generateHeatmap({
    required int width,
    required int height,
    bool gradCamPlusPlus = false,
  }) async {
    // Build an image buffer using the `image` package so we can return
    // a valid PNG instead of raw RGBA bytes (previous version caused
    // Invalid image data exceptions in Image.memory).
    final rand = Random();
    final centers = List.generate(gradCamPlusPlus ? 4 : 3,
        (_) => Offset(rand.nextDouble(), rand.nextDouble()));
    final heat = img.Image(width: width, height: height); // defaults RGBA
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final nx = x / (width - 1);
        final ny = y / (height - 1);
        double v = 0.0;
        for (final c in centers) {
          final dx = nx - c.dx;
          final dy = ny - c.dy;
          final d2 = dx * dx + dy * dy;
          v += (1.0 / (1 + d2 * (gradCamPlusPlus ? 40 : 25)));
        }
        v = (v / centers.length).clamp(0.0, 1.0);
        Color color;
        if (v < 0.5) {
          final t = v / 0.5;
            color = Color.lerp(const Color(0xFF0000FF), const Color(0xFFFFFF00), t)!;
        } else {
          final t = (v - 0.5) / 0.5;
          color = Color.lerp(const Color(0xFFFFFF00), const Color(0xFFFF0000), t)!;
        }
        final a = (160 + (v * 80)).clamp(0, 255).toInt();
        heat.setPixelRgba(x, y, color.red, color.green, color.blue, a);
      }
    }
    final png = img.encodePng(heat, level: 3);
    return Uint8List.fromList(png);
  }

  /// Simulated SHAP style ranked regions – returns a list of facial region contributions.
  Future<List<RegionAttribution>> shapAttributions() async {
    final rand = Random();
    final regions = [
      'Left Eye',
      'Right Eye',
      'Mouth',
      'Cheeks',
      'Eyebrows',
      'Nose',
      'Jawline'
    ];
    return regions
        .map((r) => RegionAttribution(
            region: r, contribution: rand.nextDouble() * 2 - 1))
        .toList()
      ..sort((a, b) => b.contribution.abs().compareTo(a.contribution.abs()));
  }

  /// Simulated LIME style binary saliency mask (1=important) with same dimensions.
  Future<Uint8List> limeMask(int width, int height) async {
    final rand = Random();
    final bytes = Uint8List(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final nx = x / width;
        final ny = y / height;
        final pattern = (sin(nx * 10) + cos(ny * 8) + rand.nextDouble()) /
            3.0; // -1..1 approx
        bytes[y * width + x] = pattern > 0.3 ? 255 : 0;
      }
    }
    return bytes;
  }
}

class RegionAttribution {
  final String region;
  final double
      contribution; // negative=negative influence, positive=positive influence
  RegionAttribution({required this.region, required this.contribution});
}
