import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class TFLiteModel {
  late Interpreter _interpreter;

  // Load the TFLite model
  Future<void> loadModel() async {
    _interpreter = await Interpreter.fromAsset('model.tflite');
    print("Model loaded successfully!");
  }

  // Run inference
  Future<List<double>> runInference(Uint8List inputImage) async {
    var inputShape = _interpreter.getInputTensor(0).shape;
    var outputShape = _interpreter.getOutputTensor(0).shape;

    // Create input and output buffers
    List.filled(inputShape.reduce((a, b) => a * b), 0.0).reshape(inputShape);
    var output = List.filled(outputShape.reduce((a, b) => a * b), 0.0)
        .reshape(outputShape);

    // Resize image if necessary and normalize
    final image = img.decodeImage(inputImage)!;
    final resizedImage = img.copyResize(image, width: 224, height: 224);
    final normalizedImage = resizedImage.data.map((e) => e / 255.0).toList();

    // Run inference
    _interpreter.run([normalizedImage], output);

    // Convert output to a list of probabilities
    return output[0].cast<double>();
  }

  // Load labels from JSON
  Future<Map<int, String>> loadLabels() async {
    final jsonString =
        await rootBundle.loadString('assets/imagenet_class_index.json');
    final Map<String, dynamic> jsonMap = json.decode(jsonString);
    return jsonMap
        .map((key, value) => MapEntry(int.parse(key), value.toString()));
  }

  // Predict and map to label
  Future<String> predict(Uint8List inputImage) async {
    final predictions = await runInference(inputImage);
    final labels = await loadLabels();

    // Find the index with the highest probability
    final predictedIndex = predictions.indexOf(predictions.reduce(max));
    return labels[predictedIndex] ?? "Unknown";
  }
}
