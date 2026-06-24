# Flutter App Update Prompt — Fish Disease Detection v3.0

> **Copy and paste this entire prompt to an AI assistant (e.g., GitHub Copilot, ChatGPT) while working on your Flutter project to get the necessary code changes.**

---

## What changed in v3.0 (read first)

The backend now runs a new model — **ATF-Net**, an attention cross-fusion network that combines a custom CNN + ResNet50 + a Vision Transformer (ViT-Tiny) with triple cross-attention. Compared to v2.0:

- **Disease classes: now 7 (was 11).** The class list below changed — update any hardcoded class names/colors.
- **The JSON response format is otherwise IDENTICAL to v2.0.** All field names (`predicted_class`, `confidence`, `all_predictions`, `gradcam_heatmap_overlay`, `gradcam_heatmap_only`, `affected_area_percentage`, `xai`, …) are unchanged, so if your app already supports v2.0 the only required change is the 7-class list. The Grad-CAM heatmap + XAI sections keep working as-is.

If your app is still on the old 7-field/pre-Grad-CAM format, follow the full guide below.

## Prompt

I need to update my Flutter fish disease detection app to work with the API v3.0 which includes **Grad-CAM heatmap visualization** and **Explainable AI (XAI)** features.

### Current API Endpoint

- **URL**: `POST /predict`
- **Request**: `multipart/form-data` with field `file` (image file)
- **Health Check**: `GET /health`

### New API Response Format (v2.0)

The `/predict` endpoint now returns this JSON structure:

```json
{
  "predicted_class": "Bacterial Red disease",
  "confidence": 0.9523,
  "confidence_percentage": "95.23%",
  "message": "Bacterial Red Disease detected (95.2% confidence). This is a serious bacterial infection...",
  "is_fish": true,
  "all_predictions": [
    {"class": "Bacterial Red disease", "confidence": 0.9523},
    {"class": "Healthy Fish", "confidence": 0.0234},
    {"class": "Parasitic diseases", "confidence": 0.0102},
    ...
  ],
  "gradcam_heatmap_overlay": "<base64_encoded_png_string — includes red bounding box around affected area with 'Affected Area' label>",
  "gradcam_heatmap_only": "<base64_encoded_png_string>",
  "affected_area_percentage": 21.1,
  "xai": {
    "summary": "The AI detected 'Bacterial Red disease' with very high confidence (95.2%). The Grad-CAM heatmap shows the areas the model focused on during analysis.",
    "affected_regions": "Body surface, skin, and fins - reddish discoloration and hemorrhagic patches",
    "confidence_level": "Very High Confidence",
    "reasoning": "The model detected red/inflamed patches on the fish body surface consistent with bacterial red disease...",
    "recommendation": "Isolate affected fish immediately. Treat with appropriate antibiotics (oxytetracycline)..."
  }
}
```

### The `/health` endpoint returns:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_classes": 7,
  "version": "3.0.0",
  "model_file": "ATF_Net_Fusion_Model_inference.h5",
  "model_architecture": "ATF-Net (Custom CNN + ResNet50 + ViT-Tiny, triple cross-attention fusion)",
  "gradcam_layer": "conv5_block3_out",
  "image_size": [224, 224]
}
```

### The 7 Disease Classes (changed from 11 in v2.0)

0. Bacterial Red disease
1. Bacterial diseases - Aeromoniasis
2. Bacterial gill disease
3. Fungal diseases - Saprolegniasis
4. Healthy Fish
5. Parasitic diseases
6. Viral diseases - White tail disease

### What I Need Updated in the Flutter App

#### 1. **API Response Model**
Update the Dart model class to include the new fields:
- `gradcam_heatmap_overlay` (nullable String — base64 PNG, includes red bounding box around affected area)
- `gradcam_heatmap_only` (nullable String — base64 PNG)
- `affected_area_percentage` (nullable double — e.g. 21.1 means 21.1% of image is affected)
- `xai` (nullable object with fields: `summary`, `affected_regions`, `confidence_level`, `reasoning`, `recommendation`)

#### 2. **Results Screen — Grad-CAM Heatmap Display**
Add a section to the results screen that shows:
- **Tab/Toggle switch** with 3 views: "Overlay", "Side by Side", "Heatmap Only"
- **Overlay view**: Display the `gradcam_heatmap_overlay` base64 image — this image already has a **red bounding box** drawn around the most affected area with an **"Affected Area" label** baked into the image by the backend
- **Side by Side view**: Show original uploaded image next to the heatmap overlay
- **Heatmap Only view**: Display the `gradcam_heatmap_only` base64 image
- **Affected Area Percentage**: Show `affected_area_percentage` (e.g. "21.1% of image affected") as a stat badge with a 🎯 icon, displayed above the heatmap images. Style it with a red-tinted background.
- Add a label/description: "The heatmap highlights regions the AI focused on. Red/yellow = high importance, Blue/green = less significant. The red bounding box marks the affected area."

To decode base64 images in Flutter:
```dart
import 'dart:convert';
import 'dart:typed_data';

Uint8List bytes = base64Decode(base64String);
Image.memory(bytes, fit: BoxFit.contain)
```

#### 3. **Results Screen — XAI Explanation Section**
Add a new expandable/collapsible section below the heatmap that shows:
- **Confidence Level Badge**: Color-coded badge (green for High/Very High, amber for Moderate, red for Low)
- **Summary**: The `xai.summary` text
- **Affected Regions** card with 📍 icon: `xai.affected_regions`
- **Reasoning** card with 🔍 icon: `xai.reasoning`
- **Recommendation** card with 💊 icon: `xai.recommendation`

#### 4. **All Predictions List**
- Update to handle 7 classes (was 11)
- Highlight the top prediction in the list
- Show percentage bars for each class

#### 5. **Disease Message**
- The `message` field contains a detailed diagnosis message — display it prominently
- Color-code the message box: green for "Healthy Fish", red for diseases, amber for low confidence

#### 6. **Color Scheme Suggestions**
- Healthy Fish: Green (#10b981)
- Disease detected (high confidence): Red (#ef4444)
- Low confidence: Amber (#f59e0b)
- Grad-CAM section background: Warm yellow (#fffbeb)
- XAI section background: Light blue (#f0f9ff)

#### 7. **Error Handling**
- Handle cases where `gradcam_heatmap_overlay` or `gradcam_heatmap_only` might be `null`
- Handle cases where `xai` might be `null`
- Show graceful fallback UI when heatmap is unavailable

### Example Dart Model Class

```dart
class PredictionResult {
  final String predictedClass;
  final double confidence;
  final String confidencePercentage;
  final String message;
  final bool isFish;
  final List<ClassPrediction> allPredictions;
  final String? gradcamHeatmapOverlay; // base64 PNG with red bounding box
  final String? gradcamHeatmapOnly;    // base64 PNG
  final double? affectedAreaPercentage; // e.g. 21.1 means 21.1%
  final XaiExplanation? xai;

  PredictionResult({
    required this.predictedClass,
    required this.confidence,
    required this.confidencePercentage,
    required this.message,
    required this.isFish,
    required this.allPredictions,
    this.gradcamHeatmapOverlay,
    this.gradcamHeatmapOnly,
    this.affectedAreaPercentage,
    this.xai,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      predictedClass: json['predicted_class'],
      confidence: json['confidence'].toDouble(),
      confidencePercentage: json['confidence_percentage'],
      message: json['message'],
      isFish: json['is_fish'] ?? true,
      allPredictions: (json['all_predictions'] as List)
          .map((p) => ClassPrediction.fromJson(p))
          .toList(),
      gradcamHeatmapOverlay: json['gradcam_heatmap_overlay'],
      gradcamHeatmapOnly: json['gradcam_heatmap_only'],
      affectedAreaPercentage: json['affected_area_percentage']?.toDouble(),
      xai: json['xai'] != null ? XaiExplanation.fromJson(json['xai']) : null,
    );
  }
}

class ClassPrediction {
  final String className;
  final double confidence;

  ClassPrediction({required this.className, required this.confidence});

  factory ClassPrediction.fromJson(Map<String, dynamic> json) {
    return ClassPrediction(
      className: json['class'],
      confidence: json['confidence'].toDouble(),
    );
  }
}

class XaiExplanation {
  final String summary;
  final String affectedRegions;
  final String confidenceLevel;
  final String reasoning;
  final String recommendation;

  XaiExplanation({
    required this.summary,
    required this.affectedRegions,
    required this.confidenceLevel,
    required this.reasoning,
    required this.recommendation,
  });

  factory XaiExplanation.fromJson(Map<String, dynamic> json) {
    return XaiExplanation(
      summary: json['summary'] ?? '',
      affectedRegions: json['affected_regions'] ?? '',
      confidenceLevel: json['confidence_level'] ?? '',
      reasoning: json['reasoning'] ?? '',
      recommendation: json['recommendation'] ?? '',
    );
  }
}
```

### UI Layout for Results Screen (suggested order)

1. **Diagnosis Card** — Disease name + confidence bar + confidence badge
2. **Diagnosis Message** — Color-coded message box
3. **🔥 Grad-CAM Heatmap Section** — With "🎯 Affected Area: X.X% of image" stat badge + Tabbed view (Overlay / Side by Side / Heatmap Only). The overlay image already has a red bounding box with "Affected Area" label drawn by the backend — just display it as-is.
4. **🧠 XAI Explanation Section** — Expandable cards (Summary, Affected Regions, Reasoning, Recommendation)
5. **📈 All Predictions** — Sorted list with percentage bars
6. **Analyze Another** button

### Important Notes
- The `gradcam_heatmap_overlay` image already contains a **red bounding box** around the most affected area with an "Affected Area" text label drawn on it by the API. You do NOT need to draw bounding boxes in Flutter — just display the base64 image directly.
- The `affected_area_percentage` is a number like `21.1` meaning 21.1% of the image pixels are in the "affected" zone. Display this as a badge/chip like "🎯 21.1% of image affected".
- Handle `null` values gracefully for `gradcam_heatmap_overlay`, `gradcam_heatmap_only`, `affected_area_percentage`, and `xai`.

Please update the existing Flutter code to incorporate all these changes. Keep the existing app design/theme but add the new sections described above.
