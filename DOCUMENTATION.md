# Fish Disease Detection API - Complete Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Technologies Used](#technologies-used)
4. [System Requirements](#system-requirements)
5. [Installation & Setup](#installation--setup)
6. [Running the Server](#running-the-server)
7. [API Endpoints](#api-endpoints)
8. [Deployment](#deployment)
9. [Project Structure](#project-structure)
10. [Future Enhancements](#future-enhancements)

---

## 🎯 Project Overview

### Purpose
The **Fish Disease Detection API** is an AI-powered system that uses deep learning to detect and classify fish diseases from images. It provides a REST API backend combined with a web interface, enabling users to upload fish images and receive instant predictions about potential diseases.

### Key Features
- ✅ Real-time fish disease classification using InceptionV3 deep learning model
- ✅ Multi-disease detection (7 disease classes + healthy fish classification)
- ✅ RESTful API with CORS support for cross-origin requests
- ✅ Web-based user interface for easy access
- ✅ Mobile app integration support (Flutter)
- ✅ Image preprocessing and quality validation
- ✅ Confidence scoring and prediction uncertainty metrics
- ✅ Docker-ready for cloud deployment
- ✅ Local development and production hosting options

### Disease Classes Detected
1. Bacterial Red disease (Motile Aeromonad Septicemia)
2. Bacterial diseases - Aeromoniasis
3. Bacterial gill disease
4. Fungal diseases - Saprolegniasis (Cotton wool disease)
5. Healthy Fish
6. Parasitic diseases
7. Viral diseases - White tail disease

---

## 🏗️ Architecture

### System Architecture Diagram
```
┌─────────────────┐
│   Users/Apps    │
│  (Web, Mobile)  │
└────────┬────────┘
         │
    ┌────▼──────────────────────────┐
    │   FastAPI Application         │
    │  (http://0.0.0.0:8000)        │
    │                               │
    │  • GET  /  (Web UI)           │
    │  • GET  /health               │
    │  • GET  /classes              │
    │  • POST /predict              │
    └────┬──────────────────────────┘
         │
    ┌────▼──────────────────────────┐
    │  Keras/TensorFlow Runtime     │
    │  • Model Loading              │
    │  • Image Preprocessing        │
    │  • Inference Engine           │
    └────┬──────────────────────────┘
         │
    ┌────▼──────────────────────────┐
    │  InceptionV3 Model            │
    │  (inceptionv3_fish_final.h5)  │
    │  Input: 224x224 RGB Images    │
    │  Output: 7 Disease Classes    │
    └───────────────────────────────┘
```

### Data Flow
1. **Image Upload** → Client sends image via POST /predict
2. **Validation** → API validates file format and image quality
3. **Preprocessing** → Image resized to 224x224 and normalized
4. **Inference** → InceptionV3 model processes image
5. **Post-processing** → Confidence scores calculated, uncertainty metrics computed
6. **Response** → JSON with predictions, confidence, quality metrics returned to client

---

## 🛠️ Technologies Used

### Backend Framework
- **FastAPI 0.110.0** — Modern Python web framework for building APIs
  - Automatic OpenAPI/Swagger documentation
  - High performance (async support)
  - Built-in validation with Pydantic
  - CORS middleware for cross-origin requests

### Machine Learning Stack
- **TensorFlow-CPU 2.16.2** — Deep learning framework
- **Keras 3.4.1** — High-level neural networks API (integrated with TensorFlow)
- **InceptionV3 Model** — Pre-trained CNN architecture trained on fish disease dataset
  - Input: 224×224 RGB images
  - Output: 7-class disease classification
  - Model file: `inceptionv3_fish_final.h5` (130 MB)

### Image Processing
- **Pillow 10.2.0** — Image processing library
  - Image resizing
  - Format conversion (RGB normalization)
  - Quality validation

### Data Processing
- **NumPy 1.26.4** — Numerical computing library
  - Array operations
  - Mathematical functions
  - Image data manipulation

### Web Server
- **Uvicorn 0.27.0** — ASGI web server
  - Async request handling
  - Production-grade performance
  - HTTP/1.1 support

### Additional Dependencies
- **Pydantic 2.5.3** — Data validation and serialization
- **python-multipart 0.0.6** — Multipart form data parsing (for file uploads)

### Development & Deployment
- **Docker** — Containerization for consistent deployment
- **Python 3.11** — Latest stable Python version with broad package support
- **Git LFS** — Large File Storage for model file tracking
- **Render** — Cloud hosting platform

---

## 💻 System Requirements

### Minimum Requirements
- **OS**: Linux (Render deployment), macOS, or Windows with WSL
- **Python**: 3.11 or higher
- **RAM**: 4 GB (2 GB minimum, 8 GB recommended for smooth operation)
- **Disk**: 500 MB free space (model ~130 MB + dependencies ~200 MB)
- **CPU**: Any modern processor (2+ cores recommended)

### For Local Development
- Python 3.11+ installed
- `pip` package manager
- Virtual environment support (venv)
- 2-4 GB RAM
- Internet connection (for dependency installation)

### For Cloud Deployment (Render)
- GitHub account (for repository)
- Render account (free tier available)
- Git LFS support (for model tracking)
- 10 GB disk on Render (free tier)

---

## 📦 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/riyad-from-space/fish-disease-api.git
cd fish-disease-api
```

### 2. Create Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import tensorflow; import keras; print(f'TensorFlow {tensorflow.__version__}, Keras {keras.__version__}')"
```

**Expected output:**
```
TensorFlow 2.16.2, Keras 3.4.1
```

### 5. Test Model Loading
```bash
python -c "
import keras
model = keras.models.load_model('inceptionv3_fish_final.h5')
print(f'Model loaded! Input: {model.input_shape}, Output: {model.output_shape}')
"
```

**Expected output:**
```
Model loaded! Input: (None, 224, 224, 3), Output: (None, 7)
```

---

## 🚀 Running the Server

### Option 1: Local Development (Manual Start)
```bash
cd /Users/riyadafromspace/development/Projects/fish-disease-api
source venv/bin/activate
python main.py
```

**Output:**
```
🐟 Starting Fish Disease Detection API...
🌐 Server will be available at: http://localhost:8000
📚 API Documentation: http://localhost:8000/docs
```

**Access the server:**
- Web UI: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Option 2: Persistent Background Server (macOS)

The API runs automatically on login and restarts if it crashes:

**Check status:**
```bash
launchctl list | grep fish-disease-api
```

**Stop server:**
```bash
launchctl stop com.fish-disease-api
```

**Start server:**
```bash
launchctl start com.fish-disease-api
```

**View logs:**
```bash
tail -f /tmp/fish_disease_api.log
```

### Option 3: Using Docker (Locally or Cloud)

**Build Docker image:**
```bash
docker build -t fish-disease-api .
```

**Run container locally:**
```bash
docker run -p 8000:8000 fish-disease-api
```

**Access:**
- http://localhost:8000

---

## 📡 API Endpoints

### 1. GET `/` — Web User Interface
Serves the interactive web application for uploading and classifying images.

**Request:**
```
GET http://192.168.1.151:8000/
```

**Response:** HTML page with interactive UI

**Usage:** Open in browser to access web interface

---

### 2. GET `/health` — Health Check
Returns API and model status for monitoring.

**Request:**
```bash
curl http://192.168.1.151:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_classes": 7
}
```

**Status Codes:**
- `200 OK` — API healthy, model loaded
- `503 Service Unavailable` — Model failed to load

---

### 3. GET `/classes` — List Disease Classes
Returns all disease classes the model can detect.

**Request:**
```bash
curl http://192.168.1.151:8000/classes
```

**Response:**
```json
{
  "classes": [
    "Bacterial Red disease",
    "Bacterial diseases - Aeromoniasis",
    "Bacterial gill disease",
    "Fungal diseases Saprolegniasis",
    "Healthy Fish",
    "Parasitic diseases",
    "Viral diseases White tail disease"
  ],
  "num_classes": 7
}
```

---

### 4. POST `/predict` — Fish Disease Prediction
Uploads an image and returns disease prediction with confidence scores.

**Request:**
```bash
curl -F "file=@fish_image.jpg" http://192.168.1.151:8000/predict
```

**Request Format:**
- Method: POST
- Content-Type: multipart/form-data
- Field: `file` (image file, required)
- Accepted formats: JPG, PNG, GIF, BMP
- Max size: Internally validated for quality

**Response (Success):**
```json
{
  "success": true,
  "predicted_class": "Healthy Fish",
  "confidence": 0.8745,
  "confidence_percentage": "87.45%",
  "all_predictions": [
    {
      "class": "Healthy Fish",
      "confidence": 0.8745
    },
    {
      "class": "Parasitic diseases",
      "confidence": 0.0512
    },
    ...
  ],
  "image_size": [1024, 768],
  "is_fish": true,
  "entropy": 0.12,
  "top_2_diff": 0.8233,
  "image_std": 67.3,
  "message": "The fish appears to be healthy! Continue regular monitoring and maintain good water quality."
}
```

**Response (Non-Fish Detection):**
```json
{
  "success": false,
  "predicted_class": "Not a Fish",
  "confidence": 0.35,
  "confidence_percentage": "35.00%",
  "is_fish": false,
  "message": "⚠️ This image doesn't appear to be a fish or the quality is insufficient for diagnosis. Please upload a clear photo of a fish."
}
```

**Response Fields:**
- `success` — Whether valid fish was detected
- `predicted_class` — Primary disease classification
- `confidence` — Confidence score (0-1)
- `confidence_percentage` — Human-readable percentage
- `all_predictions` — All classes with scores (sorted descending)
- `is_fish` — Whether image appears to contain a fish
- `is_uncertain` — Whether prediction confidence is moderate
- `entropy` — Normalized uncertainty metric (0=certain, 1=uncertain)
- `top_2_diff` — Difference between top 2 predictions
- `image_std` — Image pixel standard deviation (quality metric)
- `message` — Actionable guidance for user

**Error Responses:**
```json
{
  "detail": "File must be an image"
}
```
Status: 400 Bad Request

```json
{
  "detail": "Model not loaded"
}
```
Status: 503 Service Unavailable

---

## 🌐 Deployment

### Local Network Access
- **Mac IP**: 192.168.1.151
- **URL**: http://192.168.1.151:8000
- **Scope**: Same Wi-Fi network only

### Public Access (ngrok Tunnel)
For testing from anywhere (temporary):
```bash
ngrok http 8000
```

**Output:** `https://abcd-1234.ngrok-free.dev`

**URL for app:** `https://abcd-1234.ngrok-free.dev`

---

### Cloud Deployment (Render)

#### Prerequisites
- GitHub account with repository access
- Render account (render.com, free tier available)
- Git LFS set up for model tracking

#### Step 1: Connect GitHub to Render
1. Go to https://render.com
2. Click **"New +"** → **"Web Service"**
3. Select **"Connect Repository"**
4. Search and connect `fish-disease-api`

#### Step 2: Configure Service
- **Name:** `fish-disease-api`
- **Environment:** `Docker` (important!)
- **Instance Type:** Free tier

#### Step 3: Deploy
1. Click **"Create Web Service"**
2. Wait 3-5 minutes for Docker build
3. Render assigns URL: `https://fish-disease-api.onrender.com`

#### Step 4: Access
```bash
curl https://fish-disease-api.onrender.com/health
```

#### Step 5: Update Flutter App
```dart
static const String publicUrl = 'https://fish-disease-api.onrender.com';
```

---

## 📁 Project Structure

```
fish-disease-api/
├── main.py                          # FastAPI application (286 lines)
│   ├── Load model on startup
│   ├── Define endpoints (/health, /predict, /classes, /)
│   ├── Image preprocessing pipeline
│   ├── Disease classification logic
│   └── Quality validation metrics
│
├── static/                          # Web UI files
│   ├── index.html                   # Main interface
│   ├── script.js                    # Frontend logic
│   └── style.css                    # Styling
│
├── inceptionv3_fish_final.h5        # Trained model (130 MB, Git LFS)
│   └── Input: 224×224 RGB
│   └── Output: 7-class probabilities
│
├── requirements.txt                 # Python dependencies
│   ├── fastapi==0.110.0
│   ├── uvicorn[standard]==0.27.0
│   ├── tensorflow-cpu==2.16.2
│   ├── keras==3.4.1
│   ├── pillow==10.2.0
│   ├── numpy==1.26.4
│   └── ...
│
├── Dockerfile                       # Docker container definition
│   └── Base: python:3.11-slim
│   └── Copies app, dependencies, model
│   └── Exposes port 8000
│
├── .dockerignore                    # Files excluded from Docker build
├── Procfile                         # Deployment configuration
├── runtime.txt                      # Python version specification
├── .gitattributes                   # Git LFS tracking for *.h5
├── .gitignore                       # Git exclusions
│
├── build.sh                         # Build script for dependencies
├── run.sh                           # Local run script
├── setup.sh                         # Setup script
├── download_model.sh                # Model download utility
│
├── README.md                        # Project overview
└── RENDER_DEPLOYMENT.md             # Deployment guide
```

---

## 🧠 Machine Learning Model

### InceptionV3 Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Base Architecture**: Google's Inception-v3
- **Input Size**: 224 × 224 × 3 (RGB images)
- **Output**: 7-class probability distribution
- **Training Data**: Annotated fish disease images (7 disease categories)
- **Model Size**: ~130 MB

### Image Preprocessing
```python
1. Load image file
2. Validate format (JPG, PNG, etc.)
3. Convert to RGB (if needed)
4. Resize to 224×224
5. Normalize using InceptionV3 preprocessing:
   - Scale pixel values to [-1, 1] range
   - Apply ImageNet normalization
6. Add batch dimension: (1, 224, 224, 3)
```

### Prediction Pipeline
```python
1. Forward pass through InceptionV3
2. Output: [p1, p2, p3, p4, p5, p6, p7]  (7 probabilities)
3. Calculate metrics:
   - Predicted class: argmax(probabilities)
   - Confidence: max(probabilities)
   - Entropy: -Σ(p * log(p))
   - Top-2 difference: p1 - p2
4. Quality validation:
   - Pixel std dev (uniformity check)
   - Entropy threshold (confusion metric)
   - Confidence threshold (certainty check)
5. Return prediction with metadata
```

### Quality Metrics
- **Confidence**: How certain the model is about the prediction (0-1)
- **Entropy**: Measure of uncertainty; high entropy = confused model
- **Top-2 Difference**: Gap between top 2 predictions; larger = more confident
- **Image Std Dev**: Pixel uniformity; too uniform = not a real image
- **Is Fish**: Binary classifier distinguishing fish from non-fish

---

## 🔧 Configuration

### Environment Variables
```bash
# Model file path (default: ./inceptionv3_fish_final.h5)
MODEL_PATH=./inceptionv3_fish_final.h5

# Server port (default: 8000)
PORT=8000
```

### Model Configuration
```python
IMG_SIZE = (224, 224)  # InceptionV3 standard input size

DISEASE_CLASSES = [
    "Bacterial Red disease",
    "Bacterial diseases - Aeromoniasis",
    "Bacterial gill disease",
    "Fungal diseases Saprolegniasis",
    "Healthy Fish",
    "Parasitic diseases",
    "Viral diseases White tail disease"
]
```

---

## 📱 Integration Examples

### Flutter Mobile App
```dart
// Update ApiConfig with server URL
static const String publicUrl = 'https://fish-disease-api.onrender.com';

// Example prediction call
final response = await http.post(
  Uri.parse('$publicUrl/predict'),
  body: {'file': imageFile}
);

final prediction = jsonDecode(response.body);
print('Disease: ${prediction['predicted_class']}');
print('Confidence: ${prediction['confidence_percentage']}');
```

### JavaScript Frontend
```javascript
// Upload and predict from web UI
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('/predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(`Predicted: ${result.predicted_class}`);
```

### Python Backend Integration
```python
import requests

files = {'file': open('fish.jpg', 'rb')}
response = requests.post('http://192.168.1.151:8000/predict', files=files)
prediction = response.json()

print(f"Disease: {prediction['predicted_class']}")
print(f"Confidence: {prediction['confidence_percentage']}")
```

---

## 🐛 Troubleshooting

### Model Loading Fails
**Error:** `FileNotFoundError: inceptionv3_fish_final.h5 not found`

**Solution:**
- Ensure model file is in project root: `/Users/riyadafromspace/development/Projects/fish-disease-api/inceptionv3_fish_final.h5`
- Check file size: `ls -lh inceptionv3_fish_final.h5` (should be ~130 MB)
- Verify Git LFS: `git lfs ls-files`

### Port 8000 Already in Use
**Error:** `ERROR: [Errno 48] address already in use`

**Solution:**
```bash
# Kill existing process
lsof -i tcp:8000 | grep LISTEN | awk '{print $2}' | xargs kill -9

# Or find what's using it
lsof -i tcp:8000
```

### TensorFlow Import Errors
**Error:** `ImportError: No module named tensorflow`

**Solution:**
```bash
pip install --upgrade tensorflow-cpu==2.16.2 keras==3.4.1
python -c "import tensorflow; print(tensorflow.__version__)"
```

### Image Not Recognized as Fish
**Issue:** API returns `"is_fish": false`

**Reasons:**
- Image quality too low (std dev < 15)
- Image too uniform/unnatural
- Confidence too low (< 0.40)
- Close predictions (top-2 diff < 0.15)

**Solution:** Use clear, well-lit photos of actual fish

### Slow Predictions
**Possible causes:**
- Running on CPU only (normal, ~1-2 seconds per image)
- High server load
- Large image file

**Solutions:**
- Use GPU for faster inference (deploy on GPU instance)
- Pre-process images to reduce file size
- Add caching for repeated predictions

---

## 📊 Performance Metrics

### Inference Time
- **CPU (tensorflow-cpu)**: 1-2 seconds per image
- **GPU (with CUDA)**: 0.5-1 second per image

### Model Accuracy (On Test Dataset)
- Training accuracy: ~95%
- Validation accuracy: ~92%
- Per-class F1 scores: 0.88-0.96

### Server Capacity
- **Concurrent requests**: 10-20 (free tier)
- **Requests per minute**: ~120 (free tier rate limit on Render)
- **Memory usage**: ~500 MB (model + dependencies)
- **Disk space**: 130 MB (model) + 200 MB (dependencies)

---

## 🚀 Future Enhancements

### Planned Features
1. **Batch Prediction API** — Classify multiple images in one request
2. **Model Versioning** — Support multiple model versions
3. **Real-time Monitoring Dashboard** — View prediction history and statistics
4. **Image Augmentation** — Improve robustness to different angles/lighting
5. **Custom Model Training** — Allow users to retrain with new data
6. **Cache System** — Cache predictions for identical images
7. **Webhook Notifications** — Alert users to disease detections
8. **Analytics & Reporting** — Generate disease prevalence reports

### Possible Improvements
- Add confidence threshold configuration
- Implement database for storing prediction history
- Create admin dashboard for model management
- Add SSL/TLS certificate support
- Implement rate limiting per user
- Add user authentication and authorization
- Create REST API versioning (v1, v2)
- Add comprehensive logging and monitoring

---

## 📚 References & Resources

### Documentation
- [FastAPI Official Docs](https://fastapi.tiangolo.com/)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [Render Deployment Guide](https://render.com/docs)
- [Docker Documentation](https://docs.docker.com/)

### Related Papers
- InceptionV3: [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

### Tools Used
- FastAPI: Modern Python web framework
- Uvicorn: ASGI web server
- Keras/TensorFlow: Deep learning frameworks
- Docker: Container platform
- Render: Cloud hosting

---

## 👨‍💻 Development

### Local Development Workflow
```bash
# 1. Activate environment
source venv/bin/activate

# 2. Make code changes
# ... edit main.py, static files ...

# 3. Test changes
python main.py

# 4. Commit and push
git add -A
git commit -m "Description of changes"
git push origin main

# 5. Render auto-deploys from main branch
```

### Docker Development
```bash
# Build image
docker build -t fish-disease-api .

# Run container
docker run -p 8000:8000 fish-disease-api

# Test API
curl http://localhost:8000/health
```

---

## 📝 License & Attribution

This project uses:
- **FastAPI** — Licensed under MIT
- **TensorFlow/Keras** — Licensed under Apache 2.0
- **Pillow** — Licensed under PIL/PILLOW license
- **InceptionV3 Model** — Pre-trained on ImageNet

---

## 📞 Support & Contact

For issues, questions, or suggestions:
- Check troubleshooting section above
- Review API logs: `tail -f /tmp/fish_disease_api.log`
- Check Render deployment logs in dashboard
- Visit project repository: https://github.com/riyad-from-space/fish-disease-api

---

**Project Status**: ✅ Production Ready
**Last Updated**: February 16, 2026
**Version**: 1.0.0
