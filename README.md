# 🐟 Fish Disease Detection API

An AI-powered web application for detecting fish diseases from images using **ATF-Net**, an attention cross-fusion deep learning model (Custom CNN + ResNet50 + Vision Transformer) with Grad-CAM explainability.

## ✨ Features

- 🤖 **AI-Powered Detection**: Uses ATF-Net (CNN + ResNet50 + ViT triple cross-attention fusion) for fish disease classification
- 🔥 **Grad-CAM Heatmaps**: Visual explanation of where the model is looking (on the ResNet50 branch)
- 🧠 **Explainable AI**: Per-disease reasoning, affected regions, and treatment recommendations
- ⚡ **FastAPI Backend**: High-performance REST API with automatic documentation
- 🎨 **Modern Web Interface**: Responsive, user-friendly interface with drag-and-drop support
- 📊 **Real-time Analysis**: Instant predictions with confidence scores
- 🔍 **Non-Fish Detection**: Intelligent filtering to reject non-fish images
- 💊 **Multiple Disease Classes**: Detects 7 different fish diseases

## 🏥 Supported Disease Classes

1. Bacterial Red disease
2. Bacterial diseases - Aeromoniasis
3. Bacterial gill disease
4. Fungal diseases Saprolegniasis
5. Healthy Fish
6. Parasitic diseases
7. Viral diseases White tail disease

## 📋 Prerequisites

- Python 3.8 or higher
- Model file: `inceptionv3_fish_final.h5` (should be in `/Users/riyadafromspace/Documents/`)

## 🛠️ Installation

### Quick Setup

```bash
# Navigate to project directory
cd /Users/riyadafromspace/Documents/api_integration

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install required packages
pip install fastapi uvicorn tensorflow pillow python-multipart numpy
```

## 🚀 Running the Application

### Start the Server

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the server
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## 📡 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_classes": 7
}
```

### Predict Disease

**Using cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/fish/image.jpg"
```

**Using Python:**
```python
import requests

url = "http://localhost:8000/predict"
files = {'file': open('fish_image.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

**Response:**
```json
{
  "success": true,
  "predicted_class": "Bacterial gill disease",
  "confidence": 0.8523,
  "confidence_percentage": "85.23%",
  "is_fish": true,
  "is_uncertain": false,
  "entropy": 0.42,
  "top_2_diff": 0.65,
  "image_std": 45.3,
  "all_predictions": [...],
  "message": "Bacterial gill infection detected..."
}
```

### Get Available Classes

```bash
curl http://localhost:8000/classes
```

## 🎨 Web Interface Usage

1. **Open** http://localhost:8000 in your browser
2. **Upload** a fish image by clicking or dragging and dropping
3. **Analyze** the image by clicking the "Analyze Image" button
4. **View** detailed results including:
   - Disease prediction
   - Confidence score
   - Treatment recommendations
   - All class probabilities

## 🔧 Configuration

### Model Path

By default, the model is loaded from:
```
/Users/riyadafromspace/Documents/inceptionv3_fish_final.h5
```

To change the model location, update `main.py`:
```python
MODEL_PATH = os.getenv("MODEL_PATH", "/path/to/your/model.h5")
```

### Non-Fish Detection Thresholds

Adjust detection sensitivity in `main.py` (around line 165):

```python
is_non_fish = (
    (confidence < 0.40) or  # Minimum confidence threshold
    (confidence < 0.55 and normalized_entropy > 0.75) or
    (img_std < 15) or  # Image uniformity threshold
    (confidence < 0.50 and top_2_diff < 0.15)
)
```

## 📁 Project Structure

```
api_integration/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .env.example           # Environment variables template
├── .gitignore            # Git ignore rules
├── run.sh                # Quick start script (Unix)
├── setup.sh              # Setup script (Unix)
├── static/
│   ├── index.html        # Web interface
│   ├── style.css         # Styling
│   └── script.js         # Client-side logic
└── venv/                 # Virtual environment (created on setup)
```

## 🔍 How It Works

### Non-Fish Image Detection

The system uses multiple indicators to detect non-fish images:

1. **Confidence Score**: Model's certainty (< 40% = likely not fish)
2. **Entropy**: Prediction uncertainty (high entropy = random/confused)
3. **Image Uniformity**: Pixel variance (too uniform = solid color/pattern)
4. **Prediction Spread**: Gap between top predictions (small gap = confused)

### Image Preprocessing

1. Convert to RGB (if needed)
2. Resize to 224x224 pixels
3. Apply InceptionV3 preprocessing (scales to -1 to 1)
4. Add batch dimension for model input

### Prediction Pipeline

1. **Upload** image via web interface or API
2. **Validate** file type
3. **Preprocess** image
4. **Predict** using InceptionV3 model
5. **Validate** if image is actually a fish
6. **Return** results with confidence scores and recommendations

## 🐛 Troubleshooting

### Model Not Loading

**Issue**: `Model not loaded` error

**Solutions**:
- Check model file path in `main.py`
- Ensure model file exists: `ls -lh /Users/riyadafromspace/Documents/inceptionv3_fish_final.h5`
- Verify file permissions

### Server Won't Start

**Issue**: Port already in use

**Solutions**:
```bash
# Find process using port 8000
lsof -ti:8000

# Kill the process
kill -9 $(lsof -ti:8000)

# Or use a different port
uvicorn main:app --port 8001
```

### Predictions Always "Not a Fish"

**Issue**: Real fish images rejected

**Solutions**:
- Check image quality (blur, lighting, resolution)
- Review server logs for metrics
- Lower confidence threshold in `main.py`

### Low Confidence Predictions

**Issue**: All predictions < 50%

**Solutions**:
- Use clearer, well-lit images
- Ensure fish is clearly visible
- Check if model was trained on similar images
- Verify preprocessing matches training

## 📝 Requirements

```
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
tensorflow>=2.13.0
pillow>=10.0.0
python-multipart>=0.0.6
numpy>=1.24.0
```

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is for educational and research purposes.

## 🔗 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Interactive API documentation with request/response examples.

## 💡 Tips

- **Best Results**: Use clear, well-lit photos of fish showing visible symptoms
- **Image Quality**: Higher resolution images (but will be resized to 224x224)
- **File Formats**: JPEG, PNG, JPG supported
- **Non-Fish Images**: System will automatically reject images that don't appear to be fish

## 🚀 Quick Start Script

For Unix/Linux/macOS:

```bash
chmod +x run.sh
./run.sh
```

This will activate the virtual environment and start the server.

---

**Model**: ATF-Net — Custom CNN + ResNet50 + ViT-Tiny, triple cross-attention fusion (36M parameters, 146MB inference file)  
**Framework**: TensorFlow 2.16 / Keras 3.13  
**API**: FastAPI  
**Input**: 224×224, rescaled to [0, 1] (`/255`)  
**Grad-CAM layer**: `conv5_block3_out` (ResNet50 branch)  

For questions or issues, check the logs or API documentation.
