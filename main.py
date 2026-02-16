from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import keras
import numpy as np
from PIL import Image
import io
import os
from typing import Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fish Disease Detection API",
    description="AI-powered fish disease detection using InceptionV3",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the web app
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables
model = None
class_names = []

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "./inceptionv3_fish_final.h5")
IMG_SIZE = (224, 224)  # InceptionV3 default input size

# Define your fish disease classes here
# IMPORTANT: Order must match training data folders (alphabetical)
DISEASE_CLASSES = [
    "Bacterial Red disease",                    # Index 0
    "Bacterial diseases - Aeromoniasis",        # Index 1
    "Bacterial gill disease",                   # Index 2
    "Fungal diseases Saprolegniasis",          # Index 3
    "Healthy Fish",                             # Index 4
    "Parasitic diseases",                       # Index 5
    "Viral diseases White tail disease"        # Index 6
]


def load_model():
    """Load the trained model"""
    global model, class_names
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        model = keras.models.load_model(MODEL_PATH)
        class_names = DISEASE_CLASSES
        logger.info("Model loaded successfully")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Number of classes: {len(class_names)}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model prediction using InceptionV3 preprocessing"""
    try:
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize to model input size
        image = image.resize(IMG_SIZE)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Use InceptionV3's preprocessing (scales to -1 to 1)
        img_array = keras.applications.inception_v3.preprocess_input(img_array)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with open("static/index.html", "r") as f:
        return f.read()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "num_classes": len(class_names)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    """
    Predict fish disease from uploaded image
    
    Args:
        file: Uploaded image file
    
    Returns:
        Dictionary with prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess image first
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get prediction probabilities
        prediction_probs = predictions[0]
        
        # Calculate entropy (measures uncertainty/randomness)
        entropy = -np.sum(prediction_probs * np.log(prediction_probs + 1e-10))
        max_entropy = np.log(len(class_names))
        normalized_entropy = entropy / max_entropy
        
        # Calculate difference between top 2 predictions
        sorted_probs = np.sort(prediction_probs)[::-1]
        top_2_diff = float(sorted_probs[0] - sorted_probs[1])
        
        # Image quality checks
        img_array = np.array(image.convert("RGB"))
        img_std = float(np.std(img_array))
        
        # Non-fish detection logic (simplified and more reliable)
        # A real fish image should have:
        # 1. Reasonable confidence (> 0.40)
        # 2. Clear winner in predictions (top_2_diff > 0.15)
        # 3. Not too uniform (std > 15)
        
        is_non_fish = (
            (confidence < 0.40) or  # Very low confidence
            (confidence < 0.55 and normalized_entropy > 0.75) or  # Low confidence + high uncertainty
            (img_std < 15) or  # Too uniform
            (confidence < 0.50 and top_2_diff < 0.15)  # Low confidence + close predictions
        )
        
        # Uncertain but possibly fish
        is_uncertain_fish = not is_non_fish and (
            confidence < 0.70 or
            (normalized_entropy > 0.65 and top_2_diff < 0.25)
        )
        
        # Get all predictions
        all_predictions = [
            {
                "class": class_names[i],
                "confidence": float(predictions[0][i])
            }
            for i in range(len(class_names))
        ]
        
        # Sort by confidence
        all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Determine if this looks like a fish image
        if is_non_fish:
            result = {
                "success": False,
                "predicted_class": "Not a Fish",
                "confidence": confidence,
                "confidence_percentage": f"{confidence * 100:.2f}%",
                "all_predictions": all_predictions,
                "image_size": image.size,
                "is_fish": False,
                "entropy": float(normalized_entropy),
                "top_2_diff": float(top_2_diff),
                "image_std": float(img_std),
                "message": "⚠️ This image doesn't appear to be a fish or the quality is insufficient for diagnosis. Please upload a clear photo of a fish."
            }
            logger.warning(f"Non-fish detected - Conf: {confidence:.2%}, Std: {img_std:.1f}, Entropy: {normalized_entropy:.2f}, Top2Diff: {top_2_diff:.2f}")
        
        elif is_uncertain_fish:
            result = {
                "success": True,
                "predicted_class": class_names[predicted_class_idx],
                "confidence": confidence,
                "confidence_percentage": f"{confidence * 100:.2f}%",
                "all_predictions": all_predictions,
                "image_size": image.size,
                "is_fish": True,
                "is_uncertain": True,
                "entropy": float(normalized_entropy),
                "top_2_diff": float(top_2_diff),
                "image_std": float(img_std),
                "message": f"{get_disease_message(class_names[predicted_class_idx], confidence)}\n\n⚠️ Note: Moderate confidence ({confidence*100:.1f}%). For best results, use clear, well-lit images showing the fish clearly."
            }
            logger.info(f"Uncertain: {class_names[predicted_class_idx]} ({confidence:.2%}, entropy: {normalized_entropy:.2f})")
        
        else:
            result = {
                "success": True,
                "predicted_class": class_names[predicted_class_idx],
                "confidence": confidence,
                "confidence_percentage": f"{confidence * 100:.2f}%",
                "all_predictions": all_predictions,
                "image_size": image.size,
                "is_fish": True,
                "is_uncertain": False,
                "entropy": float(normalized_entropy),
                "top_2_diff": float(top_2_diff),
                "image_std": float(img_std),
                "message": get_disease_message(class_names[predicted_class_idx], confidence)
            }
            logger.info(f"Confident: {class_names[predicted_class_idx]} ({confidence:.2%})")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


def get_disease_message(disease: str, confidence: float) -> str:
    """Generate informative message based on prediction"""
    if confidence < 0.5:
        return "Low confidence prediction. Please try with a clearer image."
    
    messages = {
        "Healthy Fish": "The fish appears to be healthy! Continue regular monitoring and maintain good water quality.",
        "Bacterial diseases - Aeromoniasis": "Aeromonas infection detected. This bacterial disease causes ulcers and hemorrhages. Immediate antibiotic treatment and water quality improvement recommended. Quarantine affected fish.",
        "Bacterial gill disease": "Bacterial gill infection detected. This affects fish breathing. Improve water quality, increase oxygen, and consult a veterinarian for antibiotic treatment.",
        "Bacterial Red disease": "Bacterial Red disease (likely Motile Aeromonad Septicemia) detected. This serious infection causes red sores. Immediate veterinary care and antibiotics needed. Quarantine essential.",
        "Fungal diseases Saprolegniasis": "Saprolegniasis (cotton wool disease) detected. Fungal infection often secondary to injury or stress. Treat with antifungal medication and improve water conditions.",
        "Parasitic diseases": "Parasitic infection detected. Quarantine immediately, identify the specific parasite, and apply appropriate antiparasitic treatment. Monitor all tank inhabitants.",
        "Viral diseases White tail disease": "White tail viral disease detected. This serious viral infection requires immediate isolation. No direct cure available - supportive care and biosecurity measures essential."
    }
    
    return messages.get(disease, "Prediction complete. Consult a professional aquatic veterinarian for accurate diagnosis and treatment.")


@app.get("/classes")
async def get_classes():
    """Get list of disease classes"""
    return {
        "classes": class_names,
        "num_classes": len(class_names)
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
