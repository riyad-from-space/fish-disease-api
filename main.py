import os
import gc

# --- Memory Optimization (MUST be before TensorFlow import) ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # Suppress TF logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"          # Disable oneDNN (saves ~50MB)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"           # Force CPU only
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"      # Aggressive memory trimming

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Limit TensorFlow memory growth
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# --- Configuration ---
MODEL_PATH = "./best_inception.h5"
IMG_SIZE = (299, 299)

DISEASE_CLASSES = [
    "Bacterial Red disease",
    "Bacterial diseases - Aeromoniasis",
    "Bacterial gill disease",
    "Fungal diseases - Saprolegniasis",
    "Healthy Fish",
    "Parasitic diseases",
    "Tail rot and Fin rot",
    "Viral diseases - Lymphocystis",
    "Viral diseases - VHS",
    "Viral diseases - White tail disease",
    "White spot disease - Ich",
]

# --- App Setup ---
app = FastAPI(
    title="Fish Disease Detection API",
    description="Fish Disease Detection",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Load Model (lazy — only on first request to save startup memory) ---
model = None


def load_model():
    global model
    if model is not None:
        return model
    try:
        import keras
        model = keras.models.load_model(MODEL_PATH)
        print(f"Model loaded: {MODEL_PATH}")
        print(f"   Input shape : {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Classes     : {len(DISEASE_CLASSES)}")
        gc.collect()
        return model
    except Exception as e:
        print(f"Model load error: {e}")
        model = None
        return None





# --- Disease Messages ---
def get_disease_message(predicted_class, confidence):
    messages = {
        "Bacterial Red disease": f"Bacterial Red Disease detected ({confidence:.1%} confidence). Immediate antibiotic treatment recommended.",
        "Bacterial diseases - Aeromoniasis": f"Aeromoniasis detected ({confidence:.1%} confidence). Antibiotic treatment may be needed.",
        "Bacterial gill disease": f"Bacterial Gill Disease detected ({confidence:.1%} confidence). Improve water quality.",
        "Fungal diseases - Saprolegniasis": f"Saprolegniasis (fungal) detected ({confidence:.1%} confidence). Treat with antifungal agents.",
        "Healthy Fish": f"Your fish appears healthy ({confidence:.1%} confidence). Continue regular monitoring.",
        "Parasitic diseases": f"Parasitic infection detected ({confidence:.1%} confidence). Anti-parasitic treatment may be needed.",
        "Tail rot and Fin rot": f"Tail/Fin Rot detected ({confidence:.1%} confidence). Improve water quality and use antibacterial treatment.",
        "Viral diseases - Lymphocystis": f"Lymphocystis (viral) detected ({confidence:.1%} confidence). Maintain good water quality.",
        "Viral diseases - VHS": f"Viral Hemorrhagic Septicemia detected ({confidence:.1%} confidence). Quarantine immediately.",
        "Viral diseases - White tail disease": f"White Tail Disease detected ({confidence:.1%} confidence). Isolate affected fish.",
        "White spot disease - Ich": f"White Spot Disease (Ich) detected ({confidence:.1%} confidence). Raise water temperature and apply medication.",
    }
    return messages.get(
        predicted_class,
        f"Disease: {predicted_class} ({confidence:.1%} confidence). Consult a specialist."
    )


# --- Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health():
    return {
        "status": "healthy" if model else "model_not_loaded_yet",
        "model_loaded": model is not None,
        "num_classes": len(DISEASE_CLASSES),
        "version": "2.0.0",
        "model_file": os.path.basename(MODEL_PATH),
        "image_size": IMG_SIZE,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lazy load model on first prediction
    mdl = load_model()
    if mdl is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = mdl.predict(img_array, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])
        predicted_class = DISEASE_CLASSES[predicted_idx]

        # All predictions sorted
        all_preds = []
        for i, cls in enumerate(DISEASE_CLASSES):
            all_preds.append({"class": cls, "confidence": float(predictions[0][i])})
        all_preds.sort(key=lambda x: x["confidence"], reverse=True)

        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "message": get_disease_message(predicted_class, confidence),
            "all_predictions": all_preds,
            "is_fish": True,
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        gc.collect()


# --- Run ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
