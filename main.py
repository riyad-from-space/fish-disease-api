from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import keras
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import base64
import matplotlib
matplotlib.use('Agg')

# --- Configuration ---
MODEL_PATH = "./best_inception.h5"
IMG_SIZE = (299, 299)
GRADCAM_LAYER = "mixed10"

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
    description="Explainable AI Fish Disease Detection with Grad-CAM",
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

# --- Load Model ---
model = None


def load_model():
    global model
    try:
        model = keras.models.load_model(MODEL_PATH)
        print(f"Model loaded: {MODEL_PATH}")
        print(f"   Input shape : {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Classes     : {len(DISEASE_CLASSES)}")
    except Exception as e:
        print(f"Model load error: {e}")
        model = None


load_model()


# --- Grad-CAM ---
def generate_gradcam(img_array, model, layer_name=GRADCAM_LAYER):
    try:
        # Keras 3 may return model.output as a list — unwrap it
        model_output = model.output[0] if isinstance(model.output, list) else model.output

        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model_output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            predicted_class = tf.argmax(predictions[0])
            class_output = predictions[:, predicted_class]

        grads = tape.gradient(class_output, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

        return heatmap.numpy()
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


def create_heatmap_overlay(original_image, heatmap, alpha=0.4):
    try:
        img = original_image.resize(IMG_SIZE)
        img_array = np.array(img)

        heatmap_resized = np.array(
            Image.fromarray(np.uint8(heatmap * 255)).resize(IMG_SIZE)
        ) / 255.0

        colormap = matplotlib.colormaps["jet"]
        heatmap_colored = colormap(heatmap_resized)[:, :, :3]
        heatmap_colored = np.uint8(heatmap_colored * 255)

        overlay = np.uint8(img_array * (1 - alpha) + heatmap_colored * alpha)
        overlay_img = Image.fromarray(overlay)

        # Draw bounding box around the most affected area
        overlay_img = draw_affected_area_box(overlay_img, heatmap_resized)

        return overlay_img
    except Exception as e:
        print(f"Overlay error: {e}")
        return None


def draw_affected_area_box(overlay_img, heatmap_resized, threshold=0.5):
    """Draw a bounding box and label around the hottest region of the heatmap."""
    try:
        from PIL import ImageDraw, ImageFont

        # Find pixels above threshold
        hot_mask = heatmap_resized >= threshold
        coords = np.argwhere(hot_mask)

        if len(coords) == 0:
            return overlay_img

        # Bounding box: (y_min, x_min) to (y_max, x_max)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Add small padding
        pad = 6
        y_min = max(0, y_min - pad)
        x_min = max(0, x_min - pad)
        y_max = min(IMG_SIZE[1] - 1, y_max + pad)
        x_max = min(IMG_SIZE[0] - 1, x_max + pad)

        draw = ImageDraw.Draw(overlay_img)

        # Draw rectangle (red, 2px thick)
        for i in range(3):
            draw.rectangle(
                [(x_min - i, y_min - i), (x_max + i, y_max + i)],
                outline=(255, 50, 50),
            )

        # Draw label background + text
        label = "Affected Area"
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except Exception:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            except Exception:
                font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        label_x = x_min
        label_y = max(0, y_min - text_h - 8)

        draw.rectangle(
            [(label_x, label_y), (label_x + text_w + 8, label_y + text_h + 6)],
            fill=(255, 50, 50),
        )
        draw.text((label_x + 4, label_y + 2), label, fill=(255, 255, 255), font=font)

        return overlay_img
    except Exception as e:
        print(f"Bounding box error: {e}")
        return overlay_img


def create_heatmap_only(heatmap):
    try:
        heatmap_resized = np.array(
            Image.fromarray(np.uint8(heatmap * 255)).resize(IMG_SIZE)
        ) / 255.0
        colormap = matplotlib.colormaps["jet"]
        heatmap_colored = np.uint8(colormap(heatmap_resized)[:, :, :3] * 255)
        return Image.fromarray(heatmap_colored)
    except Exception as e:
        print(f"Heatmap-only error: {e}")
        return None


def image_to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --- XAI Explanation ---
def get_xai_explanation(predicted_class, confidence):
    explanations = {
        "Bacterial Red disease": {
            "affected_regions": "Body surface, skin, and fins - reddish discoloration and hemorrhagic patches",
            "reasoning": "The model detected red/inflamed patches on the fish body surface consistent with bacterial red disease caused by Yersinia ruckeri or similar pathogens.",
            "recommendation": "Isolate affected fish immediately. Treat with appropriate antibiotics (oxytetracycline). Improve water quality and reduce stocking density.",
        },
        "Bacterial diseases - Aeromoniasis": {
            "affected_regions": "Skin ulcers, fins, and internal organs - hemorrhagic lesions and fin erosion",
            "reasoning": "The model identified ulcerative lesions and hemorrhagic patterns typical of Aeromonas infections.",
            "recommendation": "Administer antibiotic treatment (florfenicol or enrofloxacin). Monitor water quality parameters. Disinfect tanks and equipment.",
        },
        "Bacterial gill disease": {
            "affected_regions": "Gill tissue - swollen, pale, or necrotic gill filaments",
            "reasoning": "The model focused on gill areas showing signs of bacterial colonization, swelling, and tissue degradation.",
            "recommendation": "Improve dissolved oxygen levels. Apply potassium permanganate or formalin bath treatment. Reduce organic load in water.",
        },
        "Fungal diseases - Saprolegniasis": {
            "affected_regions": "Skin and fins - white/grey cotton-like fungal growth patches",
            "reasoning": "The model detected cotton-like growths on the body surface characteristic of Saprolegnia fungal infection.",
            "recommendation": "Treat with malachite green or methylene blue baths. Remove dead/decaying material from tanks. Maintain optimal water temperature.",
        },
        "Healthy Fish": {
            "affected_regions": "No specific affected regions - the fish appears healthy",
            "reasoning": "The model found no significant pathological indicators. The fish shows normal coloration, intact fins, and healthy body condition.",
            "recommendation": "Continue regular monitoring and maintain good water quality. Follow standard feeding schedules and tank maintenance.",
        },
        "Parasitic diseases": {
            "affected_regions": "Skin, gills, and fins - visible parasites, irritation, or excessive mucus",
            "reasoning": "The model detected signs of parasitic infestation such as unusual spots, irritation marks, or abnormal mucus production.",
            "recommendation": "Identify the specific parasite type. Treat with appropriate anti-parasitic medication (formalin, copper sulfate, or praziquantel). Quarantine new fish.",
        },
        "Tail rot and Fin rot": {
            "affected_regions": "Tail and fin edges - progressive erosion, fraying, and discoloration",
            "reasoning": "The model identified deteriorating fin/tail edges with signs of bacterial necrosis and progressive tissue loss.",
            "recommendation": "Improve water quality immediately. Treat with antibacterial medications. Ensure proper nutrition to support tissue regeneration.",
        },
        "Viral diseases - Lymphocystis": {
            "affected_regions": "Skin and fins - white/grey nodular growths (cauliflower-like)",
            "reasoning": "The model detected characteristic nodular/wart-like growths on fins or skin caused by the Lymphocystis virus.",
            "recommendation": "No direct antiviral treatment exists. Maintain optimal water conditions. The disease is often self-limiting. Remove severely affected fish to prevent spread.",
        },
        "Viral diseases - VHS": {
            "affected_regions": "Body surface - hemorrhagic areas, darkened skin, and distended abdomen",
            "reasoning": "The model identified hemorrhagic patterns and body condition changes consistent with Viral Hemorrhagic Septicemia.",
            "recommendation": "Quarantine affected fish immediately. Report to authorities as VHS is a notifiable disease. There is no treatment; focus on biosecurity.",
        },
        "Viral diseases - White tail disease": {
            "affected_regions": "Tail region - distinctive white/opaque discoloration of the tail muscle",
            "reasoning": "The model focused on the tail area showing characteristic white/milky discoloration of muscle tissue.",
            "recommendation": "Isolate affected fish. Maintain optimal water quality. No specific treatment available; focus on prevention and biosecurity measures.",
        },
        "White spot disease - Ich": {
            "affected_regions": "Entire body, fins, and gills - small white spots (trophonts) visible on the surface",
            "reasoning": "The model detected the characteristic white spot pattern caused by the parasite Ichthyophthirius multifiliis.",
            "recommendation": "Raise water temperature gradually (to 28-30C). Treat with malachite green, formalin, or copper-based medications. Treat the entire tank.",
        },
    }

    info = explanations.get(predicted_class, {
        "affected_regions": "Unspecified regions",
        "reasoning": "Analysis based on visual pattern matching.",
        "recommendation": "Consult a fish health specialist for detailed diagnosis.",
    })

    if confidence >= 0.85:
        confidence_level = "Very High Confidence"
    elif confidence >= 0.70:
        confidence_level = "High Confidence"
    elif confidence >= 0.50:
        confidence_level = "Moderate Confidence"
    else:
        confidence_level = "Low Confidence"

    summary = (
        f"The AI detected '{predicted_class}' with {confidence_level.lower()} "
        f"({confidence * 100:.1f}%). The Grad-CAM heatmap shows the areas the model "
        f"focused on during analysis."
    )

    return {
        "summary": summary,
        "affected_regions": info["affected_regions"],
        "confidence_level": confidence_level,
        "reasoning": info["reasoning"],
        "recommendation": info["recommendation"],
    }


# --- Disease Messages ---
def get_disease_message(predicted_class, confidence):
    messages = {
        "Bacterial Red disease": f"Bacterial Red Disease detected ({confidence:.1%} confidence). This is a serious bacterial infection causing reddish discoloration. Immediate treatment with antibiotics is recommended.",
        "Bacterial diseases - Aeromoniasis": f"Aeromoniasis detected ({confidence:.1%} confidence). This bacterial infection can cause ulcers and hemorrhaging. Consult a fish health professional for antibiotic treatment.",
        "Bacterial gill disease": f"Bacterial Gill Disease detected ({confidence:.1%} confidence). The gills appear affected. Improve water quality and consider antimicrobial treatment.",
        "Fungal diseases - Saprolegniasis": f"Saprolegniasis (fungal infection) detected ({confidence:.1%} confidence). Cotton-like growths may be visible. Treat with antifungal agents.",
        "Healthy Fish": f"Your fish appears healthy ({confidence:.1%} confidence). No signs of disease detected. Continue with regular maintenance and monitoring.",
        "Parasitic diseases": f"Parasitic infection detected ({confidence:.1%} confidence). Identify the specific parasite for targeted treatment. Anti-parasitic medications may be needed.",
        "Tail rot and Fin rot": f"Tail/Fin Rot detected ({confidence:.1%} confidence). Progressive fin deterioration observed. Improve water quality and use antibacterial treatment.",
        "Viral diseases - Lymphocystis": f"Lymphocystis (viral) detected ({confidence:.1%} confidence). Nodular growths may be present. The disease is often self-limiting; maintain good water quality.",
        "Viral diseases - VHS": f"Viral Hemorrhagic Septicemia detected ({confidence:.1%} confidence). This is a serious notifiable disease. Quarantine immediately and contact authorities.",
        "Viral diseases - White tail disease": f"White Tail Disease (viral) detected ({confidence:.1%} confidence). White/opaque tail discoloration observed. Isolate affected fish immediately.",
        "White spot disease - Ich": f"White Spot Disease (Ich) detected ({confidence:.1%} confidence). Caused by Ichthyophthirius multifiliis. Raise water temperature and apply medication.",
    }
    return messages.get(
        predicted_class,
        f"Disease detected: {predicted_class} ({confidence:.1%} confidence). Please consult a fish health specialist.",
    )


# --- Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health():
    return {
        "status": "healthy" if model else "model_not_loaded",
        "model_loaded": model is not None,
        "num_classes": len(DISEASE_CLASSES),
        "version": "2.0.0",
        "model_file": os.path.basename(MODEL_PATH),
        "gradcam_layer": GRADCAM_LAYER,
        "image_size": IMG_SIZE,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        original_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess
        img = original_image.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])
        predicted_class = DISEASE_CLASSES[predicted_idx]

        # Grad-CAM
        gradcam_overlay_b64 = None
        gradcam_heatmap_b64 = None
        affected_area_pct = None

        heatmap = generate_gradcam(img_array, model)
        if heatmap is not None:
            overlay_img = create_heatmap_overlay(original_image, heatmap)
            if overlay_img:
                gradcam_overlay_b64 = image_to_base64(overlay_img)

            heatmap_img = create_heatmap_only(heatmap)
            if heatmap_img:
                gradcam_heatmap_b64 = image_to_base64(heatmap_img)

            # Calculate affected area percentage (pixels above 0.5 threshold)
            heatmap_full = np.array(
                Image.fromarray(np.uint8(heatmap * 255)).resize(IMG_SIZE)
            ) / 255.0
            affected_area_pct = round(float(np.mean(heatmap_full >= 0.5) * 100), 1)

        # XAI
        xai = get_xai_explanation(predicted_class, confidence)

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
            "gradcam_heatmap_overlay": gradcam_overlay_b64,
            "gradcam_heatmap_only": gradcam_heatmap_b64,
            "affected_area_percentage": affected_area_pct,
            "xai": xai,
            "is_fish": True,
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# --- Run ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
