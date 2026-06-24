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
import base64
import matplotlib
matplotlib.use('Agg')

# Custom layer definitions for the ATF-Net fusion model. Importing this module
# registers the 8 custom Keras layers so keras.models.load_model can rebuild
# the architecture from the .h5 file (a .h5 stores weights + config but NOT the
# Python logic of custom layers — the class definitions must be present).
import atf_layers

# Limit TensorFlow memory growth
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# --- Configuration ---
# ATF-Net = Attention Cross Fusion model: Custom CNN + ResNet50 + ViT-Tiny
# fused with triple cross-attention. Input 224x224, 7 disease classes.
# Inference-only model (optimizer state stripped: 434MB -> 146MB, identical
# predictions). Re-generate from the training export with:
#   m = keras.models.load_model("ATF_Net_Fusion_Model.h5",
#           custom_objects=atf_layers.ATF_CUSTOM_OBJECTS, compile=False)
#   m.save("ATF_Net_Fusion_Model_inference.h5")
MODEL_PATH = "./ATF_Net_Fusion_Model_inference.h5"
IMG_SIZE = (224, 224)
# Grad-CAM is computed on the last conv block of the ResNet50 branch (the only
# branch with a 2D spatial feature map; the ViT and the fused head are not
# spatial). 7x7x2048 feature map at 224px input.
GRADCAM_LAYER = "conv5_block3_out"

# The 7 classes of the "Freshwater Fish Disease (Aquaculture in South Asia)"
# dataset, in alphabetical folder order (= the model's output index order).
DISEASE_CLASSES = [
    "Bacterial Red disease",
    "Bacterial diseases - Aeromoniasis",
    "Bacterial gill disease",
    "Fungal diseases - Saprolegniasis",
    "Healthy Fish",
    "Parasitic diseases",
    "Viral diseases - White tail disease",
]

# --- App Setup ---
app = FastAPI(
    title="Fish Disease Detection API",
    description="Explainable AI Fish Disease Detection — ATF-Net fusion model (CNN + ResNet50 + ViT) with Grad-CAM",
    version="3.0.0",
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
# Cached sub-models / layers used for Grad-CAM (built once at model load).
_gradcam = None


def _build_gradcam_components(mdl):
    """Build the pieces needed for Grad-CAM on the nested ResNet50 branch.

    The conv feature map lives *inside* the `feat_ResNet50` sub-model, so a
    single keras.Model spanning the outer input to the nested conv tensor is
    "graph disconnected". Instead we split the ResNet branch at the conv layer
    and re-run the exact fusion head manually inside a GradientTape. All layers
    are reused from the loaded model, so weights are shared (no extra memory).
    """
    import keras
    resnet = mdl.get_layer("feat_ResNet50")
    conv = resnet.get_layer(GRADCAM_LAYER)
    return {
        "resnet_to_conv": keras.Model(resnet.input, conv.output),
        "resnet_from_conv": keras.Model(conv.output, resnet.output),
        "cnn": mdl.get_layer("feat_Custom_CNN"),
        "vit": mdl.get_layer("feat_ViT_B16_Tiny"),
        "fp0": mdl.get_layer("feature_projection"),    # <- Custom CNN
        "fp1": mdl.get_layer("feature_projection_1"),  # <- ResNet50
        "fp2": mdl.get_layer("feature_projection_2"),  # <- ViT
        "concat": mdl.get_layer("concat_layer"),
        "tca": mdl.get_layer("triple_cross_attention"),
        "dense_88": mdl.get_layer("dense_88"),
        "add": mdl.get_layer("add_21"),
        "head": [mdl.get_layer(n) for n in [
            "dense_89", "batch_normalization_661", "dropout_129",
            "dense_90", "batch_normalization_662", "dropout_130",
            "dense_91", "batch_normalization_663", "dropout_131",
            "dense_92",
        ]],
    }


def load_model():
    global model, _gradcam
    if model is not None:
        return model
    try:
        import keras
        model = keras.models.load_model(
            MODEL_PATH,
            custom_objects=atf_layers.ATF_CUSTOM_OBJECTS,
            compile=False,
        )
        _gradcam = _build_gradcam_components(model)
        print(f"Model loaded: {MODEL_PATH}")
        print(f"   Input shape : {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Classes     : {len(DISEASE_CLASSES)}")
        gc.collect()
        return model
    except Exception as e:
        print(f"Model load error: {e}")
        model = None
        _gradcam = None
        return None


# --- Grad-CAM ---
def _fusion_forward(x, gc_parts, tape=None):
    """Re-run the exact fusion forward pass, exposing the ResNet conv map.

    Verified to reproduce model.predict() exactly (max abs diff = 0.0)."""
    conv_map = gc_parts["resnet_to_conv"](x, training=False)
    if tape is not None:
        tape.watch(conv_map)
    rvec = gc_parts["resnet_from_conv"](conv_map, training=False)
    cvec = gc_parts["cnn"](x, training=False)
    vvec = gc_parts["vit"](x, training=False)
    p0 = gc_parts["fp0"](cvec, training=False)
    p1 = gc_parts["fp1"](rvec, training=False)
    p2 = gc_parts["fp2"](vvec, training=False)
    fused = gc_parts["add"]([
        gc_parts["tca"]([p0, p1, p2], training=False),
        gc_parts["dense_88"](gc_parts["concat"]([p0, p1, p2])),
    ])
    h = fused
    for layer in gc_parts["head"]:
        h = layer(h, training=False)
    return conv_map, h


def generate_gradcam(img_array):
    """Grad-CAM on the ResNet50 conv5_block3_out feature map of the fusion model."""
    if _gradcam is None:
        return None
    try:
        x = tf.convert_to_tensor(img_array, dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_map, preds = _fusion_forward(x, _gradcam, tape)
            preds = tf.cast(preds, tf.float32)
            predicted_class = tf.argmax(preds[0])
            class_output = preds[:, predicted_class]

        grads = tape.gradient(class_output, conv_map)
        if grads is None:
            return None
        grads = tf.cast(grads, tf.float32)
        conv_map = tf.cast(conv_map, tf.float32)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        heatmap = conv_map[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None
    finally:
        gc.collect()


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
        "Viral diseases - White tail disease": {
            "affected_regions": "Tail region - distinctive white/opaque discoloration of the tail muscle",
            "reasoning": "The model focused on the tail area showing characteristic white/milky discoloration of muscle tissue.",
            "recommendation": "Isolate affected fish. Maintain optimal water quality. No specific treatment available; focus on prevention and biosecurity measures.",
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
        "Viral diseases - White tail disease": f"White Tail Disease (viral) detected ({confidence:.1%} confidence). White/opaque tail discoloration observed. Isolate affected fish immediately.",
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
        "status": "healthy" if model else "model_not_loaded_yet",
        "model_loaded": model is not None,
        "num_classes": len(DISEASE_CLASSES),
        "version": "3.0.0",
        "model_file": os.path.basename(MODEL_PATH),
        "model_architecture": "ATF-Net (Custom CNN + ResNet50 + ViT-Tiny, triple cross-attention fusion)",
        "gradcam_layer": GRADCAM_LAYER,
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
        original_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess: resize to 224x224 and scale to [0, 1]
        img = original_image.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype("float32")

        # Predict
        predictions = mdl.predict(img_array, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])
        predicted_class = DISEASE_CLASSES[predicted_idx]

        # Grad-CAM
        gradcam_overlay_b64 = None
        gradcam_heatmap_b64 = None
        affected_area_pct = None

        heatmap = generate_gradcam(img_array)
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
    finally:
        gc.collect()


# --- Run ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
