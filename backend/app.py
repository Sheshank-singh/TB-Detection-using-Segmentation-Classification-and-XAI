import io
import os
import traceback
import base64
from typing import Tuple

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from skimage import measure
from scipy.ndimage import binary_fill_holes
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from tensorflow.keras.applications.efficientnet import preprocess_input
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
SIGMOID_THRESHOLD = 0.5
CLASS_NAMES = ["Normal", "TB"]

# Email Configuration
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "your_email@example.com") # Replace with your sender email
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "your_email_password") # Replace with your sender email password or app-specific password
RECEIVER_EMAIL = "sheshanksingh2609@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# ------------------------------------------------------------
# APP INIT
# ------------------------------------------------------------
app = FastAPI(title="TB Classification with Segmentation + Grad-CAM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------
def pil_to_numpy(img_bytes: bytes) -> np.ndarray:
    """Convert uploaded bytes to RGB NumPy array."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)

def encode_png_base64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf).decode("utf-8")

# ------------------------------------------------------------
# MODEL LOADING (with automatic shape fix)
# ------------------------------------------------------------
# def load_tf_model(path: str):
#     """Load a TensorFlow model, handling 1→3 channel mismatch if needed."""
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Model file not found: {path}")

#     try:
#         # Normal loading first
#         return tf.keras.models.load_model(path, compile=False)

#     except ValueError as e:
#         if "stem_conv" in str(e) and "shape" in str(e):
#             print("⚠️ Detected shape mismatch (1 vs 3 channels). Rebuilding EfficientNetB0 for RGB input...")
#             from tensorflow.keras.applications import EfficientNetB0
#             base = EfficientNetB0(include_top=False, input_shape=(224, 224, 3),
#                                   weights=None, pooling="avg")
#             x = tf.keras.layers.Dense(1, activation="sigmoid")(base.output)
#             model = tf.keras.Model(base.input, x)
#             model.load_weights(path, by_name=True, skip_mismatch=True)
#             print("✅ Loaded EfficientNetB0 (RGB) with partial weights.")
#             return model
#         raise RuntimeError(f"Failed loading model {path}: {e}")

# def load_tf_model(path: str):
#     """Load a TensorFlow model strictly (no partial weights)."""
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Model file not found: {path}")
#     try:
#         model = tf.keras.models.load_model(path, compile=False)
#         print(f"✅ Loaded model: {path}")
#         return model
#     except Exception as e:
#         print(f"❌ Failed to load model fully: {path}")
#         raise RuntimeError(f"Model loading error: {e}")

def load_tf_model(path: str):
    """Force-load a TensorFlow model exactly as saved (no architecture rebuild)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        # Try loading the full model directly (architecture + weights)
        model = tf.keras.models.load_model(path, compile=False)
        print(f"✅ Loaded model fully from {path}")
        print(f"   Input shape: {model.input_shape}")
        
        return model

    except Exception as e:
        raise RuntimeError(f"Model loading error: {e}")


# ------------------------------------------------------------
# POSTPROCESS MASK
# ------------------------------------------------------------
def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """Apply connected component cleaning, morphology, and hole filling."""
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape: {mask.shape}")

    binary = (mask > 0.5).astype(np.uint8)
    labels = measure.label(binary, connectivity=2)
    props = sorted(measure.regionprops(labels), key=lambda x: x.area, reverse=True)

    cleaned = np.zeros_like(binary)
    for i in range(min(2, len(props))):
        cleaned[labels == props[i].label] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    filled = binary_fill_holes(opened).astype(np.uint8)
    return filled

# ------------------------------------------------------------
# GRAD-CAM UTILITIES
# ------------------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Generate Grad-CAM heatmap for one preprocessed input image."""
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0]).numpy().item()
        loss = predictions[0][:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_heatmap_on_image(image_rgb, heatmap, alpha=0.4):
    """Overlay heatmap on RGB image (returns BGR for encoding)."""
    heatmap_resized = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    heat_uint8 = np.uint8(255 * heatmap_resized)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                              1 - alpha, heat_color, alpha, 0)
    return blended

# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------
try:
    seg_model = load_tf_model("models/segmentation_model.h5")
    cls_model = load_tf_model("models/best_efficientnetb0_stage2_rgb_fixed.h5")

    # Detect last conv layer dynamically
    conv_layers = [l.name for l in cls_model.layers if "conv" in l.name]
    last_conv_name = conv_layers[-1] if conv_layers else None

    seg_input_shape = seg_model.input_shape[1:3]
    cls_input_shape = cls_model.input_shape[1:3] + (3,)

    print(f"✅ Models loaded | seg: {seg_input_shape} | cls: {cls_input_shape} | last conv: {last_conv_name}")
    print("Classification model input shape:", cls_model.input_shape)

except Exception as e:
    print("❌ Error loading models:", e)
    raise

# ------------------------------------------------------------
# PREDICTION ENDPOINT
# ------------------------------------------------------------
@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    """Run segmentation → masking → classification → Grad-CAM."""
    try:
        # --- Read and preprocess input ---
        raw = await image.read()
        orig_rgb = pil_to_numpy(raw)
        h, w = orig_rgb.shape[:2]

        # Save original image for debugging
        cv2.imwrite("original_image.png", cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR))

        # --- SEGMENTATION ---
        seg_in = cv2.resize(orig_rgb, (seg_input_shape[1], seg_input_shape[0]))
        seg_in = np.expand_dims(seg_in[..., 0] / 255.0, axis=(0, -1))  # grayscale input
        mask_pred = seg_model.predict(seg_in, verbose=0)[0]
        mask = postprocess_mask(mask_pred)
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # --- SEGMENTED IMAGE ---
        segmented = orig_rgb * np.expand_dims(mask_resized, axis=-1)

        # --- CLASSIFICATION ---
        cls_img = cv2.resize(segmented, (cls_input_shape[1], cls_input_shape[0]))
        # Save preprocessed image for classification for debugging
        cv2.imwrite("preprocessed_classification_image.png", cv2.cvtColor(cls_img, cv2.COLOR_RGB2BGR))
        cls_x = preprocess_input(cls_img.astype(np.float32))
        cls_x = np.expand_dims(cls_x, axis=0)

        #         # --- CLASSIFICATION (Grayscale input) ---
        # cls_img = cv2.resize(segmented, (224, 224))

        # # Convert to grayscale because model expects 1-channel input
        # cls_gray = cv2.cvtColor(cls_img, cv2.COLOR_RGB2GRAY)

        # # Add channel dimension -> (224, 224, 1)
        # cls_gray = np.expand_dims(cls_gray, axis=-1)

        # # Add batch dimension -> (1, 224, 224, 1)
        # cls_x = np.expand_dims(cls_gray, axis=0)

        # # Normalize to 0-1
        # cls_x = cls_x.astype(np.float32) / 255.0

        preds = cls_model.predict(cls_x, verbose=0)
        print(f"Raw Predictions: {preds}") # Added for debugging
        prob = float(preds[0][0])
        label_idx = 1 if prob > SIGMOID_THRESHOLD else 0
        label = CLASS_NAMES[label_idx]
        print(f"Prediction Probability: {prob}, Label: {label}")

        # --- GRAD-CAM ---
        heatmap = make_gradcam_heatmap(cls_x, cls_model, last_conv_name)
        gradcam_overlay = overlay_heatmap_on_image(segmented, heatmap)

        # --- Encode outputs for frontend ---
        mask_3ch = cv2.cvtColor(mask_resized * 255, cv2.COLOR_GRAY2BGR)
        result = {
            "classification": {"label": label, "confidence": prob},
            "original_image": encode_png_base64(cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)),
            "segmentation_mask": encode_png_base64(mask_3ch),
            "segmented_lungs": encode_png_base64(cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)),
            "gradcam_on_segmented": encode_png_base64(gradcam_overlay),
        }
        return JSONResponse(result)

    except Exception as e:
        tb = traceback.format_exc()
        print("❌ Exception in /api/predict:", tb)
        return JSONResponse({"error": str(e), "trace": tb}, status_code=500)

# ------------------------------------------------------------
# MAIN ENTRY
# ------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
