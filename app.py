"""
app.py — Flask backend for Plant Disease Detection.

Serves the frontend and exposes a /predict API endpoint
that accepts an uploaded leaf image and returns disease
predictions with confidence scores.

Usage:
    python app.py
    → http://localhost:5000
"""

import os
import io
import traceback
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from predict import PlantDiseasePredictor, DISPLAY_NAMES, DEFAULT_CLASS_NAMES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "PlantVillage_model.pth")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=BASE_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Load the model once at startup (lazy — only if the model file exists)
predictor = None


def get_predictor():
    """Lazy-load the predictor so the server can start even without a model."""
    global predictor
    if predictor is None:
        if not os.path.exists(MODEL_PATH):
            return None
        predictor = PlantDiseasePredictor(MODEL_PATH)
    return predictor


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes — Static frontend
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/styles.css")
def styles():
    return send_from_directory(BASE_DIR, "styles.css")


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a multipart image upload and return disease predictions.

    Request:  POST /predict  with form-data field 'image'
    Response: JSON
        {
            "success": true,
            "disease": "Tomato — Early Blight",
            "disease_raw": "Tomato_Early_blight",
            "confidence": 94.32,
            "is_healthy": false,
            "top3": [
                {"disease": "...", "disease_raw": "...", "confidence": 94.32},
                ...
            ]
        }
    """
    # Check if model is available
    pred = get_predictor()
    if pred is None:
        return jsonify({
            "success": False,
            "error": "Model not found. Please run train.py first to train the model.",
        }), 503

    # Validate the upload
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file provided. Please upload an image with the field name 'image'.",
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "No file selected.",
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"File type not allowed. Accepted: {', '.join(ALLOWED_EXTENSIONS)}",
        }), 400

    # Read image and predict
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = pred.predict(image)

        # Check if the image was rejected as non-leaf
        if not result.get("is_leaf", True):
            return jsonify({
                "success": False,
                "error": result.get("error", "This does not appear to be a plant leaf image."),
            }), 400

        result["success"] = True
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(e)}",
        }), 500


@app.route("/api/classes", methods=["GET"])
def get_classes():
    """Return the list of all disease classes the model can detect."""
    classes = []
    for raw_name in DEFAULT_CLASS_NAMES:
        display_name = DISPLAY_NAMES.get(raw_name, raw_name.replace("_", " "))
        is_healthy = "healthy" in raw_name.lower()
        # Extract plant type from the class name
        plant = raw_name.split("_")[0].replace("_", " ")
        classes.append({
            "raw": raw_name,
            "display": display_name,
            "plant": plant,
            "is_healthy": is_healthy,
        })
    return jsonify({"classes": classes})


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    model_loaded = predictor is not None
    model_exists = os.path.exists(MODEL_PATH)
    return jsonify({
        "status": "ok",
        "model_exists": model_exists,
        "model_loaded": model_loaded,
    })


# ---------------------------------------------------------------------------
# CORS — allow local development from different ports
# ---------------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🌿 PlantDetect Server")
    print("=" * 40)
    print(f"  Model path:  {MODEL_PATH}")
    print(f"  Model found: {'✅ Yes' if os.path.exists(MODEL_PATH) else '❌ No — run train.py first'}")
    print(f"  Server:      http://localhost:5000")
    print("=" * 40 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
