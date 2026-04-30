"""
predict.py — Inference module for Plant Disease Detection.

Loads the trained ResNet18 model and predicts the disease class
for a given plant leaf image. Returns top-3 predictions with
confidence scores.

Usage (standalone):
    python predict.py path/to/leaf_image.jpg

Usage (as module):
    from predict import PlantDiseasePredictor
    predictor = PlantDiseasePredictor("PlantVillage_model.pth")
    results = predictor.predict("leaf.jpg")
"""

import os
import sys
import json
import torch
import numpy as np
from torchvision import transforms, models
from torch import nn
from PIL import Image

# Default class names for the PlantVillage dataset (used as fallback
# when the checkpoint was saved without class_names metadata)
DEFAULT_CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

# Human-readable labels for display in the UI
DISPLAY_NAMES = {
    "Pepper__bell___Bacterial_spot": "Pepper Bell — Bacterial Spot",
    "Pepper__bell___healthy": "Pepper Bell — Healthy",
    "Potato___Early_blight": "Potato — Early Blight",
    "Potato___Late_blight": "Potato — Late Blight",
    "Potato___healthy": "Potato — Healthy",
    "Tomato_Bacterial_spot": "Tomato — Bacterial Spot",
    "Tomato_Early_blight": "Tomato — Early Blight",
    "Tomato_Late_blight": "Tomato — Late Blight",
    "Tomato_Leaf_Mold": "Tomato — Leaf Mold",
    "Tomato_Septoria_leaf_spot": "Tomato — Septoria Leaf Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato — Spider Mites",
    "Tomato__Target_Spot": "Tomato — Target Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato — Yellow Leaf Curl Virus",
    "Tomato__Tomato_mosaic_virus": "Tomato — Mosaic Virus",
    "Tomato_healthy": "Tomato — Healthy",
}

# Same transforms used during validation (must match train.py)
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Confidence threshold — if the top prediction is below this, reject as non-leaf
CONFIDENCE_THRESHOLD = 40.0  # percent


class PlantDiseasePredictor:
    """Loads a trained model and performs inference on leaf images."""

    def __init__(self, model_path, device=None):
        """
        Args:
            model_path: Path to the saved .pth checkpoint.
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.class_names = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path):
        """Load model from checkpoint, handling both old and new formats."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # New format: dict with model_state_dict + metadata
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
            num_classes = checkpoint.get("num_classes", len(class_names))
            state_dict = checkpoint["model_state_dict"]
        else:
            # Old format: bare state_dict
            class_names = DEFAULT_CLASS_NAMES
            num_classes = len(class_names)
            state_dict = checkpoint

        # Build the same architecture as train.py
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, num_classes),
        )
        model.load_state_dict(state_dict)
        model = model.to(self.device)

        return model, class_names

    @staticmethod
    def _is_likely_leaf(image, green_threshold=0.15):
        """
        Quick heuristic: check if the image has enough green content
        to plausibly be a plant leaf photo.

        Returns True if the image passes the green-dominance check.
        """
        img_array = np.array(image.resize((64, 64)))  # downscale for speed
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

        # Count pixels where green is the dominant channel
        green_dominant = (g > r) & (g > b)
        green_ratio = np.mean(green_dominant)

        return green_ratio >= green_threshold

    def predict(self, image_input, top_k=3):
        """
        Predict disease from an image.

        Args:
            image_input: File path (str) or PIL.Image.Image.
            top_k: Number of top predictions to return.

        Returns:
            dict with keys:
              - disease: str (human-readable top prediction)
              - disease_raw: str (raw class name)
              - confidence: float (0–100)
              - is_healthy: bool
              - is_leaf: bool (False if the image doesn't appear to be a plant leaf)
              - top3: list of {disease, disease_raw, confidence}
        """
        # Load and preprocess
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a file path or PIL Image")

        # --- Leaf validation: color heuristic ---
        if not self._is_likely_leaf(image):
            return {
                "is_leaf": False,
                "error": "This image does not appear to be a plant leaf. "
                         "Please upload a clear photo of a plant leaf.",
            }

        tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.class_names)))

        top_predictions = []
        for prob, idx in zip(top_probs, top_indices):
            raw_name = self.class_names[idx.item()]
            display_name = DISPLAY_NAMES.get(raw_name, raw_name.replace("_", " "))
            top_predictions.append({
                "disease": display_name,
                "disease_raw": raw_name,
                "confidence": round(prob.item() * 100, 2),
            })

        top_result = top_predictions[0]

        # --- Leaf validation: confidence threshold ---
        if top_result["confidence"] < CONFIDENCE_THRESHOLD:
            return {
                "is_leaf": False,
                "error": "The model could not confidently identify a plant disease in this image. "
                         "Please upload a clear, close-up photo of a plant leaf.",
            }

        is_healthy = "healthy" in top_result["disease_raw"].lower()

        return {
            "is_leaf": True,
            "disease": top_result["disease"],
            "disease_raw": top_result["disease_raw"],
            "confidence": top_result["confidence"],
            "is_healthy": is_healthy,
            "top3": top_predictions,
        }


# --- CLI usage ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [model_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "PlantVillage_model.pth"
    )

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run train.py first to train the model.")
        sys.exit(1)

    predictor = PlantDiseasePredictor(model_path)
    result = predictor.predict(image_path)

    print("\n🌿 Plant Disease Detection Result")
    print("=" * 40)
    print(f"  Prediction:  {result['disease']}")
    print(f"  Confidence:  {result['confidence']:.1f}%")
    print(f"  Status:      {'✅ Healthy' if result['is_healthy'] else '⚠️ Diseased'}")
    print()
    print("  Top 3 Predictions:")
    for i, pred in enumerate(result["top3"], 1):
        bar = "█" * int(pred["confidence"] / 5) + "░" * (20 - int(pred["confidence"] / 5))
        print(f"    {i}. {pred['disease']:<35} {bar} {pred['confidence']:.1f}%")
    print()
