"""
Flask Web Application for Interactive Handwritten Digit Recognition.
Integrates trained PyTorch model with REST API endpoints and web interface.
"""

import os
import sys
import logging
from typing import Dict, Any, List
from flask import Flask, request, jsonify, render_template
import torch

from model.cnn import load_trained_model, make_model, resolve_model_weights_path
from utils.preprocessing import preprocess_image, EmptyCanvasError, PreprocessingError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# Global model state
MODEL = None
DEVICE = "cpu"
ACTIVATION_NAME = "ReLU"
WEIGHTS_PATH = None


def init_model():
    """
    Loads the trained model once during application initialization.
    """
    global MODEL, WEIGHTS_PATH
    try:
        WEIGHTS_PATH = resolve_model_weights_path()
        logger.info(f"Loading trained weights from: {WEIGHTS_PATH}")
        MODEL = load_trained_model(
            weights_path=WEIGHTS_PATH,
            activation_name=ACTIVATION_NAME,
            device=DEVICE
        )
        logger.info(f"Model loaded and set to eval() successfully.")
        logger.info(f"Architecture:\n{MODEL}")
    except Exception as e:
        logger.error(f"FATAL: Failed to load model weights: {e}", exc_info=True)
        raise RuntimeError(f"Model initialization failed: {e}")


# Initialize model on startup
init_model()


@app.route("/", methods=["GET"])
def index():
    """
    Serves the primary web interface.
    """
    return render_template("index.html")


@app.route("/info", methods=["GET"])
def model_info():
    """
    Returns diagnostic information about the loaded model and environment.
    """
    global MODEL, WEIGHTS_PATH, ACTIVATION_NAME, DEVICE
    total_params = sum(p.numel() for p in MODEL.parameters())
    trainable_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)

    layers_info = []
    for idx, layer in enumerate(MODEL):
        layer_dict = {"index": idx, "name": layer.__class__.__name__}
        if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
            layer_dict["in_features"] = layer.in_features
            layer_dict["out_features"] = layer.out_features
        layers_info.append(layer_dict)

    return jsonify({
        "status": "ready",
        "model_type": "Fully Connected Neural Network (MLP)",
        "weights_path": os.path.basename(WEIGHTS_PATH) if WEIGHTS_PATH else None,
        "activation": ACTIVATION_NAME,
        "device": DEVICE,
        "total_parameters": total_params,
        "layers": layers_info
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Receives canvas drawing image data, preprocesses it, runs inference,
    and returns predicted digit along with all 10 class probabilities.
    """
    global MODEL

    if MODEL is None:
        return jsonify({
            "success": False,
            "error": "Model is not initialized on the server."
        }), 500

    # 1. Parse JSON payload
    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "success": False,
            "error": "Invalid request. Expected JSON body with 'image' field."
        }), 400

    image_data = data.get("image")
    if not image_data or not isinstance(image_data, str):
        return jsonify({
            "success": False,
            "error": "No image data provided. Please draw a digit first."
        }), 400

    # 2. Preprocessing
    try:
        tensor, debug_info = preprocess_image(image_data)
    except EmptyCanvasError as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "is_empty": True
        }), 400
    except PreprocessingError as e:
        return jsonify({
            "success": False,
            "error": f"Image processing failed: {str(e)}"
        }), 400
    except Exception as e:
        logger.error(f"Unexpected preprocessing error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Unable to process canvas drawing: {str(e)}"
        }), 500

    # 3. Model Inference
    try:
        with torch.no_grad():
            tensor = tensor.to(DEVICE)
            logits = MODEL(tensor)  # Shape: [1, 10]
            probabilities = torch.softmax(logits, dim=1).squeeze(0)  # Shape: [10]

        predicted_class = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_class].item())

        # Construct detailed probability breakdown
        probs_dict: Dict[str, float] = {}
        probs_list: List[Dict[str, Any]] = []

        for digit_idx in range(10):
            prob_val = float(probabilities[digit_idx].item())
            probs_dict[str(digit_idx)] = round(prob_val, 6)
            probs_list.append({
                "digit": digit_idx,
                "probability": prob_val,
                "percentage": f"{prob_val * 100:.2f}%"
            })

        return jsonify({
            "success": True,
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2),
            "probabilities": probs_dict,
            "probabilities_list": probs_list,
            "thumbnail": debug_info.get("thumbnail"),
            "message": f"Digit successfully recognized as {predicted_class}."
        })

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Model inference error: {str(e)}"
        }), 500


if __name__ == "__main__":
    # Start local Flask development server
    port = int(os.environ.get("PORT", 5000))
    print(f"\n* Starting Handwritten Digit Recognition Web Application on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

