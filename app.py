import os
import pickle

import cv2
import numpy as np
from flask import Flask, render_template, request

# =====================================
# 1. App & model setup
# =====================================

app = Flask(__name__)

# Get absolute path to this folder (important for Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cancer_detection_model.pkl")

# Load the model ONCE at startup (global)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Class labels
LABELS = {
    0: "Benign",
    1: "Malignant",
}


# =====================================
# 2. Image preprocessing
# =====================================

def preprocess_image(file_storage):
    """
    Steps:
    1. Read uploaded file bytes from Flask's FileStorage
    2. Decode to image with OpenCV (BGR)
    3. Resize to 28x28
    4. Convert BGR -> RGB
    5. Normalize to [0, 1]
    6. Reshape to (1, 28, 28, 3) for a CNN-type model
    """

    # Make sure we read from the beginning of the stream
    file_storage.stream.seek(0)

    # Read bytes
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)

    # Decode as color image
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # Resize to match model's expected input size
    img = cv2.resize(img, (28, 28))

    # Convert BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    img = img.astype("float32") / 255.0

    # Reshape to (1, 28, 28, 3)
    img = np.expand_dims(img, axis=0)

    return img



@app.route("/", methods=["GET"])
def index():
    # index.html should have a form with:
    # <input type="file" name="image">
    # and method="POST" action="/predict"
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Check file exists in request
    if "image" not in request.files:
        return render_template("index.html", error="Please upload an image file.")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    # Preprocess image
    img_input = preprocess_image(file)
    if img_input is None:
        return render_template(
            "index.html",
            error="Could not read the image. Please upload a valid image file.",
        )

    try:
        # NOTE:
        # This assumes your pickled model has a .predict() method
        # that accepts an array shaped like (1, 28, 28, 3)
        # (for Keras CNN or similar).
        # If the model is a scikit-learn model expecting flattened input,
        # you’ll need to reshape: img_input.reshape(1, -1)

        preds = model.predict(img_input)  # shape: (1, n_classes) or (1,)

        # If model outputs probabilities (e.g., Keras softmax)
        if preds.ndim == 2 and preds.shape[1] > 1:
            class_idx = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]) * 100.0)
        else:
            # Binary model that returns single probability or class
            # Try to interpret it as probability for class 1
            prob = float(preds[0] if np.ndim(preds) == 1 else preds[0][0])
            # Threshold at 0.5
            class_idx = 1 if prob >= 0.5 else 0
            confidence = prob * 100.0 if class_idx == 1 else (100.0 - prob * 100.0)

        predicted_label = LABELS.get(class_idx, f"Class {class_idx}")

        return render_template(
            "index.html",
            prediction=predicted_label,
            confidence=f"{confidence:.2f}",
        )

    except Exception as e:
        # This will show error on the page instead of crashing worker
        print("ERROR DURING PREDICTION:", e)
        return render_template(
            "index.html",
            error=f"Prediction failed: {e}",
        )




if __name__ == "__main__":
    # For local debugging only
    app.run(host="0.0.0.0", port=5000, debug=True)
