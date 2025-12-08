from flask import Flask, render_template, request
import numpy as np
import cv2
import pickle

app = Flask(__name__)


model = pickle.load(open("cancer_detection_model.pkl", "rb"))


LABELS = {
    0: "Benign",
    1: "Malignant"
}



def preprocess_image(file_storage):
    """
    Steps:
    1. Read uploaded file bytes
    2. Decode to image with OpenCV (BGR)
    3. Resize to 28x28
    4. Convert BGR -> RGB (optional but good for Keras)
    5. Normalize to [0, 1]
    6. Reshape to (1, 28, 28, 3) for Keras CNN
    """

    # Read bytes
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)

    # Decode as color image
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # Resize to 28x28 (because your model expects 28x28x3)
    img = cv2.resize(img, (28, 28))

    # Convert BGR -> RGB (Keras usually uses RGB ordering)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    img = img.astype("float32") / 255.0

    # Reshape to (1, 28, 28, 3)
    img = np.expand_dims(img, axis=0)

    return img


# =====================================
# 3. Routes
# =====================================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
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
            error="Could not read the image. Please upload a valid image file."
        )

    try:
        # ==============================
        # Keras model prediction
        # ==============================
        # model.predict returns probabilities for each class
        preds = model.predict(img_input)  # shape: (1, n_classes)

        # Find predicted class index and confidence
        class_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0])) * 100.0

        # Map to human-readable label (or just show class index if not in LABELS)
        predicted_label = LABELS.get(class_idx, f"Class {class_idx}")

        return render_template(
            "index.html",
            prediction=predicted_label,
            confidence=f"{confidence:.2f}"
        )

    except Exception as e:
        print("ERROR DURING PREDICTION:", e)
        return render_template(
            "index.html",
            error=f"Prediction failed: {e}"
        )


# =====================================
# 4. Main
# =====================================
if __name__ == "__main__":
    app.run(debug=True)
