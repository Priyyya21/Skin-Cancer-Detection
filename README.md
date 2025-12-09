# Skin Cancer Detection using CNN
This project focuses on detecting different types of skin cancer using a Convolutional Neural Network (CNN) model. It uses the HAM10000 dataset, which contains images of various pigmented skin lesions. The goal is to classify skin cancer types and assist in early detection.

# Live Web App:
https://skin-cancer-detection-priyavashishth.onrender.com

## Objective

To classify skin lesion images into multiple categories using Deep Learning.

To support early diagnosis and improve clinical decisions.

## Dataset

Dataset Used: HAM10000 – “Human Against Machine with 10000 training images”

Contains 7 types of skin lesions:

Melanocytic nevi (nv)

Melanoma (mel)

Benign keratosis-like lesions (bkl)

Basal cell carcinoma (bcc)

Actinic keratoses (akiec)

Dermatofibroma (df)

Vascular lesions (vasc)

Images were resized and normalized before training.

## Tech Stack & Libraries
Category	Tools / Libraries
Language	Python
Framework	TensorFlow / Keras
Other Libraries	NumPy, Pandas, Matplotlib, Scikit-learn, OpenCV
 Model Training

Model: CNN

Preprocessing: Resizing, Normalization, Train-Test Split

Evaluation Metrics:

Accuracy

Precision / Recall / F1-Score

Confusion Matrix

## Results
Metric	Value
Training Accuracy	~97%
Test Accuracy	~96%


## Deployment

A Flask application is used to upload an image and predict the skin lesion type.

User uploads an image → Model processes → Cancer type prediction displayed on UI.


 
# Clone the repository
git clone https://github.com/Priyyya21/Skin-Cancer-Detection

## Future Scope

✔ Improve accuracy using transfer learning (e.g., MobileNet, EfficientNet)
✔ Add Grad-CAM heatmaps for explainability
✔ Host the model with a more powerful GPU

## Disclaimer

This project is for educational and research purposes only. Not intended for real medical diagnosis.

## Author

Priya Sharma
BTech 3rd Year Student

 Email: priyaax21@gmail.com
 LinkedIn: https://www.linkedin.com/in/priya-vashishth-1790512b2/
 
