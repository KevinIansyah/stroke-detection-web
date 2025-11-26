import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras import Model # type: ignore
import joblib
from tensorflow.keras.utils import load_img, img_to_array # type: ignore
from datetime import datetime

app = Flask(__name__)

# ================================
# LOAD MODEL
# ================================
MODEL_DIR = "model"

# Path file model
svm_model_path = os.path.join(MODEL_DIR, "svm_rbf_7015.pkl")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

base_model.trainable = False

feature_extractor = Model(
    inputs=base_model.input,
    outputs=base_model.output,
    name='mobilenet_feature_extractor'
)

# Load SVM Classifier
svm_model = joblib.load(svm_model_path)

# Class mapping
class_names = ["bleeding", "ischemia", "normal"]

# Image size
IMG_SIZE = (224, 224)


# ================================
# FEATURE EXTRACTION FUNCTION
# ================================
def extract_feature(img_path):
    """Extract features from image using MobileNetV2"""
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    features = feature_extractor.predict(img_array, verbose=0)
    return features


# ================================
# CACHE CONTROL
# ================================
@app.after_request
def add_header(response):
    """Disable caching for static files"""
    if 'static' in request.path or '/static/' in request.path:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


# ================================
# ROUTES
# ================================
@app.route("/")
def home():
    """Render homepage"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file exists
        if 'image' not in request.files:
            return render_template("error.html", error="No image uploaded"), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return render_template("error.html", error="No image selected"), 400
        
        # Create static folder if not exists
        os.makedirs("static", exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = int(datetime.now().timestamp() * 1000)  # milliseconds for uniqueness
        file_ext = os.path.splitext(file.filename)[1] or '.png'
        filename = f"uploaded_image_{timestamp}{file_ext}"
        save_path = os.path.join("static", filename)
        
        # Save uploaded file
        file.save(save_path)
        
        # Extract features
        features = extract_feature(save_path)
        
        # Predict class
        pred = svm_model.predict(features)[0]
        result = class_names[pred]
        
        return render_template(
            "result.html",
            result=result.upper(),
            img_path=filename,  # Pass only filename, not full path
            timestamp=timestamp  # Pass timestamp for additional cache busting
        )
    
    except Exception as e:
        return render_template("error.html", error=str(e)), 500


# ================================
# RUN APP
# ================================
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)