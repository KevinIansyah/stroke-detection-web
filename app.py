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
import base64
import tempfile

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
    """Disable caching for all responses"""
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
    temp_path = None  # Initialize untuk cleanup
    
    try:
        # Check if file exists
        if 'image' not in request.files:
            return render_template("error.html", error="No image uploaded"), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return render_template("error.html", error="No image selected"), 400
        
        # Baca file langsung ke memory
        img_bytes = file.read()
        
        # Deteksi format image untuk base64
        file_ext = os.path.splitext(file.filename)[1].lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif'
        }.get(file_ext, 'image/png')  # Default ke png
        
        # Convert ke base64 untuk display
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Create unique temporary file dengan timestamp
        timestamp = int(datetime.now().timestamp() * 1000000)  # microseconds
        temp_path = f"temp_image_{timestamp}.png"
        
        # Save temporary untuk feature extraction
        with open(temp_path, 'wb') as f:
            f.write(img_bytes)
        
        # Extract features
        features = extract_feature(temp_path)
        
        # Predict class
        pred = svm_model.predict(features)[0]
        result = class_names[pred]
        
        return render_template(
            "result.html",
            result=result.upper(),
            img_data=img_base64,
            mime_type=mime_type
        )
    
    except Exception as e:
        return render_template("error.html", error=str(e)), 500
    
    finally:
        # Always cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass  # Ignore cleanup errors


# ================================
# RUN APP
# ================================
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)