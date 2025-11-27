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

app = Flask(__name__)

# ================================
# LOAD MODELS
# ================================
MODEL_DIR = "model"

# Path file models
brain_classifier_path = os.path.join(MODEL_DIR, "svm_baru_7015.pkl")  # Model 1: Brain vs Non-Brain
stroke_classifier_path = os.path.join(MODEL_DIR, "svm_rbf_7015.pkl")  # Model 2: Stroke Classification

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

# Load Both SVM Classifiers
brain_classifier = joblib.load(brain_classifier_path)
stroke_classifier = joblib.load(stroke_classifier_path)

# Class mappings
brain_classes = ["non_brain", "brain"]
stroke_classes = ["bleeding", "ischemia", "normal"]

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
    """Handle image upload and prediction with 2-stage classification"""
    temp_path = None
    
    try:
        if 'image' not in request.files:
            return render_template("error.html", error="No image uploaded"), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return render_template("error.html", error="No image selected"), 400
        
        img_bytes = file.read()
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif'
        }.get(file_ext, 'image/png')
        
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        timestamp = int(datetime.now().timestamp() * 1000000)
        temp_path = f"temp_image_{timestamp}.png"
        
        with open(temp_path, 'wb') as f:
            f.write(img_bytes)
        
        features = extract_feature(temp_path)
        
        # ===== STAGE 1: Brain vs Non-Brain Classification =====
        brain_pred = brain_classifier.predict(features)[0]
        is_brain = brain_classes[brain_pred]
    
        if is_brain == "non_brain":
            return render_template(
                "invalid_image.html",
                error="Gambar yang diunggah bukan MRI otak yang valid",
                img_data=img_base64,
                mime_type=mime_type,
            )
        
        # ===== STAGE 2: Stroke Classification (only if brain MRI) =====
        stroke_pred = stroke_classifier.predict(features)[0]
        result = stroke_classes[stroke_pred]
        
        return render_template(
            "result.html",
            result=result.upper(),
            img_data=img_base64,
            mime_type=mime_type,
        )
    
    except Exception as e:
        return render_template("error.html", error=str(e)), 500
    
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


# ================================
# RUN APP
# ================================
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)