"""
app.py - Streamlit web application for Crop Disease Detection.

Run with:
    streamlit run app.py

Pages (sidebar navigation):
    1. Home           - project introduction
    2. Disease Detection - upload a leaf image and get a prediction
    3. About          - project details
"""

import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from utils.preprocessing import (is_supported_file,
                                 preprocess_uploaded_image)

# ------------------------- Paths -------------------------
MODEL_PATH = os.path.join("models", "crop_disease_model.keras")
CLASS_NAMES_PATH = os.path.join("models", "class_names.json")
DISEASE_INFO_PATH = "disease_info.json"
DISEASE_INFO_DEFAULT = "disease_info_default.json"


# ------------------------- Cached model loading -------------------------
@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    """
    Load the trained CNN model once and reuse it on every request.
    The model is NOT retrained when the app starts - only loaded.
    Returns None if the model file is missing.
    """
    if not os.path.isfile(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_class_names():
    """Load the class names saved during training. Returns [] if missing."""
    if not os.path.isfile(CLASS_NAMES_PATH):
        return []
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_disease_info():
    """Load the disease information dictionary used in the result panel."""
    for path in (DISEASE_INFO_PATH, DISEASE_INFO_DEFAULT):
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return {}


# ------------------------- Page helpers -------------------------
def apply_ui_style():
    """Small amount of CSS to give the app a clean, agriculture-themed look."""
    st.markdown(
        """
        <style>
        .main-heading {
            background: linear-gradient(90deg, #1b5e20, #2e7d32);
            color: white;
            padding: 1.2rem 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
        }
        .main-heading h1 { margin: 0; font-size: 1.9rem; }
        .main-heading p  { margin: 0.3rem 0 0 0; opacity: 0.9; }
        .result-box {
            background: #f1f8e9;
            border-left: 6px solid #2e7d32;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-top: 1rem;
            color: #1b5e20;
        }
        .result-box h3 { margin: 0 0 0.4rem 0; color: #1b5e20; }
        .info-box {
            background: #eceff1;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-top: 1rem;
            color: #263238;
        }
        .info-box h3 { color: #37474f; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_header(title, subtitle):
    st.markdown(
        f'<div class="main-heading"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def show_missing_model_message():
    """Friendly message shown when the trained model files do not exist."""
    st.error("The trained model was not found. 😔")
    st.markdown(
        """
        **To fix this:**

        1. Place your dataset inside the `dataset/` folder, e.g.:
           ```
           dataset/
               Crop___Early_blight/
                   image1.jpg
               Crop___healthy/
                   image1.jpg
           ```
        2. Train the model by running:
           ```
           python train.py
           ```
        3. Restart this app.

        The training script will automatically create:
        - `models/crop_disease_model.keras`
        - `models/class_names.json`
        """
    )


# ------------------------- Pages -------------------------
def home_page():
    show_header("AI-Based Crop Disease Detection", "Deep Learning + CNN + Streamlit")

    st.markdown(
        """
        Agriculture is the backbone of many economies, and plant diseases cause
        massive crop losses every year. This project uses **Convolutional Neural
        Networks (CNN)** to identify crop diseases from a simple photo of a leaf.

        ### How it works
        - **Step 1:** A user uploads a photo of a plant leaf.
        - **Step 2:** The image is resized and prepared for the CNN.
        - **Step 3:** A trained deep learning model analyzes the image.
        - **Step 4:** The app shows the predicted disease, the confidence
          percentage and simple prevention advice.

        ### Why it matters
        - Diseases are detected early, before they spread to the whole crop.
        - Farmers do not need expert knowledge to get a first diagnosis.
        - It is fast, free and runs entirely on a regular computer.

        👉 **Go to the "Disease Detection" page in the sidebar and try it!**
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Approach", "CNN / Deep Learning")
    col2.metric("Input", "Leaf photo (JPG/PNG)")
    col3.metric("Output", "Disease + confidence")


def detection_page():
    show_header("Disease Detection", "Upload a leaf image and let the AI diagnose it")

    model = load_model()
    class_names = load_class_names()

    # --- Guard conditions with friendly errors ---
    if model is None:
        show_missing_model_message()
        return
    if not class_names:
        st.error("`models/class_names.json` is missing. Please run `python train.py` again.")
        return

    uploaded_file = st.file_uploader(
        "Upload a crop leaf image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
    )

    if uploaded_file is None:
        st.info("👆 Upload an image to get started.")
        return

    # --- Validate the file extension ---
    if not is_supported_file(uploaded_file.name):
        st.error(
            f"Unsupported file format: **{uploaded_file.name}**. "
            "Please upload a JPG, JPEG or PNG image."
        )
        return

    # --- Try to decode the image (catches corrupted / fake files) ---
    try:
        display_image = Image.open(uploaded_file).convert("RGB")
        display_image.verify()  # quick integrity check
    except Exception as err:
        st.error(f"Could not read the image file: **{uploaded_file.name}**. Please try another image.")
        return
    # Need to reopen after verify() since it clears the file state
    uploaded_file.seek(0)

    # --- Show the uploaded image ---
    st.image(uploaded_file, caption="Your uploaded leaf image", use_column_width="always")

    if st.button("🔍 Predict Disease", type="primary"):
        with st.spinner("Analyzing the image... this may take a few seconds..."):
            try:
                # --- Preprocess: same resize (128x128) as training ---
                image_batch = preprocess_uploaded_image(uploaded_file)

                # --- Predict ---
                probabilities = model.predict(image_batch, verbose=0)[0]

            except Exception as err:
                st.error(f"Prediction failed: {err}")
                return

        # --- Build the result ---
        best_index = int(np.argmax(probabilities))
        disease_name = class_names[best_index]
        confidence = float(probabilities[best_index])

        # Healthy/Diseased status: a class is treated as healthy if
        # its folder name contains the word "healthy".
        is_healthy = "healthy" in disease_name.lower()
        status_label = "🌿 HEALTHY" if is_healthy else "⚠️ DISEASED"

        st.markdown(
            f"""
            <div class="result-box">
                <h3>Prediction Result</h3>
                <b>Predicted class:</b> {disease_name}<br>
                <b>Confidence:</b> {confidence:.1%}<br>
                <b>Status:</b> {status_label}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(min(max(confidence, 0.0), 1.0))

        # --- Disease information from disease_info.json ---
        info = load_disease_info().get(disease_name)
        if info:
            st.markdown(
                f"""
                <div class="info-box">
                    <h3>ℹ️ {info['disease_name']}</h3>
                    <p><b>Description:</b> {info['description']}</p>
                    <p><b>Symptoms:</b> {info['symptoms']}</p>
                    <p><b>Prevention / treatment:</b> {info['treatment']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info(
                f"No extra information available for **{disease_name}** yet. "
                "You can add details for this class in `disease_info.json`."
            )


def about_page():
    show_header("About This Project", "Final Year Project: Deep Learning for Agriculture")

    st.markdown(
        """
        ### Project Overview
        **AI-Based Crop Disease Detection** is a deep learning web application
        that classifies crop leaf images as healthy or diseased using a
        Convolutional Neural Network (CNN) built with TensorFlow/Keras.

        ### Technologies Used
        - **TensorFlow / Keras** - CNN model training and prediction
        - **Streamlit** - web user interface
        - **Pillow / NumPy** - image loading and preprocessing
        - **Matplotlib / scikit-learn** - graphs, metrics, confusion matrix
        - **OpenCV** - (optional) additional image processing

        ### Project Architecture
        - `train.py` - loads the dataset, trains the CNN and saves the model.
        - `predict.py` - command-line prediction for a single image.
        - `evaluate.py` - computes accuracy, precision, recall, F1-score and confusion matrix.
        - `app.py` - the Streamlit web app (Home / Detection / About).
        - `utils/preprocessing.py` - shared image preprocessing code.
        - `dataset/` - put your class folders here (one folder per disease).
        - `models/` - contains the trained model and class names.
        - `results/` - graphs, reports and confusion matrix images.

        ### Dataset
        The model was trained (or can be trained) on a folder-based dataset.
        Each subfolder name inside `dataset/` is treated as one class
        (disease name). A class whose name contains **"healthy"** is treated
        as the healthy plant class.

        ### Disclaimer
        This project is for educational and research purposes. Always
        confirm AI diagnoses with a local agricultural expert before using
        any treatment on real crops.
        """
    )


# ------------------------- App entry -------------------------
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_ui_style()

# Sidebar navigation
st.sidebar.title("🌱 Crop Disease Detection")
st.sidebar.markdown("AI-Powered Plant Leaf Diagnosis")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔍 Disease Detection", "ℹ️ About"],
)

# Show model status in the sidebar
if os.path.isfile(MODEL_PATH):
    st.sidebar.success("✅ Model loaded")
else:
    st.sidebar.warning("⚠️ Model not trained yet")

st.sidebar.markdown("---")
st.sidebar.caption("Deep Learning • CNN • Streamlit")

if page == "🏠 Home":
    home_page()
elif page == "🔍 Disease Detection":
    detection_page()
else:
    about_page()