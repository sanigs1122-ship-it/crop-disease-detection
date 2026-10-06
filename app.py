"""
app.py - CropAI: AI-Based Crop Disease Detection (Streamlit web app).

Run with:
    streamlit run app.py

Pages (sidebar navigation):
    1. Home             - hero, highlights and feature overview
    2. Disease Detection- upload a leaf image and get a prediction
    3. About            - project documentation, workflow and tech stack

All prediction logic is unchanged: the same cached CNN model, preprocessing
pipeline and disease_info.json lookup are used as before. Only the UI layer
has been redesigned (styles live in styles.css, icons in utils/ui.py).
"""

import json
import os
import sys
import textwrap

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from utils.preprocessing import (is_supported_file,
                                 preprocess_uploaded_image)
from utils.ui import (footer_section, hero_art, icon, logo,
                      model_status_card, sidebar_brand)

# ------------------------- Paths -------------------------
DATASET_DIR = "dataset"
MODEL_PATH = os.path.join("models", "crop_disease_model.keras")
CLASS_NAMES_PATH = os.path.join("models", "class_names.json")
VALIDATION_REPORT_PATH = os.path.join("results", "classification_report.txt")
DISEASE_INFO_PATH = "disease_info.json"
DISEASE_INFO_DEFAULT = "disease_info_default.json"

PAGES = ["Home", "Disease Detection", "About"]

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


def print_validation_accuracy_on_startup():
    """Print the latest evaluated model accuracy once to the app terminal."""
    if st.session_state.get("_validation_accuracy_logged", False):
        return

    def terminal(message=""):
        print(message, file=sys.stderr, flush=True)

    terminal("\n" + "=" * 60)
    terminal("CropAI model validation summary")
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), VALIDATION_REPORT_PATH)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_PATH)

    if not os.path.isfile(report_path):
        terminal("Validation accuracy is not available yet.")
        terminal("Run `python evaluate.py` from the project folder to calculate it.")
    elif os.path.isfile(model_path) and os.path.getmtime(report_path) < os.path.getmtime(model_path):
        terminal("The saved accuracy report is older than the current model.")
        terminal("Run `python evaluate.py` to refresh the validation accuracy.")
    else:
        with open(report_path, encoding="utf-8") as report_file:
            accuracy_line = report_file.readline().strip()
        terminal(accuracy_line or "Validation accuracy report is empty.")
        terminal("Metric: held-out image validation split; this is not prediction confidence.")

    terminal("Recalculate any time with: python evaluate.py")
    terminal("=" * 60 + "\n")
    st.session_state["_validation_accuracy_logged"] = True


def dataset_snapshot():
    """Return live image totals for the folder-based training dataset."""
    class_counts = {}
    if not os.path.isdir(DATASET_DIR):
        return class_counts, 0

    for class_name in sorted(os.listdir(DATASET_DIR)):
        class_path = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        class_counts[class_name] = sum(
            1 for filename in os.listdir(class_path)
            if filename.lower().endswith((".jpg", ".jpeg", ".png"))
        )
    return class_counts, sum(class_counts.values())


# ------------------------- Styling -------------------------
def load_css():
    """Inject the shared stylesheet (styles.css) into the app."""
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.css")
    if os.path.isfile(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def html(markup):
    """Shortcut for rendering an HTML fragment as raw HTML.

    The markup is dedented before rendering so that indented lines can never
    be interpreted by the Markdown parser as a code block.
    """
    st.markdown(textwrap.dedent(markup), unsafe_allow_html=True)


def page_intro(eyebrow_text, title, subtitle):
    """Standard page header: eyebrow pill + large title + subtitle."""
    html(f'<span class="eyebrow">{eyebrow_text}</span>')
    st.header(title)
    html(f'<p class="page-subtitle">{subtitle}</p>')


def section_title(title, subtitle=""):
    """Section heading helper."""
    sub = f'<p class="section-sub">{subtitle}</p>' if subtitle else ""
    html(f'<h2 class="section-head">{title}</h2>{sub}')


def show_missing_model_message():
    """Friendly message shown when the trained model files do not exist."""
    st.error("The trained model was not found.")
    html(
        """
        <div class="pane">
          <h3>How to fix this</h3>
          <p>Place your dataset inside the <code>dataset/</code> folder, one folder per class, e.g.:</p>
          <p><code>dataset/Early_blight/image1.jpg</code><br>
             <code>dataset/healthy/image1.jpg</code></p>
          <p>Then train the model by running <code>python train.py</code> which automatically
             creates <code>models/crop_disease_model.keras</code> and
             <code>models/class_names.json</code>. Finally restart this app.</p>
        </div>
        """
    )


# ------------------------- Home page -------------------------
def home_page():
    html(
        '<div class="project-strip"><span>COMPUTER VISION RESEARCH PROJECT</span>'
        '<span>MODEL DEMONSTRATION</span></div>'
    )
    html('<span class="eyebrow">PLANT HEALTH · DEEP LEARNING</span>')
    st.header("AI Crop Disease Detection")
    html(
        '<p class="page-subtitle">CropAI is a computer-vision project that classifies four common '
        'leaf conditions from an image. Explore the training dataset, model '
        'workflow and an AI-generated prediction in one place.</p>'
    )

    col_left, col_right = st.columns([1, 1], gap="large")
    with col_left:
        if st.button("Start Detection →", type="primary", key="cta_home", use_container_width=True):
            st.session_state["nav_pending"] = "Disease Detection"
            st.rerun()
        if st.button("View project methodology", key="cta_about", use_container_width=True):
            st.session_state["nav_pending"] = "About"
            st.rerun()
        html(
            """
            <p class="hero-note">Research prototype · Image-based classification · Educational use</p>
            """
        )
    with col_right:
        html(f'<div class="hero-art">{hero_art()}</div>')

    # --- Dataset and model facts ---
    class_counts, total_images = dataset_snapshot()
    section_title("Project at a glance", "Current training scope and model configuration")
    stats = [
        (f"{total_images:,}", "Training images"),
        (str(len([count for count in class_counts.values() if count > 0])), "Leaf classes"),
        ("128 × 128", "Model input size"),
        ("MobileNetV2", "Feature extractor"),
    ]
    stat_items = "".join(
        f'<div class="stat-item"><div class="stat-value">{value}</div>'
        f'<div class="stat-label">{label}</div></div>'
        for value, label in stats
    )
    html(f'<div class="stats-band">{stat_items}</div>')

    # --- Workflow cards ---
    section_title("How the analysis works", "A clear, repeatable image-classification workflow")
    features = [
        ("eye", "Image preparation",
         "A JPG, JPEG or PNG leaf photo is decoded, converted to RGB and resized to 128 × 128 pixels."),
        ("chip", "Visual feature extraction",
         "A MobileNetV2 backbone initialized with ImageNet weights extracts visual patterns."),
        ("activity", "Four-class prediction",
         "A trained classification head estimates probabilities for four leaf-condition classes."),
    ]
    cards = ""
    for name, title, desc in features:
        cards += textwrap.dedent(
            f"""
            <div class="feature-card">
              <div class="icon-badge">{icon(name, 26)}</div>
              <h3>{title}</h3>
              <p>{desc}</p>
            </div>
            """
        )
    html(f'<div class="cards-grid">{cards}</div>')

    html(
        """
        <div class="project-note">
          <div class="project-note-mark">i</div>
          <div><strong>Research note</strong><p>PlantVillage images were collected under controlled conditions. Results on field photos may differ; predictions are educational and should be checked by an agricultural expert.</p></div>
        </div>
        """
    )


# ------------------------- Detection page -------------------------
def detection_page():
    page_intro(
        "Disease Detection",
        "Diagnose your crop in seconds",
        "Run a single AI analysis on a clear photo of a plant leaf.",
    )

    model = load_model()
    class_names = load_class_names()

    # --- Guard conditions with friendly errors ---
    if model is None:
        show_missing_model_message()
        return
    if not class_names:
        st.error("`models/class_names.json` is missing. Please run `python train.py` again.")
        return

    # --- Upload card ---
    html(
        """
        <div class="upload-head">
          <div class="upload-title">Upload Crop Leaf Image</div>
          <p class="upload-sub">Upload a clear image of a crop leaf to analyze its health.</p>
        </div>
        """
    )
    uploaded_file = st.file_uploader(
        "Leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="Supported formats: JPG, JPEG, PNG.",
    )

    if uploaded_file is None:
        html(
            """
            <div class="pane">
              <p>Drag &amp; drop a leaf photo above or click <b>Browse files</b> to get
                 started. The image is processed by the trained CNN model running with
                 this app.</p>
            </div>
            """
        )
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

    # --- Reset stale results when a new image is uploaded ---
    st.session_state.setdefault("last_analyzed", None)
    if uploaded_file.name != st.session_state.last_analyzed:
        st.session_state.last_analyzed = None
        st.session_state.pop("prediction", None)

    # --- Preview + analyze ---
    col_img, col_act = st.columns([1, 1], gap="large")
    with col_img:
        st.image(uploaded_file, use_column_width="always")
        html(
            f'<div class="file-name">{icon("leaf", 13)} {uploaded_file.name}</div>'
        )
    with col_act:
        html(
            """
            <div class="pane" style="margin-top:1.5rem; display:flex; flex-direction:column; gap:0.85rem;">
              <h3 style="margin-bottom:0">Ready to analyze</h3>
              <p>Your image has been preprocessed to the size expected by the CNN
                 (128 × 128) and is ready for prediction.</p>
            </div>
            """
        )
        analyze = st.button("Analyze Image", type="primary", key="analyze_btn", use_container_width=True)

    if not analyze:
        # Keep showing the last result for the current image across reruns.
        if st.session_state.get("prediction") and st.session_state.last_analyzed == uploaded_file.name:
            render_prediction(st.session_state.prediction)
        return

    # --- Run the model (same pipeline as before) ---
    with st.spinner("Running CNN analysis — this may take a few seconds..."):
        try:
            # --- Preprocess: same resize (128x128) as training ---
            image_batch = preprocess_uploaded_image(uploaded_file)

            # --- Predict ---
            probabilities = model.predict(image_batch, verbose=0)[0]

        except Exception as err:
            st.error(f"Prediction failed: {err}")
            return

    # --- Build the result from the real model output ---
    best_index = int(np.argmax(probabilities))
    disease_name = class_names[best_index]
    confidence = float(probabilities[best_index])

    # Healthy/Diseased status: a class is treated as healthy if
    # its folder name contains the word "healthy".
    is_healthy = "healthy" in disease_name.lower()
    st.session_state.last_analyzed = uploaded_file.name
    st.session_state.prediction = {
        "disease_name": disease_name,
        "confidence": confidence,
        "is_healthy": is_healthy,
        "probabilities": probabilities.tolist(),
        "class_names": class_names,
    }

    render_prediction(st.session_state.prediction)


def readable_class_name(class_name):
    """Turn a dataset folder label into a compact name for the interface."""
    label = class_name.rsplit("___", 1)[-1].replace("_", " ").strip()
    if label.lower() == "healthy":
        return "Healthy"
    return label.title()


def render_prediction(pred):
    """Premium result dashboard driven entirely by the real model output."""
    disease_name = pred["disease_name"]
    confidence = pred["confidence"]
    is_healthy = pred["is_healthy"]
    probabilities = np.asarray(pred["probabilities"])
    class_names = pred["class_names"]

    pct = round(confidence * 100, 1)
    circ = 326.7
    offset = max(0.0, min(circ, circ * (1.0 - confidence)))
    pct_clamped = max(0.0, min(1.0, confidence)) * 100.0

    head_class = "healthy" if is_healthy else "diseased"
    if is_healthy:
        head_label = "Healthy Plant"
        status_html = (
            '<span class="status-chip healthy"><span class="pulse"></span>'
            f'{icon("check", 14)} Healthy</span>'
        )
        title = "Plant appears healthy"
        subtitle = "Your uploaded leaf was classified as healthy by the AI model."
    else:
        head_label = "Disease Detected"
        status_html = (
            '<span class="status-chip diseased"><span class="pulse"></span>'
            f'{icon("activity", 14)} Disease detected</span>'
        )
        title = readable_class_name(disease_name)
        subtitle = f"Predicted class: {title}"

    # --- Progress bars for every class (from real probabilities) ---
    order = np.argsort(probabilities)[::-1]
    class_rows = ""
    for idx in order:
        prob = float(probabilities[idx])
        name = readable_class_name(class_names[idx])
        best = "best" if idx == order[0] else ""
        class_rows += textwrap.dedent(
            f"""
            <div class="class-row">
              <div class="lbl"><span>{name}</span><span class="prob">{prob*100:.1f}%</span></div>
              <div class="track"><div class="fill {best}" style="--w: {prob*100:.2f}%"></div></div>
            </div>
            """
        )

    result_html = f"""
    <div class="result-card">
      <div class="result-head {head_class}">
        <span>{'▲' if not is_healthy else '●'} {head_label}</span>
        <span>AI Prediction</span>
      </div>
      <div class="result-body">
        <div class="result-main">
          {status_html}
          <div class="disease-name">{title}</div>
          <div class="disease-sub">{subtitle}</div>
          <div class="conf-bar">
            <div class="bar-label"><span>Model Confidence</span><span>{pct}%</span></div>
            <div class="bar"><div style="--w: {pct_clamped:.1f}%"></div></div>
          </div>
        </div>
        <div class="confidence-wrap">
          <div class="confidence-ring">
            <svg width="168" height="168" viewBox="0 0 120 120">
              <defs>
                <linearGradient id="ringGradient" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stop-color="#166534"/>
                  <stop offset="100%" stop-color="#3ec46e"/>
                </linearGradient>
              </defs>
              <circle class="ring-track" cx="60" cy="60" r="52"/>
              <circle class="ring-fill" cx="60" cy="60" r="52" style="--off: {offset:.1f}px"/>
            </svg>
            <div class="confidence-center">
              <div class="pct">{pct}%</div>
              <div class="pct-label">Confidence</div>
            </div>
          </div>
          <div class="confidence-note">Confidence from the CNN’s softmax output</div>
        </div>
      </div>
      <div style="padding: 0 1.5rem 1.4rem;" class="class-bars">{class_rows}</div>
    </div>
    """
    html(result_html)

    # --- Disease information from disease_info.json ---
    info_data = load_disease_info()
    readable_name = readable_class_name(disease_name)
    info = next(
        (value for key, value in info_data.items()
         if key.casefold() in {disease_name.casefold(), readable_name.casefold()}),
        info_data.get("default"),
    )
    if info:
        info_html = f"""
        <div class="info-cards">
          <div class="info-card">
            <div class="ic-head">{icon("info", 19)} About the Disease</div>
            <p>{info["description"]}</p>
          </div>
          <div class="info-card">
            <div class="ic-head">{icon("activity", 19)} Symptoms</div>
            <p>{info["symptoms"]}</p>
          </div>
          <div class="info-card">
            <div class="ic-head">{icon("shield", 19)} Prevention &amp; Management</div>
            <p>{info["treatment"]}</p>
          </div>
        </div>
        """
        html(info_html)
    else:
        st.info(
            f"No extra information available for **{disease_name}** yet. "
            "You can add details for this class in `disease_info.json`."
        )

    html(
        """
        <p style="color:#9db2a4; font-size:0.8rem; margin-top:1rem;">
          Looking for the result? Re-run the analysis with a new image using the
          uploader above.
        </p>
        """
    )


# ------------------------- About page -------------------------
def about_page():
    page_intro(
        "Project Documentation",
        "Project methodology",
        "A four-class plant-leaf image classifier built with TensorFlow, Keras and transfer learning.",
    )

    html(
        """
        <div class="pane">
          <h3>Project Overview</h3>
          <p>
            <b>CropAI</b> is an educational computer-vision project for leaf-condition
            classification. A MobileNetV2 feature extractor and a task-specific
            classification head predict one of four supported classes. The app returns
            the highest-scoring class, probability distribution and reference information.
          </p>
        </div>
        """
    )

    class_counts, total_images = dataset_snapshot()
    section_title("Training dataset", "Image counts currently available to the training script")
    rows = "".join(
        f'<tr><td>{readable_class_name(name)}</td><td>{count:,}</td></tr>'
        for name, count in class_counts.items()
    )
    html(
        f"""
        <div class="pane dataset-panel">
          <table class="dataset-table">
            <thead><tr><th>Leaf class</th><th>Images</th></tr></thead>
            <tbody>{rows}<tr class="dataset-total"><td>Total</td><td>{total_images:,}</td></tr></tbody>
          </table>
          <p class="source-note">Dataset: <a href="https://github.com/spMohanty/PlantVillage-Dataset" target="_blank" rel="noopener">PlantVillage color images</a> · Study: <a href="https://doi.org/10.3389/fpls.2016.01419" target="_blank" rel="noopener">Mohanty et al. (2016)</a>. Controlled-background photos may not reflect field conditions.</p>
        </div>
        """
    )

    history_path = os.path.join("results", "training_history.png")
    if os.path.isfile(history_path):
        section_title("Training history", "Training and validation curves from the latest model run")
        st.image(
            history_path,
            caption="Recorded by train.py. Validation curves use the randomized image-level split.",
            use_column_width="always",
        )

    # --- How it works ---
    section_title("How it works", "From upload to result in five steps")
    steps = [
        ("Upload Image", "A leaf photo is uploaded in JPG, JPEG or PNG format."),
        ("Image Preprocessing", "The image is resized to 128×128 and converted to RGB."),
        ("Feature Extraction", "ImageNet-initialized MobileNetV2 extracts visual features."),
        ("Class Scoring", "A task-specific head scores four leaf-condition classes."),
        ("Prediction Result", "The highest-scoring class and reference information are shown."),
    ]
    items = ""
    for i, (title, desc) in enumerate(steps, start=1):
        arrow = icon("arrow", 18) if i < len(steps) else ""
        items += textwrap.dedent(
            f"""
            <div class="workflow-item">
              <div class="step-num">0{i}</div>
              <h3>{title}</h3>
              <p>{desc}</p>
              {f'<span class="flow-arrow">{arrow}</span>' if arrow else ""}
            </div>
            """
        )
    html(f'<div class="workflow">{items}</div>')

    # --- Technology stack ---
    section_title("Technology Stack", "The tools behind the platform")
    techs = [
        ("code", "Python", "Core language for training, prediction and the web app."),
        ("braces", "TensorFlow / Keras", "Builds, trains and serves the image classifier."),
        ("network", "MobileNetV2", "ImageNet-initialized CNN used for feature extraction."),
        ("eye", "Pillow", "Decodes uploaded photos and prepares RGB model inputs."),
        ("activity", "Scikit-learn", "Computes evaluation metrics and the confusion matrix."),
        ("waves", "Streamlit", "Interactive interface for the project demonstration."),
    ]
    cards = ""
    for name, title, desc in techs:
        cards += textwrap.dedent(
            f"""
            <div class="feature-card">
              <div class="icon-badge">{icon(name, 26)}</div>
              <h3>{title}</h3>
              <p>{desc}</p>
            </div>
            """
        )
    html(f'<div class="cards-grid">{cards}</div>')

    # --- Architecture ---
    html(
        """
        <div class="pane">
          <h3>Project Architecture</h3>
          <ul class="pane-list">
            <li><code>train.py</code> — loads the dataset, trains the CNN and saves the model.</li>
            <li><code>predict.py</code> — command-line prediction for a single image.</li>
            <li><code>evaluate.py</code> — accuracy, precision, recall, F1-score and confusion matrix.</li>
            <li><code>app.py</code> — the Streamlit web app (Home / Detection / About).</li>
            <li><code>utils/preprocessing.py</code> — shared image preprocessing (128×128 resize).</li>
            <li><code>dataset/</code> — one folder per class; a class name containing “healthy” is the healthy class.</li>
            <li><code>models/</code> — trained model (<code>crop_disease_model.keras</code>) and class names.</li>
            <li><code>results/</code> — graphs, reports and confusion-matrix images.</li>
          </ul>
        </div>
        """
    )

    # --- Disclaimer ---
    html(
        """
        <div class="pane">
          <h3>Disclaimer</h3>
          <p>
            This project is for educational and research purposes. Always confirm AI
            diagnoses with a local agricultural expert before using any treatment on real crops.
          </p>
        </div>
        """
    )


# ------------------------- App entry -------------------------
st.set_page_config(
    page_title="AI Crop Disease Detection | CropAI",
    page_icon=(
        "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'"
        " fill='none'%3E%3Cpath d='M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8"
        " 0 5.5-4.78 10-10 10Z' fill='%231b7a3d'/%3E%3C/svg%3E"
    ),
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()
print_validation_accuracy_on_startup()

# Apply a queued navigation request (from a CTA button) before the sidebar
# radio widget is instantiated — writing to "nav" after that point raises
# StreamlitAPIException.
if st.session_state.get("nav_pending") in PAGES:
    st.session_state["nav"] = st.session_state.pop("nav_pending")

# ------------------------- Sidebar -------------------------
with st.sidebar:
    html(sidebar_brand())

    st.sidebar.radio(
        "Navigate",
        PAGES,
        key="nav",
        label_visibility="collapsed",
    )

    # Show model status in the sidebar
    html(model_status_card(os.path.isfile(MODEL_PATH)))

    html(
        """
        <div class="sidebar-foot">
          Built with TensorFlow &amp; Streamlit • CNN image classification
        </div>
        """
    )

# ------------------------- Page routing -------------------------
page = st.session_state.get("nav", "Home")

if page == "Home":
    home_page()
elif page == "Disease Detection":
    detection_page()
else:
    about_page()

# ------------------------- Footer -------------------------
html(footer_section())
