# 🌱 AI-Based Crop Disease Detection Using Deep Learning

A complete deep learning web application that detects crop diseases from a
simple photo of a plant leaf. The user uploads an image, and a trained
**Convolutional Neural Network (CNN)** built with **TensorFlow/Keras**
classifies it as healthy or identifies the disease, along with a confidence
percentage and simple prevention advice.

Built for beginners and suitable for a college final-year project.

---

## ✨ Features

- **Home page** - project introduction and "how it works" explanation
- **Disease Detection page** - upload a leaf image (JPG/JPEG/PNG), get the predicted disease, confidence % and healthy/diseased status
- **About page** - project details, architecture and disclaimer
- **Real CNN model** - no fake/dummy predictions; the app loads the actual trained model
- **Automatic class detection** - class names come from the dataset folder names
- **Data augmentation** - flips, rotations, zoom to make the model more robust
- **Evaluation** - accuracy, precision, recall, F1-score, confusion matrix + graph
- **Friendly error handling** - missing model, invalid image, empty dataset, etc.
- **Disease information panel** - description, symptoms and safe prevention advice for each detected class

---

## 🧰 Technologies Used

| Purpose | Library |
|---|---|
| Deep learning model | TensorFlow / Keras |
| Web application | Streamlit |
| Image loading / processing | Pillow, OpenCV |
| Numeric operations | NumPy |
| Data handling | Pandas |
| Graphs & metrics | Matplotlib, scikit-learn |

---

## 🏗️ Project Architecture

```
crop-disease-detection/
│
├── app.py                    # Streamlit web application
├── train.py                  # Trains the CNN and saves the model
├── predict.py                # Command-line prediction for one image
├── evaluate.py               # Computes metrics + confusion matrix
├── disease_info.json         # Disease info shown in the app (editable)
├── requirements.txt          # Python packages to install
├── README.md
│
├── dataset/                  # ← PUT YOUR DATASET HERE (one folder per class)
│   └── .gitkeep
│
├── models/                   # Trained model + class names are saved here
│   └── .gitkeep
│
├── results/                  # Graphs and evaluation reports
│   └── .gitkeep
│
└── utils/
    ├── __init__.py
    └── preprocessing.py      # Shared image preprocessing (used by all scripts)
```

**How the pieces work together:**

1. `train.py` reads images from `dataset/` (each subfolder = one class),
   builds a CNN and saves `models/crop_disease_model.keras` +
   `models/class_names.json`.
2. `evaluate.py` loads the saved model and reports detailed metrics.
3. `predict.py` runs the model on any single image from the terminal.
4. `app.py` (Streamlit) loads the saved model once, lets a user upload a
   leaf image, and shows the prediction with confidence + disease info.

---

## 📁 Dataset Structure

The dataset is folder-based. One subfolder per class/disease:

```
dataset/
    Crop___Early_blight/
        image1.jpg
        image2.jpg
    Crop___Late_blight/
        image1.jpg
    Crop___healthy/
        image1.jpg
```

- The **folder name becomes the class name** automatically.
- Any class whose name contains **"healthy"** is treated as the healthy class.
- Supported image formats: JPG, JPEG, PNG.
- Recommended: at least **50–100 images per class** for a decent model.
  (You can start experimenting with as few as 10 per class.)
- A public dataset that matches this structure exactly is the
  **PlantVillage dataset** (e.g. from Kaggle).

---

## 🛠️ Installation

> **IMPORTANT - Python version:**
> This project uses TensorFlow, which does **not** support Python 3.14 yet.
> Install **Python 3.10 or 3.11** (64-bit) and make sure it is the active
> version before continuing. Check with:
> ```
> python --version
> ```

1. Create and activate a virtual environment (recommended):
   ```
   python -m venv .venv
   .venv\Scripts\activate        (Windows)
   source .venv/bin/activate     (Linux / Mac)
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

---

## 📚 How to Train the Model

1. Put your images into the `dataset/` folder:
   ```
   dataset/
       Class_Name_1/
           image1.jpg
       Class_Name_2/
           image1.jpg
   ```

2. Run the training script:
   ```
   python train.py
   ```

3. When training finishes you will have:
   - `models/crop_disease_model.keras` - the trained CNN
   - `models/class_names.json` - the class names
   - `results/training_history.png` - accuracy and loss graphs

The CNN uses an input size of **128×128×3**, includes data augmentation,
Conv2D + ReLU + MaxPooling layers, dropout, flatten, dense and a softmax
output with a number of classes automatically taken from the dataset folders.

---

## 📊 How to Evaluate the Model

```
python evaluate.py
```

This prints the overall accuracy, a per-class classification report
(precision, recall, F1-score) and saves:
- `results/confusion_matrix.png`
- `results/classification_report.txt`

---

## 🖼️ How to Run Prediction (Command Line)

```
python predict.py dataset/Crop___Early_blight/example.jpg
```

Output example:

```
PREDICTION RESULT
Predicted disease : Crop___Early_blight
Confidence        : 87.4%
```

---

## 🚀 How to Run the Streamlit App

```
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

- **Home** - project introduction
- **Disease Detection** - upload a leaf image and click "Predict Disease"
- **About** - project details

⚠️ The app **does not retrain** the model on startup; it simply loads
`models/crop_disease_model.keras`. If the model is missing, the app shows
clear instructions instead of crashing.

---

## ✅ Expected Output

After uploading an image, the app displays:

- The uploaded image preview
- Predicted disease / class name
- Confidence percentage (e.g. 92.5%)
- Healthy 🌿 or Diseased ⚠️ status
- Disease description, symptoms and safe prevention/treatment advice
  (from `disease_info.json`)

---

## 📖 Customizing Disease Information

Open `disease_info.json` and add/modify entries. The key must be the exact
class name (the dataset folder name):

```json
"My_New_Disease": {
    "disease_name": "Friendly displayed name",
    "description": "Short description...",
    "symptoms": "What the farmer sees on the leaf...",
    "treatment": "Safe, general prevention advice..."
}
```

If a class has no entry, the app shows a friendly fallback message.

---

## 💡 Tips for Beginners

- Start with a small dataset (2–3 classes, ~20 images each) to see the
  pipeline work, then increase the size.
- More images = better accuracy. Data augmentation helps, but real variety
  (different angles, light, background) matters most.
- 15 epochs is a good default; watch the validation accuracy in the console.
- If training is slow on your PC, reduce `EPOCHS` or `BATCH_SIZE` in `train.py`.

---

## 🔮 Future Improvements

- Use a pretrained model (Transfer Learning with MobileNetV2 / ResNet) for
  much higher accuracy with less data.
- Add a camera capture option inside the app for mobile use.
- Show a bar chart of all class probabilities in the app.
- Deploy the app online (Streamlit Community Cloud, Hugging Face Spaces).
- Add a "retrain from the app" page for advanced users.
- Support more crops and multi-leaf images.

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Always
confirm the AI diagnosis with a local agricultural expert before applying
any treatment to real crops.

---

## 📋 Commands Summary

| Task | Command |
|---|---|
| Install packages | `pip install -r requirements.txt` |
| Train the model | `python train.py` |
| Evaluate the model | `python evaluate.py` |
| Predict one image | `python predict.py path/to/image.jpg` |
| Run the web app | `streamlit run app.py` |