# AI Crop Disease Detection with Deep Learning (CropAI)

CropAI is a Streamlit app for AI-powered crop disease detection from plant leaf
photos. It uses a **TensorFlow/Keras convolutional neural network (CNN)** to
classify supported crop conditions and show a confidence score with prevention
guidance. This educational project is intended for learning and research;
predictions are not a substitute for advice from an agricultural expert.

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
| Image loading / processing | Pillow |
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

This project currently includes **6,644 images across four leaf-condition classes**:
bacterial spot (2,134), early blight (1,009), late blight (1,908), and healthy
(1,593). The added color images are from the [PlantVillage dataset repository](https://github.com/spMohanty/PlantVillage-Dataset);
source details and a citation are in [dataset/README.md](dataset/README.md).
PlantVillage was collected under controlled conditions, so validation results
on this dataset do not establish performance on field photographs. Training
uses a random image-level validation split; it is not an independent field test.
To fetch the selected images again on another checkout, run
`python download_plantvillage.py`. The downloaded image files stay local and
are excluded from Git; the source, counts and citation are documented in
`dataset/README.md`.

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

The classifier uses **MobileNetV2** initialized with ImageNet weights as a
frozen feature extractor, followed by global average pooling and a small dense
classification head. Training applies random flips, rotations and zoom to
training images. The input size is **128×128×3**, and output classes are read
from the dataset folders. The 80/20 split is a random image-level validation
split from this dataset; it is not an independent field test.

---

## 📊 How to Evaluate the Model

```
python evaluate.py
```

This evaluates the same held-out 20% image-level validation split used during
training, prints validation accuracy and per-class precision, recall and F1,
and saves:
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
Predicted disease : Early_blight
Confidence        : 87.4%
```

---

## 🚀 How to Run the Streamlit App

```
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).
At startup, the terminal also prints the latest saved validation accuracy.
To recalculate that metric for the current model at any time, run
`python evaluate.py` from this project folder in the terminal of VS Code,
PyCharm, or another Python IDE. This reports held-out dataset accuracy;
individual image prediction confidence is a separate metric.

- **Home** - project introduction
- **Disease Detection** - upload a leaf image and click "Predict Disease"
- **About** - project details

⚠️ The app **does not retrain** the model on startup; it simply loads
`models/crop_disease_model.keras`. If the model is missing, the app shows
clear instructions instead of crashing.

## 🌐 Search visibility when deployed

For a public Streamlit Community Cloud deployment, set the app visibility to
public and choose a descriptive custom subdomain such as
`cropai-disease-detection`. The app sets a descriptive browser/search title and
starts with readable text describing crop disease detection, the supported
image workflow and the educational limitation of model predictions. Streamlit
Community Cloud indexes public apps; search engines decide how and when to show
them. A custom domain, if used, should be chosen before sharing the app URL.

This repository does not include a `robots.txt` or `sitemap.xml`: those files
must be served by the web host at the site's root, and Streamlit apps do not
serve repository files there as static website routes.

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
