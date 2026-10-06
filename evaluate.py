"""
evaluate.py - Evaluates the trained model on its held-out validation split.

It computes:
    - Accuracy
    - Precision, Recall, F1-score (per class + averages)
    - Confusion matrix
and saves the confusion matrix as results/confusion_matrix.png.

Usage:
    python evaluate.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, avoids GUI problems on Windows
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from tensorflow.keras.utils import image_dataset_from_directory

DATASET_DIR = "dataset"
MODEL_PATH = os.path.join("models", "crop_disease_model.keras")
CLASS_NAMES_PATH = os.path.join("models", "class_names.json")
RESULTS_DIR = "results"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 42


def load_model_and_class_names():
    """Load the saved model and class names with friendly error messages."""
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'.\n"
            "Train the model first with:  python train.py"
        )
    if not os.path.isfile(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"Class names file not found at '{CLASS_NAMES_PATH}'.\n"
            "Run 'python train.py' again - it creates this file."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    return model, class_names


def load_validation_dataset(class_names):
    """
    Load the same held-out image-level split used during model training.
    """
    ds = image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        class_names=class_names,
    )
    return ds


def plot_confusion_matrix(cm, class_names):
    """Draw and save the confusion matrix as results/confusion_matrix.png."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ticks = np.arange(len(class_names))
    ax.set(xticks=ticks, yticks=ticks,
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted label", ylabel="True label",
           title="Validation Confusion Matrix")

    # Rotate class name labels so long names are readable
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Show the count inside every cell
    threshold = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Confusion matrix saved to: {path}")


def main():
    print("=" * 60)
    print("CROP DISEASE DETECTION - MODEL EVALUATION")
    print("=" * 60)

    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset folder '{DATASET_DIR}' not found.\n"
            "Place your dataset folders inside it and run 'python train.py' first."
        )

    # 1. Load model + class names
    print("\n[1/3] Loading model and class names...")
    model, class_names = load_model_and_class_names()

    # 2. Evaluate only on the held-out images selected by train.py.
    print("\n[2/3] Loading held-out 20% validation split and making predictions...")
    ds = load_validation_dataset(class_names)

    # The dataset is shuffled, so read labels and predictions in the same
    # pass. Iterating twice would reshuffle the examples and misalign metrics.
    true_batches = []
    predicted_batches = []
    for images, labels in ds:
        probabilities = model(images, training=False).numpy()
        true_batches.append(labels.numpy())
        predicted_batches.append(np.argmax(probabilities, axis=1))
    y_true = np.concatenate(true_batches, axis=0)
    y_pred = np.concatenate(predicted_batches, axis=0)

    # 3. Metrics
    print("\n[3/3] Computing metrics...")
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nValidation Accuracy: {accuracy:.2%}")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report)

    # Save the report to a text file too
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Validation Accuracy: {accuracy:.2%}\n\n")
        f.write(report)
    print(f"Report saved to: {report_path}")

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as err:
        print(f"\nERROR: {err}", file=sys.stderr)
        sys.exit(1)
