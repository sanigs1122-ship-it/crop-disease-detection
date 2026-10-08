"""
predict.py - Predicts the disease of a single leaf image using the
trained CNN model.

Usage:
    python predict.py path/to/image.jpg

Example:
    python predict.py dataset/Crop___Early_blight/example.jpg

Output: predicted disease name + confidence percentage.
"""

import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf

from utils.preprocessing import load_and_preprocess_image
from utils.ood import (build_feature_extractor, check_supported_input,
                       load_ood_reference)

MODEL_PATH = os.path.join("models", "crop_disease_model.keras")
CLASS_NAMES_PATH = os.path.join("models", "class_names.json")
OOD_REFERENCE_PATH = os.path.join("models", "ood_reference.json")


def load_model_and_class_names():
    """
    Load the trained model and the class names file.
    Raises beginner-friendly errors when these files are missing.
    """
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


def predict_image(model, class_names, image_path):
    """
    Preprocess the image, run the model and return:
        (disease_name, confidence_percent, all_probabilities)
    """
    # Preprocessing here is identical to training: resize to 128x128.
    # Normalization (0-1) is done inside the model by its Rescaling layer.
    image_batch = load_and_preprocess_image(image_path)

    probabilities = model.predict(image_batch, verbose=0)[0]  # shape (num_classes,)

    best_index = int(np.argmax(probabilities))
    disease_name = class_names[best_index]
    confidence = float(probabilities[best_index])

    return disease_name, confidence, probabilities


def main():
    parser = argparse.ArgumentParser(description="Predict crop disease from an image.")
    parser.add_argument("image_path", help="Path to a leaf image (jpg/jpeg/png)")
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        print(f"ERROR: Image not found: {args.image_path}", file=sys.stderr)
        sys.exit(1)

    model, class_names = load_model_and_class_names()
    disease_name, confidence, probabilities = predict_image(model, class_names, args.image_path)
    label_key = disease_name.rsplit("___", 1)[-1].casefold().replace("-", "_").replace(" ", "_")
    is_unknown = label_key in {
        "not_a_leaf", "non_leaf", "other", "unknown",
    }
    if not is_unknown:
        reference = load_ood_reference(OOD_REFERENCE_PATH)
        if reference is not None:
            image_batch = load_and_preprocess_image(args.image_path)
            accepted, _, _, _ = check_supported_input(
                build_feature_extractor(model), image_batch, class_names, reference
            )
            is_unknown = not accepted
    if is_unknown:
        disease_name = "Not_a_leaf"
        confidence = 0.0

    print("=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"Predicted disease : {disease_name}")
    print(f"Confidence        : {confidence:.1%}")

    if not is_unknown:
        print("\nAll class probabilities:")
        for name, prob in zip(class_names, probabilities):
            print(f"  {name:30s} {prob:.1%}")

    healthy = any("healthy" in c.lower() for c in class_names)
    status = (
        "NOT A LEAF / UNSUPPORTED INPUT" if is_unknown
        else "HEALTHY" if healthy and "healthy" in disease_name.lower()
        else "DISEASED"
    )
    print(f"\nStatus: {status}")
    if is_unknown:
        print("Please upload a clear photo of a plant leaf.")


if __name__ == "__main__":
    main()
