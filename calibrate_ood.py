"""Create the non-leaf rejection reference for the currently saved model."""

import json
import os

import tensorflow as tf

from utils.ood import calibrate_ood_reference


DATASET_DIR = "dataset"
MODEL_PATH = os.path.join("models", "crop_disease_model.keras")
CLASS_NAMES_PATH = os.path.join("models", "class_names.json")
REFERENCE_PATH = os.path.join("models", "ood_reference.json")
IMAGE_SIZE = (128, 128)
VALIDATION_SPLIT = 0.2
SEED = 42


def main():
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(CLASS_NAMES_PATH):
        raise FileNotFoundError("Train the model first with `python train.py`.")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, encoding="utf-8") as names_file:
        class_names = json.load(names_file)

    calibrate_ood_reference(
        model,
        DATASET_DIR,
        class_names,
        REFERENCE_PATH,
        image_size=IMAGE_SIZE,
        validation_split=VALIDATION_SPLIT,
        seed=SEED,
    )
    print(f"Input rejection reference saved to: {REFERENCE_PATH}")


if __name__ == "__main__":
    main()
