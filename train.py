"""
train.py - Trains a CNN to classify crop diseases.

Usage:
    python train.py

Requirements:
    - A folder named "dataset/" containing one subfolder per class.
    - Each subfolder name is the class/disease name and contains images.

What this script does:
    1. Loads the dataset (class names are detected automatically).
    2. Splits data into training (80%) and validation (20%).
    3. Applies data augmentation to the training images.
    4. Builds a CNN and trains it.
    5. Plots and saves the training history graph.
    6. Saves the model to models/crop_disease_model.keras
       and the class names to models/class_names.json
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, avoids GUI problems on Windows
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import image_dataset_from_directory
from utils.ood import calibrate_ood_reference

# ------------------------- Configuration -------------------------
DATASET_DIR = "dataset"
MODEL_DIR = "models"
RESULTS_DIR = "results"
MODEL_PATH = os.path.join(MODEL_DIR, "crop_disease_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
OOD_REFERENCE_PATH = os.path.join(MODEL_DIR, "ood_reference.json")

IMAGE_SIZE = (128, 128)          # width, height (must match utils/preprocessing.py)
BATCH_SIZE = 16
EPOCHS = 40
VALIDATION_SPLIT = 0.2           # 20% of images are held out for validation
SEED = 42                        # makes the random split reproducible


# ------------------------- Helper functions -------------------------
def check_dataset_structure():
    """
    Make sure the dataset folder exists and contains class subfolders.
    Raises a beginner-friendly error otherwise.
    """
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Folder '{DATASET_DIR}' not found. Create it and put your images inside:\n"
            f"  {DATASET_DIR}/Class_Name/image1.jpg"
        )

    subdirs = [d for d in os.listdir(DATASET_DIR)
               if os.path.isdir(os.path.join(DATASET_DIR, d))]
    if not subdirs:
        raise ValueError(
            f"Folder '{DATASET_DIR}' is empty or has no class folders.\n"
            f"Structure must be:\n  {DATASET_DIR}/Class_Name/image1.jpg"
        )

    total_images = 0
    for cls in subdirs:
        count = sum(
            1 for f in os.listdir(os.path.join(DATASET_DIR, cls))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        total_images += count
        print(f"  - {cls}: {count} images")
        if count == 0:
            raise ValueError(
                f"Class folder '{cls}' contains no .jpg/.jpeg/.png images."
            )

    if total_images < 10:
        raise ValueError(
            f"Only {total_images} images found in total. "
            "Add more images (at least ~10 per class) for meaningful training."
        )


def load_datasets():
    """
    Load images from the dataset folder and split into train/validation.
    Class names are detected automatically from the folder names.
    """
    train_ds = image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    val_ds = image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    # Class names are found automatically from the subfolder names.
    return train_ds, val_ds, train_ds.class_names


def apply_augmentation(train_ds):
    """
    Data augmentation: slightly flip / rotate / zoom images.
    This makes the model more robust and works around small datasets.
    """
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )
    return train_ds.prefetch(buffer_size=AUTOTUNE)


def build_cnn(num_classes):
    """
    Build a transfer-learning model:
    MobileNetV2 (pretrained on ImageNet) extracts features, then a small
    dense head classifies the disease. The base model is frozen so the
    pretrained knowledge is kept and only the new head is trained.
    """
    base_model = MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = models.Sequential([
        layers.Rescaling(1.0 / 255.0, input_shape=IMAGE_SIZE + (3,)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_history(history):
    """Save accuracy and loss curves as results/training_history.png."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"], label="Train loss")
    axes[1].plot(history.history["val_loss"], label="Validation loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    graph_path = os.path.join(RESULTS_DIR, "training_history.png")
    fig.savefig(graph_path)
    plt.close(fig)
    print(f"\nTraining graphs saved to: {graph_path}")


# ------------------------- Main -------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("CROP DISEASE DETECTION - MODEL TRAINING")
    print("=" * 60)

    # 1. Validate the dataset structure
    print("\n[1/5] Checking dataset...")
    check_dataset_structure()

    # 2. Load and split the data
    print("\n[2/5] Loading dataset and splitting train/validation...")
    train_ds, val_ds, class_names = load_datasets()
    print(f"Classes detected: {class_names}")

    # 3. Apply data augmentation (training data only)
    print("\n[3/5] Applying data augmentation...")
    train_ds = apply_augmentation(train_ds)
    AUTOTUNE = tf.data.AUTOTUNE
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # 4. Build and train the CNN
    print("\n[4/5] Building CNN and training...")
    model = build_cnn(num_classes=len(class_names))
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        ],
    )

    # 5. Save model + class names, plot graphs
    print("\n[5/5] Saving model and results...")
    model.save(MODEL_PATH)
    print(f"Model saved to:    {MODEL_PATH}")

    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=4)
    print(f"Class names saved to: {CLASS_NAMES_PATH}")

    print("Calibrating the non-leaf input rejection check...")
    calibrate_ood_reference(
        model,
        DATASET_DIR,
        class_names,
        OOD_REFERENCE_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        seed=SEED,
    )
    print(f"Input rejection reference saved to: {OOD_REFERENCE_PATH}")

    plot_training_history(history)

    final_acc = history.history["val_accuracy"][-1]
    print(f"\nTraining complete! Final validation accuracy: {final_acc:.2%}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as err:
        print(f"\nERROR: {err}", file=sys.stderr)
        sys.exit(1)
