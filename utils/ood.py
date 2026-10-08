"""A simple embedding-distance rejection check for non-training images."""

import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory


DEFAULT_IMAGE_SIZE = (128, 128)
DEFAULT_BATCH_SIZE = 32
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_SEED = 42


def build_feature_extractor(model):
    """Return the trained model's pooled MobileNetV2 feature vectors."""
    pooling_layer = next(
        (layer for layer in model.layers
         if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)),
        None,
    )
    if pooling_layer is None:
        raise ValueError("The trained model has no global average pooling layer.")
    return tf.keras.Model(inputs=model.inputs, outputs=pooling_layer.output)


def _collect_features(feature_model, dataset):
    features = []
    labels = []
    for images, batch_labels in dataset:
        batch_features = feature_model(images, training=False).numpy()
        features.append(batch_features)
        labels.append(batch_labels.numpy())
    if not features:
        raise ValueError("No images were available to calibrate the input check.")
    return np.concatenate(features), np.concatenate(labels)


def calibrate_ood_reference(
    model,
    dataset_dir,
    class_names,
    output_path,
    image_size=DEFAULT_IMAGE_SIZE,
    batch_size=DEFAULT_BATCH_SIZE,
    validation_split=DEFAULT_VALIDATION_SPLIT,
    seed=DEFAULT_SEED,
):
    """Build per-class feature prototypes and validation-based reject limits."""
    feature_model = build_feature_extractor(model)
    dataset_options = dict(
        directory=dataset_dir,
        validation_split=validation_split,
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
        class_names=class_names,
        # Match train.py's seeded, shuffled image-level split so each class
        # appears in both train and validation subsets.
        shuffle=True,
    )
    train_ds = image_dataset_from_directory(subset="training", **dataset_options)
    val_ds = image_dataset_from_directory(subset="validation", **dataset_options)

    train_features, train_labels = _collect_features(feature_model, train_ds)
    val_features, val_labels = _collect_features(feature_model, val_ds)
    train_features = tf.math.l2_normalize(train_features, axis=1).numpy()
    val_features = tf.math.l2_normalize(val_features, axis=1).numpy()

    prototypes = []
    thresholds = []
    for class_index, class_name in enumerate(class_names):
        class_features = train_features[train_labels == class_index]
        class_validation = val_features[val_labels == class_index]
        if len(class_features) == 0 or len(class_validation) == 0:
            raise ValueError(f"Cannot calibrate class '{class_name}' without train and validation images.")

        prototype = class_features.mean(axis=0)
        prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
        similarities = class_validation @ prototype
        prototypes.append(prototype.tolist())
        # Reject only below the 5th percentile of held-out known-class examples.
        thresholds.append(float(np.percentile(similarities, 5)))

    reference = {
        "class_names": list(class_names),
        "prototypes": prototypes,
        "thresholds": thresholds,
        "method": "cosine similarity to class feature prototypes; 5th percentile validation thresholds",
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(reference, output_file)
    return reference


def load_ood_reference(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as reference_file:
        return json.load(reference_file)


def check_supported_input(feature_model, image_batch, class_names, reference):
    """Return (accepted, closest class, similarity, threshold) for one image."""
    if not reference or reference.get("class_names") != list(class_names):
        return True, None, None, None

    embedding = feature_model(image_batch, training=False).numpy().reshape(-1)
    embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
    prototypes = np.asarray(reference["prototypes"], dtype=np.float32)
    scores = prototypes @ embedding
    closest_index = int(np.argmax(scores))
    score = float(scores[closest_index])
    # Leave room for real leaf photos with different phones, backgrounds, and
    # lighting from the controlled PlantVillage training images.
    threshold = max(float(reference["thresholds"][closest_index]) - 0.15, 0.45)
    return score >= threshold, class_names[closest_index], score, threshold
