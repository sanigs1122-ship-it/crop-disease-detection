"""
Preprocessing helpers shared by train.py, predict.py and app.py.

IMPORTANT:
- The CNN model is trained on raw pixel values (0-255).
- The very first layer of the model (layers.Rescaling) normalizes the
  images to 0-1, so preprocessing only needs to RESIZE the image.
- Using exactly the same resize here (128x128) in training and prediction
  guarantees the model always sees the same type of input.
"""

import io

import numpy as np
from PIL import Image

# This size MUST match the image_size used in train.py.
DEFAULT_IMAGE_SIZE = (128, 128)

# Image formats the app accepts.
SUPPORTED_EXTENSIONS = {"jpg", "jpeg", "png"}


def _to_model_input(pil_image, target_size):
    """Convert a PIL image into the (1, height, width, 3) array the model expects."""
    # convert("RGB") removes any alpha channel (e.g. PNG) so the image is always 3 channels.
    pil_image = pil_image.convert("RGB").resize(target_size)
    img_array = np.asarray(pil_image, dtype=np.float32)
    # Add the batch dimension -> shape (1, 128, 128, 3)
    return np.expand_dims(img_array, axis=0)


def load_and_preprocess_image(image_path, target_size=DEFAULT_IMAGE_SIZE):
    """
    Load an image file from disk and prepare it for the model.

    Args:
        image_path (str): full path to a JPG/PNG image.
        target_size (tuple): (width, height) expected by the CNN.

    Returns:
        np.ndarray of shape (1, target_size[0], target_size[1], 3).
    """
    pil_image = Image.open(image_path)
    return _to_model_input(pil_image, target_size)


def decode_uploaded_image(image_bytes):
    """
    Decode raw image bytes into an RGB PIL Image for display.

    Args:
        image_bytes (bytes): raw bytes of a JPG/JPEG/PNG image.

    Returns:
        PIL.Image.Image in RGB mode.

    Raises:
        Exception: if the bytes cannot be decoded as a valid image.
    """
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image.load()  # forces a full decode so corrupt files raise an error here
    return pil_image.convert("RGB")


def preprocess_uploaded_image(uploaded_file, target_size=DEFAULT_IMAGE_SIZE):
    """
    Prepare an image uploaded through the Streamlit file uploader.

    Args:
        uploaded_file: raw bytes or a file-like object from st.file_uploader.
        target_size (tuple): (width, height) expected by the CNN.

    Returns:
        np.ndarray of shape (1, target_size[0], target_size[1], 3).

    Raises:
        ValueError: if the file is not a valid image.
    """
    if isinstance(uploaded_file, (bytes, bytearray)):
        raw = uploaded_file
    else:
        raw = uploaded_file.read()
    pil_image = Image.open(io.BytesIO(raw))
    pil_image.load()  # forces a full decode so corrupt files raise an error here
    return _to_model_input(pil_image, target_size)


def is_supported_file(filename):
    """Return True if the file name ends with jpg, jpeg or png."""
    if not filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in SUPPORTED_EXTENSIONS