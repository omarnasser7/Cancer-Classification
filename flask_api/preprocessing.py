"""
Image preprocessing for inference.

Every function here mirrors the preprocessing used during training, byte for byte.
Getting this wrong is the single most common cause of an API that returns confident
nonsense while the notebook reports high accuracy.

Two modes exist, and each model declares its own mode in model_meta.json:

  "bgr_raw"       Colon and Lung models.
                  Training used cv2.imread() -> BGR channel order, values 0-255,
                  resized with cv2.resize (INTER_LINEAR), and NO normalization.

  "mobilenet_v2"  Leukemia and Skin models.
                  Training used ImageDataGenerator with
                  tf.keras.applications.mobilenet_v2.preprocess_input, which reads
                  images as RGB, resizes with PIL nearest-neighbour interpolation
                  (the Keras default), then scales to the [-1, 1] range.
"""

import io

import cv2
import numpy as np
from PIL import Image

VALID_MODES = ("bgr_raw", "mobilenet_v2")


class PreprocessingError(ValueError):
    """Raised when the uploaded bytes cannot be decoded as an image."""


def _decode_bgr(image_bytes):
    """Decode bytes the same way cv2.imread does: BGR, 3 channels, uint8."""
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise PreprocessingError("Could not decode the uploaded file as an image.")
    return image


def _decode_rgb(image_bytes):
    """Decode bytes the same way keras load_img does: RGB PIL image."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception as exc:
        raise PreprocessingError("Could not decode the uploaded file as an image.") from exc
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def preprocess_bgr_raw(image_bytes, img_size):
    """Colon / Lung: BGR, 0-255, no normalization."""
    image = _decode_bgr(image_bytes)
    image = cv2.resize(image, (img_size, img_size))
    return image.astype("float32")


def preprocess_mobilenet_v2(image_bytes, img_size):
    """Leukemia / Skin: RGB, nearest-neighbour resize, scaled to [-1, 1]."""
    image = _decode_rgb(image_bytes)
    # Keras load_img and ImageDataGenerator both default to nearest interpolation
    image = image.resize((img_size, img_size), Image.NEAREST)
    array = np.asarray(image, dtype="float32")
    # This is exactly what mobilenet_v2.preprocess_input does
    return array / 127.5 - 1.0


_DISPATCH = {
    "bgr_raw": preprocess_bgr_raw,
    "mobilenet_v2": preprocess_mobilenet_v2,
}


def preprocess(image_bytes, img_size, mode):
    """Return a batch of shape (1, img_size, img_size, 3) ready for model.predict."""
    if mode not in _DISPATCH:
        raise PreprocessingError(
            f"Unknown preprocessing mode '{mode}'. Expected one of {VALID_MODES}."
        )
    array = _DISPATCH[mode](image_bytes, img_size)
    return np.expand_dims(array, axis=0)
