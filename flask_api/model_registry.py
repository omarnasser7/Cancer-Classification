"""
Model registry.

Loads the deployment bundles exported by the Colab notebooks. Each bundle is a folder
holding one saved model plus the metadata that describes how it must be fed:

    models/
    |-- colon/
    |   |-- colon_best.keras
    |   `-- model_meta.json
    |-- lung/
    |-- leukemia/
    `-- skin/

Copy these folders straight from Google Drive: My Projects/<project>/Deploy/
"""

import json
import os
import threading
import time

import numpy as np

# Silence TensorFlow info logs before it is imported
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

from preprocessing import preprocess  # noqa: E402

META_FILENAME = "model_meta.json"
MODEL_EXTENSIONS = (".keras", ".h5")


class ModelNotFoundError(KeyError):
    """Raised when a requested cancer type has no bundle on disk."""


class Bundle:
    """One trained model plus its metadata, loaded lazily on first use."""

    def __init__(self, cancer_type, directory):
        self.cancer_type = cancer_type
        self.directory = directory
        self._model = None
        self._lock = threading.Lock()

        with open(os.path.join(directory, META_FILENAME)) as fh:
            self.meta = json.load(fh)

        self.class_names = self.meta["class_names"]
        self.img_size = int(self.meta["img_size"])
        self.preprocessing = self.meta["preprocessing"]
        self.architecture = self.meta.get("architecture", "unknown")
        self.model_path = self._find_model_file()

    def _find_model_file(self):
        for ext in MODEL_EXTENSIONS:
            for name in os.listdir(self.directory):
                if name.endswith(ext):
                    return os.path.join(self.directory, name)
        raise FileNotFoundError(
            f"No .keras or .h5 model file found in {self.directory}"
        )

    @property
    def is_loaded(self):
        return self._model is not None

    @property
    def model(self):
        """Load once, on first request, and reuse afterwards."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    print(f"Loading model: {self.model_path}")
                    self._model = tf.keras.models.load_model(self.model_path)
        return self._model

    def predict(self, image_bytes, top_k=None):
        batch = preprocess(image_bytes, self.img_size, self.preprocessing)

        started = time.perf_counter()
        probabilities = self.model.predict(batch, verbose=0)[0]
        elapsed_ms = (time.perf_counter() - started) * 1000

        index = int(np.argmax(probabilities))
        scores = {
            name: round(float(p), 6)
            for name, p in zip(self.class_names, probabilities)
        }

        result = {
            "cancer_type": self.cancer_type,
            "architecture": self.architecture,
            "prediction": self.class_names[index],
            "confidence": round(float(probabilities[index]), 6),
            "probabilities": scores,
            "inference_time_ms": round(elapsed_ms, 1),
        }

        if top_k:
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            result["top_k"] = [
                {"label": label, "confidence": score}
                for label, score in ranked[: int(top_k)]
            ]
        return result

    def describe(self):
        return {
            "cancer_type": self.cancer_type,
            "architecture": self.architecture,
            "classes": self.class_names,
            "img_size": self.img_size,
            "preprocessing": self.preprocessing,
            "model_file": os.path.basename(self.model_path),
            "loaded": self.is_loaded,
            "training_metrics": self.meta.get("metrics", {}),
        }


class Registry:
    """Discovers every bundle under the models directory."""

    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.bundles = {}
        self._discover()

    def _discover(self):
        if not os.path.isdir(self.models_dir):
            print(f"Models directory not found: {self.models_dir}")
            return

        for name in sorted(os.listdir(self.models_dir)):
            directory = os.path.join(self.models_dir, name)
            if not os.path.isdir(directory):
                continue
            if not os.path.exists(os.path.join(directory, META_FILENAME)):
                print(f"Skipping {name}: no {META_FILENAME}")
                continue
            try:
                bundle = Bundle(name.lower(), directory)
                self.bundles[bundle.cancer_type] = bundle
                print(
                    f"Registered '{bundle.cancer_type}' "
                    f"({bundle.architecture}, {len(bundle.class_names)} classes, "
                    f"{bundle.preprocessing})"
                )
            except Exception as exc:
                print(f"Failed to register {name}: {type(exc).__name__}: {exc}")

    @property
    def available(self):
        return sorted(self.bundles)

    def get(self, cancer_type):
        key = (cancer_type or "").strip().lower()
        if key not in self.bundles:
            raise ModelNotFoundError(key)
        return self.bundles[key]

    def warm_up(self):
        """Load every model up front instead of on first request."""
        for bundle in self.bundles.values():
            _ = bundle.model
