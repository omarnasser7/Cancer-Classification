"""
Cancer Classification API.

Serves the four models trained in the Colab notebooks (colon, lung, leukemia, skin)
behind a single HTTP endpoint, using exactly the preprocessing each model was
trained with.

Run locally:
    pip install -r requirements.txt
    python app.py

Endpoints:
    GET  /                     service info
    GET  /health               liveness check
    GET  /models               list registered models and their classes
    GET  /models/<type>        details for one model
    POST /predict              form-data: image=<file>, cancer_type=<type>
    POST /predict/<type>       form-data: image=<file>

These models are for research and academic use only. They are not a medical
diagnostic tool and must not be used to make clinical decisions.
"""

import os

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from model_registry import ModelNotFoundError, Registry
from preprocessing import PreprocessingError

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.path.dirname(__file__), "models"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "16"))
WARM_UP = os.environ.get("WARM_UP", "0") == "1"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FILE_FIELDS = ("image", "file")

DISCLAIMER = (
    "Research and academic use only. This is not a medical diagnostic tool."
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["JSON_SORT_KEYS"] = False

registry = Registry(MODELS_DIR)
if WARM_UP:
    print("Warming up all models...")
    registry.warm_up()


# --------------------------------------------------------------------------- helpers
def _error(message, status, **extra):
    payload = {"error": message}
    payload.update(extra)
    return jsonify(payload), status


def _extract_file():
    """Return the uploaded file from either the 'image' or 'file' field."""
    for field in FILE_FIELDS:
        if field in request.files:
            uploaded = request.files[field]
            if uploaded and uploaded.filename:
                return uploaded
    return None


def _check_extension(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise PreprocessingError(
            f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )


def _run_prediction(cancer_type):
    uploaded = _extract_file()
    if uploaded is None:
        return _error(
            "No image uploaded. Send a multipart/form-data request with an "
            "'image' field.",
            400,
        )

    try:
        bundle = registry.get(cancer_type)
    except ModelNotFoundError:
        return _error(
            f"Unknown cancer_type '{cancer_type}'.",
            404,
            available=registry.available,
        )

    try:
        _check_extension(uploaded.filename)
        image_bytes = uploaded.read()
        if not image_bytes:
            return _error("The uploaded file is empty.", 400)

        top_k = request.form.get("top_k") or request.args.get("top_k")
        result = bundle.predict(image_bytes, top_k=top_k)
    except PreprocessingError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return _error(f"Prediction failed: {type(exc).__name__}", 500)

    result["filename"] = uploaded.filename
    result["disclaimer"] = DISCLAIMER
    return jsonify(result), 200


# --------------------------------------------------------------------------- routes
@app.get("/")
def index():
    return jsonify(
        {
            "service": "Cancer Classification API",
            "version": "1.0",
            "available_models": registry.available,
            "endpoints": {
                "GET /health": "liveness check",
                "GET /models": "list all registered models",
                "GET /models/<cancer_type>": "details for one model",
                "POST /predict": "form-data: image=<file>, cancer_type=<type>",
                "POST /predict/<cancer_type>": "form-data: image=<file>",
            },
            "disclaimer": DISCLAIMER,
        }
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok" if registry.available else "no models loaded",
            "models_dir": MODELS_DIR,
            "registered": registry.available,
            "loaded": [c for c in registry.available if registry.get(c).is_loaded],
        }
    ), (200 if registry.available else 503)


@app.get("/models")
def list_models():
    return jsonify(
        {
            "count": len(registry.available),
            "models": [registry.get(c).describe() for c in registry.available],
        }
    )


@app.get("/models/<cancer_type>")
def model_details(cancer_type):
    try:
        return jsonify(registry.get(cancer_type).describe())
    except ModelNotFoundError:
        return _error(
            f"Unknown cancer_type '{cancer_type}'.",
            404,
            available=registry.available,
        )


@app.post("/predict")
def predict():
    cancer_type = request.form.get("cancer_type") or request.args.get("cancer_type")
    if not cancer_type:
        return _error(
            "Missing 'cancer_type'.",
            400,
            available=registry.available,
        )
    return _run_prediction(cancer_type)


@app.post("/predict/<cancer_type>")
def predict_typed(cancer_type):
    return _run_prediction(cancer_type)


# --------------------------------------------------------------------------- errors
@app.errorhandler(413)
def too_large(_):
    return _error(f"File too large. Maximum size is {MAX_UPLOAD_MB} MB.", 413)


@app.errorhandler(404)
def not_found(_):
    return _error("Endpoint not found.", 404)


@app.errorhandler(405)
def method_not_allowed(_):
    return _error("Method not allowed for this endpoint.", 405)


@app.errorhandler(Exception)
def unhandled(exc):
    if isinstance(exc, HTTPException):
        return _error(exc.description, exc.code)
    app.logger.exception("Unhandled error")
    return _error("Internal server error.", 500)


if __name__ == "__main__":
    if not registry.available:
        print(
            f"\nNo models found in {MODELS_DIR}\n"
            "Copy the Deploy/<cancer_type> folders from Google Drive into that "
            "directory, then restart.\n"
        )
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("DEBUG", "0") == "1")
