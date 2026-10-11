"""
Simple client for testing a running API instance.

Usage:
    python test_api.py                                  run the health and models checks
    python test_api.py path/to/image.jpg colon          predict one image
    python test_api.py path/to/folder colon             predict every image in a folder
"""

import os
import sys

import requests

BASE_URL = os.environ.get("API_URL", "http://127.0.0.1:5000")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def check_service():
    print(f"Checking {BASE_URL}")
    health = requests.get(f"{BASE_URL}/health", timeout=30)
    print("  /health ->", health.status_code, health.json())

    models = requests.get(f"{BASE_URL}/models", timeout=30).json()
    print(f"  {models['count']} model(s) registered:")
    for m in models["models"]:
        print(
            f"    - {m['cancer_type']:<10} {m['architecture']:<10} "
            f"size={m['img_size']} preprocessing={m['preprocessing']}"
        )
        print(f"      classes: {m['classes']}")


def predict_one(image_path, cancer_type):
    with open(image_path, "rb") as fh:
        response = requests.post(
            f"{BASE_URL}/predict",
            files={"image": (os.path.basename(image_path), fh)},
            data={"cancer_type": cancer_type},
            timeout=120,
        )

    name = os.path.basename(image_path)
    if response.status_code != 200:
        print(f"  {name:<35} ERROR {response.status_code}: {response.json()}")
        return None

    body = response.json()
    print(
        f"  {name:<35} {body['prediction']:<28} "
        f"{body['confidence']:.4f}  ({body['inference_time_ms']} ms)"
    )
    return body


def main():
    if len(sys.argv) == 1:
        check_service()
        return

    if len(sys.argv) < 3:
        print("Usage: python test_api.py <image_or_folder> <cancer_type>")
        sys.exit(1)

    target, cancer_type = sys.argv[1], sys.argv[2]

    if os.path.isdir(target):
        files = sorted(
            os.path.join(target, f)
            for f in os.listdir(target)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        print(f"Predicting {len(files)} image(s) as '{cancer_type}':")
        for path in files:
            predict_one(path, cancer_type)
    else:
        result = predict_one(target, cancer_type)
        if result:
            print("\nFull probabilities:")
            for label, score in sorted(
                result["probabilities"].items(), key=lambda kv: kv[1], reverse=True
            ):
                print(f"  {label:<30} {score:.4f}")


if __name__ == "__main__":
    main()
