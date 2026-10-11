# Cancer Classification Project - 4 Cancer Types x 4 Architectures

Deep learning pipeline for classifying colon, lung, leukemia, and skin cancer images.
Each cancer type is trained with four architectures: a CNN built from scratch,
MobileNet, VGG19, and ResNet50.

```
notebooks/
|-- colon_cancer_4_models.ipynb
|-- lung_cancer_4_models.ipynb
|-- leukemia_cancer_4_models.ipynb
|-- skin_cancer_4_models.ipynb
`-- 00_master_results_dashboard.ipynb    merges all results into one report

flask_api/
|-- app.py                 REST API serving the four trained models
|-- model_registry.py      loads the deployment bundles
|-- preprocessing.py       preprocessing identical to training
|-- test_api.py            client for testing a running instance
|-- requirements.txt
`-- models/                place the exported bundles here
```

---

## Part 1 - Training notebooks (Google Colab)

### Drive layout

```
MyDrive/
`-- Colab Notebooks/
    `-- My Projects/
        `-- Cancer Detection/
            |-- Colon Dataset/
            |   |-- Adenocarcinoma/
            |   `-- Benign Tissue/
            |-- Lung Dataset/
            |   |-- Adenocarcinoma/
            |   |-- Benign Tissue/
            |   `-- Squamous Cell Carcinoma/
            |-- Leukemia Dataset/
            |   `-- Original/
            |       |-- Benign/
            |       |-- Early/
            |       |-- Pre/
            |       `-- Pro/
            |-- Skin Dataset/
            |   |-- Benign/
            |   `-- Malignant/
            |-- Results/     created automatically
            `-- Deploy/      created automatically
```

If your project folder is named something other than `Cancer Detection`, change
`PROJECT_DIR` in the first code cell of each notebook. Nothing else needs editing.

### Notebook structure

Every notebook follows the same sequence:

Import Libraries -> Load Dataset -> EDA -> Visualization -> Data Cleaning ->
Train-Test Split -> Train Model 1..4 -> Evaluation 1..4 -> Model Comparison ->
Final Evaluation -> Explainability (Grad-CAM) -> Save Model

### Training configuration

The architectures, optimizers, and epoch counts match the original scripts exactly.

| Notebook | Optimizer | Epochs (CNN / MobileNet / VGG19 / ResNet50) | Batch | Backbones |
|---|---|---|---|---|
| Colon | Adam | 5 / 3 / 5 / 5 | 64 | frozen |
| Lung | Adam | 10 / 10 / 5 / 5 | 64 | frozen |
| Leukemia | SGD (lr=0.001) | 5 / 5 / 5 / 5 | 8 | MobileNet frozen, VGG19 and ResNet50 trainable |
| Skin | SGD (lr=0.001) | 15 / 15 / 15 / 15 | 8 | MobileNet frozen, VGG19 and ResNet50 trainable |

`IMG_SIZE` is 256 in all four notebooks.

### Configuration flags

| Flag | Location | Purpose |
|---|---|---|
| `PROJECT_DIR` | cell 0 | Drive path to the project folder |
| `IMG_SIZE` | cell 0 | input resolution, 256 by default |
| `EPOCHS` | cell 0 | epochs per model |
| `DROP_DUPLICATES` | Data Cleaning | remove byte-identical duplicate images |
| `REPRODUCE_ORIGINAL_SPLIT` | Train-Test Split (leukemia, skin) | see below |

#### About `REPRODUCE_ORIGINAL_SPLIT`

The original leukemia and skin scripts contained this split:

```python
train_images, test_images = train_test_split(df, test_size=0.3, random_state=42)
train_set,    val_set     = train_test_split(df, test_size=0.2, random_state=42)  # full df
```

The second line samples from the full dataframe rather than from `train_images`,
so test images end up inside the training set and the reported scores are inflated.

- `True` (default) reproduces the original behaviour and prints how many images
  overlap between train and test.
- `False` performs a correct sequential split with stratification.

Use `False` when you need numbers you can defend in the project defence.

### Notes

- Select a GPU runtime: Runtime -> Change runtime type -> T4 GPU.
- The lung notebook is the heaviest, since all 15000 images are held in one array.
  Use a High-RAM runtime, or lower `IMG_SIZE` for that notebook only.
- Each model is saved and then released from memory right after its evaluation,
  so the four models never occupy RAM at the same time.
- The skin dataset ships with a separate official `test` folder. The notebook uses
  the training folder and splits it internally, matching the original script. The
  official test folder is available if you want an external test set.

### Output files

```
Results/<cancer_type>/
|-- comparison.csv        one row per model: accuracy, precision, recall, F1, AUC,
|                         training time, parameter counts, per-class F1 and recall
|-- all_results.json      full classification report and confusion matrix
|-- summary.xlsx          comparison sheet plus one sheet per model
|-- class_names.json      exact softmax output order
|-- models/               all four models (.keras and .h5)
|-- figures/              training curves, confusion matrices, ROC, Grad-CAM
|-- histories/            per-epoch history as CSV
|-- reports/              classification reports, EDA summary, cleaning log
`-- predictions/          y_true and y_prob, so any analysis can be redone
                          without retraining

Deploy/<cancer_type>/
|-- <cancer_type>_best.keras
|-- <cancer_type>_best.h5
`-- model_meta.json       architecture, class order, image size, preprocessing mode
```

After running all four, `00_master_results_dashboard.ipynb` produces
`Results/master_comparison.csv`, `master_summary.xlsx`, and `master_table.md`
(a Markdown table ready to paste into the project book).

---

## Part 2 - Flask API

### Setup

1. Copy the four `Deploy/<cancer_type>` folders from Drive into `flask_api/models/`,
   keeping the folder names lowercase: `colon`, `lung`, `leukemia`, `skin`.
2. Install and run:

```bash
cd flask_api
pip install -r requirements.txt
python app.py
```

Production:

```bash
gunicorn -w 1 -t 300 -b 0.0.0.0:5000 app:app
```

Use a single worker unless you have the memory for one full model set per worker.

Environment variables: `MODELS_DIR`, `PORT`, `MAX_UPLOAD_MB`, `WARM_UP=1`
(load every model at startup instead of on first request), `DEBUG=1`.

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | service info and endpoint list |
| GET | `/health` | liveness check and which models are loaded |
| GET | `/models` | all registered models with their classes and metrics |
| GET | `/models/<cancer_type>` | details for one model |
| POST | `/predict` | form-data: `image=<file>`, `cancer_type=<type>` |
| POST | `/predict/<cancer_type>` | form-data: `image=<file>` |

Optional `top_k` parameter returns the ranked classes.

### Example

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -F "image=@sample.jpeg" \
     -F "cancer_type=colon"
```

```json
{
  "cancer_type": "colon",
  "architecture": "ResNet50",
  "prediction": "Colon benign",
  "confidence": 0.9873,
  "probabilities": {
    "Colon adenocarcinoma": 0.0127,
    "Colon benign": 0.9873
  },
  "inference_time_ms": 84.2,
  "filename": "sample.jpeg",
  "disclaimer": "Research and academic use only. This is not a medical diagnostic tool."
}
```

Testing a running instance:

```bash
python test_api.py                        # health and model listing
python test_api.py sample.jpeg colon      # single image
python test_api.py ./test_images colon    # every image in a folder
```

### Why preprocessing is driven by metadata

The two model families were trained with different preprocessing:

| Models | Mode | Training pipeline |
|---|---|---|
| Colon, Lung | `bgr_raw` | `cv2.imread` (BGR, 0-255), `cv2.resize`, no normalization |
| Leukemia, Skin | `mobilenet_v2` | RGB, nearest-neighbour resize, scaled to [-1, 1] |

`model_meta.json` records which mode each model needs, and `preprocessing.py`
dispatches on that value. Both paths are byte-for-byte identical to the training
pipeline, including channel order and resize interpolation. Class labels also come
from the metadata rather than being hardcoded, so the softmax output order can
never drift out of sync with the label names.

---

These models are for research and academic use only. They are not a medical
diagnostic tool and must not be used to make clinical decisions.
