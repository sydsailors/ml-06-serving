r"""serve_medical.py

A FastAPI service that loads
a trained penguin species classifier
and exposes a /predict endpoint.

Author: Sydney Sailors
Date: 2026-08

Process:
    - Load a saved model from artifacts/.
    - Accept a POST request with penguin measurements.
    - Return the predicted species.

Data Source:
    - artifacts/model.joblib (trained in the notebook or app_case.py)

Terminal commands to run this service from the root project folder:

uv run fastapi dev src/mlstudio/serve_medical.py      # development (auto-reload)
uv run fastapi run src/mlstudio/serve_medical.py      # production

- OR -

uv run uvicorn mlstudio.serve_medical:app --reload    # development (auto-reload)
uv run uvicorn mlstudio.serve_medical:app             # production

Then send a request - open a new terminal and run

If macOS or Linux, use \ line continuation characters:

    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"bill_length_mm": 39.1, "bill_depth_mm": 18.7, "flipper_length_mm": 181, "body_mass_g": 3750}'

If Windows (PowerShell), use ` instead of \ for line continuation:

    curl -X POST http://127.0.0.1:8000/predict `
         -H "Content-Type: application/json" `
         -d '{"bill_length_mm": 39.1, "bill_depth_mm": 18.7, "flipper_length_mm": 181, "body_mass_g": 3750}'
"""

# === Section 1. IMPORTS ===

import logging
from pathlib import Path
from typing import Any, Final

from datafun_toolkit.logger import get_logger, log_header
from fastapi import FastAPI, HTTPException
import joblib  # for serializing and deserializing the model
from sklearn.ensemble import RandomForestRegressor

__all__ = ["app", "predict_from_features", "predict"]

# === Section 2. CONFIGURE LOGGER ===

LOG: logging.Logger = get_logger("M06", level="DEBUG")
log_header(LOG, "M06")

# === Section 3. CONSTANTS AND CONFIGURATION ===

# The path to the saved model artifact.
MODEL_PATH: Final[Path] = Path("artifacts") / "model.joblib"

# The feature columns the model was trained on.
# These must match exactly what was used during training.
FEATURE_COLS: Final[list[str]] = [
    "age",
    "bmi",
    "smoker",
    "diabetes",
    "hypertension",
    "heart_disease",
    "doctor_visits_per_year",
    "hospital_admissions",
    "medication_count",
    "previous_year_cost",
]

# === Section 4. LOAD THE MODEL ===

LOG.info(f"Loading model from: {MODEL_PATH}")

if not MODEL_PATH.exists():
    LOG.error(f"Model file not found: {MODEL_PATH}")
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run the training notebook or app_case.py first."
    )

MODEL = joblib.load(MODEL_PATH)
LOG.info("Model loaded successfully")

# === Section 5. CREATE THE APP ===

app = FastAPI(title="Medical Cost Prediction Regressor")

# === Section 6. DEFINE THE PREDICT ENDPOINT ===


def predict_from_features(
    model: RandomForestRegressor, payload: dict[str, Any]
) -> dict[str, Any]:
    """Pure prediction function - testable outside the web framework."""
    try:
        features = [float(payload[c]) for c in FEATURE_COLS]
    except KeyError as exc:
        raise ValueError(f"Missing required feature: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid feature value: {exc}") from exc

    prediction: float = float(model.predict([features])[0])

    LOG.info(f"Prediction made from features {features}: {prediction}")

    return {"predicted_annual_medical_cost": round(prediction, 2)}


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return predict_from_features(MODEL, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
