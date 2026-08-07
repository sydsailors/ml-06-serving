"""model_builder_medical.py

Trains and saves a Medical Cost Prediction regression.

Author: Sydney Sailors
Date: 2026-08

Process:
    - Load the Medical Cost Prediction dataset.
    - Split into train and test sets.
    - Train a RandomForestRegressor.
    - Evaluate on held-out test data.
    - Save the trained model to artifacts/model.joblib.

Terminal command to run this file from the root project folder:

    uv run python -m mlstudio.model_builder_medical
"""

# === Section 1a. DECLARE IMPORTS ===

import logging
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# === Section 1b. CONFIGURE LOGGER ONCE PER MODULE ===

LOG: logging.Logger = get_logger("M06", level="DEBUG")
log_header(LOG, "M06")

# === Section 1c. GLOBAL CONSTANTS AND CONFIGURATION ===

DATASET_NAME: Final[str] = "Medical Cost Prediction"

DATA_PATH: Final[Path] = Path("data/raw/medical_cost_prediction_dataset.csv")

# STEP 1. Pick the target variable we want to predict.
TARGET_COL: Final[str] = "annual_medical_cost"

# STEP 2. Define the feature columns used to predict the target.
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

# STEP 3. Define the test size and random state for reproducibility.
TEST_SIZE: Final[float] = 0.20
RANDOM_STATE: Final[int] = 42

# STEP 4. Define where the trained model artifact will be saved.
MODEL_PATH: Final[Path] = Path("artifacts") / "model.joblib"

# === Section 1d. PANDAS DISPLAY CONFIGURATION ===

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)


# === Section 2. Load the Data ===


def load_data() -> pd.DataFrame:
    """Load the medical cost prediction dataset."""

    LOG.info(f"Loading dataset: {DATASET_NAME}")

    df: pd.DataFrame = pd.read_csv(DATA_PATH)

    LOG.info(f"Loaded: {df.shape[0]} rows (instances), {df.shape[1]} columns")

    # Convert smoker yes/no to 1/0
    df["smoker"] = df["smoker"].map({"Yes": 1, "No": 0})

    required: list[str] = [TARGET_COL, *FEATURE_COLS]

    df_model: pd.DataFrame = df.dropna(subset=required).copy()

    LOG.info(f"Model rows after cleaning: {df_model.shape[0]}")

    return df_model


# === Section 3. Split into Train and Test ===


def split_data(
    df_model: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into stratified train and test sets.

    WHY: Scoring on held-out data is the only honest estimate
    of performance on new data.
    Stratifying preserves the class balance in both splits.
    """
    X: pd.DataFrame = df_model[FEATURE_COLS]
    y: pd.Series = df_model[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    LOG.info(f"Train instances: {len(X_train)}")
    LOG.info(f"Test instances:  {len(X_test)}")

    return X_train, X_test, y_train, y_test


# === Section 4. Train the Model ===


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Train a RandomForestRegressor on the training data."""
    LOG.info(f"Training RandomForestRegressor on {len(X_train)} instances")

    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    LOG.info("Training complete")
    return model


# === Section 5. Evaluate the Model ===


def evaluate_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Evaluate the model on held-out test data."""

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    LOG.info(f"Test MAE:  ${mae:,.2f}")
    LOG.info(f"Test RMSE: ${rmse:,.2f}")
    LOG.info(f"Test R²:   {r2:.3f}")


# === Section 6. Save the Model ===


def save_model(model: RandomForestRegressor) -> None:
    """Persist the trained model to disk with joblib.

    WHY: A server loads the saved artifact once at startup
    rather than retraining on every request.
    """
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    LOG.info(f"Saved model to: {MODEL_PATH}")


# === Section 7. Summary ===


def summarize() -> None:
    """Log a brief summary."""
    LOG.info("========================")
    LOG.info("SUMMARY")
    LOG.info("========================")
    LOG.info(f"Dataset:  {DATASET_NAME}")
    LOG.info(f"Target:   {TARGET_COL}")
    LOG.info(f"Features: {FEATURE_COLS}")
    LOG.info(f"Artifact: {MODEL_PATH}")
    LOG.info("========================")


# === MAIN ===


def main() -> None:
    """Run the model builder workflow."""

    LOG.info("Load data.................")
    df_model = load_data()

    LOG.info("Split data................")
    X_train, X_test, y_train, y_test = split_data(df_model)

    LOG.info("Train model...............")
    model = train_model(X_train, y_train)

    LOG.info("Evaluate model............")
    evaluate_model(model, X_test, y_test)

    LOG.info("Save model................")
    save_model(model)

    LOG.info("Summarize.................")
    summarize()

    LOG.info("========================")
    LOG.info("Executed successfully!")
    LOG.info("========================")


if __name__ == "__main__":
    main()
