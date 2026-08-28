from __future__ import annotations

import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "dataset.csv"
MODEL_PATH = "medbuddy_model.joblib"

TARGET_COLUMN = "disease"

TEST_SIZE = 0.20
RANDOM_STATE = 42

CONFIDENCE_THRESHOLD = 0.55


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    # Remove empty rows
    df = df.dropna(
        subset=[TARGET_COLUMN]
    )

    # Remove duplicates
    df = df.drop_duplicates()

    # Clean target
    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
    )

    # Features
    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    # Force symptom columns to numeric
    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Replace missing values
    X = X.fillna(0)

    # Force binary symptom data
    X = X.astype(float)

    print("=" * 60)
    print("MEDBUDDY DATASET")
    print("=" * 60)

    print(
        f"Rows: {len(X)}"
    )

    print(
        f"Features: {len(X.columns)}"
    )

    print(
        f"Diseases: {y.nunique()}"
    )

    return X, y


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    print()
    print("Training MedBuddy model...")

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training completed."
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print()
    print("=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print()
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # ========================================================
    # SAVE COMPLETE ARTIFACT
    # ========================================================

    artifact = {

        "model": model,

        "feature_columns":
            list(X.columns),

        "classes":
            list(model.classes_),

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "metrics": {

            "accuracy":
                float(accuracy),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1":
                float(f1)
        },

        "version":
            "medbuddy_v2"
    }

    joblib.dump(
        artifact,
        MODEL_PATH
    )

    print()
    print("=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(
        MODEL_PATH
    )

    return artifact


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(
        MODEL_PATH
    ):
        print(
            "Model file not found."
        )

        print(
            "Training a new model..."
        )

        return train_model()

    try:

        artifact = joblib.load(
            MODEL_PATH
        )

        # Verify artifact
        if not isinstance(
            artifact,
            dict
        ):
            raise ValueError(
                "Invalid model artifact."
            )

        if "model" not in artifact:
            raise ValueError(
                "Model artifact is missing 'model'."
            )

        if "feature_columns" not in artifact:
            raise ValueError(
                "Model artifact is missing feature columns."
            )

        return artifact

    except Exception as error:

        print(
            f"Old model incompatible: {error}"
        )

        print(
            "Retraining model..."
        )

        return train_model()


# ============================================================
# PREDICTION
# ============================================================

def predict_disease(
    patient_data,
    model_path=MODEL_PATH,
    top_k=5,
    confidence_threshold=None
):

    # --------------------------------------------------------
    # Load artifact
    # --------------------------------------------------------

    artifact = load_model()

    model = artifact["model"]

    feature_columns = (
        artifact["feature_columns"]
    )

    # Use the caller-supplied threshold if given, otherwise fall
    # back to the value the model was trained/saved with.
    if confidence_threshold is None:
        threshold = artifact.get(
            "confidence_threshold",
            CONFIDENCE_THRESHOLD
        )
    else:
        threshold = confidence_threshold

    # --------------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------------

    patient_df = pd.DataFrame(
        [patient_data]
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Use exactly the same features that
    # were used during training.
    # --------------------------------------------------------

    clean_df = pd.DataFrame(
        0.0,
        index=[0],
        columns=feature_columns
    )

    for feature in feature_columns:

        if feature in patient_df.columns:

            value = patient_df.iloc[0][
                feature
            ]

            try:

                clean_df.loc[
                    0,
                    feature
                ] = float(value)

            except (
                ValueError,
                TypeError
            ):

                clean_df.loc[
                    0,
                    feature
                ] = 0.0

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            clean_df
        )[0]
    )

    classes = np.asarray(
        model.classes_
    )

    # --------------------------------------------------------
    # Sort probabilities
    # --------------------------------------------------------

    indices = np.argsort(
        probabilities
    )[::-1]

    top_k = min(
        top_k,
        len(indices)
    )

    top_predictions = []

    for index in indices[:top_k]:

        top_predictions.append(
            {
                "condition":
                    str(
                        classes[index]
                    ),

                "confidence":
                    round(
                        float(
                            probabilities[index]
                        ),
                        4
                    )
            }
        )

    # --------------------------------------------------------
    # Best prediction
    # --------------------------------------------------------

    best_index = indices[0]

    best_condition = str(
        classes[best_index]
    )

    confidence = float(
        probabilities[best_index]
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    if confidence >= 0.75:

        confidence_level = "high"

    elif confidence >= threshold:

        confidence_level = "moderate"

    else:

        confidence_level = "low"

    # --------------------------------------------------------
    # Safety fallback
    # --------------------------------------------------------

    if confidence >= threshold:

        prediction = (
            best_condition
        )

        safe_to_predict = True

    else:

        prediction = (
            "Insufficient evidence"
        )

        safe_to_predict = False

    return {

        "prediction":
            prediction,

        "confidence":
            round(
                confidence,
                4
            ),

        "confidence_level":
            confidence_level,

        "safe_to_predict":
            safe_to_predict,

        "threshold_used":
            round(
                float(threshold),
                4
            ),

        "top_predictions":
            top_predictions,

        "warning":
            (
                "This AI prediction is for "
                "educational/research purposes "
                "and is not a medical diagnosis. "
                "Consult a qualified healthcare "
                "professional."
            )
    }


# ============================================================
# DATASET QUALITY REPORT
# ============================================================

def dataset_quality_report():

    df = pd.read_csv(
        DATA_PATH
    )

    print("\n" + "=" * 60)
    print("DATASET QUALITY REPORT")
    print("=" * 60)

    print(
        f"Shape: {df.shape}"
    )

    print("\nMissing values:")
    print("-" * 60)

    missing = (
        df.isnull()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing = missing[
        missing > 0
    ]

    if len(missing):

        print(
            missing.to_string()
        )

    else:

        print(
            "No missing values."
        )

    print("\nDuplicate rows:")
    print("-" * 60)

    print(
        df.duplicated().sum()
    )

    if TARGET_COLUMN in df.columns:

        print("\nDisease distribution:")
        print("-" * 60)

        print(
            df[TARGET_COLUMN]
            .value_counts()
            .to_string()
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "Starting MedBuddy AI..."
    )

    train_model()

    print()
    print(
        "MedBuddy model is ready."
    )
