import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

def _resolve_artifacts_dir(artifacts_dir: str) -> Path:
    """
    Resolve artifacts path from either:
    - absolute path
    - current working directory
    - project root (parent of src/)
    """
    p = Path(artifacts_dir)
    if p.is_absolute():
        return p

    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate

    project_root = Path(__file__).resolve().parents[1]
    root_candidate = project_root / p
    return root_candidate

def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _extract_feature_columns_from_schema(schema_obj):
    """
    Accepts schema as:
    - list[str]
    - dict with keys like feature_columns / columns / features / input_features
    """
    if isinstance(schema_obj, list):
        return [str(c) for c in schema_obj]

    if isinstance(schema_obj, dict):
        for key in ["feature_columns", "columns", "features", "input_features"]:
            val = schema_obj.get(key)
            if isinstance(val, list):
                return [str(c) for c in val]

    return None

def load_feature_columns(artifacts_path: Path):
    """
    Priority:
    1) feature_columns.json
    2) schema.json (if compatible)
    """
    feature_columns_path = artifacts_path / "feature_columns.json"
    if feature_columns_path.exists():
        obj = _load_json(feature_columns_path)
        cols = _extract_feature_columns_from_schema(obj)
        if cols:
            return cols, str(feature_columns_path)

    schema_path = artifacts_path / "schema.json"
    if schema_path.exists():
        obj = _load_json(schema_path)
        cols = _extract_feature_columns_from_schema(obj)
        if cols:
            return cols, str(schema_path)

    return None, None

def load_first_existing(artifacts_path: Path, candidates):
    for name in candidates:
        p = artifacts_path / name
        if p.exists():
            return joblib.load(p), p
    return None, None

def clean_input_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    """
    Known Telco fix:
    TotalCharges may contain blank strings -> coerce to numeric NaN
    """
    X = X.copy()
    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(
            X["TotalCharges"].astype(str).str.strip().replace("", pd.NA),
            errors="coerce",
        )
    return X

def align_to_feature_columns(X: pd.DataFrame, feature_cols):
    """
    Align input to exact feature order expected by training artifacts.
    - missing columns are added as NaN
    - extra columns are dropped
    """
    current_cols = set(X.columns)
    expected_cols = list(feature_cols)

    missing = [c for c in expected_cols if c not in current_cols]
    extra = [c for c in X.columns if c not in set(expected_cols)]

    for c in missing:
        X[c] = np.nan

    X_aligned = X[expected_cols]
    return X_aligned, missing, extra

def _safe_probability(model, X_model):
    """
    Return class-1 probability.
    Prefers predict_proba, falls back to sigmoid(decision_function) if needed.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_model)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_model)
        # sigmoid
        return 1.0 / (1.0 + np.exp(-scores))

# Last resort: hard labels -> float
    preds = model.predict(X_model)
    return np.asarray(preds, dtype=float)

def predict_batch(input_csv, output_csv, artifacts_dir="artifacts", threshold=0.40):
    artifacts_path = _resolve_artifacts_dir(artifacts_dir)
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_path}")

# Load model (priority order)
    model, model_path = load_first_existing(
        artifacts_path,
        candidates=[
            "best_day5_model.joblib",
            "model.joblib",
            "classifier.joblib",
        ],
    )
    if model is None:
        raise FileNotFoundError(
            f"No model found in {artifacts_path}. "
            "Expected one of: best_day5_model.joblib, model.joblib, classifier.joblib"
        )

# Load preprocessor (optional, but usually required)
    preprocessor, preprocessor_path = load_first_existing(
        artifacts_path,
        candidates=["preprocessor.joblib", "pipeline.joblib"],
    )

# Load data
    df = pd.read_csv(input_csv)

# Keep original columns for output, create feature frame separately
    X = df.copy()
    if "Churn" in X.columns:
        X = X.drop(columns=["Churn"], errors="ignore")

# Clean known problematic columns
    X = clean_input_dataframe(X)

# Feature-column alignment (preferred from artifact metadata)
    feature_cols, feature_cols_source = load_feature_columns(artifacts_path)

# If JSON not available, try preprocessor/model feature_names_in_
    if feature_cols is None and preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        feature_cols = [str(c) for c in preprocessor.feature_names_in_]
        feature_cols_source = "preprocessor.feature_names_in_"
    if feature_cols is None and hasattr(model, "feature_names_in_"):
        feature_cols = [str(c) for c in model.feature_names_in_]
        feature_cols_source = "model.feature_names_in_"

    missing_cols = []
    extra_cols = []

    if feature_cols is not None:
        X, missing_cols, extra_cols = align_to_feature_columns(X, feature_cols)

# Transform
    X_model = preprocessor.transform(X) if preprocessor is not None else X

# --- Guard: model/preprocessor feature mismatch ---
    model_expected = getattr(model, "n_features_in_", None)
    if model_expected is not None:
        got_features = X_model.shape[1]
        if got_features != model_expected:
            raise ValueError(
                f"Artifact mismatch: transformed input has {got_features} features, "
                f"but model expects {model_expected}. "
                "Use matching preprocessor.joblib + model.joblib from the same training run."
            )

# Predict
    churn_prob = _safe_probability(model, X_model)
    churn_pred = (churn_prob >= float(threshold)).astype(int)
    churn_label = np.where(churn_pred == 1, "Yes", "No")

# Build output
    out_df = df.copy()
    out_df["churn_probability"] = churn_prob
    out_df["churn_prediction"] = churn_pred
    out_df["churn_label"] = churn_label

# Save
    output_path = Path(output_csv)
    if output_path.parent and str(output_path.parent) not in ("", "."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

# Summary logs
    print("Prediction completed ✅")
    print(f"Threshold used: {threshold}")
    print(f"Input rows: {len(df)}")
    print(f"Output saved: {output_path}")
    print(f"Model used: {model_path.name}")
    print(
        f"Preprocessor used: {preprocessor_path.name if preprocessor_path else 'None (direct model input)'}"
    )
    print(f"Feature source: {feature_cols_source if feature_cols_source else 'input columns'}")
    print(f"Missing columns auto-added: {missing_cols}")
    print(f"Extra columns ignored: {extra_cols}")

def parse_args():
    parser = argparse.ArgumentParser(description="Batch churn prediction.")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV")
    parser.add_argument("--output_csv", required=True, help="Path to output CSV")
    parser.add_argument(
        "--artifacts_dir",
        default="artifacts",
        help="Artifacts directory path (default: artifacts)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="Classification threshold in [0.0, 1.0] (default: 0.40)",
    )
    args = parser.parse_args()

    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError("--threshold must be between 0.0 and 1.0")

    return args

if __name__ == "__main__":
    args = parse_args()
    predict_batch(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        artifacts_dir=args.artifacts_dir,
        threshold=args.threshold,
    )
