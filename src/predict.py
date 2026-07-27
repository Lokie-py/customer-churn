import argparse
import os
import json
import joblib
import pandas as pd

def load_artifacts(artifacts_dir: str):
    model = joblib.load(os.path.join(artifacts_dir, "model.joblib"))
    preprocessor = joblib.load(os.path.join(artifacts_dir, "preprocessor.joblib"))
    with open(os.path.join(artifacts_dir, "schema.json"), "r", encoding="utf-8") as f:
        schema = json.load(f)
    return model, preprocessor, schema

def align_columns(df: pd.DataFrame, expected_cols: list):
    current_cols = df.columns.tolist()
    missing_cols = [c for c in expected_cols if c not in current_cols]
    extra_cols = [c for c in current_cols if c not in expected_cols]

    for c in missing_cols:
        df[c] = pd.NA

    df = df[expected_cols]
    return df, missing_cols, extra_cols

def clean_input(df: pd.DataFrame):
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df

def predict_batch(input_csv: str, output_csv: str, artifacts_dir: str = "artifacts"):
    model, preprocessor, schema = load_artifacts(artifacts_dir)
    expected_cols = schema["raw_columns"]

    df = pd.read_csv(input_csv)
    original_df = df.copy()

    df = clean_input(df)
    df_aligned, missing_cols, extra_cols = align_columns(df, expected_cols)

    X_processed = preprocessor.transform(df_aligned)
    proba = model.predict_proba(X_processed)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = original_df.copy()
    out["churn_probability"] = proba.round(4)
    out["churn_prediction"] = pred
    out["churn_label"] = out["churn_prediction"].map({0: "No", 1: "Yes"})

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    out.to_csv(output_csv, index=False)

    print("Prediction completed ✅")
    print(f"Input rows: {len(df)}")
    print(f"Output saved: {output_csv}")
    print(f"Missing columns auto-added: {missing_cols}")
    print(f"Extra columns ignored: {extra_cols}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Artifacts directory")
    args = parser.parse_args()

predict_batch(args.input_csv, args.output_csv, args.artifacts_dir)
