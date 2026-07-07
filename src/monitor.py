# src/monitor.py

import argparse
import os
import json
import pandas as pd

def load_schema(schema_path: str):
    with open(schema_path, "r") as f:
        return json.load(f)

def get_quality_report(df: pd.DataFrame, expected_cols: list):
    current_cols = df.columns.tolist()

    missing_columns = [c for c in expected_cols if c not in current_cols]
    extra_columns = [c for c in current_cols if c not in expected_cols]

# Null report only for available expected columns
    common_cols = [c for c in expected_cols if c in current_cols]
    null_counts = df[common_cols].isnull().sum().to_dict()
    null_pct = ((df[common_cols].isnull().mean() * 100).round(2)).to_dict()

    report = {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "null_counts": null_counts,
        "null_percent": null_pct,
    }
    return report

def run_monitor(input_csv: str, output_json: str, schema_path: str = "../artifacts/schema.json"):
    schema = load_schema(schema_path)
    expected_cols = schema["raw_columns"]

    df = pd.read_csv(input_csv)

# Keep monitor cleaning consistent where needed
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    report = get_quality_report(df, expected_cols)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(report, f, indent=2)

    print("Monitoring report generated ✅")
    print(f"Saved: {output_json}")
    print(f"Rows: {report['row_count']}")
    print(f"Missing columns: {len(report['missing_columns'])}")
    print(f"Extra columns: {len(report['extra_columns'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Path to batch input CSV")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save monitor report JSON")
    parser.add_argument("--schema_path", type=str, default="../artifacts/schema.json", help="Path to schema.json")
    args = parser.parse_args()

run_monitor(args.input_csv, args.output_json, args.schema_path)
