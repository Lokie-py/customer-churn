from pathlib import Path
import json
from pyexpat import model
import joblib
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def resolve_data_path(project_root: Path) -> Path:
    """
    Resolve the churn CSV path robustly.
    Tries common filenames first, then any CSV in data/raw.
    """
    raw_dir = project_root / "data" / "raw"

    preferred_names = [
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "Telco-Customer-Churn.csv",
    ]

    for name in preferred_names:
        candidate = raw_dir / name
        if candidate.exists():
            return candidate

    csv_files = sorted(raw_dir.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV found in: {raw_dir}")

    raise FileNotFoundError(
        f"Multiple CSV files found in {raw_dir}. Please keep only one or set exact filename."
    )

def load_data(data_path: Path):
    df = pd.read_csv(data_path)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "Churn" not in df.columns:
        raise ValueError("Column 'Churn' not found in dataset.")

# Normalize churn labels safely
    df["Churn"] = (
        df["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    if df["Churn"].isna().any():
        bad_values = df.loc[df["Churn"].isna(), "Churn"]
        raise ValueError(
            "Found invalid values in 'Churn' column after mapping to Yes/No."
        )

# Convert TotalCharges if present
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing numeric values
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)
    return X, y

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X_test)
        denom = (raw_scores.max() - raw_scores.min()) + 1e-9
        y_scores = (raw_scores - raw_scores.min()) / denom
    else:
        y_scores = y_pred

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_scores), 4),
    }

def main():
    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data_path = resolve_data_path(project_root)
    print(f"Using dataset: {data_path}")

    X, y = load_data(data_path)
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "svm_rbf": SVC(probability=True, random_state=42),
    }

    results = {}
    best_model_name = None
    best_auc = -1.0
    best_model_obj = None

    for name, model in models.items():
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[name] = metrics
        print(f"{name}: {metrics}")

    if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model_name = name
            best_model_obj = model

    summary = {
        "best_model": best_model_name,
        "best_roc_auc": best_auc,
        "all_results": results,
    }

    comparison_path = artifacts_dir / "model_comparison.json"
    model_path = artifacts_dir / "best_day5_model.joblib"

    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    joblib.dump(best_model_obj, model_path)

    print("\nSaved:")
    print(f"- {comparison_path}")
    print(f"- {model_path}")

if __name__ == "__main__":
        main()