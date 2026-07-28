import json
from pathlib import Path
import joblib

artifacts = Path("artifacts")
candidates = [
    artifacts / "preprocessor.joblib",
    artifacts / "best_day5_model.joblib",
    artifacts / "model.joblib",
    artifacts / "classifier.joblib",
]

feature_cols = None
source = None

for p in candidates:
    if not p.exists():
        continue

    obj = joblib.load(p)

    if hasattr(obj, "feature_names_in_"):
        feature_cols = [str(c) for c in obj.feature_names_in_]
        source = f"{p.name}.feature_names_in_"
        break

    if hasattr(obj, "get_feature_names_out"):
        try:
            feature_cols = [str(c) for c in obj.get_feature_names_out()]
            source = f"{p.name}.get_feature_names_out()"
            break
        except Exception:
            pass

if not feature_cols:
    raise RuntimeError(
        "Could not extract feature columns from artifacts. "
        "Need feature_names_in_ or get_feature_names_out()."
    )

out_path = artifacts / "feature_columns.json"
out_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

print(f"Created: {out_path}")
print(f"Columns: {len(feature_cols)}")
print(f"Source: {source}")
