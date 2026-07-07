from pathlib import Path

folders = [
    "churn-monitor/data/raw",
    "churn-monitor/data/processed",
    "churn-monitor/notebooks",
    "churn-monitor/src",
    "churn-monitor/artifacts",
]

files = [
    "churn-monitor/notebooks/01_eda.ipynb",
    "churn-monitor/src/train.py",
    "churn-monitor/src/monitor.py",
    "churn-monitor/src/predict.py",
    "churn-monitor/src/profile_baseline.py",
    "churn-monitor/src/utils.py",
    "churn-monitor/artifacts/pipeline.joblib",
    "churn-monitor/artifacts/schema.json",
    "churn-monitor/artifacts/baseline_profile.json",
    "churn-monitor/app.py",
    "churn-monitor/requirements.txt",
    "churn-monitor/README.md",
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

for file in files:
    Path(file).touch(exist_ok=True)

print("Project structure created successfully!")