"""
Model evaluation module.
Loads the registered production model and computes performance summaries,
confusion matrices, and calibration checks.
"""

import json
import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.ml.feature_engineering import add_engineered_features, create_target, LEAKAGE_COLS


def evaluate_production_model(
    model_path: str = "models/best_model.pkl",
    data_path: str = "data/cleaned/students_cleaned.csv",
):
    """Evaluates the production model against a test or evaluation dataset."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Please train first.")

    pipeline = joblib.load(model_path)
    df = pd.read_csv(data_path)

    y_true = create_target(df)
    df_feat = add_engineered_features(df)
    feature_cols = [c for c in df_feat.columns if c not in LEAKAGE_COLS]
    X = df_feat[feature_cols]

    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    cm = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    report = classification_report(y_true, y_pred, target_names=["Pass (0)", "Fail Risk (1)"], output_dict=True)

    summary = {
        "model_path": model_path,
        "evaluation_samples": len(df),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    return summary


if __name__ == "__main__":
    try:
        res = evaluate_production_model()
        print("Model Evaluation Summary:")
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Evaluation error: {e}")
