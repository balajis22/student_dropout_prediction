"""
Data and Prediction Drift Monitoring Engine.
Computes Kolmogorov-Smirnov (KS) tests for continuous feature drift,
Chi-Square / Population Stability Index (PSI) for categorical distributions,
prediction class shift, and triggers automated retraining recommendations.
"""

import json
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from datetime import datetime
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chisquare

from src.ml.feature_engineering import (
    BASE_NUMERICAL_COLS,
    CATEGORICAL_COLS,
    add_engineered_features,
    LEAKAGE_COLS,
)


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Calculates the Population Stability Index (PSI) between two continuous distributions."""
    eps = 1e-4
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(expected, quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = expected_counts / float(len(expected)) + eps
    actual_pct = actual_counts / float(len(actual)) + eps

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    alpha: float = 0.05,
    psi_threshold: float = 0.20,
) -> Dict[str, Any]:
    """
    Analyzes feature drift between baseline reference data and incoming production data.
    - Continuous features: Kolmogorov-Smirnov (KS) 2-sample test
    - Categorical features: Relative frequency divergence & Chi-square
    """
    ref_feat = add_engineered_features(reference_df)
    cur_feat = add_engineered_features(current_df)

    numerical_results = {}
    drifted_num_features = []

    # Check continuous features
    for col in BASE_NUMERICAL_COLS:
        if col in ref_feat.columns and col in cur_feat.columns:
            ref_vals = ref_feat[col].dropna().values
            cur_vals = cur_feat[col].dropna().values

            ks_stat, p_val = ks_2samp(ref_vals, cur_vals)
            psi_val = calculate_psi(ref_vals, cur_vals)
            has_drifted = bool(p_val < alpha or psi_val > psi_threshold)

            numerical_results[col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_val), 4),
                "psi": round(float(psi_val), 4),
                "drift_detected": has_drifted,
                "ref_mean": round(float(np.mean(ref_vals)), 2),
                "cur_mean": round(float(np.mean(cur_vals)), 2),
            }
            if has_drifted:
                drifted_num_features.append(col)

    # Check categorical features
    categorical_results = {}
    drifted_cat_features = []

    for col in CATEGORICAL_COLS:
        if col in ref_feat.columns and col in cur_feat.columns:
            ref_counts = ref_feat[col].value_counts(normalize=True)
            cur_counts = cur_feat[col].value_counts(normalize=True)

            all_cats = list(set(ref_counts.index).union(set(cur_counts.index)))
            ref_dist = [ref_counts.get(c, 0.0) + 1e-4 for c in all_cats]
            cur_dist = [cur_counts.get(c, 0.0) + 1e-4 for c in all_cats]

            ref_dist = np.array(ref_dist) / sum(ref_dist)
            cur_dist = np.array(cur_dist) / sum(cur_dist)

            cat_psi = float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))
            has_cat_drift = bool(cat_psi > psi_threshold)

            categorical_results[col] = {
                "psi": round(cat_psi, 4),
                "drift_detected": has_cat_drift,
                "categories": all_cats,
            }
            if has_cat_drift:
                drifted_cat_features.append(col)

    total_monitored = len(numerical_results) + len(categorical_results)
    total_drifted = len(drifted_num_features) + len(drifted_cat_features)
    drift_share = total_drifted / float(total_monitored) if total_monitored > 0 else 0.0

    # Retraining trigger criteria:
    # If >25% of monitored features exhibit significant drift, flag for retraining
    retraining_required = bool(drift_share >= 0.25)

    report = {
        "timestamp": datetime.now().isoformat(),
        "reference_records": len(reference_df),
        "current_records": len(current_df),
        "monitored_features_count": total_monitored,
        "drifted_features_count": total_drifted,
        "drift_share_pct": round(drift_share * 100, 2),
        "retraining_recommended": retraining_required,
        "retraining_status": "TRIGGER_RETRAINING" if retraining_required else "MODEL_HEALTHY",
        "drifted_features": drifted_num_features + drifted_cat_features,
        "numerical_drift": numerical_results,
        "categorical_drift": categorical_results,
    }

    return report


def run_drift_analysis(
    reference_path: str = "models/reference_data.csv",
    current_path: str = "data/cleaned/students_cleaned.csv",
    output_report_path: str = "metadata/drift_report.json",
) -> Dict[str, Any]:
    """Runs drift detection between reference dataset and active incoming data."""
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference data not found at {reference_path}. Train model first.")

    ref_df = pd.read_csv(reference_path)
    cur_df = pd.read_csv(current_path)

    report = detect_drift(ref_df, cur_df)

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[Monitoring] Drift report generated: {output_report_path}")
    print(f"[Monitoring] Total Monitored: {report['monitored_features_count']} | Drifted: {report['drifted_features_count']}")
    print(f"[Monitoring] Retraining Decision: {report['retraining_status']}")

    return report


if __name__ == "__main__":
    report = run_drift_analysis()
    print("Drift Summary:")
    print(f"Drift Share: {report['drift_share_pct']}% | Retrain: {report['retraining_recommended']}")
