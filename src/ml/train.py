"""
Model training and experiment tracking pipeline using MLflow.
Trains Baseline (Logistic Regression) and Advanced models (Random Forest, XGBoost),
computes comprehensive evaluation metrics, logs artifacts, and registers the top model.
"""

import json
import os
import sys

# Configure UTF-8 encoding on Windows to prevent UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn

from src.ml.feature_engineering import get_preprocessor, prepare_datasets
from src.ml.versioning import record_model_registry, version_dataset

# Silence MLflow hint warning
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


def compute_metrics(y_true, y_pred, y_prob):
    """Calculates all key classification performance metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
    }
    return metrics


def save_confusion_matrix_plot(y_true, y_pred, output_path: str, title: str):
    """Generates and saves a confusion matrix visualization."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Pass (0)", "Fail Risk (1)"])
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    plt.title(f"Confusion Matrix: {title}")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_roc_curve_plot(y_true, y_prob, output_path: str, title: str, auc_val: float):
    """Generates and saves an ROC Curve visualization."""
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"ROC Curve: {title}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_feature_importance_plot(pipeline: Pipeline, feature_names: list, output_path: str, title: str):
    """Extracts and plots top feature importances or model coefficients."""
    classifier = pipeline.named_steps["classifier"]
    fig, ax = plt.subplots(figsize=(8, 6))

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        return

    # In case feature names length matches transformed features
    if len(importances) == len(feature_names):
        indices = np.argsort(importances)[-12:]
        ax.barh(range(len(indices)), importances[indices], color="#10B981", align="center")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel("Relative Importance / Absolute Coefficient")
        ax.set_title(f"Top Features: {title}")
        plt.tight_layout()
        fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_training():
    """Executes the full training, experiment logging, and registration pipeline."""
    print("=" * 70)
    print("🚀 STARTING STUDENT FAILURE RISK ML TRAINING PIPELINE")
    print("=" * 70)

    # 1. Version dataset and check integrity
    dataset_path = "data/cleaned/students_cleaned.csv"
    data_meta = version_dataset(dataset_path=dataset_path)
    print(f"📊 Dataset versioned: {data_meta['version']} (SHA-256: {data_meta['sha256_hash'][:16]}...)")

    # 2. Prepare train/val/test splits
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        cat_cols,
        num_cols,
    ) = prepare_datasets(data_path=dataset_path, test_size=0.15, val_size=0.15, random_state=42)

    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")
    print(f"Train fail rate: {y_train.mean():.2%} | Val fail rate: {y_val.mean():.2%} | Test fail rate: {y_test.mean():.2%}")

    # Set up MLflow with modern SQLite backend
    mlflow_db_path = os.path.abspath("mlflow.db").replace("\\", "/")
    tracking_uri = f"sqlite:///{mlflow_db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "Student_Performance_Failure_Risk"
    mlflow.set_experiment(experiment_name)
    print(f"[MLflow] Backend Store: {tracking_uri}")
    print(f"[MLflow] Experiment: '{experiment_name}'")

    # Calculate class imbalance weighting for positive class
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / float(pos_count) if pos_count > 0 else 1.0

    # Model candidate definitions
    models = {
        "Logistic_Regression_Baseline": {
            "estimator": LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight="balanced",
                random_state=42,
            ),
            "params": {"model_type": "LogisticRegression", "C": 1.0, "class_weight": "balanced"},
        },
        "Random_Forest": {
            "estimator": RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "params": {
                "model_type": "RandomForestClassifier",
                "n_estimators": 150,
                "max_depth": 8,
                "min_samples_split": 5,
                "class_weight": "balanced",
            },
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
            ),
            "params": {
                "model_type": "XGBClassifier",
                "n_estimators": 150,
                "max_depth": 5,
                "learning_rate": 0.08,
                "scale_pos_weight": float(scale_pos_weight),
            },
        },
    }

    artifacts_dir = "artifacts/training_plots"
    os.makedirs(artifacts_dir, exist_ok=True)

    results = {}
    fitted_pipelines = {}

    for model_name, config in models.items():
        print(f"\n--- Training: {model_name} ---")
        preprocessor = get_preprocessor()
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", config["estimator"]),
            ]
        )

        with mlflow.start_run(run_name=model_name) as run:
            # Fit pipeline
            pipeline.fit(X_train, y_train)

            # Evaluate on Validation set
            y_val_pred = pipeline.predict(X_val)
            y_val_prob = pipeline.predict_proba(X_val)[:, 1]

            val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
            print(f"Validation F1: {val_metrics['f1_score']:.4f} | ROC-AUC: {val_metrics['roc_auc']:.4f} | Recall: {val_metrics['recall']:.4f}")

            # Log parameters and metrics
            mlflow.log_params(config["params"])
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("val_samples", len(X_val))
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

            # Create visual artifacts
            cm_path = os.path.join(artifacts_dir, f"{model_name}_confusion_matrix.png")
            roc_path = os.path.join(artifacts_dir, f"{model_name}_roc_curve.png")
            feat_path = os.path.join(artifacts_dir, f"{model_name}_feature_importance.png")

            save_confusion_matrix_plot(y_val, y_val_pred, cm_path, model_name)
            save_roc_curve_plot(y_val, y_val_prob, roc_path, model_name, val_metrics["roc_auc"])

            # Feature names extraction
            try:
                preproc_fitted = pipeline.named_steps["preprocessor"]
                cat_encoder = preproc_fitted.named_transformers_["cat"].named_steps["onehot"]
                encoded_cat_names = cat_encoder.get_feature_names_out(cat_cols)
                all_feature_names = list(num_cols) + list(encoded_cat_names)
                save_feature_importance_plot(pipeline, all_feature_names, feat_path, model_name)
                if os.path.exists(feat_path):
                    mlflow.log_artifact(feat_path)
            except Exception as e:
                print(f"Note on feature names: {e}")

            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(roc_path)

            # Log pipeline in MLflow with cloudpickle serialization
            try:
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path="model",
                    serialization_format="cloudpickle",
                )
            except Exception as e:
                print(f"[Warning] MLflow log_model note: {e}")

            results[model_name] = {
                "run_id": run.info.run_id,
                "metrics": val_metrics,
                "pipeline": pipeline,
                "params": config["params"],
            }
            fitted_pipelines[model_name] = pipeline

    # Select Best Model based on Validation F1-score (harmonic mean of precision & recall for minority risk class)
    best_model_name = max(results.keys(), key=lambda m: (results[m]["metrics"]["f1_score"], results[m]["metrics"]["roc_auc"]))
    best_entry = results[best_model_name]
    best_pipeline = best_entry["pipeline"]

    print("\n" + "=" * 70)
    print(f"🏆 BEST MODEL SELECTED: {best_model_name}")
    print(f"Validation F1: {best_entry['metrics']['f1_score']:.4f} | ROC-AUC: {best_entry['metrics']['roc_auc']:.4f}")
    print("=" * 70)

    # Final Evaluation on the Unseen Hold-out Test Set
    print("\n🧪 Evaluating Best Model on Hold-Out Test Set (15% unseen)...")
    y_test_pred = best_pipeline.predict(X_test)
    y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)

    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC:    {test_metrics['pr_auc']:.4f}")

    test_report = classification_report(y_test, y_test_pred, target_names=["Pass (0)", "Fail Risk (1)"], output_dict=True)

    # Save Best Model Artifacts to models/
    os.makedirs("models", exist_ok=True)
    best_model_path = "models/best_model.pkl"
    joblib.dump(best_pipeline, best_model_path)
    print(f"💾 Serialized best pipeline saved to: {best_model_path}")

    # Save reference test data for drift monitoring
    reference_data_path = "models/reference_data.csv"
    X_train_reference = X_train.copy()
    X_train_reference["failure_risk"] = y_train
    X_train_reference.to_csv(reference_data_path, index=False)
    print(f"📁 Reference dataset saved for drift monitoring: {reference_data_path}")

    # Save comprehensive model metadata
    model_metadata = {
        "model_name": best_model_name,
        "model_version": "v1.0.0",
        "registered_stage": "Production",
        "val_metrics": best_entry["metrics"],
        "test_metrics": test_metrics,
        "classification_report": test_report,
        "parameters": best_entry["params"],
        "feature_columns": list(X_train.columns),
        "target_name": "failure_risk",
        "classes": [0, 1],
        "class_labels": ["Pass / Safe", "Fail Risk"],
        "risk_thresholds": {
            "low_risk": "< 0.20",
            "moderate_risk": "0.20 - 0.50",
            "high_risk": ">= 0.50",
        },
    }

    with open("models/model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=4)

    # Record into Model Registry
    record_model_registry(
        model_name=best_model_name,
        model_version="v1.0.0",
        metrics=test_metrics,
        parameters=best_entry["params"],
        artifact_path=best_model_path,
        stage="Production",
    )

    print("✅ Model training, MLflow logging, and registry update complete!\n")
    return model_metadata


if __name__ == "__main__":
    run_training()
