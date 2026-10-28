"""
Dataset and Model Versioning Utility.
Tracks dataset integrity with cryptographic hashing (SHA-256), git commit lineage,
and manages metadata for DVC / Git-based reproducible pipelines.
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 checksum of a file to detect any data mutation or drift."""
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_metadata() -> Dict[str, str]:
    """Retrieves current Git commit hash, branch, and status if available."""
    try:
        commit_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        commit_hash = "git-not-available"

    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        branch = "unknown"

    return {"commit_sha": commit_hash, "branch": branch}


def version_dataset(
    dataset_path: str = "data/cleaned/students_cleaned.csv",
    output_meta_path: str = "metadata/data_version.json",
    version_tag: str = "v1.0.0",
) -> Dict[str, Any]:
    """
    Creates dataset versioning metadata including SHA-256 fingerprint,
    record counts, column schema, class breakdown, and Git provenance.
    """
    os.makedirs(os.path.dirname(output_meta_path), exist_ok=True)

    file_hash = compute_file_sha256(dataset_path)
    git_info = get_git_metadata()

    metadata: Dict[str, Any] = {
        "version": version_tag,
        "timestamp": datetime.now().isoformat(),
        "dataset_path": dataset_path,
        "sha256_hash": file_hash,
        "git_commit": git_info["commit_sha"],
        "git_branch": git_info["branch"],
    }

    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        metadata["num_records"] = len(df)
        metadata["num_columns"] = len(df.columns)
        metadata["columns"] = list(df.columns)
        if "Grade" in df.columns:
            metadata["grade_distribution"] = df["Grade"].value_counts().to_dict()
        if "Final_Score" in df.columns:
            metadata["final_score_summary"] = {
                "mean": float(df["Final_Score"].mean()),
                "min": float(df["Final_Score"].min()),
                "max": float(df["Final_Score"].max()),
                "std": float(df["Final_Score"].std()),
            }

    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # Also create a standard DVC pointer file representation
    dvc_pointer_path = f"{dataset_path}.dvc"
    dvc_manifest = {
        "outs": [
            {
                "md5": file_hash[:32],
                "size": os.path.getsize(dataset_path) if os.path.exists(dataset_path) else 0,
                "path": os.path.basename(dataset_path),
            }
        ]
    }
    with open(dvc_pointer_path, "w", encoding="utf-8") as f:
        json.dump(dvc_manifest, f, indent=2)

    return metadata


def record_model_registry(
    model_name: str,
    model_version: str,
    metrics: Dict[str, float],
    parameters: Dict[str, Any],
    artifact_path: str,
    stage: str = "Production",
    output_registry_path: str = "models/registry.json",
) -> Dict[str, Any]:
    """
    Records and versions the trained model metadata into a local model registry.
    """
    os.makedirs(os.path.dirname(output_registry_path), exist_ok=True)

    registry_data = {"models": []}
    if os.path.exists(output_registry_path):
        try:
            with open(output_registry_path, "r", encoding="utf-8") as f:
                registry_data = json.load(f)
        except Exception:
            registry_data = {"models": []}

    entry = {
        "model_name": model_name,
        "version": model_version,
        "stage": stage,
        "registered_at": datetime.now().isoformat(),
        "metrics": metrics,
        "parameters": parameters,
        "artifact_path": artifact_path,
        "git_metadata": get_git_metadata(),
    }

    # Demote previous Production models if this is Production
    if stage == "Production":
        for item in registry_data.get("models", []):
            if item.get("stage") == "Production":
                item["stage"] = "Archived"

    registry_data["models"].append(entry)

    with open(output_registry_path, "w", encoding="utf-8") as f:
        json.dump(registry_data, f, indent=4)

    return entry


if __name__ == "__main__":
    meta = version_dataset()
    print("Dataset versioned successfully:")
    print(json.dumps(meta, indent=2))
