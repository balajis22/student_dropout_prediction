"""
Feature engineering and preprocessing pipeline for student failure risk prediction.
Ensures reproducible transformations and prevents target leakage.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Categorical and numerical column specifications
CATEGORICAL_COLS = [
    "Gender",
    "Department",
    "Extracurricular_Activities",
    "Internet_Access_at_Home",
    "Parent_Education_Level",
    "Family_Income_Level",
]

BASE_NUMERICAL_COLS = [
    "Age",
    "Attendance (%)",
    "Midterm_Score",
    "Assignments_Avg",
    "Quizzes_Avg",
    "Participation_Score",
    "Projects_Score",
    "Study_Hours_per_Week",
    "Stress_Level (1-10)",
    "Sleep_Hours_per_Night",
]

ENGINEERED_NUMERICAL_COLS = [
    "Study_Sleep_Ratio",
    "Academic_Engagement",
    "Continuous_Assessment_Avg",
    "Exam_to_Coursework_Ratio",
    "Stress_Sleep_Index",
]

ALL_NUMERICAL_COLS = BASE_NUMERICAL_COLS + ENGINEERED_NUMERICAL_COLS

# Columns that represent post-course outcomes or target variables (MUST BE DROPPED TO PREVENT LEAKAGE)
LEAKAGE_COLS = [
    "Student_ID",
    "Final_Score",
    "Total_Score",
    "Grade",
    "Average_Assessment_Score",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific interaction and behavioral features:
    - Study_Sleep_Ratio: Balance between study hours and sleep
    - Academic_Engagement: Weighted composite of attendance, participation, and assignments
    - Continuous_Assessment_Avg: Average performance across formative assessments
    - Exam_to_Coursework_Ratio: Ratio of midterm exam to continuous coursework
    - Stress_Sleep_Index: Ratio of reported stress to sleep duration
    """
    data = df.copy()

    # Avoid zero division with small epsilon
    eps = 1e-5

    data["Study_Sleep_Ratio"] = data["Study_Hours_per_Week"] / (
        data["Sleep_Hours_per_Night"] * 7.0 + eps
    )

    data["Academic_Engagement"] = (
        (data["Attendance (%)"] / 100.0) * 0.4
        + (data["Participation_Score"] / 100.0) * 0.3
        + (data["Assignments_Avg"] / 100.0) * 0.3
    )

    data["Continuous_Assessment_Avg"] = (
        data["Assignments_Avg"] + data["Quizzes_Avg"] + data["Projects_Score"]
    ) / 3.0

    data["Exam_to_Coursework_Ratio"] = data["Midterm_Score"] / (
        data["Continuous_Assessment_Avg"] + eps
    )

    data["Stress_Sleep_Index"] = data["Stress_Level (1-10)"] / (
        data["Sleep_Hours_per_Night"] + 1.0
    )

    return data


def create_target(df: pd.DataFrame) -> pd.Series:
    """
    Defines the prediction target: Failure Risk.
    A student is classified as at failure risk (1) if their Grade is 'F' or Total_Score < 60,
    and passing/safe (0) otherwise.
    """
    if "Grade" in df.columns:
        return (df["Grade"] == "F").astype(int)
    elif "Total_Score" in df.columns:
        return (df["Total_Score"] < 60.0).astype(int)
    raise ValueError("Neither 'Grade' nor 'Total_Score' found to construct target.")


def get_preprocessor() -> ColumnTransformer:
    """
    Builds the ColumnTransformer for numeric and categorical pipelines.
    - Numeric features: StandardScaler
    - Categorical features: OneHotEncoder (ignoring unseen categories during inference)
    """
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ALL_NUMERICAL_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )

    return preprocessor


def prepare_datasets(
    data_path: str = "data/cleaned/students_cleaned.csv",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    """
    Loads raw cleaned data, creates target and engineered features,
    and splits into reproducible stratified Train, Validation, and Test sets.
    """
    df = pd.read_csv(data_path)

    # 1. Construct target
    y = create_target(df)

    # 2. Add engineered features
    df_engineered = add_engineered_features(df)

    # 3. Drop leakage columns and target
    feature_cols = [c for c in df_engineered.columns if c not in LEAKAGE_COLS]
    X = df_engineered[feature_cols]

    # First split: train+val (85%), test (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train (70% of total), val (15% of total)
    val_relative_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        CATEGORICAL_COLS,
        ALL_NUMERICAL_COLS,
    )
