"""
FastAPI Microservice for Student Failure Risk Prediction.
Provides real-time inference, batch scoring, health diagnostics, model metadata,
and actionable intervention recommendations.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional
import io

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.ml.feature_engineering import add_engineered_features, LEAKAGE_COLS

# Initialize FastAPI application
app = FastAPI(
    title="Student Failure Risk Prediction API",
    description="MLOps Inference Service for Early Student Academic Failure Risk Detection & Intervention",
    version="1.0.0",
)

# Enable CORS for Streamlit and external frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model state and telemetry
MODEL_PIPELINE = None
MODEL_METADATA = {}
START_TIME = time.time()
REQUEST_METRICS = {
    "total_predictions": 0,
    "high_risk_count": 0,
    "moderate_risk_count": 0,
    "low_risk_count": 0,
    "total_latency_ms": 0.0,
}

MODEL_PATH = os.path.join("models", "best_model.pkl")
META_PATH = os.path.join("models", "model_metadata.json")


def load_artifacts():
    """Loads the registered model and metadata at startup."""
    global MODEL_PIPELINE, MODEL_METADATA
    if os.path.exists(MODEL_PATH):
        try:
            MODEL_PIPELINE = joblib.load(MODEL_PATH)
            print(f"[Serving] Loaded model pipeline from {MODEL_PATH}")
        except Exception as e:
            print(f"[Serving Error] Failed to load model: {e}")

    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                MODEL_METADATA = json.load(f)
            print(f"[Serving] Loaded model metadata: {MODEL_METADATA.get('model_name')}")
        except Exception as e:
            print(f"[Serving Error] Failed to load metadata: {e}")


@app.on_event("startup")
def startup_event():
    load_artifacts()


class StudentInput(BaseModel):
    Student_ID: Optional[str] = Field("S9999", description="Optional student identifier")
    Gender: str = Field("Female", description="Gender: Female, Male, or Other")
    Age: int = Field(20, ge=15, le=60, description="Student age")
    Department: str = Field("Engineering", description="Academic Department")
    Attendance_pct: float = Field(85.0, ge=0.0, le=100.0, alias="Attendance (%)", description="Attendance percentage")
    Midterm_Score: float = Field(65.0, ge=0.0, le=100.0, description="Midterm exam score")
    Assignments_Avg: float = Field(70.0, ge=0.0, le=100.0, description="Assignments average score")
    Quizzes_Avg: float = Field(68.0, ge=0.0, le=100.0, description="Quizzes average score")
    Participation_Score: float = Field(75.0, ge=0.0, le=100.0, description="Class participation score")
    Projects_Score: float = Field(72.0, ge=0.0, le=100.0, description="Course projects score")
    Study_Hours_per_Week: float = Field(15.0, ge=0.0, le=80.0, description="Self-study hours per week")
    Extracurricular_Activities: str = Field("Yes", description="Yes or No")
    Internet_Access_at_Home: str = Field("Yes", description="Yes or No")
    Parent_Education_Level: str = Field("Bachelor's", description="Highest parental education")
    Family_Income_Level: str = Field("Medium", description="Low, Medium, or High")
    Stress_Level_1_10: int = Field(5, ge=1, le=10, alias="Stress_Level (1-10)", description="Stress level from 1 to 10")
    Sleep_Hours_per_Night: float = Field(7.0, ge=1.0, le=16.0, description="Average sleep hours per night")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Student_ID": "S1234",
                "Gender": "Female",
                "Age": 21,
                "Department": "Computer Science",
                "Attendance (%)": 82.5,
                "Midterm_Score": 58.0,
                "Assignments_Avg": 68.0,
                "Quizzes_Avg": 62.0,
                "Participation_Score": 70.0,
                "Projects_Score": 65.0,
                "Study_Hours_per_Week": 12.0,
                "Extracurricular_Activities": "No",
                "Internet_Access_at_Home": "Yes",
                "Parent_Education_Level": "High School",
                "Family_Income_Level": "Medium",
                "Stress_Level (1-10)": 6,
                "Sleep_Hours_per_Night": 6.0,
            }
        }


class PredictionResponse(BaseModel):
    student_id: Optional[str]
    failure_risk: int
    risk_label: str
    risk_probability: float
    risk_tier: str
    confidence_pct: float
    recommendations: List[str]
    model_version: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    students: List[StudentInput]


def generate_recommendations(student: Dict[str, Any], prob: float) -> List[str]:
    """Generates personalized academic and wellness guidance based on student indicators."""
    recs = []
    if prob >= 0.50:
        recs.append("CRITICAL: Student is flagged as High Academic Failure Risk. Schedule mandatory advising intervention.")
    elif prob >= 0.20:
        recs.append("ATTENTION: Student is in Moderate Risk tier. Monitor upcoming coursework and quiz milestones.")
    else:
        recs.append("ON TRACK: Student performance is currently stable and in Low Risk tier.")

    if student.get("Attendance (%)", 100) < 75.0:
        recs.append("Attendance is below 75% threshold. Recommend meeting with academic mentor to resolve attendance gaps.")

    if student.get("Midterm_Score", 100) < 60.0:
        recs.append("Midterm exam score was unsatisfactory (<60). Recommend enrollment in peer tutoring or review sessions.")

    if student.get("Study_Hours_per_Week", 20) < 10.0:
        recs.append("Study time is under 10 hours/week. Suggest weekly study timetable and homework groups.")

    if student.get("Stress_Level (1-10)", 1) >= 7:
        recs.append("High reported stress level. Recommend referral to campus counseling & student wellness services.")

    if student.get("Sleep_Hours_per_Night", 8) < 6.0:
        recs.append("Chronic sleep deficit (<6 hrs/night) identified. Advise on sleep hygiene and workload balance.")

    return recs


@app.get("/health", tags=["System"])
def health_check():
    """Healthcheck endpoint returning system uptime and model readiness."""
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "model_loaded": MODEL_PIPELINE is not None,
        "model_name": MODEL_METADATA.get("model_name", "Unknown"),
        "model_version": MODEL_METADATA.get("model_version", "v1.0.0"),
    }


@app.get("/model-info", tags=["Governance"])
def model_info():
    """Returns production model metadata, hyperparameters, and hold-out test metrics."""
    if not MODEL_METADATA:
        return {"message": "Model metadata not available. Please train model."}
    return MODEL_METADATA


@app.get("/metrics", tags=["Monitoring"])
def get_metrics():
    """Returns inference usage statistics and throughput metrics."""
    total = REQUEST_METRICS["total_predictions"]
    avg_latency = (
        REQUEST_METRICS["total_latency_ms"] / total if total > 0 else 0.0
    )
    return {
        "total_predictions": total,
        "risk_breakdown": {
            "high_risk": REQUEST_METRICS["high_risk_count"],
            "moderate_risk": REQUEST_METRICS["moderate_risk_count"],
            "low_risk": REQUEST_METRICS["low_risk_count"],
        },
        "average_latency_ms": round(avg_latency, 2),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(student: StudentInput):
    """
    Predicts student course failure risk given demographic, academic, and behavioral features.
    """
    global MODEL_PIPELINE
    if MODEL_PIPELINE is None:
        load_artifacts()
        if MODEL_PIPELINE is None:
            raise HTTPException(status_code=503, detail="Model is not loaded. Run training first.")

    t0 = time.time()
    # Convert input to dict matching feature names
    data_dict = student.dict(by_alias=True)
    df_raw = pd.DataFrame([data_dict])

    # Apply engineered features
    df_feat = add_engineered_features(df_raw)

    # Drop leakage columns if present
    feat_cols = [c for c in df_feat.columns if c not in LEAKAGE_COLS]
    X_input = df_feat[feat_cols]

    # Predict probability and class
    try:
        prob = float(MODEL_PIPELINE.predict_proba(X_input)[0, 1])
        pred = int(prob >= 0.50)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}")

    latency_ms = (time.time() - t0) * 1000.0

    # Risk tier determination
    if prob >= 0.50:
        tier = "High Risk"
        REQUEST_METRICS["high_risk_count"] += 1
    elif prob >= 0.20:
        tier = "Moderate Risk"
        REQUEST_METRICS["moderate_risk_count"] += 1
    else:
        tier = "Low Risk"
        REQUEST_METRICS["low_risk_count"] += 1

    REQUEST_METRICS["total_predictions"] += 1
    REQUEST_METRICS["total_latency_ms"] += latency_ms

    recs = generate_recommendations(data_dict, prob)

    return PredictionResponse(
        student_id=student.Student_ID,
        failure_risk=pred,
        risk_label="Fail Risk" if pred == 1 else "Pass / Safe",
        risk_probability=round(prob, 4),
        risk_tier=tier,
        confidence_pct=round(prob * 100 if pred == 1 else (1.0 - prob) * 100, 2),
        recommendations=recs,
        model_version=MODEL_METADATA.get("model_version", "v1.0.0"),
        latency_ms=round(latency_ms, 2),
    )


@app.post("/batch-predict", tags=["Inference"])
def batch_predict(batch: BatchPredictionRequest):
    """Performs batch failure risk prediction for a list of student records."""
    results = []
    for item in batch.students:
        res = predict_single(item)
        results.append(res)
    return {
        "batch_size": len(results),
        "results": results,
    }


@app.post("/upload-csv-predict", tags=["Inference"])
async def upload_csv_predict(file: UploadFile = File(...)):
    """Uploads a CSV file of students and returns scored risk predictions."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV format.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    df_feat = add_engineered_features(df)
    feat_cols = [c for c in df_feat.columns if c not in LEAKAGE_COLS]
    X_input = df_feat[feat_cols]

    probs = MODEL_PIPELINE.predict_proba(X_input)[:, 1]
    preds = (probs >= 0.50).astype(int)

    df["failure_risk_pred"] = preds
    df["risk_probability"] = probs.round(4)
    df["risk_tier"] = [
        "High Risk" if p >= 0.50 else ("Moderate Risk" if p >= 0.20 else "Low Risk")
        for p in probs
    ]

    summary = {
        "total_records": len(df),
        "high_risk_count": int((df["risk_tier"] == "High Risk").sum()),
        "moderate_risk_count": int((df["risk_tier"] == "Moderate Risk").sum()),
        "low_risk_count": int((df["risk_tier"] == "Low Risk").sum()),
        "high_risk_rate_pct": round(float((df["risk_tier"] == "High Risk").mean() * 100), 2),
    }

    return {
        "summary": summary,
        "sample_predictions": df[["Student_ID", "Department", "Attendance (%)", "Midterm_Score", "failure_risk_pred", "risk_probability", "risk_tier"]].head(25).to_dict(orient="records"),
    }
