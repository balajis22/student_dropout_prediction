# 🎓 Student Performance & Dropout/Failure Risk MLOps Platform

An end-to-end Enterprise MLOps system that ingests student academic and behavioral data, tracks experiments with MLflow, versions datasets and models, serves real-time predictions via FastAPI, monitors feature drift, and delivers proactive early-warning insights through an interactive Streamlit dashboard.

---

## 📌 Part 2 Machine Learning & MLOps Highlights

- **🎯 Prediction Target:** Early Course Failure Risk (`failure_risk` = 1 if Grade is 'F' / Total Score < 60, 0 otherwise).
- **🛡️ Leakage-Free Feature Engineering:** Isolates and drops final post-course outcome metrics (`Final_Score`, `Total_Score`, `Grade`) during early-warning inference, and computes domain-specific behavioral features (Study-Sleep ratio, Academic Engagement Index, Exam-to-Coursework ratio).
- **🤖 Models Evaluated:**
  - **Baseline:** Logistic Regression (Balanced)
  - **Advanced Ensemble:** Random Forest Classifier (**Selected Best Model: 93.3% Test Accuracy, 93.9% Test ROC-AUC, 0.5106 Validation F1**)
  - **Advanced Gradient Boosting:** XGBoost Classifier
- **🔬 MLflow Experiment Tracking:** SQLite tracking database (`sqlite:///mlflow.db`) with parameter logging, evaluation metrics, confusion matrices, ROC curves, and feature importance artifacts.
- **📦 Data & Model Versioning:** Cryptographic SHA-256 fingerprinting, Git provenance, DVC pointer manifest (`students_cleaned.csv.dvc`), and local registry metadata (`models/registry.json`).
- **🚀 Real-Time Serving:** Production-ready **FastAPI** microservice with `/predict`, `/batch-predict`, `/upload-csv-predict`, `/health`, `/model-info`, and `/metrics`.
- **🐳 Containerization:** Dockerfile and Docker Compose orchestration.
- **📡 Drift Monitoring & Governance:** Automated Kolmogorov-Smirnov (KS) test and Population Stability Index (PSI) drift monitoring with automated retraining trigger flags.
- **🖥️ Streamlit Dashboard Integration:** Multi-tab analytical app integrating Part 1 descriptive analytics with Part 2 interactive What-If risk simulation, batch CSV scoring, and model governance dashboards.

---

## 🏗️ Repository Architecture

```
student-performance-analysis/
├── artifacts/
│   └── training_plots/       # Confusion matrix, ROC curve, and Feature Importance plots
├── dashboard/
│   └── app.py                # Streamlit Dashboard (Descriptive + ML Predictor + Governance)
├── data/
│   ├── raw/                  # Original raw student performance dataset
│   └── cleaned/              # Cleaned and processed dataset
├── docs/
│   └── part2_mlops_architecture.md # Full MLOps technical and governance documentation
├── metadata/
│   ├── data_version.json     # SHA-256 dataset checksum and Git provenance
│   └── drift_report.json     # Feature drift detection metrics and retraining decisions
├── models/
│   ├── best_model.pkl        # Serialized production pipeline
│   ├── model_metadata.json   # Production model metrics, thresholds, and schema
│   ├── reference_data.csv    # Training baseline distribution for drift monitoring
│   └── registry.json         # Model versioning registry
├── src/
│   ├── ml/
│   │   ├── feature_engineering.py # Pipeline, leak prevention & feature calculations
│   │   ├── versioning.py          # SHA-256 and DVC dataset/model versioning
│   │   ├── train.py               # MLflow training pipeline & multi-model evaluation
│   │   └── evaluate.py            # Model checkpoint evaluation
│   ├── monitoring/
│   │   └── drift_monitor.py       # KS-test & PSI feature drift monitoring engine
│   └── serving/
│       └── api.py                 # FastAPI microservice for real-time inference
├── Dockerfile                # Multi-stage production container image
├── docker-compose.yml        # Docker Compose configuration for API and Dashboard
├── requirements.txt          # Python dependencies
└── mlflow.db                 # MLflow SQLite tracking store
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```powershell
# Activate your virtual environment
.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

---

### 2. Run Feature Engineering & Model Training (MLflow)
Train baseline and advanced models, compute metrics, generate visual artifacts, and register the best model:
```powershell
python -m src.ml.train
```

To view the interactive MLflow experiment tracking dashboard:
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

---

### 3. Run Drift Detection & Retraining Monitor
Run Kolmogorov-Smirnov statistical tests between the baseline reference dataset and incoming data:
```powershell
python -m src.monitoring.drift_monitor
```

---

### 4. Launch the FastAPI Prediction Microservice
Start the production inference API:
```powershell
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)
- Model Metadata: [http://localhost:8000/model-info](http://localhost:8000/model-info)
- Metrics: [http://localhost:8000/metrics](http://localhost:8000/metrics)

#### Sample Prediction Request (`POST /predict`):
```json
{
  "Department": "Computer Science",
  "Gender": "Female",
  "Age": 21,
  "Attendance (%)": 62.0,
  "Midterm_Score": 48.0,
  "Assignments_Avg": 55.0,
  "Quizzes_Avg": 50.0,
  "Participation_Score": 60.0,
  "Projects_Score": 58.0,
  "Study_Hours_per_Week": 8.0,
  "Extracurricular_Activities": "No",
  "Internet_Access_at_Home": "Yes",
  "Parent_Education_Level": "High School",
  "Family_Income_Level": "Low",
  "Stress_Level (1-10)": 8,
  "Sleep_Hours_per_Night": 5.0
}
```

---

### 5. Launch the Streamlit Dashboard
```powershell
streamlit run dashboard/app.py
```
Open [http://localhost:8501](http://localhost:8501) to explore:
1. **Descriptive Analytics (Part 1):** Grade distributions, attendance correlations, and student explorer.
2. **ML Risk Prediction (Part 2):** Real-time interactive student What-If risk simulator and batch CSV roster scoring.
3. **MLOps Model Governance (Part 2):** Production model card, MLflow multi-model comparison, artifact plots, and live Kolmogorov-Smirnov drift tests.

---

### 6. Run with Docker & Docker Compose
```powershell
# Build and run both FastAPI and Streamlit services
docker-compose up --build
```
- FastAPI Serving: `http://localhost:8000`
- Streamlit Dashboard: `http://localhost:8501`
