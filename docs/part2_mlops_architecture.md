# MLOps Part 2 Architecture & Model Lifecycle Specification

## 1. Problem Formulation & Prediction Target Definition

### 1.1 Objective
The goal is to implement an end-to-end Machine Learning Operations (MLOps) pipeline for **Student Academic Failure & Dropout Risk Prediction**. The system acts as an institutional early-warning mechanism, flagging students at risk of failing the course prior to final exams so academic advisors and faculty can intervene proactively.

### 1.2 Target Variable Formulation
- **Target Name:** `failure_risk` (Binary Classification)
- **Mathematical Definition:**
  $$\text{failure\_risk} = \begin{cases} 1 & \text{if Grade} = \text{'F'} \lor \text{Total\_Score} < 60.0 \\ 0 & \text{otherwise (Passing / Low Risk)} \end{cases}$$
- **Class Distribution:**
  - Class 0 (Pass / Low Risk): 4,721 students (94.42%)
  - Class 1 (Fail Risk / At Risk): 279 students (5.58%)
- **Risk Tiers:**
  - **High Risk:** Failure Probability $P \ge 0.50$ (Classified as Fail Risk)
  - **Moderate Risk:** $0.20 \le P < 0.50$ (Academic Watchlist)
  - **Low Risk:** $P < 0.20$ (On Track)

### 1.3 Data Leakage Prevention
To ensure authentic early-warning utility in production, all post-course or final outcome features are explicitly isolated and dropped from the feature space during training and inference:
- `Final_Score`: Not known during early-term advisory.
- `Total_Score`: Derived directly from final exam score.
- `Grade`: Post-hoc letter grade outcome.
- `Average_Assessment_Score`: Composite metric containing final assessment.

---

## 2. Feature Engineering & Preprocessing Pipeline

### 2.1 Feature Definitions
- **Demographics & Environment:** `Gender`, `Department`, `Age`, `Parent_Education_Level`, `Family_Income_Level`, `Internet_Access_at_Home`, `Extracurricular_Activities`.
- **Academic Milestones:** `Attendance (%)`, `Midterm_Score`, `Assignments_Avg`, `Quizzes_Avg`, `Participation_Score`, `Projects_Score`.
- **Behavioral & Lifestyle:** `Study_Hours_per_Week`, `Stress_Level (1-10)`, `Sleep_Hours_per_Night`.

### 2.2 Domain-Specific Engineered Features
1. **Study to Sleep Ratio:**
   $$\text{Study\_Sleep\_Ratio} = \frac{\text{Study\_Hours\_per\_Week}}{(\text{Sleep\_Hours\_per\_Night} \times 7) + 10^{-5}}$$
2. **Academic Engagement Index:**
   $$\text{Academic\_Engagement} = 0.4 \times \frac{\text{Attendance}}{100} + 0.3 \times \frac{\text{Participation}}{100} + 0.3 \times \frac{\text{Assignments\_Avg}}{100}$$
3. **Continuous Assessment Average:**
   $$\text{Continuous\_Assessment\_Avg} = \frac{\text{Assignments\_Avg} + \text{Quizzes\_Avg} + \text{Projects\_Score}}{3.0}$$
4. **Exam to Coursework Ratio:**
   $$\text{Exam\_to\_Coursework\_Ratio} = \frac{\text{Midterm\_Score}}{\text{Continuous\_Assessment\_Avg} + 10^{-5}}$$
5. **Stress Sleep Index:**
   $$\text{Stress\_Sleep\_Index} = \frac{\text{Stress\_Level}}{\text{Sleep\_Hours\_per\_Night} + 1.0}$$

### 2.3 Data Transformation Pipeline
- Continuous variables are standardized using `StandardScaler`.
- Categorical variables are encoded with `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- All steps are encapsulated in a Scikit-Learn `ColumnTransformer` and serialized inside a single composite `Pipeline`.

---

## 3. Data Splitting & Reproducibility
- **Dataset Size:** 5,000 records.
- **Split Strategy:** Stratified hold-out split preserving class proportion:
  - **Train Set (70%):** 3,500 samples (5.57% fail rate)
  - **Validation Set (15%):** 750 samples (5.60% fail rate)
  - **Hold-out Test Set (15%):** 750 samples (5.60% fail rate)
- **Random Seed:** Fixed `random_state=42` across all splitting and estimator steps.

---

## 4. Model Training & Evaluation Metrics

Three candidate models were evaluated with class imbalance weighting:
1. **Logistic Regression (Baseline):**
   - L2 penalty, balanced class weighting, maximum iterations = 1000.
2. **Random Forest Classifier (Advanced Ensemble):**
   - 150 estimators, max depth 8, min samples split 5, balanced class weighting.
3. **XGBoost Classifier (Gradient Boosted Decision Trees):**
   - 150 estimators, max depth 5, learning rate 0.08, `scale_pos_weight = 16.95`.

### 4.1 Comparative Validation Performance
| Model | Validation Accuracy | Validation Recall | Validation Precision | Validation F1-Score | Validation ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 86.13% | 85.71% | 28.12% | 0.4235 | 0.9499 |
| **Random Forest (Selected)** | **94.40%** | **57.14%** | **46.15%** | **0.5106** | **0.9392** |
| **XGBoost Classifier** | 93.87% | 54.76% | 44.23% | 0.4894 | 0.9395 |

### 4.2 Selected Production Model: Random Forest
- **Selection Justification:** Random Forest achieved the highest harmonic **F1-Score (0.5106)** on the minority failure class, achieving superior balance between false alarms and true positive detection, while maintaining **94.40% accuracy** and **0.9392 ROC-AUC**.
- **Final Hold-Out Test Set Performance (Unseen Data):**
  - **Accuracy:** 93.33%
  - **ROC-AUC:** 93.89%
  - **Recall:** 57.14%
  - **Precision:** 42.86%
  - **F1-Score:** 0.4898
  - **PR-AUC:** 0.4446

---

## 5. Experiment Tracking & Artifact Management (MLflow)
- **Tracking Store Backend:** `sqlite:///mlflow.db`
- **Experiment Name:** `Student_Performance_Failure_Risk`
- **Tracked Entities:**
  - Hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight`, etc.
  - Metrics: `val_accuracy`, `val_precision`, `val_recall`, `val_f1_score`, `val_roc_auc`, `val_log_loss`.
  - Artifacts: Confusion matrices, ROC curves, and Feature Importance bar charts saved to `artifacts/training_plots/`.
  - Model serialization: Exported with cloudpickle format.

---

## 6. Dataset and Model Versioning

### 6.1 Cryptographic Dataset Hashing
- Every ingestion/training run records the **SHA-256 checksum** of `data/cleaned/students_cleaned.csv` into `metadata/data_version.json`:
  - `SHA-256:` `7da8bddaa82bf5e7ad8e5ecaab9393a632206960a9c9e6836ac55474d59c1335`
- A DVC pointer manifest (`students_cleaned.csv.dvc`) stores file size and hash for remote synchronization.

### 6.2 Model Registry Lineage
The production model is recorded in `models/registry.json`:
- Version: `v1.0.0`
- Stage: `Production`
- Artifact Path: `models/best_model.pkl`
- Associated Git metadata and validation metrics.

---

## 7. Model Serving & Containerization

### 7.1 FastAPI Inference Service (`src/serving/api.py`)
- `POST /predict`: Single student real-time prediction returning risk probability, risk category, and personalized intervention recommendations.
- `POST /batch-predict`: Batch inference for list of student objects.
- `POST /upload-csv-predict`: Direct CSV upload risk scoring.
- `GET /health`: Healthcheck diagnostic, memory status, and model availability.
- `GET /model-info`: Production model metadata, test set performance, and parameter configuration.
- `GET /metrics`: Service latency and request distribution statistics.

### 7.2 Docker Deployment
- Containerized via multi-stage `Dockerfile` (Python 3.11-slim) with non-root security.
- Orchestrated with `docker-compose.yml` defining the API service (`:8000`) and the Streamlit dashboard (`:8501`).

---

## 8. Continuous Monitoring & Automated Retraining Criteria

### 8.1 Drift Detection Engine (`src/monitoring/drift_monitor.py`)
- **Continuous Features:** Two-sample Kolmogorov-Smirnov (KS) test ($\alpha = 0.05$) and Population Stability Index (PSI).
- **Categorical Features:** Distribution divergence and Chi-Square frequency testing.
- **Prediction Drift:** Shifts in the ratio of High/Moderate/Low risk classifications.

### 8.2 Retraining Decision Triggers
Retraining is automatically flagged (`retraining_recommended: true`) under any of the following criteria:
1. **Feature Drift Threshold:** Over **25%** of monitored input features exhibit statistically significant drift ($p < 0.05$ or $\text{PSI} \ge 0.20$).
2. **Performance Degradation:** Observed F1-score on newly labeled course completion cohorts drops by more than **10%** from baseline (threshold: $\text{F1} < 0.44$).
3. **Data Quality Degradation:** Missing value frequency or schema violations exceed 5% of records.
4. **Scheduled Retraining:** Bi-annual retraining cycle at the conclusion of each academic semester.
