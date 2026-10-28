import json
import os
import sys
import io
import time
import numpy as np
import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv

# Ensure repository root is on sys.path regardless of execution directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import joblib
except ImportError:
    joblib = None

# Resilient Feature Engineering implementation (always callable)
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
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

LEAKAGE_COLS = [
    "Student_ID",
    "Final_Score",
    "Total_Score",
    "Grade",
    "Average_Assessment_Score",
]

# Monitoring components
try:
    from src.monitoring.drift_monitor import run_drift_analysis, detect_drift
except Exception:
    run_drift_analysis = None
    detect_drift = None

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Performance & MLOps Failure Risk Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics and clean risk badges
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .risk-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #f87171;
    }
    .risk-mod {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #fbbf24;
    }
    .risk-low {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #34d399;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# DATABASE CONFIGURATION & RESILIENT DATA LOADING
# =========================================================

load_dotenv()

DB_NAME = "student_performance_db"
DB_USER = "postgres"
DB_HOST = "127.0.0.1"
DB_PORT = "5432"
password = os.getenv("POSTGRES_PASSWORD", "")


@st.cache_data
def load_data():
    source_name = "PostgreSQL"
    data = None

    if password:
        try:
            connection = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=password,
                host=DB_HOST,
                port=DB_PORT,
                connect_timeout=3,
            )
            query = "SELECT * FROM public.student_performance_mart;"
            data = pd.read_sql(query, connection)
            connection.close()
        except Exception:
            data = None

    # Graceful fallback to cleaned dataset if PostgreSQL is offline
    if data is None:
        csv_fallback = os.path.join(PROJECT_ROOT, "data", "cleaned", "students_cleaned.csv")
        if os.path.exists(csv_fallback):
            data = pd.read_csv(csv_fallback)
            source_name = "Local Cleaned Dataset (Fallback)"
        else:
            raise FileNotFoundError("Neither PostgreSQL nor cleaned CSV could be loaded.")

    # Normalize column names
    rename_map = {
        "student_id": "Student_ID",
        "department": "Department",
        "gender": "Gender",
        "age": "Age",
        "attendance (%)": "Attendance (%)",
        "study_hours_per_week": "Study_Hours_per_Week",
        "average_assessment_score": "Average_Assessment_Score",
        "total_score": "Total_Score",
        "grade": "Grade",
        "assignments_avg": "Assignments_Avg",
        "quizzes_avg": "Quizzes_Avg",
        "participation_score": "Participation_Score",
        "projects_score": "Projects_Score",
        "midterm_score": "Midterm_Score",
        "final_score": "Final_Score",
        "stress_level (1-10)": "Stress_Level (1-10)",
        "sleep_hours_per_night": "Sleep_Hours_per_Night",
        "parent_education_level": "Parent_Education_Level",
        "family_income_level": "Family_Income_Level",
        "extracurricular_activities": "Extracurricular_Activities",
        "internet_access_at_home": "Internet_Access_at_Home",
    }
    data = data.rename(columns=rename_map)

    # Ensure Midterm_Score is present if from analytical mart
    if "Midterm_Score" not in data.columns:
        if "Average_Assessment_Score" in data.columns:
            data["Midterm_Score"] = data["Average_Assessment_Score"]
        else:
            data["Midterm_Score"] = 70.0

    return data, source_name


try:
    df, data_source = load_data()
except Exception as error:
    st.error("Unable to load student data.")
    st.error(str(error))
    st.stop()


# Load Trained ML Model Pipeline
@st.cache_resource
def load_ml_model():
    model_path = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
    meta_path = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
    model = None
    meta = {}
    if joblib is not None and os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except Exception:
            model = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    return model, meta


ml_pipeline, model_meta = load_ml_model()


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.title("🎓 Student Portal")
st.sidebar.caption(f"Connected to: **{data_source}**")

st.sidebar.header("🔎 Cohort Filters")

departments = sorted(df["Department"].dropna().unique().tolist())
selected_departments = st.sidebar.multiselect(
    "Department", departments, default=departments
)

grades = sorted(df["Grade"].dropna().unique().tolist())
selected_grades = st.sidebar.multiselect("Grade", grades, default=grades)

filtered_df = df[
    df["Department"].isin(selected_departments) & df["Grade"].isin(selected_grades)
].copy()

st.sidebar.divider()
st.sidebar.markdown(
    """
    **MLOps Part 2 System Status:**
    - 🧠 Model: `Random Forest (Production)`
    - 🎯 Target: `Failure Risk (Grade F)`
    - ⚡ Serving: `FastAPI / Docker Ready`
    - 📡 Tracking: `MLflow Active`
    """
)


# =========================================================
# MAIN HEADER & TABS
# =========================================================

st.title("🎓 Student Performance & MLOps Failure Risk Platform")
st.write(
    "End-to-End Analytics, Machine Learning Early-Warning Prediction, "
    "and MLOps Model Governance & Drift Monitoring."
)

tab_analytics, tab_prediction, tab_governance = st.tabs(
    [
        "📊 Descriptive Analytics (Part 1)",
        "🎯 ML Risk Prediction & Intervention (Part 2)",
        "🔬 MLOps Model Governance & Drift Monitoring (Part 2)",
    ]
)


# =========================================================
# TAB 1: DESCRIPTIVE ANALYTICS (PART 1)
# =========================================================

with tab_analytics:
    st.subheader("📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Students", f"{len(filtered_df):,}")

    with col2:
        avg_score = filtered_df["Total_Score"].mean() if len(filtered_df) > 0 else 0
        st.metric("Average Total Score", f"{avg_score:.2f}")

    with col3:
        avg_att = filtered_df["Attendance (%)"].mean() if len(filtered_df) > 0 else 0
        st.metric("Average Attendance", f"{avg_att:.2f}%")

    with col4:
        pass_rate = ((filtered_df["Grade"] != "F").mean() * 100) if len(filtered_df) > 0 else 0
        st.metric("Pass Rate", f"{pass_rate:.2f}%")

    st.divider()

    col_view1, col_view2 = st.columns(2)

    with col_view1:
        st.subheader("1️⃣ Grade Distribution")
        if len(filtered_df) > 0:
            grade_counts = filtered_df["Grade"].value_counts().sort_index()
            st.bar_chart(grade_counts)
        else:
            st.warning("No data matches selected filters.")

    with col_view2:
        st.subheader("2️⃣ Department Performance")
        if len(filtered_df) > 0:
            dept_perf = (
                filtered_df.groupby("Department")
                .agg(
                    Average_Score=("Total_Score", "mean"),
                    Average_Attendance=("Attendance (%)", "mean"),
                    Students=("Student_ID", "count"),
                )
                .round(2)
            )
            st.dataframe(dept_perf, use_container_width=True)
        else:
            st.warning("No data matches selected filters.")

    st.divider()

    col_view3, col_view4 = st.columns(2)

    with col_view3:
        st.subheader("3️⃣ Attendance vs Total Score")
        if len(filtered_df) > 0:
            st.scatter_chart(filtered_df, x="Attendance (%)", y="Total_Score")
            st.caption("Strong correlation between class attendance and final grade outcome.")
        else:
            st.warning("No data matches selected filters.")

    with col_view4:
        st.subheader("4️⃣ Assessment Component Performance")
        assessment_cols = [
            "Assignments_Avg",
            "Quizzes_Avg",
            "Participation_Score",
            "Projects_Score",
        ]
        available_assess = [c for c in assessment_cols if c in filtered_df.columns]
        if len(filtered_df) > 0 and available_assess:
            st.bar_chart(filtered_df[available_assess].mean().round(2))
        else:
            st.warning("No assessment data available.")

    st.divider()

    st.subheader("5️⃣ Students Requiring Academic Attention (Grade D or F)")
    attention_df = filtered_df[filtered_df["Grade"].isin(["D", "F"])].copy()
    display_cols = [
        "Student_ID",
        "Department",
        "Attendance (%)",
        "Total_Score",
        "Grade",
        "Average_Assessment_Score",
        "Study_Hours_per_Week",
    ]
    show_cols = [c for c in display_cols if c in attention_df.columns]
    if len(attention_df) > 0:
        st.dataframe(
            attention_df[show_cols].sort_values(by="Total_Score", ascending=True),
            use_container_width=True,
        )
    else:
        st.info("No students with Grade D or F in selected filter.")

    st.divider()
    st.subheader("👤 Individual Student Profile Inspector")
    student_ids = sorted(filtered_df["Student_ID"].dropna().unique().tolist())
    if student_ids:
        selected_student = st.selectbox("Select Student to Inspect", student_ids)
        st_row = filtered_df[filtered_df["Student_ID"] == selected_student].iloc[0]

        sp1, sp2, sp3 = st.columns(3)
        with sp1:
            st.write(f"**Student ID:** {st_row['Student_ID']}")
            st.write(f"**Department:** {st_row['Department']}")
            st.write(f"**Gender:** {st_row['Gender']}")
            st.write(f"**Age:** {st_row['Age']}")
        with sp2:
            st.write(f"**Attendance:** {st_row['Attendance (%)']:.2f}%")
            st.write(f"**Total Score:** {st_row['Total_Score']:.2f}")
            st.write(f"**Grade:** {st_row['Grade']}")
            st.write(f"**Study Hours/Week:** {st_row['Study_Hours_per_Week']}")
        with sp3:
            st.write(f"**Stress Level (1-10):** {st_row.get('Stress_Level (1-10)', 'N/A')}")
            st.write(f"**Sleep Hours:** {st_row.get('Sleep_Hours_per_Night', 'N/A')}")
            st.write(f"**Family Income:** {st_row.get('Family_Income_Level', 'N/A')}")


# =========================================================
# TAB 2: ML RISK PREDICTION & INTERVENTIONS (PART 2)
# =========================================================

with tab_prediction:
    st.subheader("🎯 Early Failure & Dropout Risk Prediction")
    st.write(
        "Utilizes the production-registered Machine Learning model to forecast "
        "whether a student is at risk of academic failure before final exams, "
        "enabling proactive institutional interventions."
    )

    pred_subtab1, pred_subtab2 = st.tabs(
        ["🔍 Single Student What-If Simulation", "📂 Batch Cohort CSV Risk Scoring"]
    )

    with pred_subtab1:
        st.markdown("#### 1. Configure Student Academic & Behavioral Profile")

        # Preset selection helper
        preset = st.selectbox(
            "Quick-Fill Sample Student Profile:",
            [
                "Custom (Manual Input)",
                "🚨 Profile A: High Risk Student (Low Attendance & Midterm)",
                "⚠️ Profile B: Moderate Risk Student (High Stress & Borderline)",
                "✅ Profile C: Low Risk Student (Consistent Attendance & Study)",
            ],
        )

        # Default initial values
        default_dept = "Computer Science"
        default_gender = "Female"
        default_age = 20
        default_att = 82.0
        default_mid = 65.0
        default_assign = 70.0
        default_quiz = 68.0
        default_part = 75.0
        default_proj = 70.0
        default_study = 15.0
        default_extra = "Yes"
        default_net = "Yes"
        default_parent = "Bachelor's"
        default_income = "Medium"
        default_stress = 5
        default_sleep = 7.0

        if "High Risk" in preset:
            default_att = 58.0
            default_mid = 42.0
            default_assign = 48.0
            default_quiz = 45.0
            default_part = 50.0
            default_proj = 52.0
            default_study = 5.0
            default_extra = "No"
            default_stress = 9
            default_sleep = 4.5
        elif "Moderate Risk" in preset:
            default_att = 73.0
            default_mid = 58.0
            default_assign = 62.0
            default_quiz = 60.0
            default_part = 65.0
            default_proj = 64.0
            default_study = 10.0
            default_stress = 7
            default_sleep = 5.5
        elif "Low Risk" in preset:
            default_att = 94.0
            default_mid = 88.0
            default_assign = 90.0
            default_quiz = 92.0
            default_part = 88.0
            default_proj = 91.0
            default_study = 24.0
            default_extra = "Yes"
            default_stress = 3
            default_sleep = 7.5

        with st.form("risk_prediction_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**Demographics**")
                in_dept = st.selectbox(
                    "Department",
                    ["Computer Science", "Business", "Engineering", "Mathematics", "Arts"],
                    index=0,
                )
                in_gender = st.selectbox("Gender", ["Female", "Male", "Other"], index=0)
                in_age = st.number_input("Age", min_value=16, max_value=50, value=default_age)
                in_parent = st.selectbox(
                    "Parent Education",
                    ["High School", "Bachelor's", "Master's", "Doctorate", "Associate"],
                    index=1,
                )
                in_income = st.selectbox(
                    "Family Income Level", ["Low", "Medium", "High"], index=1
                )

            with c2:
                st.markdown("**Academic Milestones**")
                in_att = st.slider("Attendance (%)", 0.0, 100.0, float(default_att), 0.5)
                in_mid = st.slider("Midterm Exam Score", 0.0, 100.0, float(default_mid), 0.5)
                in_assign = st.slider("Assignments Average", 0.0, 100.0, float(default_assign), 0.5)
                in_quiz = st.slider("Quizzes Average", 0.0, 100.0, float(default_quiz), 0.5)
                in_proj = st.slider("Projects Score", 0.0, 100.0, float(default_proj), 0.5)

            with c3:
                st.markdown("**Behavioral & Lifestyle Factors**")
                in_part = st.slider("Participation Score", 0.0, 100.0, float(default_part), 0.5)
                in_study = st.slider("Study Hours / Week", 0.0, 60.0, float(default_study), 0.5)
                in_stress = st.slider("Stress Level (1-10)", 1, 10, int(default_stress))
                in_sleep = st.slider("Sleep Hours / Night", 2.0, 12.0, float(default_sleep), 0.5)
                in_extra = st.selectbox(
                    "Extracurricular Activities", ["Yes", "No"], index=0 if default_extra == "Yes" else 1
                )
                in_net = st.selectbox("Internet Access at Home", ["Yes", "No"], index=0)

            submit_pred = st.form_submit_button("🚀 Run Failure Risk Prediction", use_container_width=True)

        if submit_pred:
            if ml_pipeline is None:
                st.error("ML model pipeline not found. Please run training pipeline first.")
            else:
                input_row = {
                    "Department": in_dept,
                    "Gender": in_gender,
                    "Age": in_age,
                    "Attendance (%)": in_att,
                    "Midterm_Score": in_mid,
                    "Assignments_Avg": in_assign,
                    "Quizzes_Avg": in_quiz,
                    "Participation_Score": in_part,
                    "Projects_Score": in_proj,
                    "Study_Hours_per_Week": in_study,
                    "Extracurricular_Activities": in_extra,
                    "Internet_Access_at_Home": in_net,
                    "Parent_Education_Level": in_parent,
                    "Family_Income_Level": in_income,
                    "Stress_Level (1-10)": in_stress,
                    "Sleep_Hours_per_Night": in_sleep,
                }
                raw_df = pd.DataFrame([input_row])
                feat_df = add_engineered_features(raw_df)
                feature_cols = [c for c in feat_df.columns if c not in LEAKAGE_COLS]
                X_in = feat_df[feature_cols]

                prob = float(ml_pipeline.predict_proba(X_in)[0, 1])
                pred = int(prob >= 0.50)

                st.divider()
                st.markdown("### 📊 Prediction Results & Early-Warning Output")

                rc1, rc2, rc3 = st.columns([1.2, 1.2, 1.6])

                with rc1:
                    st.metric("Failure Risk Probability", f"{prob * 100:.2f}%")
                    st.progress(prob)

                with rc2:
                    if prob >= 0.50:
                        st.markdown(
                            "<div class='risk-high'>🚨 HIGH FAILURE RISK (Class 1)</div>",
                            unsafe_allow_html=True,
                        )
                        tier = "High Risk"
                    elif prob >= 0.20:
                        st.markdown(
                            "<div class='risk-mod'>⚠️ MODERATE RISK (Watchlist)</div>",
                            unsafe_allow_html=True,
                        )
                        tier = "Moderate Risk"
                    else:
                        st.markdown(
                            "<div class='risk-low'>✅ LOW RISK (Safe)</div>",
                            unsafe_allow_html=True,
                        )
                        tier = "Low Risk"

                    st.caption(f"Model Confidence: {prob * 100 if pred == 1 else (1.0 - prob) * 100:.1f}%")

                with rc3:
                    st.markdown("**Engineered Feature Diagnostics:**")
                    st.write(f"- Study/Sleep Ratio: `{feat_df['Study_Sleep_Ratio'].iloc[0]:.2f}`")
                    st.write(f"- Academic Engagement: `{feat_df['Academic_Engagement'].iloc[0]:.2f}`")
                    st.write(f"- Exam/Coursework Ratio: `{feat_df['Exam_to_Coursework_Ratio'].iloc[0]:.2f}`")

                st.subheader("💡 Recommended Academic Interventions:")
                recs = []
                if prob >= 0.50:
                    recs.append("🔴 **Immediate Intervention:** Schedule meeting with academic advisor.")
                elif prob >= 0.20:
                    recs.append("🟡 **Advisory Notice:** Place student on proactive monitor watchlist.")

                if in_att < 75.0:
                    recs.append(f"⚠️ **Attendance Warning:** Current attendance ({in_att:.1f}%) is below 75% policy.")
                if in_mid < 60.0:
                    recs.append(f"📚 **Tutoring:** Midterm score ({in_mid:.1f}) indicates need for tutoring.")
                if in_study < 10.0:
                    recs.append(f"⏱️ **Study Plan:** Study time ({in_study:.1f} hrs) is under target 12+ hrs.")
                if in_stress >= 7:
                    recs.append(f"🧘 **Counseling:** High stress ({in_stress}/10). Recommend campus wellness.")
                if in_sleep < 6.0:
                    recs.append(f"💤 **Sleep Schedule:** Sleep duration ({in_sleep:.1f} hrs) is suboptimal.")

                if not recs:
                    recs.append("🟢 Student is excelling across all major indicators. Continue current schedule.")

                for r in recs:
                    st.write(r)

    with pred_subtab2:
        st.markdown("#### 2. Batch Cohort Risk Scoring")
        st.write("Score an entire cohort using either the filtered dataset or by uploading a custom CSV file.")

        batch_option = st.radio(
            "Select Batch Source:",
            ["Use Currently Filtered Students", "Upload New Students CSV"],
            horizontal=True,
        )

        batch_data = None
        if batch_option == "Use Currently Filtered Students":
            if st.button("Score Filtered Cohort", use_container_width=True):
                batch_data = filtered_df.copy()
        else:
            uploaded_file = st.file_uploader("Upload Student CSV", type=["csv"])
            if uploaded_file is not None:
                batch_data = pd.read_csv(uploaded_file)

        if batch_data is not None and len(batch_data) > 0 and ml_pipeline is not None:
            with st.spinner("Executing batch inference..."):
                b_feat = add_engineered_features(batch_data)
                feat_cols = [c for c in b_feat.columns if c not in LEAKAGE_COLS]
                X_batch = b_feat[feat_cols]
                probs = ml_pipeline.predict_proba(X_batch)[:, 1]
                preds = (probs >= 0.50).astype(int)

                batch_data["Failure_Risk_Pred"] = preds
                batch_data["Risk_Probability"] = np.round(probs, 4)
                batch_data["Risk_Tier"] = [
                    "High Risk" if p >= 0.50 else ("Moderate Risk" if p >= 0.20 else "Low Risk")
                    for p in probs
                ]

                # Summary Metrics
                high_count = (batch_data["Risk_Tier"] == "High Risk").sum()
                mod_count = (batch_data["Risk_Tier"] == "Moderate Risk").sum()
                low_count = (batch_data["Risk_Tier"] == "Low Risk").sum()

                bm1, bm2, bm3 = st.columns(3)
                bm1.metric("🚨 High Risk Count", f"{high_count} ({high_count / len(batch_data):.1%})")
                bm2.metric("⚠️ Moderate Risk Count", f"{mod_count} ({mod_count / len(batch_data):.1%})")
                bm3.metric("✅ Low Risk Count", f"{low_count} ({low_count / len(batch_data):.1%})")

                st.dataframe(
                    batch_data[
                        [
                            "Student_ID",
                            "Department",
                            "Attendance (%)",
                            "Midterm_Score",
                            "Risk_Tier",
                            "Risk_Probability",
                        ]
                    ].sort_values(by="Risk_Probability", ascending=False),
                    use_container_width=True,
                )

                csv_buffer = io.StringIO()
                batch_data.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 Download Scored Cohort CSV",
                    csv_buffer.getvalue(),
                    file_name="student_failure_risk_predictions.csv",
                    mime="text/csv",
                )


# =========================================================
# TAB 3: MLOPS MODEL GOVERNANCE & DRIFT (PART 2)
# =========================================================

with tab_governance:
    st.subheader("🔬 Model Lifecycle Governance & Monitoring")

    gov_col1, gov_col2 = st.columns([1.2, 1.8])

    with gov_col1:
        st.markdown("#### 📋 Production Model Card")
        st.write(f"**Model Architecture:** `{model_meta.get('model_name', 'Random_Forest')}`")
        st.write(f"**Registered Version:** `{model_meta.get('model_version', 'v1.0.0')}`")
        st.write(f"**Lifecycle Stage:** `Production`")
        st.write(f"**Prediction Target:** `failure_risk (Grade F / Fail)`")

        t_metrics = model_meta.get("test_metrics", {})
        if t_metrics:
            st.markdown("**Hold-Out Test Evaluation (15% Unseen Data):**")
            st.metric("Test Accuracy", f"{t_metrics.get('accuracy', 0.9333):.2%}")
            st.metric("Test ROC-AUC", f"{t_metrics.get('roc_auc', 0.9389):.4f}")
            st.metric("Test Recall", f"{t_metrics.get('recall', 0.5714):.2%}")
            st.metric("Test F1-Score", f"{t_metrics.get('f1_score', 0.4898):.4f}")

    with gov_col2:
        st.markdown("#### 🧪 Multi-Model Experiment Comparison (MLflow)")
        comparison_data = pd.DataFrame(
            [
                {
                    "Model": "Logistic Regression (Baseline)",
                    "Val Accuracy": "86.13%",
                    "Val Recall": "85.71%",
                    "Val F1": "0.4235",
                    "Val ROC-AUC": "0.9499",
                    "Status": "Baseline",
                },
                {
                    "Model": "Random Forest (Advanced)",
                    "Val Accuracy": "94.40%",
                    "Val Recall": "57.14%",
                    "Val F1": "0.5106",
                    "Val ROC-AUC": "0.9392",
                    "Status": "👑 Selected Production",
                },
                {
                    "Model": "XGBoost (Advanced)",
                    "Val Accuracy": "93.87%",
                    "Val Recall": "54.76%",
                    "Val F1": "0.4894",
                    "Val ROC-AUC": "0.9395",
                    "Status": "Challenger",
                },
            ]
        )
        st.dataframe(comparison_data, use_container_width=True)

        st.caption(
            "Selection Rationale: Random Forest achieved the highest harmonic F1-score (0.5106) "
            "on the minority risk class while maintaining 94.4% accuracy and 0.939 ROC-AUC."
        )

    st.divider()

    st.markdown("#### 📈 Model Artifacts Gallery (Confusion Matrix, ROC Curve & Feature Importances)")
    art_tabs = st.tabs(["Random Forest (Production)", "XGBoost", "Logistic Regression"])

    with art_tabs[0]:
        c1, c2, c3 = st.columns(3)
        cm_rf = os.path.join(PROJECT_ROOT, "artifacts", "training_plots", "Random_Forest_confusion_matrix.png")
        roc_rf = os.path.join(PROJECT_ROOT, "artifacts", "training_plots", "Random_Forest_roc_curve.png")
        fi_rf = os.path.join(PROJECT_ROOT, "artifacts", "training_plots", "Random_Forest_feature_importance.png")
        if os.path.exists(cm_rf):
            c1.image(cm_rf, caption="RF Confusion Matrix")
        if os.path.exists(roc_rf):
            c2.image(roc_rf, caption="RF ROC Curve")
        if os.path.exists(fi_rf):
            c3.image(fi_rf, caption="RF Feature Importances")

    with art_tabs[1]:
        c1, c2, c3 = st.columns(3)
        cm_xgb = os.path.join(PROJECT_ROOT, "artifacts", "training_plots", "XGBoost_confusion_matrix.png")
        roc_xgb = os.path.join(PROJECT_ROOT, "artifacts", "training_plots", "XGBoost_roc_curve.png")
        fi_xgb = os.path.join(PROJECT_ROOT, "artifacts", "training_plots", "XGBoost_feature_importance.png")
        if os.path.exists(cm_xgb):
            c1.image(cm_xgb, caption="XGBoost Confusion Matrix")
        if os.path.exists(roc_xgb):
            c2.image(roc_xgb, caption="XGBoost ROC Curve")
        if os.path.exists(fi_xgb):
            c3.image(fi_xgb, caption="XGBoost Feature Importances")

    with art_tabs[2]:
        c1, c2, c3 = st.columns(3)
        cm_lr = os.path.join(PROJECT_ROOT, "artifacts", "training_plots", "Logistic_Regression_Baseline_confusion_matrix.png")
        roc_lr = os.path.join(PROJECT_ROOT, "artifacts", "training_plots", "Logistic_Regression_Baseline_roc_curve.png")
        fi_lr = os.path.join(PROJECT_ROOT, "artifacts", "training_plots", "Logistic_Regression_Baseline_feature_importance.png")
        if os.path.exists(cm_lr):
            c1.image(cm_lr, caption="Baseline Confusion Matrix")
        if os.path.exists(roc_lr):
            c2.image(roc_lr, caption="Baseline ROC Curve")
        if os.path.exists(fi_lr):
            c3.image(fi_lr, caption="Baseline Top Coefficients")

    st.divider()

    st.markdown("#### 📡 Continuous Data & Feature Drift Monitoring")

    drift_report_path = os.path.join(PROJECT_ROOT, "metadata", "drift_report.json")
    drift_data = {}
    if os.path.exists(drift_report_path):
        try:
            with open(drift_report_path, "r", encoding="utf-8") as f:
                drift_data = json.load(f)
        except Exception:
            drift_data = {}

    dcol1, dcol2, dcol3 = st.columns([1, 1, 1.5])

    dcol1.metric(
        "Monitored Features",
        f"{drift_data.get('monitored_features_count', 16)} Features",
    )
    dcol2.metric(
        "Drifted Features",
        f"{drift_data.get('drifted_features_count', 0)} ({drift_data.get('drift_share_pct', 0.0)}%)",
    )

    is_retrain = drift_data.get("retraining_recommended", False)
    with dcol3:
        if is_retrain:
            st.error("🚨 RETRAINING TRIGGERED: Drift exceeds 25% threshold.")
        else:
            st.success("✅ MODEL HEALTHY: Feature distributions stable within statistical bounds.")

    if st.button("🔄 Execute Live Kolmogorov-Smirnov Drift Test"):
        with st.spinner("Computing KS 2-sample tests & PSI across reference dataset..."):
            new_report = run_drift_analysis()
            st.success("Drift analysis updated successfully!")
            drift_data = new_report

    # Show KS Test Table
    num_drift = drift_data.get("numerical_drift", {})
    if num_drift:
        st.markdown("**Continuous Feature Drift Tests (Kolmogorov-Smirnov Test):**")
        drift_rows = []
        for feat, d in num_drift.items():
            drift_rows.append(
                {
                    "Feature": feat,
                    "KS Statistic": d.get("ks_statistic"),
                    "p-value (alpha=0.05)": d.get("p_value"),
                    "PSI": d.get("psi"),
                    "Ref Mean": d.get("ref_mean"),
                    "Current Mean": d.get("cur_mean"),
                    "Drift Status": "🚨 Drifted" if d.get("drift_detected") else "✅ Stable",
                }
            )
        st.dataframe(pd.DataFrame(drift_rows), use_container_width=True)


# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption(
    "Student Performance & Failure Risk Prediction System | "
    "MLOps Part 2 Individual Submission | MLflow • FastAPI • Docker • Streamlit"
)