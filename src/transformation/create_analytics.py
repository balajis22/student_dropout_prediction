import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEANED_DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
cleaned_file = CLEANED_DATA_DIR / "students_cleaned.csv"


def create_analytics():

    analytics_df = pd.read_csv(cleaned_file)

    analytics_columns = [
        "Student_ID",
        "Department",
        "Gender",
        "Age",
        "Attendance (%)",
        "Study_Hours_per_Week",
        "Average_Assessment_Score",
        "Total_Score",
        "Grade",
        "Assignments_Avg",
        "Quizzes_Avg",
        "Participation_Score",
        "Projects_Score",
        "Stress_Level (1-10)",
        "Sleep_Hours_per_Night",
        "Parent_Education_Level",
        "Family_Income_Level",
        "Extracurricular_Activities",
        "Internet_Access_at_Home"
    ]

    analytics_df = analytics_df[analytics_columns]

    analytics_path = (
        PROJECT_ROOT
        / "data"
        / "analytics"
        / "student_performance_mart.csv"
    )

    analytics_path.parent.mkdir(parents=True, exist_ok=True)

    analytics_df.to_csv(
        analytics_path,
        index=False
    )

    analytics_check = pd.read_csv(analytics_path)

    print("Analytics mart created successfully!")
    print("Analytics dataset shape:", analytics_check.shape)
    print("Missing values:", analytics_check.isnull().sum().sum())

    return analytics_check


if __name__ == "__main__":
    create_analytics()