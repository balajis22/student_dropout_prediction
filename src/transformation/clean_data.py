import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
raw_file = RAW_DATA_DIR / "Students Performance Dataset.csv"


def clean_data():

    cleaned_df = pd.read_csv(raw_file)

    # Transformations

    cleaned_df["Parent_Education_Level"] = (
        cleaned_df["Parent_Education_Level"].fillna("Unknown")
    )

    pii_columns = ["First_Name", "Last_Name", "Email"]
    cleaned_df = cleaned_df.drop(columns=pii_columns)

    assessment_columns = [
        "Midterm_Score",
        "Final_Score",
        "Assignments_Avg",
        "Quizzes_Avg",
        "Participation_Score",
        "Projects_Score"
    ]

    cleaned_df["Average_Assessment_Score"] = (
        cleaned_df[assessment_columns].mean(axis=1)
    )

    cleaned_path = (
        PROJECT_ROOT
        / "data"
        / "cleaned"
        / "students_cleaned.csv"
    )

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(
        cleaned_path,
        index=False
    )

    print("Cleaning successful!")
    print("Cleaned dataset shape:", cleaned_df.shape)
    print("Missing values:", cleaned_df.isnull().sum().sum())

    return cleaned_df


if __name__ == "__main__":
    clean_data()