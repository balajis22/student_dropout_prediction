import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
LOGS_DIR = PROJECT_ROOT / "logs"

RAW_FILE = RAW_DATA_DIR / "Students Performance Dataset.csv"
REJECTED_FILE = LOGS_DIR / "rejected_records.csv"

def validate_data():
    df = pd.read_csv(RAW_FILE)

    invalid_mask = (
        df["Student_ID"].isna()
        | df["Student_ID"].duplicated(keep=False)
        | ~df["Attendance (%)"].between(0, 100)
        | ~df["Midterm_Score"].between(0, 100)
        | ~df["Final_Score"].between(0, 100)
        | ~df["Assignments_Avg"].between(0, 100)
        | ~df["Quizzes_Avg"].between(0, 100)
        | ~df["Participation_Score"].between(0, 100)
        | ~df["Projects_Score"].between(0, 100)
        | ~df["Age"].between(18, 25)
    )

    rejected_df = df[invalid_mask].copy()

    if len(rejected_df) > 0:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        rejected_df["rejection_time"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        rejected_df.to_csv(
            REJECTED_FILE,
            index=False
        )

        print("Validation failed!")
        print("Rejected records:", len(rejected_df))
        print("Rejected records saved to:", REJECTED_FILE)

        raise ValueError(
            f"Validation failed with {len(rejected_df)} rejected records."
        )

    print("Validation successful!")
    print("Rows validated:", len(df))

if __name__ == "__main__":
    validate_data()