import os
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

password = os.getenv("POSTGRES_PASSWORD")

if not password:
    raise ValueError("POSTGRES_PASSWORD was not found in .env")


# --------------------------------------------------
# 2. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

analytics_file = (
    PROJECT_ROOT
    / "data"
    / "analytics"
    / "student_performance_mart.csv"
)


# --------------------------------------------------
# 3. PostgreSQL settings
# --------------------------------------------------

DB_NAME = "student_performance_db"
DB_USER = "postgres"
DB_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
DB_PORT = "5432"


# --------------------------------------------------
# 4. Read analytical CSV
# --------------------------------------------------

analytics_df = pd.read_csv(analytics_file)

print("Rows to load:", len(analytics_df))
print("Columns in CSV:", len(analytics_df.columns))


# --------------------------------------------------
# 5. Keep the exact column order
# --------------------------------------------------

columns = [
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

analytics_df = analytics_df[columns]

print("Column order verified.")


# --------------------------------------------------
# 6. Convert rows to tuples
# --------------------------------------------------

rows = list(
    analytics_df.itertuples(
        index=False,
        name=None
    )
)

print("Values per row:", len(rows[0]))


# --------------------------------------------------
# 7. Connect to PostgreSQL
# --------------------------------------------------

connection = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=password,
    host=DB_HOST,
    port=DB_PORT
)

print("Connected to PostgreSQL successfully!")


# --------------------------------------------------
# 8. Create cursor
# --------------------------------------------------

cursor = connection.cursor()


# --------------------------------------------------
# 9. SQL INSERT statement
# --------------------------------------------------

insert_query = """
INSERT INTO public.student_performance_mart (
    Student_ID,
    Department,
    Gender,
    Age,
    "Attendance (%%)",
    Study_Hours_per_Week,
    Average_Assessment_Score,
    Total_Score,
    Grade,
    Assignments_Avg,
    Quizzes_Avg,
    Participation_Score,
    Projects_Score,
    "Stress_Level (1-10)",
    Sleep_Hours_per_Night,
    Parent_Education_Level,
    Family_Income_Level,
    Extracurricular_Activities,
    Internet_Access_at_Home
)
VALUES %s
"""


# --------------------------------------------------
# 10. Load the data
# --------------------------------------------------

try:

    execute_values(
        cursor,
        insert_query,
        rows
    )

    connection.commit()

    print("Data loaded successfully!")
    print("Total rows inserted:", len(rows))


    # --------------------------------------------------
    # 11. Verify the inserted data
    # --------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM public.student_performance_mart;
    """)

    count = cursor.fetchone()[0]

    print("Rows currently in PostgreSQL:", count)


except Exception as error:

    connection.rollback()

    print("Error while loading data:")
    print(error)


finally:

    cursor.close()
    connection.close()

    print("PostgreSQL connection closed.")