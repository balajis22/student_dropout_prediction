import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

file_path = RAW_DATA_DIR / "Students Performance Dataset.csv"

df = pd.read_csv(file_path)
print(df.head())
print(df.columns.to_list())
print(df.info())
print(df.describe())