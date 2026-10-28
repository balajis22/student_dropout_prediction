import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
LOGS_DIR = PROJECT_ROOT / "logs"

RAW_FILE = RAW_DATA_DIR / "Students Performance Dataset.csv"
LOG_FILE = LOGS_DIR / "ingestion_log.csv"

def ingest_data():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    extraction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        df = pd.read_csv(RAW_FILE)

        log_record = pd.DataFrame([{
            "extraction_time": extraction_time,
            "source": "Local CSV",
            "file": RAW_FILE.name,
            "status": "SUCCESS",
            "row_count": len(df),
            "error": ""
        }])

        if LOG_FILE.exists():
            log_record.to_csv(
                LOG_FILE,
                mode="a",
                header=False,
                index=False
            )
        else:
            log_record.to_csv(
                LOG_FILE,
                index=False
            )

        print("Ingestion successful!")
        print("Rows ingested:", len(df))

        return df

    except Exception as error:

        log_record = pd.DataFrame([{
            "extraction_time": extraction_time,
            "source": "Local CSV",
            "file": RAW_FILE.name,
            "status": "FAILED",
            "row_count": 0,
            "error": str(error)
        }])

        if LOG_FILE.exists():
            log_record.to_csv(
                LOG_FILE,
                mode="a",
                header=False,
                index=False
            )
        else:
            log_record.to_csv(
                LOG_FILE,
                index=False
            )

        print("Ingestion failed!")
        print("Error:", error)

        raise

if __name__ == "__main__":
    ingest_data()