from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="student_performance_pipeline",
    start_date=datetime(2026, 9, 1),
    schedule="@daily",
    catchup=False,
) as dag:

    ingestion_task = BashOperator(
        task_id="ingestion",
        bash_command="python /opt/airflow/src/ingestion/ingest_data.py",
    )

    validation_task = BashOperator(
        task_id="validation",
        bash_command="python /opt/airflow/src/validation/validate_data.py",
    )

    cleaning_task = BashOperator(
        task_id="cleaning",
        bash_command="python /opt/airflow/src/transformation/clean_data.py",
    )

    analytics_task = BashOperator(
        task_id="analytics",
        bash_command="python /opt/airflow/src/transformation/create_analytics.py",
    )

    postgres_task = BashOperator(
        task_id="load_to_postgres",
        bash_command="python /opt/airflow/src/ingestion/load_to_postgres.py",
    )

    ingestion_task >> validation_task >> cleaning_task >> analytics_task >> postgres_task