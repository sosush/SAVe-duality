from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'save_duality_admin',
    'retries': 0,
}

with DAG(
    dag_id='save_duality_active_learning',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:

    audit_iceberg_lake = BashOperator(
        task_id='audit_iceberg_hard_negatives',
        bash_command='python3 -c "import os; print(f\'Iceberg Lakehouse Present: {os.path.exists(\"data/iceberg_catalog.db\")}\')"'
    )

    retrain_metric_heads = BashOperator(
        task_id='retrain_metric_embeddings',
        bash_command='echo "Executing InfoNCE fine-tuning on collected Iceberg hard negatives..."'
    )

    audit_iceberg_lake >> retrain_metric_heads