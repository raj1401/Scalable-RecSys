"""
Airflow DAG for ETL, Training, and Testing Pipeline (Using SparkSubmitOperator)
This DAG runs once and executes the complete ML pipeline:
1. ETL: Process raw data into parquet format
2. Create Futures: Generate future predictions dataset
3. Train: Train the ALS recommender model
4. Find Latest Model: Dynamically discover the trained model version
5. Test: Evaluate the trained model using the discovered version

This version uses SparkSubmitOperator which provides:
- Better integration with Spark
- Cleaner error handling
- Direct connection management
- No need for docker exec wrapper commands
- Dynamic model version resolution via XCom
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.python import PythonOperator

from airflow.dags.common.config import (
    DEFAULT_DAG_ARGS,
    SPARK_MASTER,
    TRAIN_FILE,
    TEST_FILE,
    FUTURE_FILE,
    PARQUET_TRAIN_PATH,
    PARQUET_TEST_PATH,
    FUTURE_CSV_PATH,
    MODEL_ARTIFACTS_PATH,
    DEFAULT_K,
    DEFAULT_RATING_THRESHOLD,
    SPARK_APPS_PATH,
)
from airflow.dags.common.model_utils import get_latest_model_version_from_path


def find_latest_model_version(**context):
    """
    Find the latest model version after training completes and push to XCom.
    This allows the test task to dynamically use the newly trained model.
    """
    latest_version = get_latest_model_version_from_path(MODEL_ARTIFACTS_PATH)
    print(f"Found latest model version: {latest_version}")
    return latest_version


# DAG definition
dag = DAG(
    'etl_train_test_pipeline',
    default_args=DEFAULT_DAG_ARGS,
    description='Complete ETL, training, and testing pipeline using SparkSubmitOperator',
    schedule=None,  # Run only when triggered manually (runs once)
    start_date=datetime(2025, 10, 9),
    catchup=False,
    tags=['spark', 'ml', 'etl', 'training', 'testing'],
    max_active_runs=1,
)


# Task 1: ETL - Process raw data into parquet format
etl_task = SparkSubmitOperator(
    task_id='spark_etl',
    application=f'{SPARK_APPS_PATH}/etl.py',
    name='netflix-etl',
    conn_id='spark_default',  # Can be configured in Airflow connections
    deploy_mode='client',
    verbose=True,
    application_args=[
        '--train-file', TRAIN_FILE,
        '--test-file', TEST_FILE,
        '--parquet-train-path', PARQUET_TRAIN_PATH,
        '--parquet-test-path', PARQUET_TEST_PATH,
    ],
    dag=dag,
)


# Task 2: Create Futures - Generate future predictions dataset
create_futures_task = SparkSubmitOperator(
    task_id='spark_create_futures',
    application=f'{SPARK_APPS_PATH}/create_futures.py',
    name='netflix-create-futures',
    conn_id='spark_default',
    verbose=True,
    application_args=[
        '--future-file', FUTURE_FILE,
        '--csv-output-path', FUTURE_CSV_PATH,
    ],
    dag=dag,
)


# Task 3: Train - Train the ALS recommender model
train_task = SparkSubmitOperator(
    task_id='spark_train_recommender',
    application=f'{SPARK_APPS_PATH}/train_recommender.py',
    name='als-training',
    conn_id='spark_default',
    verbose=True,
    application_args=[
        '--parquet-train-path', PARQUET_TRAIN_PATH,
        '--model-save-path', MODEL_ARTIFACTS_PATH,
    ],
    dag=dag,
)


# Task 4: Find Latest Model Version - Dynamically discover the trained model
find_model_task = PythonOperator(
    task_id='find_latest_model',
    python_callable=find_latest_model_version,
    dag=dag,
)


# Task 5: Test - Evaluate the trained model using dynamic model version from XCom
# The model path is retrieved from the find_latest_model task via XCom
test_task = SparkSubmitOperator(
    task_id='spark_test_recommender',
    application=f'{SPARK_APPS_PATH}/test_recommender.py',
    name='als-evaluation',
    conn_id='spark_default',
    verbose=True,
    application_args=[
        '--model-path', "{{ ti.xcom_pull(task_ids='find_latest_model') }}",
        '--test-parquet', PARQUET_TEST_PATH,
        '--k', str(DEFAULT_K),
        '--rating-threshold', str(DEFAULT_RATING_THRESHOLD),
    ],
    dag=dag,
)


# Define task dependencies
# ETL must complete before create_futures and train
etl_task >> [create_futures_task, train_task]

# After training, find the latest model version, then run tests
train_task >> find_model_task >> test_task
