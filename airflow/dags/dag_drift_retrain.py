"""
Airflow DAG for Drift Detection and Model Retraining Pipeline

This DAG monitors for new streaming data and automatically retrains the model when drift is detected.

Task Flow:
1. process_streaming_data - Process CSV files from streaming directory
2. find_current_model - Get the latest trained model version
3. detect_drift - Detect data and model drift
4. check_retrain - Decide if retraining is needed based on drift metrics
5. merge_streaming_data - Merge streaming data into processed data (if retraining)
6. train_recommender - Retrain the model (if retraining)
7. find_latest_model - Get the newly trained model version
8. test_recommender - Evaluate the retrained model (if retraining)

The pipeline uses XCom for passing data between tasks:
- Model paths are discovered dynamically
- Drift detection results are passed to the retraining decision task
- Conditional execution based on drift detection results
"""
from datetime import datetime
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

from airflow.dags.common.config import (
    DEFAULT_DAG_ARGS,
    SPARK_APPS_PATH,
    DATA_PATH,
    PARQUET_TRAIN_PATH,
    PARQUET_TEST_PATH,
    MODEL_ARTIFACTS_PATH,
    DEFAULT_K,
    DEFAULT_RATING_THRESHOLD,
    STREAMING_BASE_PATH,
    STREAMING_TRAIN_PATH,
    STREAMING_TEST_PATH,
    DEFAULT_KL_THRESHOLD,
    DEFAULT_MEAN_SHIFT_THRESHOLD,
    DEFAULT_MEDIAN_SHIFT_THRESHOLD,
    DEFAULT_KL_WEIGHT,
    DEFAULT_MEAN_WEIGHT,
    DEFAULT_MEDIAN_WEIGHT,
)
from airflow.dags.common.model_utils import get_latest_model_version_from_path


# Drift detection thresholds (can override config defaults)
KL_THRESHOLD = DEFAULT_KL_THRESHOLD
MEAN_SHIFT_THRESHOLD = DEFAULT_MEAN_SHIFT_THRESHOLD
MEDIAN_SHIFT_THRESHOLD = DEFAULT_MEDIAN_SHIFT_THRESHOLD
KL_WEIGHT = DEFAULT_KL_WEIGHT
MEAN_WEIGHT = DEFAULT_MEAN_WEIGHT
MEDIAN_WEIGHT = DEFAULT_MEDIAN_WEIGHT


def find_current_model_version(**context):
    """
    Find the current model version before drift detection.
    This model will be used to detect model drift.
    """
    latest_version = get_latest_model_version_from_path(MODEL_ARTIFACTS_PATH)
    print(f"Found current model version: {latest_version}")
    return latest_version


def find_latest_model_version(**context):
    """
    Find the latest model version after retraining completes.
    This allows the test task to use the newly trained model.
    """
    latest_version = get_latest_model_version_from_path(MODEL_ARTIFACTS_PATH)
    print(f"Found latest model version after retraining: {latest_version}")
    return latest_version


def check_retrain_decision(**context):
    """
    Check if retraining is needed based on drift detection results.
    Uses XCom to pull drift results and makes a retraining decision.
    
    Returns:
        str: Task ID to branch to ('merge_streaming_data' if retrain, 'skip_retrain' if not)
    """
    # Import here to avoid issues with Airflow's task execution
    import sys
    sys.path.insert(0, SPARK_APPS_PATH)
    from drift_detection import should_retrain
    
    ti = context['ti']
    
    # Pull drift detection results from XCom
    drift_results = ti.xcom_pull(task_ids='detect_drift')
    
    if not drift_results:
        print("⚠️  No drift detection results found. Skipping retraining.")
        return 'skip_retrain'
    
    # Make retraining decision
    retrain_decision = should_retrain(
        data_drift_result=drift_results.get('data_drift'),
        model_drift_result=drift_results.get('model_drift'),
        kl_threshold=KL_THRESHOLD,
        mean_shift_threshold=MEAN_SHIFT_THRESHOLD,
        median_shift_threshold=MEDIAN_SHIFT_THRESHOLD,
        kl_weight=KL_WEIGHT,
        mean_weight=MEAN_WEIGHT,
        median_weight=MEDIAN_WEIGHT,
    )
    
    # Push decision to XCom for reference
    ti.xcom_push(key='retrain_decision', value=retrain_decision)
    
    should_retrain_flag = retrain_decision.get('should_retrain', False)
    drift_score = retrain_decision.get('drift_score', 0.0)
    
    if should_retrain_flag:
        print(f"✓ Retraining needed - Drift score: {drift_score:.4f}")
        print(f"  Reasons: {retrain_decision.get('reasons', [])}")
        return 'merge_streaming_data'
    else:
        print(f"✓ No retraining needed - Drift score: {drift_score:.4f}")
        return 'skip_retrain'


# DAG definition
dag = DAG(
    'drift_detection_retrain_pipeline',
    default_args=DEFAULT_DAG_ARGS,
    description='Drift detection and automated model retraining pipeline',
    schedule=None,  # Run daily to check for new streaming data
    start_date=datetime(2025, 10, 9),
    catchup=False,
    tags=['spark', 'ml', 'drift-detection', 'retraining', 'monitoring'],
    max_active_runs=1,
)


# Task 1: Process streaming data from CSV files
process_streaming_task = SparkSubmitOperator(
    task_id='process_streaming_data',
    application=f'{SPARK_APPS_PATH}/drift_detection.py',
    name='process-streaming-data',
    conn_id='spark_default',
    verbose=True,
    application_args=[
        'process',
        '--csv_path', f'{STREAMING_BASE_PATH}/current.csv',  # Expected CSV filename
        '--output_path', STREAMING_BASE_PATH,
        '--train_ratio', '0.8',
        '--shuffle_partitions', '200',
    ],
    dag=dag,
)


# Task 2: Find current model version for drift detection
find_current_model_task = PythonOperator(
    task_id='find_current_model',
    python_callable=find_current_model_version,
    dag=dag,
)


# Task 3: Detect data and model drift
detect_drift_task = SparkSubmitOperator(
    task_id='detect_drift',
    application=f'{SPARK_APPS_PATH}/drift_detection.py',
    name='detect-drift',
    conn_id='spark_default',
    verbose=True,
    application_args=[
        'detect_drift',
        '--model_path', "{{ ti.xcom_pull(task_ids='find_current_model') }}",
        '--original_train_path', PARQUET_TRAIN_PATH,
        '--current_train_path', STREAMING_TRAIN_PATH,
        '--original_test_path', PARQUET_TEST_PATH,
        '--current_test_path', STREAMING_TEST_PATH,
        '--shuffle_partitions', '200',
    ],
    dag=dag,
)


# Task 4: Check if retraining is needed (branching task)
check_retrain_task = BranchPythonOperator(
    task_id='check_retrain',
    python_callable=check_retrain_decision,
    dag=dag,
)


# Task 5: Skip retraining (dummy task for "no retrain" branch)
skip_retrain_task = EmptyOperator(
    task_id='skip_retrain',
    dag=dag,
)


# Task 6: Merge streaming data into processed data
merge_streaming_task = SparkSubmitOperator(
    task_id='merge_streaming_data',
    application=f'{SPARK_APPS_PATH}/drift_detection.py',
    name='merge-streaming-data',
    conn_id='spark_default',
    verbose=True,
    application_args=[
        'merge',
        '--streaming_path', STREAMING_BASE_PATH,
        '--processed_path', f'{DATA_PATH}/processed/parquet',
        '--shuffle_partitions', '200',
    ],
    dag=dag,
)


# Task 7: Train new model with merged data
train_task = SparkSubmitOperator(
    task_id='train_recommender',
    application=f'{SPARK_APPS_PATH}/train_recommender.py',
    name='als-retraining',
    conn_id='spark_default',
    verbose=True,
    application_args=[
        '--parquet-train-path', PARQUET_TRAIN_PATH,
        '--model-save-path', MODEL_ARTIFACTS_PATH,
    ],
    dag=dag,
)


# Task 8: Find latest model version after retraining
find_latest_model_task = PythonOperator(
    task_id='find_latest_model',
    python_callable=find_latest_model_version,
    dag=dag,
)


# Task 9: Test the retrained model
test_task = SparkSubmitOperator(
    task_id='test_recommender',
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
# Linear flow until drift detection
process_streaming_task >> find_current_model_task >> detect_drift_task >> check_retrain_task

# Branch: If retraining is needed
check_retrain_task >> merge_streaming_task >> train_task >> find_latest_model_task >> test_task

# Branch: If retraining is not needed
check_retrain_task >> skip_retrain_task
