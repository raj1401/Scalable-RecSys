"""
Common configuration for Airflow DAGs
"""
from datetime import datetime

# Spark configuration
SPARK_MASTER = "spark://spark-master:7077"
SPARK_DEPLOY_MODE = "client"
SPARK_CONTAINER = "spark-master"

# Path configurations (inside container)
WORKSPACE_PATH = "/workspace"
DATA_PATH = f"{WORKSPACE_PATH}/data"
MODELS_PATH = f"{WORKSPACE_PATH}/models"
SPARK_APPS_PATH = f"{WORKSPACE_PATH}/spark/apps"

# Data file paths
TRAIN_FILE = f"{DATA_PATH}/combined/train.txt"
TEST_FILE = f"{DATA_PATH}/combined/test.txt"
FUTURE_FILE = f"{DATA_PATH}/combined/future.txt"

# Processed data paths
PARQUET_TRAIN_PATH = f"{DATA_PATH}/processed/parquet/train"
PARQUET_TEST_PATH = f"{DATA_PATH}/processed/parquet/test"
FUTURE_CSV_PATH = f"{DATA_PATH}/processed/future.csv"

# Streaming data paths
STREAMING_BASE_PATH = f"{DATA_PATH}/streaming"
STREAMING_TRAIN_PATH = f"{STREAMING_BASE_PATH}/train"
STREAMING_TEST_PATH = f"{STREAMING_BASE_PATH}/test"

# Model paths
MODEL_ARTIFACTS_PATH = f"{MODELS_PATH}/artifacts"

# Model testing parameters
DEFAULT_K = 10
DEFAULT_RATING_THRESHOLD = 4.0

# Drift detection thresholds
DEFAULT_KL_THRESHOLD = 0.1
DEFAULT_MEAN_SHIFT_THRESHOLD = 0.2
DEFAULT_MEDIAN_SHIFT_THRESHOLD = 0.5
DEFAULT_KL_WEIGHT = 0.5
DEFAULT_MEAN_WEIGHT = 0.3
DEFAULT_MEDIAN_WEIGHT = 0.2

# Default DAG arguments
DEFAULT_DAG_ARGS = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}
