# Airflow DAGs

Airflow DAGs for orchestrating the ML recommendation pipeline.

## Structure

```
airflow/dags/
├── common/                    # Shared utilities
│   ├── config.py              # Configuration constants
│   ├── spark_utils.py         # Spark command builders
│   └── model_utils.py         # Model version utilities
├── dag_etl_train.py           # Main ML pipeline
└── dag_drift_retrain.py       # Drift detection & retraining
```

## DAGs Overview

### 1. `etl_train_test_pipeline` (dag_etl_train.py)
Initial model training pipeline - runs once on manual trigger.

### 2. `drift_detection_retrain_pipeline` (dag_drift_retrain.py)
Automated drift monitoring and retraining - runs daily to check for new data.

## Main DAG: `etl_train_test_pipeline`

**File:** `dag_etl_train.py`

Runs the complete ML pipeline (manual trigger only):

1. `spark_etl` - Process raw data to Parquet
2. `spark_create_futures` - Generate future predictions dataset
3. `spark_train_recommender` - Train ALS model
4. `find_latest_model` - Discover newly trained model version
5. `spark_test_recommender` - Evaluate model with dynamic version

**Task Flow:**
```
etl → create_futures
  └→ train → find_latest_model → test
```

**Key Features:**
- Uses `SparkSubmitOperator` for native Spark integration
- XCom passes latest model version from training to testing
- Runs once per trigger (no scheduling)
- Tags: `spark`, `ml`, `etl`, `training`, `testing`

---

## Drift Detection DAG: `drift_detection_retrain_pipeline`

**File:** `dag_drift_retrain.py`

Monitors for new streaming data and automatically retrains the model when drift is detected (runs daily):

1. `process_streaming_data` - Process new CSV files from streaming directory
2. `find_current_model` - Get the current trained model version
3. `detect_drift` - Detect data and model drift using the current model
4. `check_retrain` - Decide if retraining is needed based on drift metrics
5. `merge_streaming_data` - Merge streaming data into processed data (if retraining needed)
6. `train_recommender` - Retrain the ALS model (if retraining needed)
7. `find_latest_model` - Get the newly trained model version
8. `test_recommender` - Evaluate the retrained model (if retraining needed)

**Task Flow:**
```
process_streaming_data → find_current_model → detect_drift → check_retrain
                                                                    ├→ skip_retrain (no drift)
                                                                    └→ merge_streaming_data → train_recommender
                                                                        → find_latest_model → test_recommender
```

**Key Features:**
- **Automated Drift Detection**: Monitors data and model performance drift
- **Conditional Retraining**: Only retrains when drift exceeds thresholds
- **XCom Integration**: Passes drift metrics and model paths between tasks
- **Branch Operator**: Uses `BranchPythonOperator` for conditional execution
- **Daily Schedule**: Checks for new streaming data every day
- Tags: `spark`, `ml`, `drift-detection`, `retraining`, `monitoring`

**Drift Thresholds:**
- KL Divergence: 0.1
- Mean Shift: 0.2
- Median Shift: 0.5
- Weighted scoring: KL (50%), Mean (30%), Median (20%)

**Input Data:**
- Expects CSV files in `/workspace/data/streaming/current.csv`
- CSV format: `DATE,CUST_ID,MOVIE_ID,RATING`
- Processed into train/test splits (80/20)

**Drift Detection Logic:**
The DAG compares:
1. **Data Drift**: Original training data vs. new streaming data
2. **Model Drift**: Model predictions on original test vs. new test data

If combined drift score > 1.0, retraining is triggered.

---

## How It Works

### SparkSubmitOperator

Uses Airflow's native Spark operator instead of BashOperator:
- Better Spark integration and error handling
- Uses Airflow connections (`spark_default`)
- No docker exec wrappers needed
- Cleaner code and logs

**Example:**
```python
SparkSubmitOperator(
    task_id='spark_etl',
    application='/workspace/spark/apps/etl.py',
    conn_id='spark_default',
    application_args=['--train-file', '/workspace/data/combined/train.txt', ...],
)
```

### Dynamic Model Versioning (XCom)

The `find_latest_model` task automatically discovers the latest trained model and passes it to the test task:

```python
def find_latest_model_version(**context):
    latest = get_latest_model_version_from_path('/workspace/models/artifacts/')
    return latest  # Pushed to XCom

test_task = SparkSubmitOperator(
    application_args=['--model-path', "{{ ti.xcom_pull(task_ids='find_latest_model') }}"]
)
```

No hardcoded model versions needed.

### Drift Detection with XCom

The drift detection DAG uses XCom to pass drift results between tasks:

```python
# Detect drift task returns results
detect_drift_task = SparkSubmitOperator(
    task_id='detect_drift',
    application='drift_detection.py',
    application_args=['detect_drift', '--model_path', '...']
)

# Check retrain task pulls results from XCom
def check_retrain_decision(**context):
    drift_results = context['ti'].xcom_pull(task_ids='detect_drift')
    
    retrain_decision = should_retrain(
        data_drift_result=drift_results['data_drift'],
        model_drift_result=drift_results['model_drift']
    )
    
    if retrain_decision['should_retrain']:
        return 'merge_streaming_data'  # Branch to retraining
    else:
        return 'skip_retrain'  # Skip retraining
```

The `BranchPythonOperator` dynamically routes the workflow based on drift detection.

## Common Utilities

### `common/config.py`
- Spark connection settings
- File paths (data, models, apps)
- Default DAG arguments
- Model parameters (k=10, rating_threshold=4.0)

### `common/spark_utils.py`
- `build_spark_submit_command()` - Build spark-submit commands
- `build_docker_spark_command()` - Build docker exec commands

### `common/model_utils.py`
- `get_latest_model_version_from_path()` - Scan and find latest model
- `validate_model_version()` - Validate model directory structure

---

## Setup

### 1. Install Spark Provider
```bash
pip install apache-airflow-providers-apache-spark
```

### 2. Configure Spark Connection

**Via Environment Variable (Docker):**
```yaml
AIRFLOW_CONN_SPARK_DEFAULT: spark://spark-master:7077
```

**Via Airflow UI:**
- Admin → Connections → Add
- Connection Id: `spark_default`
- Type: `Spark`
- Host: `spark-master`, Port: `7077`

### 3. Run

```bash
# Start services
make run-all

# Trigger initial training DAG
make airflow-trigger-dag dag=etl_train_test_pipeline

# Trigger drift detection DAG
make airflow-trigger-dag dag=drift_detection_retrain_pipeline

# Or via Airflow UI
open http://localhost:8081
```

### 4. Add Streaming Data

To test the drift detection pipeline, add new data to the streaming directory:

```bash
# Copy or create a CSV file with new ratings
# Format: DATE,CUST_ID,MOVIE_ID,RATING
cp new_ratings.csv data/streaming/current.csv

# The next scheduled run will process this file
# Or trigger manually via Airflow UI
```

---

## Task Arguments Reference

### ETL Pipeline

**ETL:**
```
--train-file /workspace/data/combined/train.txt
--test-file /workspace/data/combined/test.txt
--parquet-train-path /workspace/data/processed/parquet/train
--parquet-test-path /workspace/data/processed/parquet/test
```

**Create Futures:**
```
--future-file /workspace/data/combined/future.txt
--csv-output-path /workspace/data/processed/future.csv
```

**Train:**
```
--parquet-train-path /workspace/data/processed/parquet/train
--model-output-path /workspace/models/artifacts/version_{timestamp}
```

**Test:**
```
--model-path {dynamic from XCom}
--test-parquet /workspace/data/processed/parquet/test
--k 10
--rating-threshold 4.0
```

### Drift Detection Pipeline

**Process Streaming:**
```
process
--csv_path /workspace/data/streaming/current.csv
--output_path /workspace/data/streaming
--train_ratio 0.8
--shuffle_partitions 200
```

**Detect Drift:**
```
detect_drift
--model_path {dynamic from XCom}
--original_train_path /workspace/data/processed/parquet/train
--current_train_path /workspace/data/streaming/train
--original_test_path /workspace/data/processed/parquet/test
--current_test_path /workspace/data/streaming/test
--shuffle_partitions 200
```

**Merge Streaming:**
```
merge
--streaming_path /workspace/data/streaming
--processed_path /workspace/data/processed/parquet
--shuffle_partitions 200
```

---

---

## Workflow Diagrams

### Initial Training Pipeline
```
┌─────────────┐
│     ETL     │
└──────┬──────┘
       │
       ├──────────────────┐
       │                  │
       ▼                  ▼
┌──────────────┐   ┌────────────┐
│Create Futures│   │   Train    │
└──────────────┘   └─────┬──────┘
                         │
                         ▼
                  ┌──────────────┐
                  │Find Latest   │
                  │Model         │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │    Test      │
                  └──────────────┘
```

### Drift Detection & Retraining Pipeline
```
┌──────────────────┐
│  Process         │
│  Streaming Data  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Find Current    │
│  Model           │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Detect Drift    │
│  (Data+Model)    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Check Retrain   │◄──── Decision Point
│  (Branch)        │
└────┬────────┬────┘
     │        │
     │        │ No Drift
     │        └──────────► Skip (End)
     │
     │ Drift Detected
     ▼
┌──────────────────┐
│  Merge Streaming │
│  Data            │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Train           │
│  Recommender     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Find Latest     │
│  Model           │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Test            │
│  Recommender     │
└──────────────────┘
```

---

## Adding New DAGs

```python
from common.config import DEFAULT_DAG_ARGS, SPARK_APPS_PATH

dag = DAG(
    'my_dag',
    default_args=DEFAULT_DAG_ARGS,
    schedule_interval=None,
    tags=['my-tag'],
)

task = SparkSubmitOperator(
    task_id='my_task',
    application=f'{SPARK_APPS_PATH}/my_app.py',
    conn_id='spark_default',
    application_args=['--arg', 'value'],
    dag=dag,
)
```

## Notes

- All paths use container paths (`/workspace/...`)
- Docker Compose ensures Airflow and Spark are on same network (`spark-net`)
- See `../DOCKER_SETUP.md` for comprehensive Docker configuration

### Drift Detection Behavior

**Daily Monitoring:**
- The drift detection DAG runs daily (`schedule_interval='@daily'`)
- Checks for new CSV files in `/workspace/data/streaming/`
- If no new data, the pipeline will fail at the `process_streaming_data` step

**Retraining Decision:**
- Drift score is computed as a weighted combination of:
  - KL divergence between rating distributions
  - Mean shift in ratings
  - Median shift in ratings
- If drift score > 1.0, retraining is triggered
- Otherwise, the pipeline skips retraining and ends

**Data Management:**
- Streaming data is merged into processed data only when retraining occurs
- Original CSV files are deleted after successful processing
- Train/test splits are maintained at 80/20 ratio

**Model Versioning:**
- Each training run creates a new model version with timestamp
- Model paths are discovered dynamically via XCom
- Latest model is always used for drift detection and testing

### Monitoring and Alerts

To monitor the drift detection pipeline:

1. **Check Airflow UI** for task status and logs
2. **Review XCom values** to see drift metrics and retraining decisions
3. **Monitor model versions** in `/workspace/models/artifacts/`
4. **Set up Airflow alerts** for task failures or retraining events

Example: Check drift decision in XCom:
```python
# In Airflow UI → Admin → XCom
# Key: retrain_decision
# Value: {"should_retrain": true, "drift_score": 1.25, ...}
```
