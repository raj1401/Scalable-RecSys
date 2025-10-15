# Spark Applications for Large-Scale Recommendation System

This directory contains PySpark applications for building, training, and maintaining a scalable Netflix-style movie recommendation system using Apache Spark's ALS (Alternating Least Squares) collaborative filtering algorithm.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Spark Applications](#spark-applications)
- [Interacting with the Spark Cluster](#interacting-with-the-spark-cluster)
- [Data Flow](#data-flow)
- [Common Utilities](#common-utilities)
- [Configuration](#configuration)

## 🎯 Overview

This project implements a complete ML pipeline for collaborative filtering at scale, processing the Netflix Prize dataset (100M+ ratings) using distributed computing. The system supports:

- **ETL**: Parsing and preprocessing large text files into partitioned Parquet datasets
- **Training**: Distributed ALS model training with hyperparameter tuning
- **Evaluation**: Comprehensive model testing with regression and ranking metrics
- **Drift Detection**: Monitoring data and model drift for automated retraining
- **Streaming**: Processing new data and merging into existing datasets

## 🏗️ Architecture

### Spark Cluster Setup

The Spark cluster is deployed using Docker Compose with the following components:

- **Spark Master** (`spark-master`): Cluster coordinator and job scheduler

  - Web UI: http://localhost:8080
  - Master URL: `spark://spark-master:7077`
  - REST API: http://localhost:6066

- **Spark Workers** (`spark-worker`): Distributed compute nodes

  - Configurable cores and memory (default: 2 cores, 2GB RAM)
  - Scalable via `docker compose up --scale spark-worker=N`
  - Worker Web UI: http://localhost:8081

- **Spark History Server** (`spark-history`): Historical job monitoring
  - Web UI: http://localhost:18080
  - Persistent event logs in shared volume

All components are containerized using Python 3.13 with Spark 4.0.1 and Java 21, managed via the `spark/Dockerfile`.

### Integration with Airflow

The Spark jobs are orchestrated via Apache Airflow using `SparkSubmitOperator`, which:

- Submits jobs directly to the Spark cluster (no shell wrapper needed)
- Manages connection lifecycle and error handling
- Enables dynamic parameter passing via XCom
- Provides detailed logging and monitoring

## 📦 Spark Applications

### 1. ETL Pipeline (`etl.py`)

**Purpose**: Parse raw Netflix-format text files and create optimized Parquet datasets.

**Key Features**:

- Parses Netflix Prize format (movie IDs as headers, ratings as rows)
- Handles malformed/missing data gracefully
- Partitions output by year and month for efficient querying
- Upserts new data into existing datasets with deduplication
- Configurable shuffle partitions and repartitioning for performance tuning

**Input Format**:

```
12345:
1234567,5,2005-12-25
9876543,4,2005-12-26
```

**Output Schema**:

```
MOVIE_ID (int), CUST_ID (int), RATING (int), DATE (date), year (int), month (int)
```

**Usage**:

```bash
spark-submit spark/apps/etl.py \
  --train-file /workspace/data/combined/train.txt \
  --test-file /workspace/data/combined/test.txt \
  --parquet-train-path /workspace/data/processed/parquet/train \
  --parquet-test-path /workspace/data/processed/parquet/test \
  --shuffle-partitions 400 \
  --target-repartition 400
```

**Parameters**:

- `--train-file`, `--test-file`: Input text files in Netflix format
- `--parquet-train-path`, `--parquet-test-path`: Output paths for Parquet datasets
- `--shuffle-partitions`: Number of partitions for shuffle operations (default: 400)
- `--target-repartition`: Repartition count before writing (controls file sizes)
- `--extra-write-options-json`: JSON dict for Parquet writer options

### 2. Training Pipeline (`train_recommender.py`)

**Purpose**: Train ALS collaborative filtering model on distributed ratings data.

**Key Features**:

- Implements matrix factorization via Spark MLlib's ALS algorithm
- Learns latent user and item factors from rating patterns
- Supports implicit feedback and nonnegative constraints
- Automatic model versioning with timestamps
- Saves complete model artifacts for inference and evaluation

**ALS Algorithm**:

- Alternating Least Squares collaborative filtering
- Factorizes user-item rating matrix into latent factors
- Enables scalable training on 100M+ ratings
- Handles cold start scenarios via configurable strategies

**Usage**:

```bash
spark-submit spark/apps/train_recommender.py \
  --parquet-train-path /workspace/data/processed/parquet/train \
  --model-save-path models/artifacts \
  --als-rank 64 \
  --als-reg 0.1 \
  --als-max-iter 15 \
  --shuffle-partitions 400
```

**Hyperparameters**:

- `--als-rank`: Dimensionality of latent factors (default: 64)
- `--als-reg`: Regularization parameter to prevent overfitting (default: 0.1)
- `--als-max-iter`: Maximum training iterations (default: 15)
- `--als-implicit`: Use implicit feedback mode (boolean flag)
- `--als-nonnegative`: Enforce nonnegative constraints (boolean flag)
- `--als-cold-start`: Strategy for cold start users/items (default: "drop")

**Model Output**:

- Saved to: `models/artifacts/version_YYYYMMDD_HHMMSS/`
- Contains: User factors, item factors, model metadata

### 3. Model Evaluation (`test_recommender.py`)

**Purpose**: Comprehensive evaluation using regression and ranking metrics.

**Metrics Computed**:

1. **Regression Metrics** (rating prediction accuracy):

   - **RMSE** (Root Mean Squared Error): Measures prediction error magnitude
   - **MAE** (Mean Absolute Error): Average absolute deviation from actual ratings

2. **Ranking Metrics** (recommendation quality):
   - **Precision@K**: Fraction of top-K recommendations that are relevant
   - **Recall@K**: Fraction of relevant items found in top-K recommendations
   - **NDCG@K** (Normalized Discounted Cumulative Gain): Ranking quality with position-based discounting

**Usage**:

```bash
spark-submit spark/apps/test_recommender.py \
  --model-path models/artifacts/version_20250930_212327 \
  --test-parquet /workspace/data/processed/parquet/test \
  --k 10 \
  --rating-threshold 4.0
```

**Parameters**:

- `--model-path`: Path to trained model artifacts
- `--test-parquet`: Path to test dataset (Parquet format)
- `--k`: Number of recommendations for ranking metrics (default: 10)
- `--rating-threshold`: Minimum rating to consider item "relevant" (default: 4.0)

**Output Example**:

```
================================
📊 Model Evaluation Results
================================
Model: models/artifacts/version_20250930_212327
Test Dataset: /workspace/data/processed/parquet/test

Regression Metrics (Rating Prediction):
  RMSE: 0.8542
  MAE:  0.6721

Ranking Metrics (Recommendation Quality):
  Precision@10: 0.7234
  Recall@10:    0.4521
  NDCG@10:      0.8123
```

### 4. Drift Detection (`drift_detection.py`)

**Purpose**: Monitor data distribution and model performance for automated retraining decisions.

**Drift Types Detected**:

1. **Data Drift**: Changes in rating distribution

   - **KL Divergence**: Measures distribution shift between original and current ratings
   - **Mean Shift**: Absolute difference in average ratings
   - **Median Shift**: Absolute difference in median ratings

2. **Model Drift**: Changes in prediction quality
   - **RMSE Comparison**: Prediction error on original vs. current test data
   - **MAE Comparison**: Average error magnitude comparison

**Operations**:

#### a) Process Streaming Data

Loads CSV files, splits into train/test, and saves as Parquet:

```bash
spark-submit spark/apps/drift_detection.py process \
  --csv_path data/streaming/current.csv \
  --output_path data/streaming \
  --train_ratio 0.8
```

#### b) Detect Drift

Analyzes data and model drift, returns results for retraining decision:

```bash
spark-submit spark/apps/drift_detection.py detect_drift \
  --model_path models/artifacts/version_20250930_212327 \
  --original_train_path data/processed/parquet/train \
  --original_test_path data/processed/parquet/test \
  --streaming_train_path data/streaming/train \
  --streaming_test_path data/streaming/test
```

#### c) Merge Data

Upserts streaming data into processed datasets:

```bash
spark-submit spark/apps/drift_detection.py merge \
  --streaming_path data/streaming \
  --processed_path data/processed/parquet
```

**Retraining Decision Logic**:

- Weighted heuristic combining all drift metrics
- Configurable thresholds and weights
- Returns boolean decision via `should_retrain()` function
- Integrated with Airflow for automated pipeline triggering

**Default Thresholds**:

- KL Divergence: 0.1 (weight: 0.5)
- Mean Shift: 0.2 (weight: 0.3)
- Median Shift: 0.5 (weight: 0.2)

### 5. Future Data Processing (`create_futures.py`)

**Purpose**: Process future/holdout data for online simulation and production testing.

**Key Features**:

- Parses Netflix-format future data file
- Converts to CSV format for streaming simulation
- Sorts by date for temporal ordering
- Single-file output for sequential processing

**Usage**:

```bash
spark-submit spark/apps/create_futures.py \
  --future-file /workspace/data/combined/future.txt \
  --csv-output-path /workspace/data/processed/future.csv
```

**Output Format** (CSV with header):

```
DATE,CUST_ID,MOVIE_ID,RATING
2005-12-31,1234567,5678,4
2006-01-01,9876543,1234,5
```

## 🔧 Interacting with the Spark Cluster

### Job Submission Methods

#### 1. Direct `spark-submit` (via Docker Exec)

Execute from the Spark master container:

```bash
docker exec spark-master spark-submit \
  --master spark://spark-master:7077 \
  --deploy-mode client \
  /workspace/spark/apps/etl.py \
  --train-file /workspace/data/combined/train.txt \
  --test-file /workspace/data/combined/test.txt \
  --parquet-train-path /workspace/data/processed/parquet/train \
  --parquet-test-path /workspace/data/processed/parquet/test
```

#### 2. Makefile Commands (Simplified Wrappers)

Predefined commands in the root `Makefile`:

```bash
# ETL pipeline
make submit-standalone-spark-etl

# Train model
make submit-standalone-spark-train

# Test model (specify version)
make submit-standalone-spark-test

# Process streaming data
make submit-standalone-spark-drift

# Create future CSV
make submit-standalone-spark-future

# Merge streaming data
make submit-spark-merge-streaming
```

#### 3. Airflow DAGs (Recommended for Production)

Two orchestrated pipelines using `SparkSubmitOperator`:

**ETL-Train-Test Pipeline** (`dag_etl_train.py`):

```
etl_task → [create_futures_task, train_task] → find_model_task → test_task
```

**Drift Detection & Retraining Pipeline** (`dag_drift_retrain.py`):

```
process_streaming → find_current_model → detect_drift → check_retrain →
  [merge_streaming → train → find_latest → test] (if drift detected)
```

Trigger via Airflow UI at http://localhost:8081 or CLI:

```bash
make airflow-trigger-dag dag=etl_train_test_pipeline
make airflow-trigger-dag dag=drift_detection_retrain_pipeline
```

### Monitoring and Debugging

**Spark Web UIs**:

- Master UI: http://localhost:8080 - View cluster status, running applications
- Worker UI: http://localhost:8081 - Monitor worker health and tasks
- History Server: http://localhost:18080 - Review completed jobs

**Airflow Integration**:

- Airflow UI: http://localhost:8081 - DAG monitoring, task logs
- Task logs: `airflow/logs/dag_id=<dag_name>/run_id=<run_id>/`

**View Logs**:

```bash
# Spark master logs
docker logs spark-master

# Spark worker logs
docker logs <worker_container_id>

# Airflow scheduler logs
make airflow-scheduler-logs

# Airflow API server logs
make airflow-apiserver-logs
```

## 📊 Data Flow

### Initial Pipeline (One-time Setup)

```
Raw Text Files
    ↓
[ETL] → Parquet Datasets (partitioned by year/month)
    ↓
[Train] → ALS Model (user/item factors)
    ↓
[Test] → Evaluation Metrics
```

### Continuous Drift Monitoring Pipeline

```
Streaming CSV Data
    ↓
[Process] → Streaming Parquet (train/test split)
    ↓
[Detect Drift] → Drift Metrics (KL, mean shift, RMSE, etc.)
    ↓
[Check Retrain] → Decision (boolean)
    ↓ (if True)
[Merge] → Updated Parquet Datasets
    ↓
[Train] → New ALS Model
    ↓
[Test] → Validation Metrics
```

## 🛠️ Common Utilities (`common.py`)

Shared functions used across all Spark applications:

### Core Functions

**`build_spark(app_name, shuffle_partitions)`**

- Creates or retrieves a SparkSession with optimized configuration
- Configures shuffle partitions for performance tuning

**`parse_netflix_file(spark, path)`**

- Parses Netflix Prize format (header lines with colons, rating rows)
- Handles malformed data gracefully
- Returns DataFrame with schema: `MOVIE_ID, CUST_ID, RATING, DATE`

**`upsert_ratings_parquet(new_df, output_path, ...)`**

- Merges new data into existing Parquet datasets
- Deduplicates by key columns, keeping latest/highest rating
- Partitions by year and month for efficient querying
- Supports schema evolution via `mergeSchema` option

**`train_als(train_df, rank, reg_param, max_iter, ...)`**

- Trains ALS model with specified hyperparameters
- Saves complete model to timestamped directory
- Configurable for implicit/explicit feedback

**`load_als_model(model_path)`**

- Loads pre-trained ALS model for inference
- Returns `ALSModel` ready for predictions/recommendations

## ⚙️ Configuration

### Spark Configuration

**Default Settings** (via `common.py`):

```python
spark.sql.shuffle.partitions = 400  # Tunable per job
spark.eventLog.enabled = true
spark.eventLog.dir = /opt/spark/history
```

**Resource Allocation** (via `docker-compose.yml`):

```yaml
spark-worker:
  environment:
    SPARK_WORKER_CORES: 2
    SPARK_WORKER_MEMORY: 2g
```

### Airflow Configuration

**Spark Connection** (`SPARK_MASTER`):

```python
SPARK_MASTER = "spark://spark-master:7077"
```

**Paths** (inside container):

```python
WORKSPACE_PATH = "/workspace"
SPARK_APPS_PATH = "/workspace/spark/apps"
MODEL_ARTIFACTS_PATH = "/workspace/models/artifacts"
PARQUET_TRAIN_PATH = "/workspace/data/processed/parquet/train"
PARQUET_TEST_PATH = "/workspace/data/processed/parquet/test"
```

**Drift Thresholds**:

```python
DEFAULT_KL_THRESHOLD = 0.1
DEFAULT_MEAN_SHIFT_THRESHOLD = 0.2
DEFAULT_MEDIAN_SHIFT_THRESHOLD = 0.5
```

## 🔍 Performance Tuning

### Key Parameters

1. **Shuffle Partitions** (`--shuffle-partitions`):

   - Controls parallelism for wide transformations (joins, aggregations)
   - Default: 400
   - Tune based on data size: `num_partitions ≈ data_size_GB × 2-4`

2. **Target Repartition** (`--target-repartition`):

   - Controls output file count and size
   - Default: 400
   - Balance between file count and file size (target: 128MB-1GB per file)

3. **ALS Rank** (`--als-rank`):

   - Latent factor dimensionality
   - Higher = more expressive but slower training
   - Default: 64

4. **Worker Scaling**:
   ```bash
   docker compose up -d --scale spark-worker=5
   ```

### Best Practices

- **Data Partitioning**: Use year/month partitions for time-based queries
- **Caching**: Cache frequently accessed DataFrames in iterative jobs
- **Broadcasting**: Leverage broadcast joins for small lookup tables
- **Compression**: Use Snappy for balanced compression/speed tradeoff
- **Coalescing**: Use before writing to reduce small files

## 📝 Example Workflows

### Complete ML Pipeline (Manual)

```bash
# 1. ETL: Process raw data
make submit-standalone-spark-etl

# 2. Create future data CSV
make submit-standalone-spark-future

# 3. Train model
make submit-standalone-spark-train

# 4. Test model (replace version with actual timestamp)
docker exec spark-master spark-submit spark/apps/test_recommender.py \
  --model-path models/artifacts/version_20250930_212327 \
  --test-parquet /workspace/data/processed/parquet/test

# 5. Monitor at http://localhost:8080
```

### Automated Drift Detection (via Airflow)

```bash
# 1. Place new data in streaming directory
cp new_ratings.csv data/streaming/current.csv

# 2. Trigger drift detection DAG
make airflow-trigger-dag dag=drift_detection_retrain_pipeline

# 3. Monitor progress at http://localhost:8081
# Pipeline automatically retrains if drift exceeds thresholds
```

## 🐳 Docker Environment

### Container Structure

```
spark-master:
  - Runs Spark master process
  - Accepts job submissions
  - Coordinates worker tasks
  - Mounts: /workspace (project root)

spark-worker (scalable):
  - Executes distributed tasks
  - Configurable cores/memory
  - Auto-registers with master
  - Mounts: /workspace (shared)

spark-history:
  - Web UI for completed jobs
  - Persistent event logs
  - Volume: spark-history
```

### Environment Variables

```bash
SPARK_HOME=/opt/spark
SPARK_MASTER=spark://spark-master:7077
VIRTUAL_ENV=/opt/app/.venv
PYTHONPATH=/workspace
```

### Python Environment

- **Manager**: UV (fast package installer)
- **Dependencies**: Defined in `pyproject.toml` and `uv.lock`
- **Virtual Env**: `/opt/app/.venv` (activated by default)

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose
- 8GB+ RAM (16GB recommended for full cluster)
- Netflix Prize dataset in `data/combined/` directory

### Quick Start

```bash
# 1. Build and start cluster
make build
make run

# 2. Verify cluster health
curl http://localhost:8080  # Spark Master UI
curl http://localhost:8081  # Airflow UI

# 3. Run complete pipeline via Airflow
make airflow-trigger-dag dag=etl_train_test_pipeline

# 4. View results
make airflow-logs
```

### Scaling the Cluster

```bash
# Start with 5 workers
make down
docker compose up -d --scale spark-worker=5

# Monitor resource usage
docker stats
```

## 📚 Additional Resources

- **Spark Documentation**: https://spark.apache.org/docs/latest/
- **MLlib Guide**: https://spark.apache.org/docs/latest/ml-guide.html
- **ALS Algorithm**: https://spark.apache.org/docs/latest/ml-collaborative-filtering.html
- **Airflow SparkSubmitOperator**: https://airflow.apache.org/docs/apache-airflow-providers-apache-spark/

---
