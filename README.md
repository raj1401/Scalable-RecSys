# Scalable Recommendation System

A production-ready, scalable movie recommendation system built with Apache Spark, Apache Kafka and Apache Airflow, deployed via Docker containers, orchestrated via Docker Compose.

<!-- Project architecture diagram -->
<p align="center">
   <img src="diagram.png" alt="Architecture Diagram" style="max-width:100%;height:auto;" />
</p>

## Overview

This project implements a complete end-to-end ML pipeline for movie recommendations:

- **ETL Pipeline**: Process and partition Netflix Prize dataset
- **Collaborative Filtering**: Train ALS (Alternating Least Squares) model
- **Batch Inference**: Generate future recommendations
- **Model Serving**: gRPC inference server
- **Orchestration**: Apache Airflow DAG automation
- **Streaming**: Kafka-based real-time processing
- **Deployment**: Fully containerized with Docker Compose

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.13+
- Make

### 1. Start the Complete Stack

```bash
# Build all images
make build

# Start Spark + Kafka + Airflow + PostgreSQL
make run-all
```

### 2. Access Services

| Service              | URL                    | Credentials   |
| -------------------- | ---------------------- | ------------- |
| Airflow UI           | http://localhost:8081  | admin / admin |
| Spark Master UI      | http://localhost:8080  | -             |
| Spark History Server | http://localhost:18080 | -             |
| Kafka Producer API   | http://localhost:8082  | -             |
| Kafka Result API     | http://localhost:8083  | -             |
| Kafka Broker         | http://localhost:9092  | -             |

### 3. Run the ML Pipeline

**Via Airflow (Recommended):**

```bash
# List DAGs
make airflow-list-dags

# Trigger ETL → Train → Test pipeline
make airflow-trigger-dag dag=etl_train_test_pipeline

# Monitor in Airflow UI
open http://localhost:8081
```

**Via Spark Directly:**

```bash
# Run individual jobs
make submit-standalone-spark-etl
make submit-standalone-spark-train
make submit-standalone-spark-test
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network (spark-net)               │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Airflow    │───►│ Spark Master │───►│ Spark Workers│   │
│  │ Orchestrator │    │              │    │              │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  PostgreSQL  │    │    Kafka     │    │ gRPC Server  │   │
│  │  (Metadata)  │    │  (Streaming) │    │  (Serving)   │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
└─────────────────────────────────────────────────────────────┘

Within the Kafka layer now live three services: the producer FastAPI gateway (`8082`), the result FastAPI gateway (`8083`), and the ALS worker that bridges Kafka with the compiled model artifacts.
```

## Project Structure

```
.
├── airflow/                    # Airflow orchestration
│   ├── dags/                   # DAG definitions
│   │   ├── dag_etl_train.py    # Main ML pipeline DAG
│   │   └── common/             # Shared utilities
│   ├── Dockerfile              # Airflow container
│   └── DOCKER_SETUP.md         # Detailed Airflow setup guide
├── spark/                      # Spark applications
│   ├── apps/
│   │   ├── etl.py              # Data processing
│   │   ├── train_recommender.py # Model training
│   │   ├── test_recommender.py  # Model evaluation
│   │   ├── create_futures.py    # Batch predictions
│   │   └── drift_detection.py   # Model monitoring
│   └── Dockerfile              # Spark container
├── inference/                  # Model serving
│   ├── server.py               # gRPC inference server
│   ├── client.py               # Test client
│   └── recs.proto              # Service definition
├── kafka/                      # Kafka pub-sub services + worker
│   ├── producer_service.py     # HTTP → Kafka gateway (POST /recommendations)
│   ├── result_service.py       # HTTP ← Kafka results (GET /recommendations/{job_id})
│   ├── worker.py               # ALS consumer that drives inference
│   ├── als_engine.py           # Wrapper around compiled ALS bundle
│   └── Dockerfile              # Base image for Kafka services
├── models/                     # Trained models
│   ├── artifacts/              # Model checkpoints
│   └── compiled_artifacts/     # Optimized models (NumPy)
├── data/                       # Datasets
│   ├── processed/              # Parquet data
│   └── streaming/              # Kafka data
├── docker-compose.yml          # Multi-service orchestration
└── Makefile                    # Command shortcuts
```

## ML Pipeline

### 1. ETL (`spark/apps/etl.py`)

- Reads Netflix Prize dataset
- Partitions by year (1999-2004)
- Splits into train/test sets
- Saves as Parquet

### 2. Create Futures (`spark/apps/create_futures.py`)

- Loads test data
- Generates future prediction dataset
- Used for batch inference

### 3. Train (`spark/apps/train_recommender.py`)

- Trains ALS collaborative filtering model
- Hyperparameters: rank=50, maxIter=10, regParam=0.1
- Saves user/item factors
- Versioned model artifacts

### 4. Test (`spark/apps/test_recommender.py`)

- Loads trained model
- Evaluates on test set
- Computes RMSE metric
- Uses XCom for dynamic model version

## Airflow DAG

**DAG: `etl_train_test_pipeline`**

```
etl_task
   │
   ▼
create_futures_task
   │
   ▼
train_task
   │
   ▼
find_latest_model_version ──(XCom)──┐
                                     │
                                     ▼
                                 test_task
```

**Features:**

- **SparkSubmitOperator**: Native Spark job submission
- **XCom**: Dynamic model version passing
- **Dependencies**: Sequential execution with retries
- **Idempotent**: Can safely re-run

See [airflow/DOCKER_SETUP.md](airflow/DOCKER_SETUP.md) for detailed setup.

## Model Serving

### gRPC Inference Server

```bash
# Start server (compiles model artifacts first)
python inference/compile_artifacts.py
python inference/server.py

# Test client
python client.py --user-id 12345
```

**API:**

```protobuf
service Recommender {
  rpc GetRecommendations (UserRequest) returns (RecommendationResponse);
}
```

### Kafka-Powered Async APIs

The Kafka services expose an asynchronous workflow for recommendation jobs:

1. **POST** `http://localhost:8082/recommendations` with either a `user_id` or fold-in feedback to enqueue work.
2. Receive a `job_id` immediately while the ALS worker processes the request in the background.
3. **GET** `http://localhost:8083/recommendations/{job_id}` (optionally with `?wait_seconds=10`) to retrieve the latest status or final recommendations.

Example payloads:

```bash
# Submit a user-based job
curl -s -X POST http://localhost:8082/recommendations \
   -H "Content-Type: application/json" \
   -d '{"user_id": 123, "k": 20, "exclude_item_ids": [456,789]}'

# Fold-in job using implicit feedback
curl -s -X POST http://localhost:8082/recommendations \
   -H "Content-Type: application/json" \
   -d '{"feedback": [{"item_id": 101, "rating": 4.5}, {"item_id": 202, "rating": 3.0}], "k": 15}'

# Poll for the result (long polling 10 seconds)
curl -s "http://localhost:8083/recommendations/<job_id>?wait_seconds=10"
```

Responses mirror the schemas in `kafka/models.py`, with statuses `PENDING`, `RUNNING`, `DONE`, or `FAILED` and optional `items` once recommendations are ready.

## Makefile Commands

### Quick Commands

```bash
make build              # Build all images
make run-all            # Start everything
make down               # Stop everything
make logs-all           # View all logs
```

### Airflow

```bash
make airflow-init                          # Initialize Airflow
make airflow-list-dags                     # List DAGs
make airflow-trigger-dag dag=<dag_id>      # Trigger DAG
make airflow-logs                          # View logs
make airflow-bash                          # Access container
```

### Spark

```bash
make submit-standalone-spark-etl           # Run ETL
make submit-standalone-spark-train         # Train model
make submit-standalone-spark-test          # Test model
make submit-standalone-spark-future        # Batch predictions
```

### Development

```bash
make airflow-clean                         # Clean Airflow data
make build-nc                              # Build without cache
```

## Configuration

### Spark Settings

- **Master**: `spark://spark-master:7077`
- **Workers**: 2 (scalable via `--scale spark-worker=N`)
- **Driver Memory**: 2g
- **Executor Memory**: 2g

### Airflow Settings

- **Executor**: LocalExecutor (switch to Celery for production)
- **Database**: PostgreSQL
- **DAG Directory**: `./airflow/dags`
- **Logs**: `./airflow/logs`

## Data

### Netflix Prize Dataset

Expected structure:

```
data/
├── combined/
│   ├── train.txt
│   ├── test.txt
│   └── future.txt
└── movie_titles.csv
```

Format: `user_id,movie_id,rating,date`

## Development

### Testing DAGs Locally

```bash
# Test DAG imports
python airflow/dags/dag_etl_train.py

# Test in Airflow container
make airflow-bash
airflow dags test etl_train_test_pipeline 2025-01-01
```

### Debugging Spark Jobs

```bash
# Check Spark logs
docker compose logs spark-master

# Access Spark container
docker compose exec spark-master bash

# View history server
open http://localhost:18080
```

### Adding New DAGs

1. Create DAG file in `airflow/dags/`
2. Use `common/` utilities for consistency
3. Wait 30 seconds for scheduler detection
4. Test: `make airflow-list-dags`

## Monitoring

### Metrics

- **Airflow UI**: Task success/failure, duration, logs
- **Spark UI**: Job stages, executors, storage
- **History Server**: Completed applications

### Logs

```bash
make airflow-logs              # Airflow logs
docker compose logs -f         # All services
make airflow-webserver-logs    # Webserver only
```

## Troubleshooting

### Airflow UI not loading

```bash
# Check containers
docker ps | grep airflow

# Check logs
make airflow-webserver-logs

# Reinitialize
make airflow-clean && make run-all
```

### Spark connection errors

```bash
# Verify network
docker network inspect spark-net

# Test connectivity
docker compose exec airflow-webserver ping spark-master
```

### DAG not appearing

```bash
# Check for syntax errors
docker compose exec airflow-webserver airflow dags list-import-errors

# Verify file location
docker compose exec airflow-webserver ls /opt/airflow/dags/
```

See [airflow/DOCKER_SETUP.md](airflow/DOCKER_SETUP.md) for comprehensive troubleshooting.

## Production Deployment

### Scaling

**Spark:**

```bash
docker compose up -d --scale spark-worker=10
```

**Airflow:**

- Switch to CeleryExecutor
- Add Redis/RabbitMQ
- Deploy worker pool

### Security

1. **Change default passwords**
2. **Enable TLS/SSL**
3. **Use secrets management**
4. **Implement RBAC**
5. **Restrict network access**

### Best Practices

- ✅ Use versioned model artifacts
- ✅ Implement monitoring/alerting
- ✅ Enable autoscaling
- ✅ Use external metadata stores
- ✅ Implement CI/CD pipelines

## Resources

- [Airflow Docker Setup Guide](airflow/DOCKER_SETUP.md)
- [Apache Airflow Docs](https://airflow.apache.org/docs/)
- [Apache Spark Docs](https://spark.apache.org/docs/latest/)
- [gRPC Python Guide](https://grpc.io/docs/languages/python/)

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test with `make run-all`
5. Submit pull request

---
