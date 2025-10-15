# Kafka Services for ALS Recommendations

This package wires Kafka into the recommendation stack so that clients can submit asynchronous jobs and retrieve results once ALS workers finish scoring.

## Components

- **`producer_service.py`** – FastAPI app that exposes `POST /recommendations` to enqueue jobs on `recs.requests`.
- **`result_service.py`** – FastAPI app that exposes `GET /recommendations/{job_id}` to stream the latest status from `recs.results` with optional long polling.
- **`worker.py`** – Async consumer that pulls jobs from `recs.requests`, runs ALS inference via `als_engine.py`, and publishes status updates back to `recs.results`.
- **`als_engine.py`** – Thin wrapper around the existing ALS bundle loader that offers convenient methods for scoring users or fold-in feedback within the worker.
- **`job_cache.py`** – In-memory cache used by the result API to serve the latest job snapshot per `job_id`.
- **`config.py` / `models.py`** – Shared configuration and Pydantic models for typed Kafka payloads.

## Topics and Message Flow

1. `POST /recommendations` validates the request, assigns a `job_id`, and produces a JSON payload keyed by `job_id` to the `recs.requests` topic.
2. One or more workers (same consumer group) mark the job `RUNNING`, execute ALS scoring, and emit either `DONE` or `FAILED` records to the `recs.results` compacted topic.
3. The result API tails `recs.results`, caching the latest envelope per `job_id` so clients can poll (or long poll) and obtain their recommendations.

Each envelope stored in Kafka is JSON encoded and matches the schemas in `models.py`; the key is always the `job_id` string to take advantage of log compaction.

## Running Locally

1. Launch Kafka/ZooKeeper (e.g. via Docker Compose) and create `recs.requests` and `recs.results` topics with log retention suited to your workload (`recs.results` should be compacted).
2. Start at least one ALS worker:

   ```powershell
   uvicorn kafka.producer_service:app --reload --port 8080
   uvicorn kafka.result_service:app --reload --port 8081
   python -m kafka.worker
   ```

   Adjust environment variables defined in `config.py` (`KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_TOPIC_*`, etc.) to point at your brokers.

3. Submit a request:

   ```powershell
   Invoke-RestMethod -Method Post -Uri http://localhost:8080/recommendations -Body (@{
       user_id = 123
       k = 20
       exclude_item_ids = @(456,789)
   } | ConvertTo-Json) -ContentType "application/json"
   ```

4. Poll for the result:

   ```powershell
   Invoke-RestMethod -Method Get -Uri http://localhost:8081/recommendations/<job_id>?wait_seconds=10
   ```

## Required Python Packages

Install these extras in your environment before running any of the services above:

- `aiokafka`
- `fastapi`
- `uvicorn[standard]`
- `pydantic>=2`

Depending on your Kafka security mode you may also need `python-snappy`, `lz4`, or `confluent-kafka` extras, but they are optional.
