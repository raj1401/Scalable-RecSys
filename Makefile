build:
	docker compose build

build-nc:
	docker compose build --no-cache

build-progress:
	docker compose build --no-cache --progress=plain

down:
	docker compose down --volumes

run:
	make down && docker compose up -d

run-scaled:
	make down && docker compose up -d --scale spark-worker=3

stop:
	docker compose stop

submit:
	docker exec spark-master spark-submit --master spark://spark-master:7077 --deploy-mode client ./apps/$(app)

submit-standalone-spark-etl:
	docker compose exec spark-master spark-submit \
		spark/apps/etl.py \
		--train-file /workspace/data/combined/train.txt \
		--test-file /workspace/data/combined/test.txt \
		--parquet-train-path /workspace/data/processed/parquet/train \
		--parquet-test-path /workspace/data/processed/parquet/test \

submit-standalone-spark-train:
	docker compose exec spark-master spark-submit \
		spark/apps/train_recommender.py \
		--parquet-train-path /workspace/data/processed/parquet/train \
		--model-save-path models/artifacts

submit-standalone-spark-test:
	docker compose exec spark-master spark-submit \
		spark/apps/test_recommender.py \
		--model-path models/artifacts/version_20250930_212327 \
		--test-parquet /workspace/data/processed/parquet/test \
		--k 10 \
		--rating-threshold 4.0

submit-standalone-spark-drift:
	docker compose exec spark-master spark-submit \
		spark/apps/drift_detection.py \
		--csv_path /workspace/data/streaming/current.csv \
		--output_path /workspace/data/streaming \
		--train_ratio 0.8

submit-standalone-spark-future:
	docker compose exec spark-master spark-submit \
		spark/apps/create_futures.py \
		--future-file /workspace/data/combined/future.txt \
		--csv-output-path /workspace/data/processed/future.csv

submit-spark-process-detect-drift:
	docker compose exec spark-master spark-submit \
		spark/apps/drift_detection.py \
		process_and_detect \
		--csv_path /workspace/data/streaming/current.csv \
		--output_path /workspace/data/streaming \
		--train_ratio 0.8 \
		--model_path /workspace/models/artifacts/version_20250930_212327 \
		--original_train_path /workspace/data/processed/parquet/train \
		--original_test_path /workspace/data/processed/parquet/test \
		--kl_threshold 0.1 \
		--mean_shift_threshold 0.2 \
		--median_shift_threshold 0.5

submit-spark-merge-streaming:
	docker compose exec spark-master spark-submit \
		spark/apps/drift_detection.py \
		merge_job \
		--streaming_path /workspace/data/streaming \
		--processed_path /workspace/data/processed/parquet

# ============================================================================
# Airflow Commands
# ============================================================================

airflow-logs:
	docker compose logs -f airflow-webserver airflow-scheduler

airflow-webserver-logs:
	docker compose logs -f airflow-webserver

airflow-scheduler-logs:
	docker compose logs -f airflow-scheduler

airflow-cli:
	docker compose exec airflow-webserver airflow $(cmd)

airflow-list-dags:
	docker compose exec airflow-webserver airflow dags list

airflow-trigger-dag:
	docker compose exec airflow-webserver airflow dags trigger $(dag)

airflow-test-dag:
	docker compose exec airflow-webserver airflow dags test $(dag)

airflow-bash:
	docker compose exec airflow-webserver bash

airflow-clean:
	docker compose down -v
	rm -rf airflow/logs/*
	rm -rf airflow/config/*

# Combined commands
run-all:
	make down && docker compose up -d

run-all-scaled:
	make down && docker compose up -d --scale spark-worker=3

run-spark-only:
	make down && docker compose up -d spark-master spark-worker spark-history

run-airflow-only:
	make down && docker compose up -d postgres airflow-webserver airflow-scheduler

logs-all:
	docker compose logs -f
