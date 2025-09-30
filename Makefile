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
		--test-file /workspace/data/combined/test.txt \
		--k 10 \
		--rating-threshold 4.0

