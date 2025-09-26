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

submit-standalone-spark:
	docker compose exec spark-master spark-submit \
		spark/apps/etl.py \
		--input-files /workspace/data/combined_data_1.txt \
		             /workspace/data/combined_data_2.txt \
		             /workspace/data/combined_data_3.txt \
		             /workspace/data/combined_data_4.txt \
		--parquet-train-path /workspace/data/processed/parquet/train \
		--parquet-test-path /workspace/data/processed/parquet/test \
      --model-save-path /workspace/models/artifacts/ \
