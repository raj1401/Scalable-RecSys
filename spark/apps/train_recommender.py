from __future__ import annotations

import argparse
from typing import List, Optional

from spark.apps.common import build_spark, train_als


def main(argv: Optional[List[str]] = None) -> None:
	parser = argparse.ArgumentParser(
		description="Train ALS recommender from a persisted Parquet ratings dataset"
	)

	parser.add_argument(
		"--parquet-train-path",
		required=True,
		help="Input Parquet dataset containing ratings for ALS training.",
	)
	parser.add_argument(
		"--model-save-path",
		required=False,
		default="models/artifacts/",
		help="Base path to save ALS model factors (Parquet).",
	)

	parser.add_argument(
		"--shuffle-partitions",
		type=int,
		default=400,
		help="spark.sql.shuffle.partitions to use during training.",
	)

	parser.add_argument("--als-rank", type=int, default=64)
	parser.add_argument("--als-reg", type=float, default=0.1)
	parser.add_argument("--als-max-iter", type=int, default=15)
	parser.add_argument(
		"--als-implicit",
		action="store_true",
		help="Use implicitPrefs=True if set.",
	)
	parser.add_argument(
		"--als-nonnegative",
		action="store_true",
		help="Use nonnegative=True if set.",
	)
	parser.add_argument(
		"--als-cold-start",
		default="drop",
		help='coldStartStrategy parameter, default "drop".',
	)

	# MLflow tracking options
	parser.add_argument(
		"--mlflow-tracking-uri",
		default=None,
		help="MLflow tracking URI (e.g., http://localhost:5000 or file:///path/to/mlruns)",
	)
	parser.add_argument(
		"--mlflow-experiment",
		default="netflix-als-training",
		help="MLflow experiment name",
	)

	args = parser.parse_args(argv)

	spark = build_spark(app_name="netflix-train-als", shuffle_partitions=args.shuffle_partitions)

	try:
		train_df = spark.read.parquet(args.parquet_train_path)

		train_als(
			train_df=train_df,
			user_col="CUST_ID",
			item_col="MOVIE_ID",
			rating_col="RATING",
			rank=args.als_rank,
			reg_param=args.als_reg,
			max_iter=args.als_max_iter,
			implicit_prefs=bool(args.als_implicit),
			cold_start_strategy=args.als_cold_start,
			nonnegative=bool(args.als_nonnegative),
			model_save_path=args.model_save_path,
			mlflow_tracking_uri=args.mlflow_tracking_uri,
			mlflow_experiment=args.mlflow_experiment,
		)
	finally:
		spark.stop()


if __name__ == "__main__":
	main()
