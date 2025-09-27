from __future__ import annotations

import argparse
import json
from typing import List, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import to_date, col

from spark.apps.common import (
    build_spark,
    parse_netflix_file,
    upsert_ratings_parquet,
)


def _parse_ratings_file(spark: SparkSession, path: str) -> DataFrame:
    df = parse_netflix_file(spark, path)
    return df.withColumn("DATE", to_date(col("DATE")))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Netflix ETL: parse pre-split train/test files and upsert Parquet datasets"
    )
    # Inputs
    parser.add_argument(
        "--train-file",
        required=True,
        help="Path to the pre-split TRAIN Netflix-style text file.",
    )
    parser.add_argument(
        "--test-file",
        required=True,
        help="Path to the pre-split TEST Netflix-style text file.",
    )

    # Output paths
    parser.add_argument("--parquet-train-path", required=True, help="Output path for TRAIN Parquet dataset.")
    parser.add_argument("--parquet-test-path", required=True, help="Output path for TEST Parquet dataset.")
    parser.add_argument("--parquet-table-train", default=None, help="Optional metastore table name for TRAIN Parquet dataset.")
    parser.add_argument("--parquet-table-test", default=None, help="Optional metastore table name for TEST Parquet dataset.")

    # Spark tuning
    parser.add_argument("--shuffle-partitions", type=int, default=400, help="spark.sql.shuffle.partitions to use.")
    parser.add_argument("--target-repartition", type=int, default=400,
                        help="Repartition before write to control file sizes (co-partitioned by year/month).")

    # Extra writer options
    parser.add_argument(
        "--extra-write-options-json",
        default='{"mergeSchema":"true","compression":"snappy"}',
        help="JSON dict of extra writer options for the Parquet writer (e.g. '{\"mergeSchema\":\"true\"}')",
    )

    args = parser.parse_args(argv)

    try:
        extra_write_options = json.loads(args.extra_write_options_json) if args.extra_write_options_json else {}
        if not isinstance(extra_write_options, dict):
            raise ValueError("extra-write-options-json must parse to a JSON object/dict.")
    except Exception as e:
        raise SystemExit(f"Failed to parse --extra-write-options-json: {e}")

    spark = build_spark(app_name="netflix-etl", shuffle_partitions=args.shuffle_partitions)

    try:
        # 1) Parse provided train/test files
        train_df_new = _parse_ratings_file(spark, args.train_file)
        test_df_new = _parse_ratings_file(spark, args.test_file)

        # 2) Upsert into existing Parquet datasets (union + dedupe)
        upsert_ratings_parquet(
            new_df=train_df_new,
            output_path=args.parquet_train_path,
            subset_cols=("MOVIE_ID", "CUST_ID", "DATE"),
            target_col="RATING",
            partition_cols=("year", "month"),
            num_shuffle_partitions=args.shuffle_partitions,
            target_repartition=args.target_repartition,
            table_name=args.parquet_table_train,
            extra_write_options=extra_write_options,
        )

        upsert_ratings_parquet(
            new_df=test_df_new,
            output_path=args.parquet_test_path,
            subset_cols=("MOVIE_ID", "CUST_ID", "DATE"),
            target_col="RATING",
            partition_cols=("year", "month"),
            num_shuffle_partitions=args.shuffle_partitions,
            target_repartition=args.target_repartition,
            table_name=args.parquet_table_test,
            extra_write_options=extra_write_options,
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()