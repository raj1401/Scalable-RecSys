from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import to_date, col

from spark.apps.common import (
    parse_netflix_file,
    split_dataframe_by_date,
    combine_dataframes,
    write_dataframes_as_parquet,
    train_als,
)


# ---------- Spark builder ----------
def build_spark(app_name: str,
                shuffle_partitions: Optional[int] = None) -> SparkSession:
    builder = (
        SparkSession.builder
        .appName(app_name)
    )
    if shuffle_partitions is not None:
        builder = builder.config("spark.sql.shuffle.partitions", int(shuffle_partitions))
    return builder.getOrCreate()


def _read_and_split_files(
    spark: SparkSession,
    input_files: List[str],
    max_train_date: str,
) -> Tuple[List[DataFrame], List[DataFrame]]:
    """
    For each input text file:
      - parse → DataFrame(MOVIE_ID, CUST_ID, RATING, DATE(str))
      - cast DATE to DateType
      - split into (train_i, test_i) with split_dataframe_by_date
    Returns two lists: [train_i], [test_i]
    """
    train_parts, test_parts = [], []
    for path in input_files:
        df = parse_netflix_file(spark, path)
        # Ensure DATE is DateType for downstream Parquet writer
        df = df.withColumn("DATE", to_date(col("DATE")))  # expects yyyy-MM-dd by default
        tdf, vdf = split_dataframe_by_date(df, "DATE", max_train_date)
        train_parts.append(tdf)
        test_parts.append(vdf)
    return train_parts, test_parts


def _combine_and_write(
    train_parts: List[DataFrame],
    test_parts: List[DataFrame],
    *,
    parquet_train_path: str,
    parquet_test_path: str,
    parquet_table_train: Optional[str],
    parquet_table_test: Optional[str],
    shuffle_partitions: Optional[int],
    target_repartition: Optional[int],
    extra_write_options: Optional[dict],
) -> Tuple[DataFrame, DataFrame]:
    """
    Combine (dedup by MOVIE_ID, CUST_ID, DATE taking max RATING) and write each split as Parquet.
    """
    subset_cols = ["MOVIE_ID", "CUST_ID", "DATE"]
    train_df = combine_dataframes(train_parts, subset_cols=subset_cols, target_col="RATING")
    test_df = combine_dataframes(test_parts, subset_cols=subset_cols, target_col="RATING")

    # Drop rows with nulls in any column
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # Write to Parquet; partitioned year/month is handled inside
    write_dataframes_as_parquet(
        df=train_df,
        output_path=parquet_train_path,
        mode="overwrite",
        partition_cols=("year", "month"),
        num_shuffle_partitions=shuffle_partitions,
        target_repartition=target_repartition,
        table_name=parquet_table_train,
        extra_write_options=extra_write_options,
    )

    write_dataframes_as_parquet(
        df=test_df,
        output_path=parquet_test_path,
        mode="overwrite",
        partition_cols=("year", "month"),
        num_shuffle_partitions=shuffle_partitions,
        target_repartition=target_repartition,
        table_name=parquet_table_test,
        extra_write_options=extra_write_options,
    )

    return train_df, test_df


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Netflix ETL: parse → split → combine → Parquet write → train ALS"
    )
    # Inputs
    parser.add_argument(
        "--input-files",
        nargs="+",
        required=True,
        help="Paths to Netflix-style text files (local/HDFS/S3/etc.). Provide one or many.",
    )
    parser.add_argument(
        "--max-train-date",
        default="2005-12-31",
        help="Inclusive max date for train split; rows after this go to test. Format yyyy-MM-dd.",
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

    # ALS training outputs
    parser.add_argument("--model-save-path", required=False, default="models/artifacts/", help="Base path to save ALS model artifacts (Parquet).")

    # ALS hyperparams
    parser.add_argument("--als-rank", type=int, default=64)
    parser.add_argument("--als-reg", type=float, default=0.1)
    parser.add_argument("--als-max-iter", type=int, default=15)
    parser.add_argument("--als-implicit", action="store_true", help="Use implicitPrefs=True if set.")
    parser.add_argument("--als-nonnegative", action="store_true", help="Use nonnegative=True if set.")
    parser.add_argument("--als-cold-start", default="drop", help='coldStartStrategy, default "drop".')

    args = parser.parse_args(argv)

    if not args.input_files:
        raise SystemExit("No --input-files provided.")

    try:
        extra_write_options = json.loads(args.extra_write_options_json) if args.extra_write_options_json else {}
        if not isinstance(extra_write_options, dict):
            raise ValueError("extra-write-options-json must parse to a JSON object/dict.")
    except Exception as e:
        raise SystemExit(f"Failed to parse --extra-write-options-json: {e}")

    spark = build_spark(app_name="netflix-etl", shuffle_partitions=args.shuffle_partitions)

    try:
        # 1) Parse & split per file
        train_parts, test_parts = _read_and_split_files(
            spark=spark,
            input_files=args.input_files,
            max_train_date=args.max_train_date,
        )

        # 2) Combine & write Parquet
        train_df, _ = _combine_and_write(
            train_parts=train_parts,
            test_parts=test_parts,
            parquet_train_path=args.parquet_train_path,
            parquet_test_path=args.parquet_test_path,
            parquet_table_train=args.parquet_table_train,
            parquet_table_test=args.parquet_table_test,
            shuffle_partitions=args.shuffle_partitions,
            target_repartition=args.target_repartition,
            extra_write_options=extra_write_options,
        )

        # 3) Train ALS on combined TRAIN split
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
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()