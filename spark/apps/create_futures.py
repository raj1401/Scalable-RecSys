from __future__ import annotations

import argparse
from typing import List, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import to_date, col

from spark.apps.common import (
    build_spark,
    parse_netflix_file,
)


def _parse_ratings_file(spark: SparkSession, path: str) -> DataFrame:
    """Parse a Netflix ratings file and convert DATE column to date type."""
    df = parse_netflix_file(spark, path)
    return df.withColumn("DATE", to_date(col("DATE")))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Netflix Future Data ETL: parse future.txt and create CSV file"
    )
    # Input
    parser.add_argument(
        "--future-file",
        default="/workspace/data/combined/future.txt",
        help="Path to the future Netflix-style text file (default: /workspace/data/combined/future.txt).",
    )

    # Output path
    parser.add_argument(
        "--csv-output-path",
        default="/workspace/data/processed/future.csv",
        help="Output path for FUTURE CSV file (default: /workspace/data/processed/future.csv).",
    )

    # Spark tuning
    parser.add_argument(
        "--shuffle-partitions",
        type=int,
        default=400,
        help="spark.sql.shuffle.partitions to use (default: 400).",
    )

    args = parser.parse_args(argv)

    spark = build_spark(app_name="netflix-future-etl", shuffle_partitions=args.shuffle_partitions)

    try:
        # 1) Parse future file
        print(f"Parsing future data from: {args.future_file}")
        future_df = _parse_ratings_file(spark, args.future_file)
        
        # Show sample of parsed data
        print(f"Parsed {future_df.count()} rows from future file")
        print("Sample data:")
        future_df.show(5)

        # 2) Select columns in the specified order: DATE, CUST_ID, MOVIE_ID, RATING
        # 3) Order by DATE (ascending)
        output_df = (future_df
                     .select("DATE", "CUST_ID", "MOVIE_ID", "RATING")
                     .orderBy(col("DATE").asc())
                     .dropna())  # Remove any rows with null values
        
        print(f"Writing CSV file to: {args.csv_output_path}")
        print(f"Total rows to write: {output_df.count()}")
        
        # 4) Write as CSV with header, coalesce to single file for proper ordering
        (output_df
         .coalesce(1)  # Single file to maintain order
         .write
         .mode("overwrite")
         .option("header", "true")
         .csv(args.csv_output_path))
        
        print("Future CSV file created successfully!")
        print("\nFirst few rows of output:")
        output_df.show(10)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
