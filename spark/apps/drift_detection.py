"""
Drift Detection Module for Scalable RecSys

This module processes streaming data from CSV files and splits them into 
training and test datasets stored as Parquet files.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.functions import col, rand, to_date
from common import build_spark, upsert_ratings_parquet
import argparse
import os


def load_streaming_csv(spark: SparkSession, csv_path: str) -> DataFrame:
    """
    Load streaming CSV data with schema: DATE, CUST_ID, MOVIE_ID, RATING
    
    Args:
        spark: SparkSession instance
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with proper schema and data types
    """
    # Define schema for the CSV file
    schema = StructType([
        StructField("DATE", StringType(), True),
        StructField("CUST_ID", IntegerType(), True),
        StructField("MOVIE_ID", IntegerType(), True),
        StructField("RATING", IntegerType(), True)
    ])
    
    # Read CSV with header
    df = spark.read.format("csv") \
        .option("header", "true") \
        .schema(schema) \
        .load(csv_path)
    
    # Convert DATE column to proper date type
    df = df.withColumn("DATE", to_date(col("DATE")))
    
    # Filter out any null values
    df = df.dropna()
    
    return df


def split_streaming_data(df: DataFrame, train_ratio: float = 0.8) -> tuple[DataFrame, DataFrame]:
    """
    Split the streaming data into train and test sets based on probability.
    
    Args:
        df: Input DataFrame
        train_ratio: Probability of a row going to training set (default 0.8)
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Add a random column for splitting
    df_with_random = df.withColumn("random_val", rand())
    
    # Split based on random value
    train_df = df_with_random.filter(col("random_val") < train_ratio).drop("random_val")
    test_df = df_with_random.filter(col("random_val") >= train_ratio).drop("random_val")
    
    return train_df, test_df


def process_streaming_data(
    spark: SparkSession,
    csv_path: str,
    output_base_path: str,
    train_ratio: float = 0.8
) -> None:
    """
    Main function to process streaming CSV data and create Parquet files.
    Upserts data into existing Parquet files to avoid duplicates.
    
    Args:
        spark: SparkSession instance
        csv_path: Path to the input CSV file
        output_base_path: Base path for output (will create train/test subdirectories)
        train_ratio: Proportion of data for training (default 0.8)
    """
    print(f"Loading streaming data from: {csv_path}")
    
    # Load the CSV data
    df = load_streaming_csv(spark, csv_path)
    
    print(f"Loaded {df.count()} rows from CSV")
    
    # Split the data
    train_df, test_df = split_streaming_data(df, train_ratio)
    
    # Define output paths
    train_output_path = os.path.join(output_base_path, "streaming_train")
    test_output_path = os.path.join(output_base_path, "streaming_test")
    
    # Get counts for logging
    train_count = train_df.count()
    test_count = test_df.count()
    
    print(f"Split data: {train_count} training samples, {test_count} test samples")
    
    # Flag to track if all writes were successful
    write_success = True
    
    try:
        # Upsert training data to existing Parquet files (or create new if they don't exist)
        print(f"Upserting training data to: {train_output_path}")
        upsert_ratings_parquet(
            new_df=train_df,
            output_path=train_output_path,
            subset_cols=("MOVIE_ID", "CUST_ID", "DATE"),
            target_col="RATING",
            partition_cols=("year", "month"),
            target_repartition=4
        )
        print(f"Successfully upserted {train_count} training samples")
        
        # Upsert test data to existing Parquet files (or create new if they don't exist)
        print(f"Upserting test data to: {test_output_path}")
        upsert_ratings_parquet(
            new_df=test_df,
            output_path=test_output_path,
            subset_cols=("MOVIE_ID", "CUST_ID", "DATE"),
            target_col="RATING",
            partition_cols=("year", "month"),
            target_repartition=4
        )
        print(f"Successfully upserted {test_count} test samples")
        
        print("Streaming data processing completed successfully!")
        
    except Exception as e:
        print(f"Error during data processing: {str(e)}")
        write_success = False
        raise
    
    # Delete the CSV file only if all writes were successful
    if write_success:
        try:
            if os.path.exists(csv_path):
                os.remove(csv_path)
                print(f"Successfully deleted source CSV file: {csv_path}")
            else:
                print(f"Source CSV file not found for deletion: {csv_path}")
        except Exception as e:
            print(f"Warning: Could not delete source CSV file {csv_path}: {str(e)}")
            # Don't raise the exception since the main processing was successful


def main():
    """Main entry point for the drift detection script."""
    parser = argparse.ArgumentParser(description="Process streaming CSV data for drift detection")
    parser.add_argument(
        "--csv_path",
        default="data/streaming/current.csv",
        help="Path to the input CSV file (default: data/streaming/current.csv)"
    )
    parser.add_argument(
        "--output_path",
        default="data/streaming",
        help="Base output path for Parquet files (default: data/streaming)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Proportion of data for training (default: 0.8)"
    )
    parser.add_argument(
        "--shuffle_partitions",
        type=int,
        default=200,
        help="Number of shuffle partitions for Spark (default: 200)"
    )
    
    args = parser.parse_args()
    
    # Validate train_ratio
    if not 0 < args.train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    
    # Build Spark session
    spark = build_spark("DriftDetection", shuffle_partitions=args.shuffle_partitions)
    
    try:
        # Check if input file exists
        if not os.path.exists(args.csv_path):
            raise FileNotFoundError(f"Input CSV file not found: {args.csv_path}")
        
        # Process the streaming data
        process_streaming_data(
            spark=spark,
            csv_path=args.csv_path,
            output_base_path=args.output_path,
            train_ratio=args.train_ratio
        )
        
    except Exception as e:
        print(f"Error processing streaming data: {str(e)}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
