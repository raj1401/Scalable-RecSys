"""
Drift Detection Module for Scalable RecSys

This module provides functionality for:
1. Processing streaming data from CSV files and splitting them into training and test datasets
2. Detecting data drift by comparing rating distributions between original and current training data
3. Detecting model drift by comparing model predictions on original and current test data
4. Making automated retraining decisions based on drift metrics
5. Merging streaming data into processed data directories

Features:
- Load and process streaming CSV data with proper schema
- Split data into train/test sets and save as Parquet files
- Compute KL divergence between rating distributions
- Compare statistical metrics (mean, median) across datasets
- Generate predictions using trained ALS models for drift analysis
- Weighted heuristic-based retraining decision (KL divergence, mean shift, median shift)
- Upsert streaming data into processed data directories

Usage:
    # Process streaming CSV data
    python drift_detection.py process --csv_path data/streaming/current.csv
    
    # Detect data drift
    python drift_detection.py data_drift --original_train_path data/processed/parquet/train \\
                                         --current_train_path data/streaming/train
    
    # Detect model drift
    python drift_detection.py model_drift --model_path models/artifacts/version_XXX \\
                                          --original_test_path data/processed/parquet/test \\
                                          --current_test_path data/streaming/test
    
    # Check if retraining is needed (data drift only)
    python drift_detection.py check_retrain
    
    # Check if retraining is needed (data + model drift)
    python drift_detection.py check_retrain --model_path models/artifacts/version_XXX \\
                                            --kl_threshold 0.1 --mean_shift_threshold 0.2
    
    # Merge streaming data into processed data
    python drift_detection.py merge --streaming_path data/streaming \\
                                   --processed_path data/processed/parquet
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.functions import col, rand, to_date, round as spark_round
from common import build_spark, upsert_ratings_parquet, load_als_model
import argparse
import os
import numpy as np
from typing import Dict, List, Tuple


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
    train_output_path = os.path.join(output_base_path, "train")
    test_output_path = os.path.join(output_base_path, "test")
    
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


def compute_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute KL divergence between two probability distributions.
    
    Args:
        p: Original distribution (reference)
        q: Current distribution (to be compared)
        epsilon: Small value to avoid division by zero
        
    Returns:
        KL divergence value
    """
    # Normalize to ensure they are valid probability distributions
    p_norm = p / (p.sum() + epsilon)
    q_norm = q / (q.sum() + epsilon)
    
    # Add epsilon to avoid log(0)
    p_norm = p_norm + epsilon
    q_norm = q_norm + epsilon
    
    # Compute KL divergence: D_KL(P||Q) = sum(P * log(P/Q))
    kl_div = np.sum(p_norm * np.log(p_norm / q_norm))
    
    return float(kl_div)


def detect_data_drift(
    spark: SparkSession,
    original_train_path: str = "data/processed/parquet/train",
    current_train_path: str = "data/streaming/train"
) -> Dict[str, any]:
    """
    Detect data drift by comparing rating distributions between original and current training data.
    
    Computes:
    - Rating counts for each rating value (1-5)
    - KL divergence between distributions
    - Mean and median ratings for both datasets
    
    Args:
        spark: SparkSession instance
        original_train_path: Path to original training data
        current_train_path: Path to current streaming training data
        
    Returns:
        Dictionary containing:
            - original_counts: Array of counts for ratings 1-5
            - current_counts: Array of counts for ratings 1-5
            - kl_divergence: KL divergence value
            - original_mean: Mean rating in original data
            - original_median: Median rating in original data
            - current_mean: Mean rating in current data
            - current_median: Median rating in current data
    """
    print("=" * 80)
    print("DATA DRIFT DETECTION")
    print("=" * 80)
    
    # Load original training data
    print(f"\nLoading original training data from: {original_train_path}")
    try:
        original_df = spark.read.parquet(original_train_path)
        original_count = original_df.count()
        print(f"Loaded {original_count:,} rows from original training data")
    except Exception as e:
        print(f"Error loading original training data: {str(e)}")
        raise
    
    # Load current training data
    print(f"\nLoading current training data from: {current_train_path}")
    try:
        current_df = spark.read.parquet(current_train_path)
        current_count = current_df.count()
        print(f"Loaded {current_count:,} rows from current training data")
    except Exception as e:
        print(f"Error loading current training data: {str(e)}")
        raise
    
    # Get rating distributions (counts for ratings 1-5)
    rating_values = [1, 2, 3, 4, 5]
    
    # Original distribution
    original_rating_counts = original_df.groupBy("RATING").count().collect()
    original_counts_dict = {row["RATING"]: row["count"] for row in original_rating_counts}
    original_counts = np.array([original_counts_dict.get(r, 0) for r in rating_values])
    
    # Current distribution
    current_rating_counts = current_df.groupBy("RATING").count().collect()
    current_counts_dict = {row["RATING"]: row["count"] for row in current_rating_counts}
    current_counts = np.array([current_counts_dict.get(r, 0) for r in rating_values])
    
    # Compute KL divergence
    kl_div = compute_kl_divergence(original_counts, current_counts)
    
    # Compute statistics
    original_stats = original_df.select(
        spark_round(col("RATING").cast("double"), 4).alias("RATING")
    ).agg(
        {"RATING": "mean"}
    ).collect()[0]
    original_mean = float(original_stats["avg(RATING)"])
    
    current_stats = current_df.select(
        spark_round(col("RATING").cast("double"), 4).alias("RATING")
    ).agg(
        {"RATING": "mean"}
    ).collect()[0]
    current_mean = float(current_stats["avg(RATING)"])
    
    # Compute median using percentile
    original_median = float(
        original_df.stat.approxQuantile("RATING", [0.5], 0.01)[0]
    )
    current_median = float(
        current_df.stat.approxQuantile("RATING", [0.5], 0.01)[0]
    )
    
    # Print results
    print("\n" + "-" * 80)
    print("RATING DISTRIBUTION COMPARISON")
    print("-" * 80)
    print(f"{'Rating':<10} {'Original Count':<20} {'Current Count':<20} {'Difference':<15}")
    print("-" * 80)
    for i, rating in enumerate(rating_values):
        diff = int(current_counts[i]) - int(original_counts[i])
        diff_sign = "+" if diff > 0 else ""
        print(f"{rating:<10} {int(original_counts[i]):<20,} {int(current_counts[i]):<20,} {diff_sign}{diff:,}")
    
    print("\n" + "-" * 80)
    print("DRIFT METRICS")
    print("-" * 80)
    print(f"KL Divergence:        {kl_div:.6f}")
    print(f"\nOriginal Data:")
    print(f"  Mean:               {original_mean:.4f}")
    print(f"  Median:             {original_median:.1f}")
    print(f"\nCurrent Data:")
    print(f"  Mean:               {current_mean:.4f}")
    print(f"  Median:             {current_median:.1f}")
    print(f"\nDifferences:")
    print(f"  Mean Difference:    {current_mean - original_mean:+.4f}")
    print(f"  Median Difference:  {current_median - original_median:+.1f}")
    print("-" * 80)
    
    result = {
        "original_counts": original_counts.tolist(),
        "current_counts": current_counts.tolist(),
        "kl_divergence": kl_div,
        "original_mean": original_mean,
        "original_median": original_median,
        "current_mean": current_mean,
        "current_median": current_median,
        "original_total": int(original_count),
        "current_total": int(current_count)
    }
    
    return result


def detect_model_drift(
    spark: SparkSession,
    model_path: str,
    original_test_path: str = "data/processed/parquet/test",
    current_test_path: str = "data/streaming/test"
) -> Dict[str, any]:
    """
    Detect model drift by comparing prediction distributions on original vs current test data.
    
    Uses a trained ALS model to make predictions on both test sets and compares the
    distributions of predicted ratings.
    
    Args:
        spark: SparkSession instance
        model_path: Path to the trained ALS model
        original_test_path: Path to original test data
        current_test_path: Path to current streaming test data
        
    Returns:
        Dictionary containing:
            - original_pred_counts: Array of prediction counts (binned to 1-5)
            - current_pred_counts: Array of prediction counts (binned to 1-5)
            - kl_divergence: KL divergence value
            - original_pred_mean: Mean predicted rating in original test
            - original_pred_median: Median predicted rating in original test
            - current_pred_mean: Mean predicted rating in current test
            - current_pred_median: Median predicted rating in current test
    """
    print("\n" + "=" * 80)
    print("MODEL DRIFT DETECTION")
    print("=" * 80)
    
    # Load the ALS model
    print(f"\nLoading ALS model from: {model_path}")
    try:
        model = load_als_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    
    # Load original test data
    print(f"\nLoading original test data from: {original_test_path}")
    try:
        original_test_df = spark.read.parquet(original_test_path)
        original_count = original_test_df.count()
        print(f"Loaded {original_count:,} rows from original test data")
    except Exception as e:
        print(f"Error loading original test data: {str(e)}")
        raise
    
    # Load current test data
    print(f"\nLoading current test data from: {current_test_path}")
    try:
        current_test_df = spark.read.parquet(current_test_path)
        current_count = current_test_df.count()
        print(f"Loaded {current_count:,} rows from current test data")
    except Exception as e:
        print(f"Error loading current test data: {str(e)}")
        raise
    
    # Make predictions on original test data
    print("\nGenerating predictions on original test data...")
    original_predictions = model.transform(original_test_df)
    original_predictions = original_predictions.filter(col("prediction").isNotNull())
    
    # Make predictions on current test data
    print("Generating predictions on current test data...")
    current_predictions = model.transform(current_test_df)
    current_predictions = current_predictions.filter(col("prediction").isNotNull())
    
    # Round predictions to nearest integer (1-5) for distribution comparison
    original_predictions = original_predictions.withColumn(
        "pred_rounded",
        spark_round(col("prediction")).cast("int")
    )
    current_predictions = current_predictions.withColumn(
        "pred_rounded",
        spark_round(col("prediction")).cast("int")
    )
    
    # Clip predictions to valid range [1, 5]
    from pyspark.sql.functions import when, lit
    original_predictions = original_predictions.withColumn(
        "pred_rounded",
        when(col("pred_rounded") < 1, lit(1))
        .when(col("pred_rounded") > 5, lit(5))
        .otherwise(col("pred_rounded"))
    )
    current_predictions = current_predictions.withColumn(
        "pred_rounded",
        when(col("pred_rounded") < 1, lit(1))
        .when(col("pred_rounded") > 5, lit(5))
        .otherwise(col("pred_rounded"))
    )
    
    # Get prediction distributions (counts for predicted ratings 1-5)
    rating_values = [1, 2, 3, 4, 5]
    
    # Original prediction distribution
    original_pred_counts_rows = original_predictions.groupBy("pred_rounded").count().collect()
    original_pred_counts_dict = {row["pred_rounded"]: row["count"] for row in original_pred_counts_rows}
    original_pred_counts = np.array([original_pred_counts_dict.get(r, 0) for r in rating_values])
    
    # Current prediction distribution
    current_pred_counts_rows = current_predictions.groupBy("pred_rounded").count().collect()
    current_pred_counts_dict = {row["pred_rounded"]: row["count"] for row in current_pred_counts_rows}
    current_pred_counts = np.array([current_pred_counts_dict.get(r, 0) for r in rating_values])
    
    # Compute KL divergence
    kl_div = compute_kl_divergence(original_pred_counts, current_pred_counts)
    
    # Compute statistics on actual prediction values (not rounded)
    original_pred_stats = original_predictions.agg(
        {"prediction": "mean"}
    ).collect()[0]
    original_pred_mean = float(original_pred_stats["avg(prediction)"])
    
    current_pred_stats = current_predictions.agg(
        {"prediction": "mean"}
    ).collect()[0]
    current_pred_mean = float(current_pred_stats["avg(prediction)"])
    
    # Compute median
    original_pred_median = float(
        original_predictions.stat.approxQuantile("prediction", [0.5], 0.01)[0]
    )
    current_pred_median = float(
        current_predictions.stat.approxQuantile("prediction", [0.5], 0.01)[0]
    )
    
    # Print results
    print("\n" + "-" * 80)
    print("PREDICTION DISTRIBUTION COMPARISON")
    print("-" * 80)
    print(f"{'Predicted Rating':<20} {'Original Count':<20} {'Current Count':<20} {'Difference':<15}")
    print("-" * 80)
    for i, rating in enumerate(rating_values):
        diff = int(current_pred_counts[i]) - int(original_pred_counts[i])
        diff_sign = "+" if diff > 0 else ""
        print(f"{rating:<20} {int(original_pred_counts[i]):<20,} {int(current_pred_counts[i]):<20,} {diff_sign}{diff:,}")
    
    print("\n" + "-" * 80)
    print("MODEL DRIFT METRICS")
    print("-" * 80)
    print(f"KL Divergence:        {kl_div:.6f}")
    print(f"\nOriginal Test Predictions:")
    print(f"  Mean:               {original_pred_mean:.4f}")
    print(f"  Median:             {original_pred_median:.4f}")
    print(f"\nCurrent Test Predictions:")
    print(f"  Mean:               {current_pred_mean:.4f}")
    print(f"  Median:             {current_pred_median:.4f}")
    print(f"\nDifferences:")
    print(f"  Mean Difference:    {current_pred_mean - original_pred_mean:+.4f}")
    print(f"  Median Difference:  {current_pred_median - original_pred_median:+.4f}")
    print("-" * 80)
    
    result = {
        "original_pred_counts": original_pred_counts.tolist(),
        "current_pred_counts": current_pred_counts.tolist(),
        "kl_divergence": kl_div,
        "original_pred_mean": original_pred_mean,
        "original_pred_median": original_pred_median,
        "current_pred_mean": current_pred_mean,
        "current_pred_median": current_pred_median,
        "original_total": int(original_predictions.count()),
        "current_total": int(current_predictions.count())
    }
    
    return result


def should_retrain(
    data_drift_result: Dict[str, any] = None,
    model_drift_result: Dict[str, any] = None,
    kl_threshold: float = 0.1,
    mean_shift_threshold: float = 0.2,
    median_shift_threshold: float = 0.5,
    kl_weight: float = 0.5,
    mean_weight: float = 0.3,
    median_weight: float = 0.2
) -> Dict[str, any]:
    """
    Determine if model retraining is needed based on drift metrics using a weighted heuristic.
    
    The function computes a drift score based on:
    - KL divergence (normalized by threshold)
    - Mean shift (normalized by threshold)
    - Median shift (normalized by threshold)
    
    Each metric is weighted and combined into a final drift score. If the score exceeds 1.0,
    retraining is recommended.
    
    Args:
        data_drift_result: Result dictionary from detect_data_drift()
        model_drift_result: Result dictionary from detect_model_drift()
        kl_threshold: Threshold for KL divergence (default: 0.1)
        mean_shift_threshold: Threshold for mean shift (default: 0.2)
        median_shift_threshold: Threshold for median shift (default: 0.5)
        kl_weight: Weight for KL divergence in final score (default: 0.5)
        mean_weight: Weight for mean shift in final score (default: 0.3)
        median_weight: Weight for median shift in final score (default: 0.2)
        
    Returns:
        Dictionary containing:
            - should_retrain: Boolean indicating if retraining is recommended
            - drift_score: Combined drift score (>1.0 means retrain)
            - data_drift_score: Score from data drift (if provided)
            - model_drift_score: Score from model drift (if provided)
            - reasons: List of reasons for the decision
    """
    print("\n" + "=" * 80)
    print("RETRAINING DECISION ANALYSIS")
    print("=" * 80)
    
    if data_drift_result is None and model_drift_result is None:
        raise ValueError("At least one of data_drift_result or model_drift_result must be provided")
    
    # Validate weights sum to 1.0
    total_weight = kl_weight + mean_weight + median_weight
    if not abs(total_weight - 1.0) < 0.001:
        raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    reasons = []
    drift_scores = []
    
    # Analyze data drift
    if data_drift_result:
        print("\nData Drift Analysis:")
        print("-" * 80)
        
        kl_div = data_drift_result["kl_divergence"]
        mean_shift = abs(data_drift_result["current_mean"] - data_drift_result["original_mean"])
        median_shift = abs(data_drift_result["current_median"] - data_drift_result["original_median"])
        
        # Normalize by thresholds
        kl_score = kl_div / kl_threshold
        mean_score = mean_shift / mean_shift_threshold
        median_score = median_shift / median_shift_threshold
        
        # Weighted score
        data_drift_score = (kl_weight * kl_score + 
                           mean_weight * mean_score + 
                           median_weight * median_score)
        
        drift_scores.append(data_drift_score)
        
        print(f"  KL Divergence:      {kl_div:.6f} (threshold: {kl_threshold:.6f}, score: {kl_score:.2f})")
        print(f"  Mean Shift:         {mean_shift:.4f} (threshold: {mean_shift_threshold:.4f}, score: {mean_score:.2f})")
        print(f"  Median Shift:       {median_shift:.1f} (threshold: {median_shift_threshold:.1f}, score: {median_score:.2f})")
        print(f"  Data Drift Score:   {data_drift_score:.4f}")
        
        if kl_div > kl_threshold:
            reasons.append(f"Data KL divergence ({kl_div:.4f}) exceeds threshold ({kl_threshold})")
        if mean_shift > mean_shift_threshold:
            reasons.append(f"Data mean shift ({mean_shift:.4f}) exceeds threshold ({mean_shift_threshold})")
        if median_shift > median_shift_threshold:
            reasons.append(f"Data median shift ({median_shift:.1f}) exceeds threshold ({median_shift_threshold})")
    else:
        data_drift_score = None
    
    # Analyze model drift
    if model_drift_result:
        print("\nModel Drift Analysis:")
        print("-" * 80)
        
        kl_div = model_drift_result["kl_divergence"]
        mean_shift = abs(model_drift_result["current_pred_mean"] - model_drift_result["original_pred_mean"])
        median_shift = abs(model_drift_result["current_pred_median"] - model_drift_result["original_pred_median"])
        
        # Normalize by thresholds
        kl_score = kl_div / kl_threshold
        mean_score = mean_shift / mean_shift_threshold
        median_score = median_shift / median_shift_threshold
        
        # Weighted score
        model_drift_score = (kl_weight * kl_score + 
                            mean_weight * mean_score + 
                            median_weight * median_score)
        
        drift_scores.append(model_drift_score)
        
        print(f"  KL Divergence:      {kl_div:.6f} (threshold: {kl_threshold:.6f}, score: {kl_score:.2f})")
        print(f"  Mean Shift:         {mean_shift:.4f} (threshold: {mean_shift_threshold:.4f}, score: {mean_score:.2f})")
        print(f"  Median Shift:       {median_shift:.4f} (threshold: {median_shift_threshold:.4f}, score: {median_score:.2f})")
        print(f"  Model Drift Score:  {model_drift_score:.4f}")
        
        if kl_div > kl_threshold:
            reasons.append(f"Model KL divergence ({kl_div:.4f}) exceeds threshold ({kl_threshold})")
        if mean_shift > mean_shift_threshold:
            reasons.append(f"Model mean shift ({mean_shift:.4f}) exceeds threshold ({mean_shift_threshold})")
        if median_shift > median_shift_threshold:
            reasons.append(f"Model median shift ({median_shift:.4f}) exceeds threshold ({median_shift_threshold})")
    else:
        model_drift_score = None
    
    # Compute final drift score (max of available scores)
    final_drift_score = max(drift_scores)
    
    # Decision: retrain if drift score > 1.0
    retrain_decision = final_drift_score > 1.0
    
    print("\n" + "=" * 80)
    print("FINAL DECISION")
    print("=" * 80)
    print(f"Final Drift Score:    {final_drift_score:.4f}")
    print(f"Threshold:            1.0000")
    print(f"Decision:             {'RETRAIN RECOMMENDED ⚠️' if retrain_decision else 'NO RETRAINING NEEDED ✓'}")
    
    if reasons:
        print(f"\nReasons:")
        for reason in reasons:
            print(f"  • {reason}")
    else:
        print(f"\nAll drift metrics are within acceptable thresholds.")
    
    print("=" * 80)
    
    result = {
        "should_retrain": retrain_decision,
        "drift_score": final_drift_score,
        "data_drift_score": data_drift_score,
        "model_drift_score": model_drift_score,
        "reasons": reasons,
        "thresholds": {
            "kl_divergence": kl_threshold,
            "mean_shift": mean_shift_threshold,
            "median_shift": median_shift_threshold
        },
        "weights": {
            "kl": kl_weight,
            "mean": mean_weight,
            "median": median_weight
        }
    }
    
    return result


def merge_streaming_to_processed(
    spark: SparkSession,
    streaming_base_path: str = "data/streaming",
    processed_base_path: str = "data/processed/parquet",
    delete_streaming_after_merge: bool = True
) -> None:
    """
    Upsert streaming data (train and test) into processed data directories.
    
    This function merges new streaming data into the main processed datasets,
    removing duplicates and optionally cleaning up the streaming directory.
    
    Args:
        spark: SparkSession instance
        streaming_base_path: Base path for streaming data (default: data/streaming)
        processed_base_path: Base path for processed data (default: data/processed/parquet)
        delete_streaming_after_merge: Whether to delete streaming data after merge (default: True)
    """
    print("\n" + "=" * 80)
    print("MERGING STREAMING DATA TO PROCESSED DATA")
    print("=" * 80)
    
    # Define paths
    streaming_train_path = os.path.join(streaming_base_path, "train")
    streaming_test_path = os.path.join(streaming_base_path, "test")
    processed_train_path = os.path.join(processed_base_path, "train")
    processed_test_path = os.path.join(processed_base_path, "test")
    
    merge_results = {"train": False, "test": False}
    
    # Merge training data
    print(f"\nMerging training data:")
    print(f"  From: {streaming_train_path}")
    print(f"  To:   {processed_train_path}")
    
    try:
        # Check if streaming train data exists
        streaming_train_df = spark.read.parquet(streaming_train_path)
        train_count = streaming_train_df.count()
        print(f"  Found {train_count:,} rows in streaming training data")
        
        # Upsert into processed train data
        print(f"  Upserting data...")
        upsert_ratings_parquet(
            new_df=streaming_train_df,
            output_path=processed_train_path,
            subset_cols=("MOVIE_ID", "CUST_ID", "DATE"),
            target_col="RATING",
            partition_cols=("year", "month"),
            target_repartition=4
        )
        print(f"  ✓ Successfully merged {train_count:,} training samples")
        merge_results["train"] = True
        
    except Exception as e:
        print(f"  ✗ Error merging training data: {str(e)}")
        if "Path does not exist" in str(e):
            print(f"  No streaming training data found, skipping...")
        else:
            raise
    
    # Merge test data
    print(f"\nMerging test data:")
    print(f"  From: {streaming_test_path}")
    print(f"  To:   {processed_test_path}")
    
    try:
        # Check if streaming test data exists
        streaming_test_df = spark.read.parquet(streaming_test_path)
        test_count = streaming_test_df.count()
        print(f"  Found {test_count:,} rows in streaming test data")
        
        # Upsert into processed test data
        print(f"  Upserting data...")
        upsert_ratings_parquet(
            new_df=streaming_test_df,
            output_path=processed_test_path,
            subset_cols=("MOVIE_ID", "CUST_ID", "DATE"),
            target_col="RATING",
            partition_cols=("year", "month"),
            target_repartition=4
        )
        print(f"  ✓ Successfully merged {test_count:,} test samples")
        merge_results["test"] = True
        
    except Exception as e:
        print(f"  ✗ Error merging test data: {str(e)}")
        if "Path does not exist" in str(e):
            print(f"  No streaming test data found, skipping...")
        else:
            raise
    
    # Optionally delete streaming data after successful merge
    if delete_streaming_after_merge and (merge_results["train"] or merge_results["test"]):
        print(f"\nCleaning up streaming data...")
        
        import shutil
        
        if merge_results["train"]:
            try:
                if os.path.exists(streaming_train_path):
                    shutil.rmtree(streaming_train_path)
                    print(f"  ✓ Deleted streaming train directory: {streaming_train_path}")
            except Exception as e:
                print(f"  ✗ Warning: Could not delete {streaming_train_path}: {str(e)}")
        
        if merge_results["test"]:
            try:
                if os.path.exists(streaming_test_path):
                    shutil.rmtree(streaming_test_path)
                    print(f"  ✓ Deleted streaming test directory: {streaming_test_path}")
            except Exception as e:
                print(f"  ✗ Warning: Could not delete {streaming_test_path}: {str(e)}")
    
    print("\n" + "=" * 80)
    print("MERGE COMPLETED")
    print("=" * 80)
    print(f"Training data merged: {'✓' if merge_results['train'] else '✗'}")
    print(f"Test data merged:     {'✓' if merge_results['test'] else '✗'}")
    print("=" * 80)


def process_and_detect_drift(
    csv_path: str,
    output_base_path: str = "data/streaming",
    train_ratio: float = 0.8,
    model_path: str = None,
    original_train_path: str = "data/processed/parquet/train",
    original_test_path: str = "data/processed/parquet/test",
    kl_threshold: float = 0.1,
    mean_shift_threshold: float = 0.2,
    median_shift_threshold: float = 0.5,
    kl_weight: float = 0.5,
    mean_weight: float = 0.3,
    median_weight: float = 0.2,
    shuffle_partitions: int = 200
) -> Dict[str, any]:
    """
    Spark job to process streaming CSV data and detect drift.
    
    This is a complete pipeline that:
    1. Loads and processes streaming CSV data
    2. Splits into train/test sets
    3. Saves as Parquet files
    4. Detects data drift
    5. Detects model drift (if model_path provided)
    6. Makes retraining decision
    
    Args:
        csv_path: Path to the input CSV file
        output_base_path: Base path for output (default: data/streaming)
        train_ratio: Proportion of data for training (default: 0.8)
        model_path: Path to trained ALS model (optional, for model drift)
        original_train_path: Path to original training data
        original_test_path: Path to original test data
        kl_threshold: Threshold for KL divergence
        mean_shift_threshold: Threshold for mean shift
        median_shift_threshold: Threshold for median shift
        kl_weight: Weight for KL divergence in retraining decision
        mean_weight: Weight for mean shift in retraining decision
        median_weight: Weight for median shift in retraining decision
        shuffle_partitions: Number of shuffle partitions for Spark
        
    Returns:
        Dictionary containing:
            - data_drift: Data drift detection results
            - model_drift: Model drift detection results (if model_path provided)
            - retrain_decision: Retraining decision results
            - csv_processed: Boolean indicating if CSV was processed successfully
    """
    print("\n" + "=" * 80)
    print("SPARK JOB: PROCESS AND DETECT DRIFT")
    print("=" * 80)
    
    # Build Spark session
    spark = build_spark("ProcessAndDetectDrift", shuffle_partitions=shuffle_partitions)
    
    results = {
        "data_drift": None,
        "model_drift": None,
        "retrain_decision": None,
        "csv_processed": False
    }
    
    try:
        # Step 1: Process streaming CSV data
        print("\n" + "-" * 80)
        print("STEP 1: PROCESSING STREAMING CSV DATA")
        print("-" * 80)
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Input CSV file not found: {csv_path}")
        
        process_streaming_data(
            spark=spark,
            csv_path=csv_path,
            output_base_path=output_base_path,
            train_ratio=train_ratio
        )
        results["csv_processed"] = True
        print("✓ CSV processing completed")
        
        # Define paths for drift detection
        current_train_path = os.path.join(output_base_path, "train")
        current_test_path = os.path.join(output_base_path, "test")
        
        # Step 2: Detect data drift
        print("\n" + "-" * 80)
        print("STEP 2: DETECTING DATA DRIFT")
        print("-" * 80)
        
        data_drift_result = detect_data_drift(
            spark=spark,
            original_train_path=original_train_path,
            current_train_path=current_train_path
        )
        results["data_drift"] = data_drift_result
        print("✓ Data drift detection completed")
        
        # Step 3: Detect model drift (if model path provided)
        model_drift_result = None
        if model_path:
            print("\n" + "-" * 80)
            print("STEP 3: DETECTING MODEL DRIFT")
            print("-" * 80)
            
            model_drift_result = detect_model_drift(
                spark=spark,
                model_path=model_path,
                original_test_path=original_test_path,
                current_test_path=current_test_path
            )
            results["model_drift"] = model_drift_result
            print("✓ Model drift detection completed")
        else:
            print("\n" + "-" * 80)
            print("STEP 3: SKIPPING MODEL DRIFT (no model path provided)")
            print("-" * 80)
        
        # Step 4: Make retraining decision
        print("\n" + "-" * 80)
        print("STEP 4: RETRAINING DECISION")
        print("-" * 80)
        
        retrain_result = should_retrain(
            data_drift_result=data_drift_result,
            model_drift_result=model_drift_result,
            kl_threshold=kl_threshold,
            mean_shift_threshold=mean_shift_threshold,
            median_shift_threshold=median_shift_threshold,
            kl_weight=kl_weight,
            mean_weight=mean_weight,
            median_weight=median_weight
        )
        results["retrain_decision"] = retrain_result
        
        print("\n" + "=" * 80)
        print("SPARK JOB COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"CSV Processed:        ✓")
        print(f"Data Drift Detected:  ✓")
        print(f"Model Drift Detected: {'✓' if model_path else 'N/A'}")
        print(f"Retrain Decision:     {'RETRAIN ⚠️' if retrain_result['should_retrain'] else 'NO RETRAIN ✓'}")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error in drift detection job: {str(e)}")
        raise
    finally:
        spark.stop()


def merge_streaming_data_job(
    streaming_base_path: str = "data/streaming",
    processed_base_path: str = "data/processed/parquet",
    delete_streaming_after_merge: bool = True,
    shuffle_partitions: int = 200
) -> Dict[str, bool]:
    """
    Spark job to merge streaming data into processed data.
    
    This is a standalone job that upserts streaming train and test data
    into the processed data directories.
    
    Args:
        streaming_base_path: Base path for streaming data (default: data/streaming)
        processed_base_path: Base path for processed data (default: data/processed/parquet)
        delete_streaming_after_merge: Whether to delete streaming data after merge
        shuffle_partitions: Number of shuffle partitions for Spark
        
    Returns:
        Dictionary containing:
            - train_merged: Boolean indicating if train data was merged
            - test_merged: Boolean indicating if test data was merged
    """
    print("\n" + "=" * 80)
    print("SPARK JOB: MERGE STREAMING DATA")
    print("=" * 80)
    
    # Build Spark session
    spark = build_spark("MergeStreamingData", shuffle_partitions=shuffle_partitions)
    
    try:
        # Call the merge function
        merge_streaming_to_processed(
            spark=spark,
            streaming_base_path=streaming_base_path,
            processed_base_path=processed_base_path,
            delete_streaming_after_merge=delete_streaming_after_merge
        )
        
        # Check what was merged by trying to read the paths
        results = {"train_merged": False, "test_merged": False}
        
        streaming_train_path = os.path.join(streaming_base_path, "train")
        streaming_test_path = os.path.join(streaming_base_path, "test")
        
        # If delete_streaming_after_merge is True and paths don't exist, merge was successful
        if delete_streaming_after_merge:
            results["train_merged"] = not os.path.exists(streaming_train_path)
            results["test_merged"] = not os.path.exists(streaming_test_path)
        else:
            # If keeping streaming data, assume success (merge function would have raised exception otherwise)
            results["train_merged"] = True
            results["test_merged"] = True
        
        print("\n" + "=" * 80)
        print("SPARK JOB COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error in merge job: {str(e)}")
        raise
    finally:
        spark.stop()


def main():
    """Main entry point for the drift detection script."""
    parser = argparse.ArgumentParser(description="Process streaming CSV data and perform drift detection")
    
    # Add subparsers for different operations
    subparsers = parser.add_subparsers(dest="operation", help="Operation to perform")
    
    # Process streaming data subcommand
    process_parser = subparsers.add_parser("process", help="Process streaming CSV data")
    process_parser.add_argument(
        "--csv_path",
        default="data/streaming/current.csv",
        help="Path to the input CSV file (default: data/streaming/current.csv)"
    )
    process_parser.add_argument(
        "--output_path",
        default="data/streaming",
        help="Base output path for Parquet files (default: data/streaming)"
    )
    process_parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Proportion of data for training (default: 0.8)"
    )
    process_parser.add_argument(
        "--shuffle_partitions",
        type=int,
        default=200,
        help="Number of shuffle partitions for Spark (default: 200)"
    )
    
    # Data drift detection subcommand
    data_drift_parser = subparsers.add_parser("data_drift", help="Detect data drift")
    data_drift_parser.add_argument(
        "--original_train_path",
        default="data/processed/parquet/train",
        help="Path to original training data (default: data/processed/parquet/train)"
    )
    data_drift_parser.add_argument(
        "--current_train_path",
        default="data/streaming/train",
        help="Path to current training data (default: data/streaming/train)"
    )
    data_drift_parser.add_argument(
        "--shuffle_partitions",
        type=int,
        default=200,
        help="Number of shuffle partitions for Spark (default: 200)"
    )
    
    # Model drift detection subcommand
    model_drift_parser = subparsers.add_parser("model_drift", help="Detect model drift")
    model_drift_parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the trained ALS model"
    )
    model_drift_parser.add_argument(
        "--original_test_path",
        default="data/processed/parquet/test",
        help="Path to original test data (default: data/processed/parquet/test)"
    )
    model_drift_parser.add_argument(
        "--current_test_path",
        default="data/streaming/test",
        help="Path to current test data (default: data/streaming/test)"
    )
    model_drift_parser.add_argument(
        "--shuffle_partitions",
        type=int,
        default=200,
        help="Number of shuffle partitions for Spark (default: 200)"
    )
    
    # Check retraining subcommand
    retrain_parser = subparsers.add_parser("check_retrain", help="Determine if retraining is needed")
    retrain_parser.add_argument(
        "--model_path",
        help="Path to the trained ALS model (optional, for model drift)"
    )
    retrain_parser.add_argument(
        "--original_train_path",
        default="data/processed/parquet/train",
        help="Path to original training data (default: data/processed/parquet/train)"
    )
    retrain_parser.add_argument(
        "--current_train_path",
        default="data/streaming/train",
        help="Path to current training data (default: data/streaming/train)"
    )
    retrain_parser.add_argument(
        "--original_test_path",
        default="data/processed/parquet/test",
        help="Path to original test data (default: data/processed/parquet/test)"
    )
    retrain_parser.add_argument(
        "--current_test_path",
        default="data/streaming/test",
        help="Path to current test data (default: data/streaming/test)"
    )
    retrain_parser.add_argument(
        "--kl_threshold",
        type=float,
        default=0.1,
        help="Threshold for KL divergence (default: 0.1)"
    )
    retrain_parser.add_argument(
        "--mean_shift_threshold",
        type=float,
        default=0.2,
        help="Threshold for mean shift (default: 0.2)"
    )
    retrain_parser.add_argument(
        "--median_shift_threshold",
        type=float,
        default=0.5,
        help="Threshold for median shift (default: 0.5)"
    )
    retrain_parser.add_argument(
        "--kl_weight",
        type=float,
        default=0.5,
        help="Weight for KL divergence (default: 0.5)"
    )
    retrain_parser.add_argument(
        "--mean_weight",
        type=float,
        default=0.3,
        help="Weight for mean shift (default: 0.3)"
    )
    retrain_parser.add_argument(
        "--median_weight",
        type=float,
        default=0.2,
        help="Weight for median shift (default: 0.2)"
    )
    retrain_parser.add_argument(
        "--shuffle_partitions",
        type=int,
        default=200,
        help="Number of shuffle partitions for Spark (default: 200)"
    )
    
    # Merge streaming data subcommand
    merge_parser = subparsers.add_parser("merge", help="Merge streaming data into processed data")
    merge_parser.add_argument(
        "--streaming_path",
        default="data/streaming",
        help="Base path for streaming data (default: data/streaming)"
    )
    merge_parser.add_argument(
        "--processed_path",
        default="data/processed/parquet",
        help="Base path for processed data (default: data/processed/parquet)"
    )
    merge_parser.add_argument(
        "--keep_streaming",
        action="store_true",
        help="Keep streaming data after merge (default: delete)"
    )
    merge_parser.add_argument(
        "--shuffle_partitions",
        type=int,
        default=200,
        help="Number of shuffle partitions for Spark (default: 200)"
    )
    
    # Process and detect drift job (standalone Spark job)
    process_detect_parser = subparsers.add_parser(
        "process_and_detect",
        help="[SPARK JOB] Process streaming data and detect drift"
    )
    process_detect_parser.add_argument(
        "--csv_path",
        required=True,
        help="Path to the input CSV file"
    )
    process_detect_parser.add_argument(
        "--output_path",
        default="data/streaming",
        help="Base output path for Parquet files (default: data/streaming)"
    )
    process_detect_parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Proportion of data for training (default: 0.8)"
    )
    process_detect_parser.add_argument(
        "--model_path",
        help="Path to the trained ALS model (optional, for model drift)"
    )
    process_detect_parser.add_argument(
        "--original_train_path",
        default="data/processed/parquet/train",
        help="Path to original training data (default: data/processed/parquet/train)"
    )
    process_detect_parser.add_argument(
        "--original_test_path",
        default="data/processed/parquet/test",
        help="Path to original test data (default: data/processed/parquet/test)"
    )
    process_detect_parser.add_argument(
        "--kl_threshold",
        type=float,
        default=0.1,
        help="Threshold for KL divergence (default: 0.1)"
    )
    process_detect_parser.add_argument(
        "--mean_shift_threshold",
        type=float,
        default=0.2,
        help="Threshold for mean shift (default: 0.2)"
    )
    process_detect_parser.add_argument(
        "--median_shift_threshold",
        type=float,
        default=0.5,
        help="Threshold for median shift (default: 0.5)"
    )
    process_detect_parser.add_argument(
        "--kl_weight",
        type=float,
        default=0.5,
        help="Weight for KL divergence (default: 0.5)"
    )
    process_detect_parser.add_argument(
        "--mean_weight",
        type=float,
        default=0.3,
        help="Weight for mean shift (default: 0.3)"
    )
    process_detect_parser.add_argument(
        "--median_weight",
        type=float,
        default=0.2,
        help="Weight for median shift (default: 0.2)"
    )
    process_detect_parser.add_argument(
        "--shuffle_partitions",
        type=int,
        default=200,
        help="Number of shuffle partitions for Spark (default: 200)"
    )
    
    # Merge streaming data job (standalone Spark job)
    merge_job_parser = subparsers.add_parser(
        "merge_job",
        help="[SPARK JOB] Merge streaming data into processed data"
    )
    merge_job_parser.add_argument(
        "--streaming_path",
        default="data/streaming",
        help="Base path for streaming data (default: data/streaming)"
    )
    merge_job_parser.add_argument(
        "--processed_path",
        default="data/processed/parquet",
        help="Base path for processed data (default: data/processed/parquet)"
    )
    merge_job_parser.add_argument(
        "--keep_streaming",
        action="store_true",
        help="Keep streaming data after merge (default: delete)"
    )
    merge_job_parser.add_argument(
        "--shuffle_partitions",
        type=int,
        default=200,
        help="Number of shuffle partitions for Spark (default: 200)"
    )
    
    args = parser.parse_args()
    
    if not args.operation:
        parser.print_help()
        return
    
    # Build Spark session
    shuffle_partitions = getattr(args, 'shuffle_partitions', 200)
    spark = build_spark("DriftDetection", shuffle_partitions=shuffle_partitions)
    
    try:
        if args.operation == "process":
            # Validate train_ratio
            if not 0 < args.train_ratio < 1:
                raise ValueError("train_ratio must be between 0 and 1")
            
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
            
        elif args.operation == "data_drift":
            # Detect data drift
            result = detect_data_drift(
                spark=spark,
                original_train_path=args.original_train_path,
                current_train_path=args.current_train_path
            )
            print("\n✓ Data drift detection completed successfully")
            
        elif args.operation == "model_drift":
            # Detect model drift
            result = detect_model_drift(
                spark=spark,
                model_path=args.model_path,
                original_test_path=args.original_test_path,
                current_test_path=args.current_test_path
            )
            print("\n✓ Model drift detection completed successfully")
            
        elif args.operation == "check_retrain":
            # Check if retraining is needed
            data_drift_result = None
            model_drift_result = None
            
            # Always run data drift detection
            print("Running data drift detection...")
            data_drift_result = detect_data_drift(
                spark=spark,
                original_train_path=args.original_train_path,
                current_train_path=args.current_train_path
            )
            
            # Run model drift detection if model path is provided
            if args.model_path:
                print("\nRunning model drift detection...")
                model_drift_result = detect_model_drift(
                    spark=spark,
                    model_path=args.model_path,
                    original_test_path=args.original_test_path,
                    current_test_path=args.current_test_path
                )
            
            # Make retraining decision
            retrain_result = should_retrain(
                data_drift_result=data_drift_result,
                model_drift_result=model_drift_result,
                kl_threshold=args.kl_threshold,
                mean_shift_threshold=args.mean_shift_threshold,
                median_shift_threshold=args.median_shift_threshold,
                kl_weight=args.kl_weight,
                mean_weight=args.mean_weight,
                median_weight=args.median_weight
            )
            
            # Exit with code 0 if no retrain needed, 1 if retrain needed
            # This allows shell scripts to check the exit code
            import sys
            sys.exit(0 if not retrain_result["should_retrain"] else 1)
            
        elif args.operation == "merge":
            # Merge streaming data into processed data
            merge_streaming_to_processed(
                spark=spark,
                streaming_base_path=args.streaming_path,
                processed_base_path=args.processed_path,
                delete_streaming_after_merge=not args.keep_streaming
            )
            print("\n✓ Merge completed successfully")
            
        elif args.operation == "process_and_detect":
            # Standalone Spark job: Process and detect drift
            # Note: This operation manages its own Spark session
            spark.stop()  # Stop the session created above
            
            # Validate train_ratio
            if not 0 < args.train_ratio < 1:
                raise ValueError("train_ratio must be between 0 and 1")
            
            result = process_and_detect_drift(
                csv_path=args.csv_path,
                output_base_path=args.output_path,
                train_ratio=args.train_ratio,
                model_path=args.model_path,
                original_train_path=args.original_train_path,
                original_test_path=args.original_test_path,
                kl_threshold=args.kl_threshold,
                mean_shift_threshold=args.mean_shift_threshold,
                median_shift_threshold=args.median_shift_threshold,
                kl_weight=args.kl_weight,
                mean_weight=args.mean_weight,
                median_weight=args.median_weight,
                shuffle_partitions=args.shuffle_partitions
            )
            
            # Exit with appropriate code based on retraining decision
            import sys
            if result["retrain_decision"] and result["retrain_decision"]["should_retrain"]:
                sys.exit(1)  # Exit code 1 indicates retraining needed
            else:
                sys.exit(0)  # Exit code 0 indicates no retraining needed
            
        elif args.operation == "merge_job":
            # Standalone Spark job: Merge streaming data
            # Note: This operation manages its own Spark session
            spark.stop()  # Stop the session created above
            
            result = merge_streaming_data_job(
                streaming_base_path=args.streaming_path,
                processed_base_path=args.processed_path,
                delete_streaming_after_merge=not args.keep_streaming,
                shuffle_partitions=args.shuffle_partitions
            )
            
            print(f"\nMerge Job Results:")
            print(f"  Training data merged: {'✓' if result['train_merged'] else '✗'}")
            print(f"  Test data merged:     {'✓' if result['test_merged'] else '✗'}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
    finally:
        # Only stop spark if it's still running (job operations manage their own sessions)
        if args.operation not in ["process_and_detect", "merge_job"]:
            spark.stop()


if __name__ == "__main__":
    main()
