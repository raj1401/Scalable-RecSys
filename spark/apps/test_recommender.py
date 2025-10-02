"""
ALS Model Evaluation Script for Netflix-style Recommendation System

This script evaluates trained ALS models using multiple metrics:
- Regression metrics: RMSE, MAE
- Ranking metrics: Precision@K, Recall@K, NDCG@K

Usage examples:
    # Evaluate latest model with default parameters
    python spark/apps/test_recommender.py --model-path models/artifacts/version_20250930_212327
    
    # Evaluate with custom parameters
    python spark/apps/test_recommender.py \
        --model-path models/artifacts/version_20250930_212327 \
        --test-parquet data/processed/parquet/test \
        --k 20 \
        --rating-threshold 3.5
        
    # Using Docker/Makefile
    make submit-test-model version=version_20250930_212327 k=10 threshold=4.0
"""

from __future__ import annotations

import argparse
import math
from typing import List, Optional

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import col, collect_list, explode, lit, when, desc, row_number, size, slice
from pyspark.sql.types import DoubleType, ArrayType, IntegerType
from pyspark.ml.evaluation import RegressionEvaluator

from spark.apps.common import build_spark, load_als_model


def compute_rmse_mae(predictions_df: DataFrame, rating_col: str = "RATING", prediction_col: str = "prediction") -> tuple[float, float]:
    """
    Compute RMSE and MAE metrics for rating predictions.
    
    Args:
        predictions_df: DataFrame with actual ratings and predictions
        rating_col: Column name for actual ratings
        prediction_col: Column name for predictions
        
    Returns:
        Tuple of (RMSE, MAE)
    """
    # Remove rows with null predictions (cold start users/items)
    clean_predictions = predictions_df.filter(
        col(prediction_col).isNotNull() & col(rating_col).isNotNull()
    )
    
    # Compute RMSE using built-in evaluator
    rmse_evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol=rating_col,
        predictionCol=prediction_col
    )
    rmse = rmse_evaluator.evaluate(clean_predictions)
    
    # Compute MAE using built-in evaluator
    mae_evaluator = RegressionEvaluator(
        metricName="mae",
        labelCol=rating_col,
        predictionCol=prediction_col
    )
    mae = mae_evaluator.evaluate(clean_predictions)
    
    return rmse, mae


def compute_ranking_metrics(
    test_df: DataFrame,
    model,
    user_col: str = "CUST_ID",
    item_col: str = "MOVIE_ID", 
    rating_col: str = "RATING",
    k: int = 10,
    rating_threshold: float = 4.0
) -> tuple[float, float, float]:
    """
    Compute Precision@K, Recall@K, and NDCG@K metrics.
    
    Args:
        test_df: Test DataFrame with user-item-rating triples
        model: Trained ALS model
        user_col: User column name
        item_col: Item column name
        rating_col: Rating column name
        k: Number of recommendations to consider
        rating_threshold: Threshold above which an item is considered relevant
        
    Returns:
        Tuple of (Precision@K, Recall@K, NDCG@K)
    """
    
    # Get unique users in test set
    test_users = test_df.select(user_col).distinct()
    
    # Generate top-K recommendations for each user
    user_recs = model.recommendForUserSubset(test_users, k)
    
    # Extract recommended item IDs
    user_recs_expanded = user_recs.select(
        col(user_col),
        explode(col("recommendations")).alias("rec")
    ).select(
        col(user_col),
        col("rec.MOVIE_ID").alias("rec_item_id"),
        col("rec.rating").alias("rec_score"),
        row_number().over(
            Window.partitionBy(user_col).orderBy(desc("rec.rating"))
        ).alias("rank")
    ).filter(col("rank") <= k)
    
    # Get relevant items for each user (rating >= threshold)
    relevant_items = test_df.filter(col(rating_col) >= rating_threshold).select(
        col(user_col),
        col(item_col).alias("relevant_item_id")
    ).groupBy(user_col).agg(
        collect_list("relevant_item_id").alias("relevant_items")
    )
    
    # Get recommended items for each user
    recommended_items = user_recs_expanded.groupBy(user_col).agg(
        collect_list("rec_item_id").alias("recommended_items"),
        collect_list("rec_score").alias("rec_scores")
    )
    
    # Join relevant and recommended items
    metrics_df = relevant_items.join(recommended_items, user_col, "inner")
    
    # Define functions using Spark SQL expressions with explicit array operations
    def compute_precision_recall_ndcg(df: DataFrame) -> DataFrame:
        # Create a helper DataFrame to compute DCG and IDCG using array operations
        return df.withColumn(
            "hits",
            F.size(F.array_intersect(col("relevant_items"), col("recommended_items")))
        ).withColumn(
            "precision_at_k",
            col("hits") / F.size(col("recommended_items"))
        ).withColumn(
            "recall_at_k", 
            col("hits") / F.size(col("relevant_items"))
        ).withColumn(
            # Create an array of relevance scores (1 if relevant, 0 if not)
            "relevance_scores",
            F.expr(f"""
                transform(
                    sequence(0, {k-1}),
                    i -> case 
                        when i < size(recommended_items) and array_contains(relevant_items, recommended_items[i])
                        then 1.0
                        else 0.0
                    end
                )
            """)
        ).withColumn(
            # Create an array of discount factors
            "discount_factors", 
            F.expr(f"""
                transform(
                    sequence(0, {k-1}),
                    i -> 1.0 / log(2.0, cast(i + 2 as double))
                )
            """)
        ).withColumn(
            # Compute DCG by element-wise multiplication and sum
            "dcg",
            F.expr("""
                aggregate(
                    zip_with(relevance_scores, discount_factors, (x, y) -> x * y),
                    cast(0.0 as double),
                    (acc, x) -> acc + x
                )
            """)
        ).withColumn(
            # Compute IDCG for perfect ranking
            "idcg",
            F.expr(f"""
                aggregate(
                    slice(
                        transform(sequence(0, {k-1}), i -> 1.0 / log(2.0, cast(i + 2 as double))),
                        1,
                        least({k}, size(relevant_items))
                    ),
                    cast(0.0 as double),
                    (acc, x) -> acc + x
                )
            """)
        ).withColumn(
            "ndcg_at_k",
            when(col("idcg") > 0, col("dcg") / col("idcg")).otherwise(0.0)
        ).drop("relevance_scores", "discount_factors")
    
    # Compute metrics for all users
    user_metrics = compute_precision_recall_ndcg(metrics_df)
    
    # Aggregate metrics across all users
    avg_metrics = user_metrics.agg(
        F.avg("precision_at_k").alias("avg_precision"),
        F.avg("recall_at_k").alias("avg_recall"), 
        F.avg("ndcg_at_k").alias("avg_ndcg")
    ).collect()[0]
    
    return (
        float(avg_metrics["avg_precision"]) if avg_metrics["avg_precision"] else 0.0,
        float(avg_metrics["avg_recall"]) if avg_metrics["avg_recall"] else 0.0,
        float(avg_metrics["avg_ndcg"]) if avg_metrics["avg_ndcg"] else 0.0
    )


def evaluate_model(
    model_path: str,
    test_parquet_path: str,
    user_col: str = "CUST_ID",
    item_col: str = "MOVIE_ID",
    rating_col: str = "RATING",
    k: int = 10,
    rating_threshold: float = 4.0
) -> dict:
    """
    Comprehensive evaluation of an ALS model.
    
    Args:
        model_path: Path to saved ALS model
        test_parquet_path: Path to test data in Parquet format
        user_col: User column name
        item_col: Item column name  
        rating_col: Rating column name
        k: Number of recommendations for ranking metrics
        rating_threshold: Threshold for relevant items
        
    Returns:
        Dictionary containing all computed metrics
    """
    
    spark = build_spark("test-recommender-evaluation")
    
    try:
        # Load the trained model
        print(f"Loading model from: {model_path}")
        model = load_als_model(model_path)
        print(f"✓ Model loaded successfully! Rank: {model.rank}")
        
        # Load test data from Parquet
        print(f"Loading test data from: {test_parquet_path}")
        test_df = spark.read.parquet(test_parquet_path)
        
        # Remove rows with null values
        test_df = test_df.filter(
            col(user_col).isNotNull() & 
            col(item_col).isNotNull() & 
            col(rating_col).isNotNull()
        )
        
        test_count = test_df.count()
        print(f"✓ Test data loaded: {test_count:,} ratings")
        
        # Generate predictions for test set
        print("Generating predictions...")
        predictions = model.transform(test_df)
        
        # Compute RMSE and MAE
        print("Computing RMSE and MAE...")
        rmse, mae = compute_rmse_mae(predictions, rating_col, "prediction")
        
        # Compute ranking metrics (Precision@K, Recall@K, NDCG@K)
        print(f"Computing ranking metrics (K={k})...")
        precision_k, recall_k, ndcg_k = compute_ranking_metrics(
            test_df, model, user_col, item_col, rating_col, k, rating_threshold
        )
        
        # Compile results
        results = {
            "model_path": model_path,
            "test_file": test_parquet_path,
            "test_ratings_count": test_count,
            "model_rank": model.rank,
            "rmse": rmse,
            "mae": mae,
            f"precision_at_{k}": precision_k,
            f"recall_at_{k}": recall_k, 
            f"ndcg_at_{k}": ndcg_k,
            "k": k,
            "rating_threshold": rating_threshold
        }
        
        return results
        
    finally:
        spark.stop()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ALS recommender model on test dataset"
    )
    
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the saved ALS model directory"
    )
    
    parser.add_argument(
        "--test-parquet",
        required=False,
        default="data/processed/parquet/test",
        help="Path to test data in Parquet format"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of recommendations for ranking metrics (default: 10)"
    )
    
    parser.add_argument(
        "--rating-threshold",
        type=float,
        default=4.0,
        help="Rating threshold for relevance (default: 4.0)"
    )
    
    parser.add_argument(
        "--shuffle-partitions",
        type=int,
        default=400,
        help="spark.sql.shuffle.partitions to use during evaluation"
    )

    args = parser.parse_args(argv)
    
    # Build Spark session for evaluation
    spark = build_spark(
        app_name="netflix-test-als", 
        shuffle_partitions=args.shuffle_partitions
    )
    
    try:
        # Run evaluation
        results = evaluate_model(
            model_path=args.model_path,
            test_parquet_path=args.test_parquet,
            k=args.k,
            rating_threshold=args.rating_threshold
        )
        
        # Print results
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Model Path: {results['model_path']}")
        print(f"Test File: {results['test_file']}")
        print(f"Test Ratings: {results['test_ratings_count']:,}")
        print(f"Model Rank: {results['model_rank']}")
        print(f"K (for ranking metrics): {results['k']}")
        print(f"Rating Threshold: {results['rating_threshold']}")
        print()
        print("REGRESSION METRICS:")
        print(f"  RMSE: {results['rmse']:.4f}")
        print(f"  MAE:  {results['mae']:.4f}")
        print()
        print("RANKING METRICS:")
        k_val = results['k']
        print(f"  Precision@{k_val}: {results[f'precision_at_{k_val}']:.4f}")
        print(f"  Recall@{k_val}:    {results[f'recall_at_{k_val}']:.4f}")
        print(f"  NDCG@{k_val}:      {results[f'ndcg_at_{k_val}']:.4f}")
        print("="*60)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
