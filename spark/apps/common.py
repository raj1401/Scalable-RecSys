from pyspark.sql import SparkSession, Row, DataFrame, Window
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
import pyspark.sql.functions as F
from pyspark.sql.functions import col, year, month
from pyspark.ml.recommendation import ALS
from pyspark.errors.exceptions.base import AnalysisException
from typing import Iterable, Optional
import datetime

try:
    import mlflow
    from mlflow import spark as mlflow_spark
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def build_spark(app_name: str, shuffle_partitions: Optional[int] = None) -> SparkSession:
    """Create (or get) a SparkSession configured for this project."""
    builder = SparkSession.builder.appName(app_name)
    if shuffle_partitions is not None:
        builder = builder.config("spark.sql.shuffle.partitions", int(shuffle_partitions))
    return builder.getOrCreate()


def parse_netflix_file(spark: SparkSession, path: str) -> DataFrame:
    """
    Parse a Netflix-style ratings text file into a DataFrame with schema:
    MOVIE_ID (int), CUST_ID (int), RATING (int), DATE (string).
    Handles malformed / missing values by setting them to None (NaN in Spark).
    
    Args:
        spark (SparkSession): active Spark session
        path (str): path to the text file (local, HDFS, S3, etc.)
        
    Returns:
        pyspark.sql.DataFrame
    """
    lines = spark.sparkContext.textFile(path)

    def parse_line(line):
        if line.endswith(":"):
            # Header line
            try:
                return ("H", int(line[:-1]))
            except Exception:
                return ("H", None)
        else:
            try:
                cust_id, rating, date = line.split(",")
                cust_id = int(cust_id) if cust_id else None
                rating = int(rating) if rating else None
                date = date if date else None
                return ("R", (cust_id, rating, date))
            except Exception:
                # Malformed rating row
                return ("R", (None, None, None))

    tagged = lines.map(parse_line)

    def attach_movie_id(partition):
        current_movie = None
        for tag, value in partition:
            if tag == "H":
                current_movie = value
            else:
                cust_id, rating, date = value
                yield Row(
                    MOVIE_ID=current_movie,
                    CUST_ID=cust_id,
                    RATING=rating,
                    DATE=date,
                )

    rows = tagged.mapPartitions(attach_movie_id)

    schema = StructType([
        StructField("MOVIE_ID", IntegerType(), True),
        StructField("CUST_ID", IntegerType(), True),
        StructField("RATING", IntegerType(), True),
        StructField("DATE", StringType(), True),
    ])

    return spark.createDataFrame(rows, schema)


def _deduplicate_by_max(
    df: DataFrame,
    subset_cols: Iterable[str],
    target_col: str = "RATING",
) -> DataFrame:
    """Remove duplicates keeping the row with the highest `target_col` (ties resolved deterministically)."""
    window = Window.partitionBy(*subset_cols).orderBy(F.desc(target_col), F.desc("DATE"))
    ranked = df.withColumn("_rank", F.row_number().over(window))
    return ranked.filter(F.col("_rank") == 1).drop("_rank")


def _drop_partition_columns(df: DataFrame, partition_cols: Iterable[str]) -> DataFrame:
    cols_to_drop = [c for c in partition_cols if c in df.columns]
    return df.drop(*cols_to_drop) if cols_to_drop else df


def train_als(
    train_df,
    user_col="CUST_ID",
    item_col="MOVIE_ID",
    rating_col="RATING",
    rank=64,
    reg_param=0.1,
    max_iter=15,
    implicit_prefs=False,
    cold_start_strategy="drop",
    nonnegative=False,
    model_save_path="models/artifacts/",
    mlflow_tracking_uri=None,
    mlflow_experiment="netflix-als-training",
) -> None:
    """
    Train ALS model, extract user/item factors, and save as Parquet files.
    Args:
        train_df: Input DataFrame for ALS training
        user_col, item_col, rating_col: Column names
        rank, reg_param, max_iter, implicit_prefs, cold_start_strategy, nonnegative: ALS params
        model_save_path: Base path to save model artifacts
        mlflow_tracking_uri: MLflow tracking URI (optional)
        mlflow_experiment: MLflow experiment name
    """
    # Setup MLflow if available and requested
    if MLFLOW_AVAILABLE and mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment)

    # Training parameters for logging
    params = {
        "rank": rank,
        "reg_param": reg_param,
        "max_iter": max_iter,
        "implicit_prefs": implicit_prefs,
        "cold_start_strategy": cold_start_strategy,
        "nonnegative": nonnegative,
        "user_col": user_col,
        "item_col": item_col,
        "rating_col": rating_col,
    }

    def _train_and_save():
        als = ALS(
            userCol=user_col,
            itemCol=item_col,
            ratingCol=rating_col,
            rank=rank,
            regParam=reg_param,
            maxIter=max_iter,
            implicitPrefs=implicit_prefs,
            coldStartStrategy=cold_start_strategy,
            nonnegative=nonnegative
        )
        model = als.fit(train_df)

        # Extract factors
        user_f = model.userFactors.withColumnRenamed("id", user_col)
        item_f = model.itemFactors.withColumnRenamed("id", item_col)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        user_f_path = model_save_path.rstrip("/") + f"/version_{timestamp}/user_factors"
        item_f_path = model_save_path.rstrip("/") + f"/version_{timestamp}/item_factors"

        user_f.write.mode("overwrite").parquet(user_f_path)
        item_f.write.mode("overwrite").parquet(item_f_path)

        # Log metrics if MLflow is available
        if MLFLOW_AVAILABLE:
            # Get basic dataset metrics
            train_count = train_df.count()
            user_count = train_df.select(user_col).distinct().count()
            item_count = train_df.select(item_col).distinct().count()
            
            mlflow.log_metric("train_count", train_count)
            mlflow.log_metric("user_count", user_count)
            mlflow.log_metric("item_count", item_count)
            
            # Log model artifacts paths
            mlflow.log_param("user_factors_path", user_f_path)
            mlflow.log_param("item_factors_path", item_f_path)
            mlflow.log_param("model_timestamp", timestamp)

        return model, user_f_path, item_f_path

    # Execute training with or without MLflow tracking
    if MLFLOW_AVAILABLE and mlflow_tracking_uri:
        with mlflow.start_run():
            mlflow.log_params(params)
            model, user_f_path, item_f_path = _train_and_save()
            
            # Log the Spark ML model
            try:
                mlflow_spark.log_model(model, "als_model")
            except Exception as e:
                print(f"Warning: Could not log Spark model to MLflow: {e}")
    else:
        model, user_f_path, item_f_path = _train_and_save()


def write_dataframe_as_parquet(
    df: DataFrame,
    output_path: str,
    *,
    mode: str = "overwrite",
    partition_cols: Iterable[str] = ("year", "month"),
    num_shuffle_partitions: Optional[int] = None,
    target_repartition: Optional[int] = None,
    table_name: Optional[str] = None,
    extra_write_options: Optional[dict] = None,
) -> None:
    """
    Write the ratings dataframe to Parquet format, partitioned by year/month.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame with columns: MOVIE_ID, CUST_ID, RATING, DATE (DateType).
    output_path : str
        Filesystem path (e.g., "s3://…", "abfss://…", or "/data/…") to store the Parquet dataset.
    mode : str, default "overwrite"
        Write mode: "overwrite" or "append".
    partition_cols : Iterable[str], default ("year", "month")
        Partition columns to add and use.
    num_shuffle_partitions : int, optional
        If provided, temporarily sets spark.sql.shuffle.partitions for this write.
    target_repartition : int, optional
        If provided, repartitions the data to this many partitions before write (helps avoid small files).
    table_name : str, optional
        If provided, also registers/creates a table in the metastore that points to `output_path`.
    extra_write_options : dict, optional
        Any additional .option(key, value) pairs to pass to the writer.
    """
    spark = df.sparkSession

    # Optionally tweak shuffle partitions just for this write
    prev_shuffle = spark.conf.get("spark.sql.shuffle.partitions", None)
    if num_shuffle_partitions is not None:
        spark.conf.set("spark.sql.shuffle.partitions", int(num_shuffle_partitions))

    try:
        # Add partition columns if missing
        need_year = "year" in partition_cols and "year" not in df.columns
        need_month = "month" in partition_cols and "month" not in df.columns

        df_out = df
        if need_year:
            df_out = df_out.withColumn("year", year(col("DATE")))
        if need_month:
            df_out = df_out.withColumn("month", month(col("DATE")))

        # Optional global repartition to control file sizes
        if target_repartition is not None:
            # Repartition by partition columns to co-locate data
            df_out = df_out.repartition(int(target_repartition), *[col(c) for c in partition_cols])

        writer = (df_out.write
                  .format("parquet")
                  .mode(mode)
                  .partitionBy(*partition_cols))

        # Sensible defaults; can be overridden via extra_write_options
        writer = writer.option("compression", "snappy")

        if extra_write_options:
            for k, v in extra_write_options.items():
                writer = writer.option(k, v)

        # Write to path
        writer.save(output_path)

        # Optionally register as a table in the metastore
        if table_name:
            # CREATE/REPLACE managed metadata pointing to the path
            spark.sql(f"""
                CREATE TABLE IF NOT EXISTS {table_name}
                USING PARQUET
                LOCATION '{output_path}'
            """)
            if mode == "overwrite":
                # Ensure table metadata reflects current schema/partitions after overwrite
                spark.sql(f"MSCK REPAIR TABLE {table_name}")

    finally:
        # Restore previous shuffle setting
        if num_shuffle_partitions is not None and prev_shuffle is not None:
            spark.conf.set("spark.sql.shuffle.partitions", prev_shuffle)


def upsert_ratings_parquet(
    new_df: DataFrame,
    output_path: str,
    *,
    subset_cols: Iterable[str] = ("MOVIE_ID", "CUST_ID", "DATE"),
    target_col: str = "RATING",
    partition_cols: Iterable[str] = ("year", "month"),
    num_shuffle_partitions: Optional[int] = None,
    target_repartition: Optional[int] = None,
    table_name: Optional[str] = None,
    extra_write_options: Optional[dict] = None,
) -> DataFrame:
    """Merge `new_df` into an existing Parquet dataset, deduplicate, and persist the result."""

    spark = new_df.sparkSession

    partition_cols = tuple(partition_cols)
    base_new_df = _drop_partition_columns(new_df, partition_cols)

    try:
        existing_df = spark.read.parquet(output_path)
        base_existing_df = _drop_partition_columns(existing_df, partition_cols)
        combined_df = base_existing_df.unionByName(base_new_df, allowMissingColumns=True)
    except AnalysisException:
        combined_df = base_new_df

    deduped_df = _deduplicate_by_max(combined_df, subset_cols=subset_cols, target_col=target_col)
    deduped_df = deduped_df.dropna(subset=list(subset_cols) + [target_col])

    write_dataframe_as_parquet(
        df=deduped_df,
        output_path=output_path,
        mode="overwrite",
        partition_cols=partition_cols,
        num_shuffle_partitions=num_shuffle_partitions,
        target_repartition=target_repartition,
        table_name=table_name,
        extra_write_options=extra_write_options,
    )

    return deduped_df