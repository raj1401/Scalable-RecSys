from pyspark.sql import SparkSession, Row, DataFrame, Window
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
import pyspark.sql.functions as F
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, year, month
from typing import Tuple
import datetime
from typing import Iterable, Optional


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


def split_dataframe_by_date(df: DataFrame, date_column: str, max_date: str) -> Tuple[DataFrame, DataFrame]:
    """
    Split a DataFrame into two DataFrames based on a date threshold.
    
    Args:
        df (DataFrame): Input DataFrame to split
        date_column (str): Name of the date column to use for splitting
        max_date (str): Date threshold in string format (e.g., '2005-12-31')
        
    Returns:
        Tuple[DataFrame, DataFrame]: A tuple containing:
            - DataFrame with dates <= max_date
            - DataFrame with dates > max_date
    """
    # Filter for dates less than or equal to max_date
    before_df = df.filter(col(date_column) <= max_date)
    
    # Filter for dates greater than max_date
    after_df = df.filter(col(date_column) > max_date)
    
    return before_df, after_df


def combine_dataframes(dfs: list, subset_cols: list, target_col: str = "RATING") -> DataFrame:
    """
    Combine multiple DataFrames into one, keeping only the row with the max value for the target column for duplicates.
    
    Args:
        dfs (list): List of DataFrames to combine
        subset_cols (list): Columns to consider as duplicates (e.g., ["MOVIE_ID", "CUST_ID", "DATE"])
        target_col (str): Column to maximize (default: "RATING")
    Returns:
        DataFrame: Combined DataFrame with duplicates resolved by max target_col
    """
    if not dfs:
        raise ValueError("No DataFrames provided")
    
    from functools import reduce
    combined_df = reduce(lambda df1, df2: df1.unionByName(df2), dfs)

    window = Window.partitionBy(*subset_cols).orderBy(F.desc(target_col))
    ranked = combined_df.withColumn("_rank", F.row_number().over(window))
    result = ranked.filter(F.col("_rank") == 1).drop("_rank")

    return result


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
) -> None:
    """
    Train ALS model, extract user/item factors, and save as Parquet files.
    Args:
        train_df: Input DataFrame for ALS training
        user_col, item_col, rating_col: Column names
        rank, reg_param, max_iter, implicit_prefs, cold_start_strategy, nonnegative: ALS params
        user_factors_path, item_factors_path: Output Parquet paths
    """
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

    user_f = model.userFactors.withColumnRenamed("id", user_col)
    item_f = model.itemFactors.withColumnRenamed("id", item_col)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    user_f_path = model_save_path.rstrip("/") + f"/version_{timestamp}/user_factors"
    item_f_path = model_save_path.rstrip("/") + f"/version_{timestamp}/item_factors"

    user_f.write.mode("overwrite").parquet(user_f_path)
    item_f.write.mode("overwrite").parquet(item_f_path)


def write_dataframes_as_parquet(
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