"""
Helper functions for dynamic model version resolution.
Use these in PythonOperator tasks or XCom for passing model versions between tasks.
"""
import os
from typing import Optional
from pathlib import Path


def get_latest_model_version_from_path(base_path: str = "/workspace/models/artifacts") -> str:
    """
    Dynamically find the latest model version by scanning the artifacts directory.
    Assumes version directories follow the pattern: version_YYYYMMDD_HHMMSS
    
    Args:
        base_path: Base path to the model artifacts directory
    
    Returns:
        Full path to the latest model version directory
        
    Raises:
        ValueError: If no model versions are found
    """
    base_dir = Path(base_path)
    
    if not base_dir.exists():
        raise ValueError(f"Model artifacts directory does not exist: {base_path}")
    
    # Find all version directories
    version_dirs = [
        d for d in base_dir.iterdir() 
        if d.is_dir() and d.name.startswith("version_")
    ]
    
    if not version_dirs:
        raise ValueError(f"No model versions found in {base_path}")
    
    # Sort by directory name (which contains timestamp)
    # Latest version will be last when sorted
    latest_version = sorted(version_dirs, key=lambda x: x.name)[-1]
    
    return str(latest_version)


def extract_version_timestamp(version_path: str) -> str:
    """
    Extract the timestamp from a version path.
    
    Args:
        version_path: Path like /workspace/models/artifacts/version_20250930_212327
    
    Returns:
        Version timestamp like "20250930_212327"
    """
    version_name = Path(version_path).name
    if version_name.startswith("version_"):
        return version_name.replace("version_", "")
    return version_name


def validate_model_version(version_path: str) -> bool:
    """
    Validate that a model version directory exists and contains expected files.
    
    Args:
        version_path: Full path to the model version directory
    
    Returns:
        True if valid, False otherwise
    """
    version_dir = Path(version_path)
    
    # Check if directory exists
    if not version_dir.exists() or not version_dir.is_dir():
        return False
    
    # Check for expected subdirectories (for ALS model)
    expected_dirs = ["userFactors", "itemFactors"]
    alternative_dirs = ["user_factors", "item_factors"]
    
    has_expected = all((version_dir / d).exists() for d in expected_dirs)
    has_alternative = all((version_dir / d).exists() for d in alternative_dirs)
    
    return has_expected or has_alternative


# Example usage in Airflow PythonOperator:
"""
from airflow.operators.python import PythonOperator

def push_latest_model_version(**context):
    '''Push the latest model version to XCom'''
    from common.model_utils import get_latest_model_version_from_path
    latest_version = get_latest_model_version_from_path()
    return latest_version

find_model_task = PythonOperator(
    task_id='find_latest_model',
    python_callable=push_latest_model_version,
    dag=dag,
)

# Then in a downstream task:
def use_model_version(**context):
    '''Retrieve model version from XCom'''
    ti = context['ti']
    model_version = ti.xcom_pull(task_ids='find_latest_model')
    print(f"Using model version: {model_version}")
    return model_version
"""
