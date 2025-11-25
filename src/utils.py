"""
Utility functions for NBA prediction pipeline.

Includes logging, error handling, and helper functions.
"""

import logging
import os
from typing import Optional

from config import LOG_LEVEL, LOG_FORMAT, DATA_DIR

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logger(name: str) -> logging.Logger:
    """
    Setup and return a configured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(LOG_LEVEL)

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)

    # Add handler to logger
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


logger = setup_logger(__name__)

# ============================================================================
# FILE PATH UTILITIES
# ============================================================================

def get_data_filepath(filename: str) -> str:
    """
    Get full path to a data file.

    Args:
        filename: Name of the file (e.g., 'my_data.csv')

    Returns:
        Full path to the file in the data directory
    """
    return os.path.join(DATA_DIR, filename)


def ensure_data_dir_exists() -> None:
    """Ensure the data directory exists."""
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.debug(f"Data directory ensured: {DATA_DIR}")


# ============================================================================
# ERROR HANDLING & VALIDATION
# ============================================================================

class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class DataAcquisitionError(PipelineError):
    """Raised when data acquisition fails."""
    pass


class FeatureEngineeringError(PipelineError):
    """Raised when feature engineering fails."""
    pass


class ModelTrainingError(PipelineError):
    """Raised when model training fails."""
    pass


def validate_dataframe(df, required_columns: Optional[list] = None) -> bool:
    """
    Validate a dataframe.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if valid

    Raises:
        ValueError: If dataframe is invalid
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")

    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    return True


# ============================================================================
# STRING UTILITIES
# ============================================================================

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing/replacing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Remove special characters
    filename = ''.join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))
    return filename


# ============================================================================
# PRINTING & REPORTING
# ============================================================================

def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(label: str, value: any) -> None:
    """Print a formatted result line."""
    print(f"{label:.<40} {value}")


def print_model_results(xgb_mae: float, naive_mae: float) -> None:
    """Print formatted model evaluation results."""
    print_section("MODEL EVALUATION")
    print_result("XGBoost MAE", f"{xgb_mae:.2f} points")
    print_result("Naive Baseline MAE", f"{naive_mae:.2f} points")

    if xgb_mae < naive_mae:
        improvement = ((naive_mae - xgb_mae) / naive_mae) * 100
        print(f"\n✅ SUCCESS: Model beats baseline by {improvement:.1f}%")
    else:
        diff = ((xgb_mae - naive_mae) / naive_mae) * 100
        print(f"\n❌ FAIL: Model underperforms baseline by {diff:.1f}%")


def print_prediction(predicted_score: float) -> None:
    """Print formatted prediction."""
    print_section("PREDICTION FOR NEXT GAME")
    print(f"Projected Points: {predicted_score:.1f}")
    print_section("")
