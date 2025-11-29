# Force re-cache
"""
Utility functions for NBA prediction pipeline.

Includes logging, error handling, and helper functions.
"""

import logging
import os
import sys
from typing import Optional

from src.config import LOG_LEVEL, LOG_FORMAT, DATA_DIR, LOGS_DIR

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logger(name: str = 'NBA_Model') -> logging.Logger:
    """
    Setup and return a configured logger instance.
    Writes to both Console and automation.log.
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        return logger

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # --- ENCODING FIX FOR WINDOWS ---
    if sys.platform == 'win32':
        # This line prevents the UnicodeEncodeError crash
        sys.stdout.reconfigure(encoding='utf-8')
    # --------------------------------

    # 1. Create Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)

    # 2. Create File Handler
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, 'automation.log')
    
    # Use 'utf-8' encoding for the file handler too
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# --- CRITICAL FIX: Instantiate the logger globally ---
# This allows other files to run 'from src.utils import logger'
logger = setup_logger()


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
# PRINTING & REPORTING (UPDATED TO USE LOGGER)
# ============================================================================

def print_section(title: str) -> None:
    """Print a formatted section header to log and console."""
    separator = "=" * 60
    logger.info(separator)
    logger.info(f"  {title}")
    logger.info(separator)


def print_result(label: str, value: any) -> None:
    """Print a formatted result line to log and console."""
    logger.info(f"{label:.<40} {value}")


def print_model_results(target_name: str, xgb_mae: float, naive_mae: float) -> None:
    """Print formatted model evaluation results."""
    print_section(f"MODEL EVALUATION ({target_name})")
    print_result(f"XGBoost MAE ({target_name})", f"{xgb_mae:.2f}")
    print_result(f"Naive Baseline MAE ({target_name})", f"{naive_mae:.2f}")

    if xgb_mae < naive_mae:
        improvement = ((naive_mae - xgb_mae) / naive_mae) * 100
        logger.info(f"\n✅ SUCCESS: Model beats baseline by {improvement:.1f}%")
    else:
        diff = ((xgb_mae - naive_mae) / naive_mae) * 100
        logger.info(f"\n❌ FAIL: Model underperforms baseline by {diff:.1f}%")


def print_prediction(predictions: dict) -> None:
    """Print formatted prediction."""
    print_section("PREDICTION FOR NEXT GAME")
    for target, value in predictions.items():
        logger.info(f"Projected {target}: {value:.1f}")
    print_section("")