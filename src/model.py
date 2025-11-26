"""
Model training and evaluation module for NBA prediction pipeline.

Handles XGBoost model training, evaluation, and prediction.
"""

import logging
from typing import Tuple, Optional

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from .config import (
    XGBOOST_PARAMS,
    TRAIN_TEST_RATIO,
    PLOT_FIGSIZE,
    PLOT_ALPHA,
)
from .utils import ModelTrainingError, logger

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(
    df: pd.DataFrame,
    features: list,
    target: str,
) -> Tuple[xgb.XGBRegressor, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train an XGBoost regression model with sample weighting.

    Performs an 80/20 temporal train/test split (respecting time series structure)
    and applies sample weights based on season recency (data decay).

    Args:
        df: DataFrame with engineered features (must include SAMPLE_WEIGHT column)
        features: List of feature column names
        target: Name of target column (e.g., 'PTS')

    Returns:
        Tuple of (trained_model, X_train, X_test, y_train, y_test)

    Raises:
        ModelTrainingError: If training fails
    """
    try:
        logger.info("=" * 60)
        logger.info("TRAINING MODEL")
        logger.info("=" * 60)

        # Validate inputs
        if df is None or df.empty:
            raise ValueError("DataFrame is empty")

        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        if target not in df.columns:
            raise ValueError(f"Target '{target}' not in DataFrame")

        # Extract features and target
        X = df[features]
        y = df[target]

        logger.debug(f"Features shape: {X.shape}")
        logger.debug(f"Target shape: {y.shape}")

        # Temporal train/test split (preserving time series order)
        split_index = int(len(df) * TRAIN_TEST_RATIO)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        logger.info(f"Training on {len(X_train)} games, "
                    f"testing on {len(X_test)} games")

        # Extract sample weights if available (for data decay)
        sample_weights = None
        if 'SAMPLE_WEIGHT' in df.columns:
            sample_weights = df['SAMPLE_WEIGHT'].iloc[:split_index].values
            weight_stats = f"min={sample_weights.min():.2f}, " \
                          f"mean={sample_weights.mean():.2f}, " \
                          f"max={sample_weights.max():.2f}"
            logger.info(f"Sample weights applied (data decay): {weight_stats}")
        else:
            logger.debug("No SAMPLE_WEIGHT column found; using uniform weights")

        # Create and train model
        model = xgb.XGBRegressor(**XGBOOST_PARAMS)

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        logger.info("Model training complete!")

        return model, X_train, X_test, y_train, y_test

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise ModelTrainingError(f"Invalid model input: {e}")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise ModelTrainingError(f"Training failed: {e}")


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target: str,
) -> Tuple[float, float, bool]:
    """
    Evaluate model performance using Mean Absolute Error (MAE).

    Compares XGBoost model to naive baseline (using previous 5-game average).

    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test target values
        target: The name of the target variable being evaluated

    Returns:
        Tuple of (xgb_mae, naive_mae, success_flag)
    """
    try:
        logger.info("=" * 60)
        logger.info(f"EVALUATING MODEL ({target})")
        logger.info("=" * 60)

        # Get predictions
        predictions = model.predict(X_test)
        xgb_mae = mean_absolute_error(y_test, predictions)

        # Calculate naive baseline (using 5-game average)
        naive_predictions = X_test[f'{target}_L5']
        naive_mae = mean_absolute_error(y_test, naive_predictions)

        # Compare
        success = xgb_mae < naive_mae

        logger.info(f"XGBoost MAE ({target}): {xgb_mae:.2f}")
        logger.info(f"Naive MAE ({target}):   {naive_mae:.2f}")

        if success:
            improvement = ((naive_mae - xgb_mae) / naive_mae) * 100
            logger.info(f"✅ Model beats baseline by {improvement:.1f}%")
        else:
            underperformance = ((xgb_mae - naive_mae) / naive_mae) * 100
            logger.warning(f"❌ Model underperforms baseline by {underperformance:.1f}%")

        return xgb_mae, naive_mae, success

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise ModelTrainingError(f"Model evaluation failed: {e}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_model_results(
    y_test: pd.Series,
    predictions: np.ndarray,
    figsize: Tuple[int, int] = PLOT_FIGSIZE,
) -> None:
    """
    Plot actual vs. predicted values.

    Args:
        y_test: Actual test values
        predictions: Model predictions
        figsize: Figure size (width, height)
    """
    try:
        logger.debug("Creating results visualization...")

        plt.figure(figsize=figsize)
        plt.scatter(y_test, predictions, alpha=PLOT_ALPHA, label='XGBoost Predictions')
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--',
            lw=2,
            label='Perfect Prediction'
        )
        plt.xlabel('Actual Points')
        plt.ylabel('Predicted Points')
        plt.title('XGBoost Model: Actual vs. Predicted Points')
        plt.legend()
        plt.grid(True)
        plt.show()

        logger.debug("Visualization complete")

    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        # Don't raise - visualization is optional


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def get_feature_importance(
    model: xgb.XGBRegressor,
    features: list,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Get feature importance from trained model.

    Args:
        model: Trained XGBoost model
        features: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with features and their importance scores
    """
    try:
        logger.debug("Extracting feature importance...")

        importance_dict = model.get_booster().get_score(importance_type='weight')

        # Create dataframe
        importance_df = pd.DataFrame([
            {'feature': feat, 'importance': importance_dict.get(feat, 0)}
            for feat in features
        ])

        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)

        # Return top N
        return importance_df.head(top_n)

    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        return pd.DataFrame()


# ============================================================================
# PREDICTION
# ============================================================================

def predict_next_game(
    model: xgb.XGBRegressor,
    last_game_features: dict,
    features: list,
) -> float:
    """
    Make a prediction for the next game.

    Args:
        model: Trained model
        last_game_features: Dictionary of features for next game
        features: List of feature names (for column ordering)

    Returns:
        Predicted score (float)

    Raises:
        ModelTrainingError: If prediction fails
    """
    try:
        logger.debug("Making prediction for next game...")

        # Create DataFrame
        future_data = pd.DataFrame([last_game_features])

        # Ensure correct column order
        future_data = future_data[features]

        # Make prediction
        predicted_score = model.predict(future_data)[0]

        logger.debug(f"Predicted score: {predicted_score:.1f}")

        return predicted_score

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise ModelTrainingError(f"Failed to make prediction: {e}")


def prepare_prediction_data(
    engineered_df: pd.DataFrame,
    next_game_is_home: int = 1,
) -> dict:
    """
    Prepare feature data for next game prediction.

    Uses last game's statistics as baseline for next game.

    Args:
        engineered_df: DataFrame with engineered features
        next_game_is_home: 1 for home, 0 for away (default: 1)

    Returns:
        Dictionary with all features for prediction
    """
    logger.debug("Preparing prediction data...")

    last_game = engineered_df.iloc[-1]

    future_data = {}

    # Lag features
    for stat in ['PTS', 'MIN', 'REB', 'AST', 'STL', 'BLK', 'FG3M']:
        future_data[f'{stat}_L5'] = last_game[f'{stat}_L5']

    # Home/away
    future_data['HOME_GAME'] = next_game_is_home

    # Opponent context
    future_data['OPP_DEF_RATING'] = last_game['OPP_DEF_RATING']
    future_data['OPP_PACE'] = last_game['OPP_PACE']

    # Travel and rest
    future_data['TRAVEL_DISTANCE'] = last_game['TRAVEL_DISTANCE']
    future_data['DAYS_REST'] = 1  # Assume 1 day rest
    future_data['BACK_TO_BACK'] = 0

    # Usage rate
    future_data['USAGE_RATE'] = last_game['USAGE_RATE']

    return future_data
