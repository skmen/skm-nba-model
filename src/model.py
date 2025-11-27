"""
Model training and evaluation module for NBA prediction pipeline.

Handles XGBoost and Ridge model training (Tournament Style), evaluation, and prediction.
"""

import logging
from typing import Tuple, Optional, Union

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    XGBOOST_PARAMS,
    RIDGE_PARAMS,
    TRAIN_TEST_RATIO,
    PLOT_FIGSIZE,
    PLOT_ALPHA,
)
from .utils import ModelTrainingError, logger

# ============================================================================
# MODEL TRAINING (THE TOURNAMENT)
# ============================================================================

def train_model(
    df: pd.DataFrame,
    features: list,
    target: str,
) -> Tuple[object, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Trains both XGBoost and Ridge Regression, compares them, and returns the winner.
    """
    try:
        logger.info("=" * 60)
        logger.info(f"TRAINING MODEL TOURNAMENT ({target})")
        logger.info("=" * 60)

        # Validate inputs
        if df is None or df.empty:
            raise ValueError("DataFrame is empty")

        # Extract features and target
        X = df[features]
        y = df[target]

        # Temporal train/test split
        split_index = int(len(df) * TRAIN_TEST_RATIO)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        logger.info(f"Training on {len(X_train)} games, testing on {len(X_test)} games")

        # Extract weights
        train_weights = None
        if 'SAMPLE_WEIGHT' in df.columns:
            train_weights = df['SAMPLE_WEIGHT'].iloc[:split_index].values
            # Log weight stats
            w_min, w_max = train_weights.min(), train_weights.max()
            logger.info(f"Sample weights applied: min={w_min:.2f}, max={w_max:.2f}")

        # --- CONTESTANT 1: RIDGE REGRESSION ---
        # Pipeline: Scale Data -> Ridge Regression
        ridge_model = make_pipeline(StandardScaler(), Ridge(**RIDGE_PARAMS))
        # Note: 'ridge__sample_weight' passes weights to the 'ridge' step
        ridge_model.fit(X_train, y_train, ridge__sample_weight=train_weights)
        
        ridge_pred = ridge_model.predict(X_test)
        ridge_mae = mean_absolute_error(y_test, ridge_pred)

        # --- CONTESTANT 2: XGBOOST ---
        xgb_model = xgb.XGBRegressor(**XGBOOST_PARAMS)
        xgb_model.fit(
            X_train, y_train,
            sample_weight=train_weights,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        xgb_pred = xgb_model.predict(X_test)
        xgb_mae = mean_absolute_error(y_test, xgb_pred)

        # --- THE DECISION ---
        logger.info(f"Ridge MAE: {ridge_mae:.2f} | XGBoost MAE: {xgb_mae:.2f}")

        if ridge_mae < xgb_mae:
            diff = xgb_mae - ridge_mae
            logger.info(f"🏆 WINNER: Ridge Regression (Better by {diff:.2f})")
            return ridge_model, X_train, X_test, y_train, y_test
        else:
            diff = ridge_mae - xgb_mae
            logger.info(f"🏆 WINNER: XGBoost (Better by {diff:.2f})")
            return xgb_model, X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        # Re-raise so pipeline knows to stop for this target
        raise ModelTrainingError(f"Training failed: {e}")


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target: str,
) -> Tuple[float, float, bool]:
    """
    Evaluate model performance using Mean Absolute Error (MAE).
    """
    try:
        logger.info("=" * 60)
        logger.info(f"EVALUATING MODEL ({target})")
        logger.info("=" * 60)

        # Get predictions
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)

        # Calculate naive baseline (using 5-game average)
        naive_predictions = X_test[f'{target}_L5']
        naive_mae = mean_absolute_error(y_test, naive_predictions)

        success = mae < naive_mae

        logger.info(f"Model MAE ({target}): {mae:.2f}")
        logger.info(f"Naive MAE ({target}): {naive_mae:.2f}")

        if success:
            improvement = ((naive_mae - mae) / naive_mae) * 100
            logger.info(f"✅ Model beats baseline by {improvement:.1f}%")
        else:
            underperformance = ((mae - naive_mae) / naive_mae) * 100
            logger.warning(f"❌ Model underperforms baseline by {underperformance:.1f}%")

        return mae, naive_mae, success

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise ModelTrainingError(f"Model evaluation failed: {e}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_model_results(
    y_test: pd.Series,
    predictions: np.ndarray,
    target: str,
    figsize: Tuple[int, int] = PLOT_FIGSIZE,
) -> None:
    """Plot actual vs. predicted values."""
    try:
        # Don't block execution if plotting fails
        plt.figure(figsize=figsize)
        plt.scatter(y_test, predictions, alpha=PLOT_ALPHA, label='Predictions')
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--', lw=2, label='Perfect Prediction'
        )
        plt.xlabel(f'Actual {target}')
        plt.ylabel(f'Predicted {target}')
        plt.title(f'Model Results: Actual vs. Predicted {target}')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        logger.warning(f"Visualization skipped: {e}")


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def get_feature_importance(
    model: object,
    features: list,
    top_n: int = 10,
) -> pd.DataFrame:
    """Get feature importance (Coefficients for Ridge, Gain for XGBoost)."""
    try:
        logger.debug("Extracting feature importance...")
        importance_data = []

        # Case 1: Ridge Pipeline
        if isinstance(model, Pipeline):
            # Extract coefficients from the 'ridge' step
            ridge_step = model.named_steps['ridge']
            coefs = np.abs(ridge_step.coef_) # Use absolute value
            importance_data = [
                {'feature': feat, 'importance': coef}
                for feat, coef in zip(features, coefs)
            ]
        
        # Case 2: XGBoost
        elif isinstance(model, xgb.XGBRegressor):
            importance_dict = model.get_booster().get_score(importance_type='weight')
            importance_data = [
                {'feature': feat, 'importance': importance_dict.get(feat, 0)}
                for feat in features
            ]

        # Create DataFrame
        importance_df = pd.DataFrame(importance_data)
        
        if not importance_df.empty:
            importance_df = importance_df.sort_values('importance', ascending=False)
            return importance_df.head(top_n)
        
        return pd.DataFrame()

    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
        return pd.DataFrame()


# ============================================================================
# PREDICTION
# ============================================================================

def predict_next_game(
    model: object,
    last_game_features: dict,
    features: list,
) -> float:
    """Make a prediction for the next game."""
    try:
        logger.debug("Making prediction for next game...")
        
        # Create DataFrame to ensure feature order matches training
        future_data = pd.DataFrame([last_game_features])
        future_data = future_data[features]

        prediction = model.predict(future_data)
        
        # Handle different return types (Scalar vs Array)
        if isinstance(prediction, (np.ndarray, list)):
            predicted_score = float(prediction[0])
        else:
            predicted_score = float(prediction)

        return predicted_score

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise ModelTrainingError(f"Failed to make prediction: {e}")


def prepare_prediction_data(
    engineered_df: pd.DataFrame,
    next_game_is_home: int = 1,
) -> dict:
    """Prepare feature data for next game prediction."""
    last_game = engineered_df.iloc[-1]
    future_data = {}

    # Lag features - ADDED 'PRA' TO THIS LIST
    stats_to_lag = ['PTS', 'MIN', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'PRA']
    
    for stat in stats_to_lag:
        # Check if the feature exists before trying to access it
        feature_name = f'{stat}_L5'
        if feature_name in last_game:
            future_data[feature_name] = last_game[feature_name]

    # Context
    future_data['HOME_GAME'] = next_game_is_home
    future_data['OPP_DEF_RATING'] = last_game['OPP_DEF_RATING']
    future_data['OPP_PACE'] = last_game['OPP_PACE']
    future_data['USAGE_RATE'] = last_game['USAGE_RATE']
    # Efficiency Features
    safe_min = last_game['MIN_L5'] if last_game['MIN_L5'] > 0 else 1
    future_data['PTS_PER_MIN'] = last_game['PTS_L5'] / safe_min
    future_data['REB_PER_MIN'] = last_game['REB_L5'] / safe_min
    
    if 'PRA_L5' in last_game:
        future_data['PRA_PER_MIN'] = last_game['PRA_L5'] / safe_min
    
    return future_data