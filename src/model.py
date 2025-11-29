"""
Model training and evaluation module for NBA prediction pipeline.

Updates: Implements Level 3 Stacking Architecture (XGB + RF -> Bayesian Ridge).
"""

import logging
import os
import joblib
from typing import Tuple, Optional, Union, List

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# --- LEVEL 3 IMPORTS ---
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from src.config import (
    TRAIN_TEST_RATIO,
    PLOT_FIGSIZE,
    PLOT_ALPHA,
    MODELS_DIR
)
from src.utils import ModelTrainingError, logger

# ============================================================================
# LEVEL 3: STACKED MODEL CLASS
# ============================================================================

class NBAPlayerStack:
    """
    The 'Golden Trio' Stacked Model.
    Combines XGBoost (Specialist) and Random Forest (Generalist)
    using Bayesian Ridge (The Judge) as the meta-learner.
    """
    def __init__(self, name: str = "Stacked_Model"):
        self.name = name
        self.model = self._build_model()
        
    def _build_model(self):
        # 1. Base Learner A: XGBoost (Captures non-linear, sharp trends)
        xgb_reg = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            objective='reg:absoluteerror',
            n_jobs=-1,
            random_state=42
        )
        
        # 2. Base Learner B: Random Forest (Captures general stability, handles noise)
        rf_reg = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            n_jobs=-1,
            random_state=42
        )
        
        # 3. Meta Learner: Bayesian Ridge (Weighs the two inputs probabilistically)
        # Using BayesianRidge prevents overfitting in the stacking layer
        meta_learner = BayesianRidge()
        
        # Define the Stack
        estimators = [
            ('xgb', xgb_reg),
            ('rf', rf_reg)
        ]
        
        stack = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5, # 5-fold CV to generate meta-features safely
            n_jobs=-1,
            passthrough=False # Meta-learner only sees predictions, not raw features
        )
        
        return stack

    def fit(self, X, y, sample_weight=None):
        # Note: StackingRegressor's fit supports sample_weight but applying it 
        # to internal CV splits is complex. For stability, we fit without explicit 
        # sample weights in the meta-layer for now, or pass them if supported.
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filename: str):
        """Save the trained stack to the models directory."""
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        path = os.path.join(MODELS_DIR, filename)
        joblib.dump(self.model, path)
        logger.info(f"Saved model to {path}")

    def load(self, filename: str):
        """Load a pretrained stack."""
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            self.model = joblib.load(path)
            logger.info(f"Loaded model from {path}")
        else:
            raise FileNotFoundError(f"Model {filename} not found")


# ============================================================================
# TRAIN MODEL (Backward Compatible Wrapper)
# ============================================================================

def train_model(
    df: pd.DataFrame,
    features: list,
    target: str,
) -> Tuple[object, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Trains the Level 3 Stacked Model.
    
    Retains the original signature so existing scripts don't break.
    Instead of a tournament, it returns the Stacked Model.
    """
    try:
        logger.info("=" * 60)
        logger.info(f"TRAINING LEVEL 3 STACK ({target})")
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

        # Extract weights (Optional)
        train_weights = None
        if 'SAMPLE_WEIGHT' in df.columns:
            train_weights = df['SAMPLE_WEIGHT'].iloc[:split_index].values
        
        # --- INITIALIZE AND TRAIN STACK ---
        # We replace the Ridge vs XGB tournament with the single Stack
        stack = NBAPlayerStack(name=f"Stack_{target}")
        
        # Train the model
        stack.fit(X_train, y_train, sample_weight=train_weights)
        
        # Generate predictions for evaluation log
        preds = stack.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        
        # Log results (Safe formatting, no emojis to prevent Windows crash)
        logger.info(f"[RESULT] Stacked Model MAE: {mae:.2f}")

        # Return tuple matching original signature
        return stack, X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Model training failed: {e}")
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
        # Handle case where L5 column might be missing in older datasets
        naive_col = f'{target}_L5'
        if naive_col in X_test.columns:
            naive_predictions = X_test[naive_col]
            naive_mae = mean_absolute_error(y_test, naive_predictions)
        else:
            logger.warning(f"Naive baseline column {naive_col} missing. Using global mean.")
            naive_mae = mean_absolute_error(y_test, [y_test.mean()]*len(y_test))

        success = mae < naive_mae

        logger.info(f"Model MAE ({target}): {mae:.2f}")
        logger.info(f"Naive MAE ({target}): {naive_mae:.2f}")

        if success:
            improvement = ((naive_mae - mae) / naive_mae) * 100
            logger.info(f"[SUCCESS] Model beats baseline by {improvement:.1f}%")
        else:
            underperformance = ((mae - naive_mae) / naive_mae) * 100
            logger.info(f"[FAIL] Model underperforms baseline by {underperformance:.1f}%")

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
        # plt.show() # Commented out for automation safety
        plt.close() # Close figure to free memory

    except Exception as e:
        logger.warning(f"Visualization skipped: {e}")


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def get_feature_importance(
    model_obj: object,
    features: list,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Get feature importance.
    Note: StackingRegressor doesn't have a direct 'feature_importance_'.
    We typically inspect the base learners or the meta-learner coefficients.
    """
    try:
        logger.debug("Extracting feature importance...")
        
        # Unwrap if it's our NBAPlayerStack class
        if hasattr(model_obj, 'model'):
            model = model_obj.model
        else:
            model = model_obj

        # If it's the StackingRegressor
        if isinstance(model, StackingRegressor):
            # We can return the coefficients of the Meta-Learner (BayesianRidge)
            # This tells us how much the stack trusts XGB vs RF
            meta = model.final_estimator_
            if hasattr(meta, 'coef_'):
                # Coefs correspond to the estimators: [XGB_Weight, RF_Weight]
                importance_data = [
                    {'feature': 'XGBoost_Influence', 'importance': abs(meta.coef_[0])},
                    {'feature': 'RandomForest_Influence', 'importance': abs(meta.coef_[1])}
                ]
                return pd.DataFrame(importance_data)

        # Fallback for base learners if accessed directly
        if hasattr(model, 'feature_importances_'):
             importance_data = [
                {'feature': feat, 'importance': imp}
                for feat, imp in zip(features, model.feature_importances_)
            ]
             df = pd.DataFrame(importance_data)
             return df.sort_values('importance', ascending=False).head(top_n)

        return pd.DataFrame()

    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
        return pd.DataFrame()


# ============================================================================
# PREDICTION
# ============================================================================

def predict_next_game(
    model_obj: object,
    last_game_features: dict,
    features: list,
) -> float:
    """Make a prediction for the next game."""
    try:
        logger.debug("Making prediction for next game...")
        
        # Create DataFrame to ensure feature order matches training
        future_data = pd.DataFrame([last_game_features])
        
        # Ensure all columns exist (fill missing with 0)
        for f in features:
            if f not in future_data.columns:
                future_data[f] = 0
                
        future_data = future_data[features]

        prediction = model_obj.predict(future_data)
        
        # Handle different return types
        if isinstance(prediction, (np.ndarray, list)):
            predicted_score = float(prediction[0])
        else:
            predicted_score = float(prediction)

        return predicted_score

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        # Return a safe default (e.g., L5 average) if model fails
        if 'PTS_L5' in last_game_features:
             logger.warning("Falling back to PTS_L5 due to error")
             return float(last_game_features['PTS_L5'])
        raise ModelTrainingError(f"Failed to make prediction: {e}")


def prepare_prediction_data(
    engineered_df: pd.DataFrame,
    next_game_is_home: int = 1,
) -> dict:
    """Prepare feature data for next game prediction."""
    last_game = engineered_df.iloc[-1]
    future_data = {}

    # Lag features
    stats_to_lag = ['PTS', 'MIN', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'PRA']
    
    for stat in stats_to_lag:
        feature_name = f'{stat}_L5'
        if feature_name in last_game:
            future_data[feature_name] = last_game[feature_name]

    # Context
    future_data['HOME_GAME'] = next_game_is_home
    
    # Safe access for opponent metrics
    future_data['OPP_DEF_RATING'] = last_game.get('OPP_DEF_RATING', 110.0)
    future_data['OPP_PACE'] = last_game.get('OPP_PACE', 100.0)
    future_data['USAGE_RATE'] = last_game.get('USAGE_RATE', 20.0)
    future_data['DVP_MULTIPLIER'] = last_game.get('DVP_MULTIPLIER', 1.0)
    future_data['DAYS_REST'] = last_game.get('DAYS_REST', 1)
    future_data['TRAVEL_DISTANCE'] = last_game.get('TRAVEL_DISTANCE', 0)

    # Efficiency Features
    safe_min = last_game['MIN_L5'] if last_game.get('MIN_L5', 0) > 0 else 1
    future_data['PTS_PER_MIN'] = last_game.get('PTS_L5', 0) / safe_min
    future_data['REB_PER_MIN'] = last_game.get('REB_L5', 0) / safe_min
    
    if 'PRA_L5' in last_game:
        future_data['PRA_PER_MIN'] = last_game['PRA_L5'] / safe_min
    
    return future_data