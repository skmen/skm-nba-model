"""
Hyperparameter Tuning Script for NBA Prediction Models

This script uses GridSearchCV to systematically search for the optimal
hyperparameters for both the XGBoost and Ridge models.

Usage:
    python tune_hyperparameters.py --stat PTS
    python tune_hyperparameters.py --stat REB --player "LeBron James"
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import sys
import warnings

# Add parent directory to path for local imports
sys.path.insert(0, sys.path[0] + '/src')

from src.data_fetcher import acquire_all_data
from src.feature_engineer import engineer_features
from src.config import DEFAULT_PLAYER_NAME, FEATURES, SEASONS_TO_FETCH, SEASON_WEIGHTS
from src.utils import print_section, print_result

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def tune_models(stat_target='PTS', player_name=DEFAULT_PLAYER_NAME):
    """
    Performs hyperparameter tuning for XGBoost and Ridge models for a given stat.
    """
    print_section(f"TUNING MODELS FOR '{stat_target}' (Player: {player_name})")

    # --- 1. Data Acquisition and Preparation ---
    print("Step 1: Acquiring and Engineering Features...")
    try:
        # Use existing functions to get the data
        game_log_df, opponent_defense, _, usage_rate = acquire_all_data(
            player_name=player_name,
            season=SEASONS_TO_FETCH[0], # Primary season
            use_multi_season=True,
            seasons_to_fetch=SEASONS_TO_FETCH
        )
        
        if game_log_df is None or game_log_df.empty:
            print_result(f"❌ Could not retrieve data for {player_name}. Aborting.", False)
            return
            
        # Engineer features
        engineered_df = engineer_features(game_log_df, opponent_defense, usage_rate)
        engineered_df.dropna(subset=FEATURES + [stat_target], inplace=True)
        
        if engineered_df.empty:
            print_result(f"❌ No valid data after feature engineering for {player_name}. Aborting.", False)
            return

        X = engineered_df[FEATURES]
        y = engineered_df[stat_target]
        sample_weight = engineered_df['SAMPLE_WEIGHT']
        
        print_result(f"✅ Data prepared: {len(engineered_df)} games", True)

    except Exception as e:
        print_result(f"❌ Failed during data preparation: {e}", False)
        return

    # --- 2. Define Hyperparameter Grids ---
    # Define a smaller, more focused grid for faster tuning
    # A wider grid might yield better results but take significantly longer
    xgb_param_grid = {
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [200, 500], # Reduced for speed
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
        'reg_lambda': [1, 1.5] # L2 regularization
    }

    ridge_param_grid = {
        'alpha': [0.1, 1.0, 10.0, 50.0, 100.0],
        'solver': ['auto', 'svd', 'cholesky']
    }

    # TimeSeriesSplit is crucial for time-series data to prevent data leakage
    tscv = TimeSeriesSplit(n_splits=5)

    # --- 3. Tune XGBoost ---
    print("\nStep 2: Tuning XGBoost Model (this may take several minutes)...")
    xgb = XGBRegressor(random_state=42)
    grid_search_xgb = GridSearchCV(
        estimator=xgb, 
        param_grid=xgb_param_grid, 
        cv=tscv, 
        scoring='neg_mean_absolute_error', 
        n_jobs=-1, # Use all available CPU cores
        verbose=1
    )
    # Use sample_weight during the fit process
    grid_search_xgb.fit(X, y, sample_weight=sample_weight)
    
    print_result("✅ XGBoost Tuning Complete!", True)
    print("Best Parameters Found:")
    print(grid_search_xgb.best_params_)
    print(f"Best MAE: {-grid_search_xgb.best_score_:.3f}")

    # --- 4. Tune Ridge ---
    print("\nStep 3: Tuning Ridge Model...")
    ridge = Ridge()
    grid_search_ridge = GridSearchCV(
        estimator=ridge,
        param_grid=ridge_param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search_ridge.fit(X, y, sample_weight=sample_weight)

    print_result("✅ Ridge Tuning Complete!", True)
    print("Best Parameters Found:")
    print(grid_search_ridge.best_params_)
    print(f"Best MAE: {-grid_search_ridge.best_score_:.3f}")

    print_section("TUNING COMPLETE")
    print("ACTION: To use these parameters, update the `XGBOOST_PARAMS` dictionary in `src/config.py` and the Ridge instantiation in `src/model.py`.")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning script for NBA prediction models.")
    parser.add_argument(
        '--stat', 
        type=str, 
        default='PTS', 
        choices=['PTS', 'REB', 'AST', 'STL', 'BLK', 'PRA'],
        help="The statistic to tune the models for (e.g., PTS, REB)."
    )
    parser.add_argument(
        '--player',
        type=str,
        default=DEFAULT_PLAYER_NAME,
        help="The player to use for data fetching."
    )
    args = parser.parse_args()
    tune_models(stat_target=args.stat, player_name=args.player)

if __name__ == "__main__":
    main()
