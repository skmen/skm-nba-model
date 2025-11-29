"""
Hyperparameter Tuning Script for NBA Prediction Models

This script uses GridSearchCV to systematically search for the optimal
hyperparameters for both the XGBoost and Ridge models for ALL stats.

Usage:
    python tune_hyperparameters.py
    python tune_hyperparameters.py --player-name "LeBron James"
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
from src.config import DEFAULT_PLAYER_NAME, FEATURES, SEASONS_TO_FETCH, TARGETS
from src.utils import print_section, print_result

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def tune_all_models(player_name=DEFAULT_PLAYER_NAME):
    """
    Performs hyperparameter tuning for XGBoost and Ridge models for all stats.
    """
    print_section(f"STARTING HYPERPARAMETER TUNING (Player: {player_name})")

    # --- 1. Data Acquisition and Preparation (Done Once) ---
    print("Step 1: Acquiring and Engineering Features...")
    try:
        # Use existing functions to get the data
        game_log_df, opponent_defense, _, usage_rate = acquire_all_data(
            player_name=player_name,
            season=SEASONS_TO_FETCH[0], # Primary season
            use_multi_season=True
        )
        
        if game_log_df is None or game_log_df.empty:
            print_result(f"❌ Could not retrieve data for {player_name}. Aborting.", False)
            return
            
        # Engineer features
        base_engineered_df = engineer_features(game_log_df, opponent_defense, usage_rate)
        print_result(f"✅ Base data prepared with {len(base_engineered_df)} games", True)

    except Exception as e:
        print_result(f"❌ Failed during data preparation: {e}", False)
        return

    # --- 2. Define Hyperparameter Grids ---
    xgb_param_grid = {
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [200, 500],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
        'reg_lambda': [1, 1.5]
    }

    ridge_param_grid = {
        'alpha': [0.1, 1.0, 10.0, 50.0, 100.0],
        'solver': ['auto', 'svd', 'cholesky']
    }

    # TimeSeriesSplit is crucial for time-series data
    tscv = TimeSeriesSplit(n_splits=5)

    # --- 3. Loop Through Each Stat to Tune ---
    for stat_target in TARGETS:
        print_section(f"TUNING MODELS FOR '{stat_target}'")
        
        # Prepare data for this specific stat
        engineered_df = base_engineered_df.copy()
        engineered_df.dropna(subset=FEATURES + [stat_target], inplace=True)
        
        if engineered_df.empty:
            print_result(f"⚠️ No valid data for '{stat_target}', skipping.", False)
            continue

        X = engineered_df[FEATURES]
        y = engineered_df[stat_target]
        sample_weight = engineered_df['SAMPLE_WEIGHT']
        
        # --- Tune XGBoost ---
        print(f"\nTuning XGBoost for {stat_target}...")
        xgb = XGBRegressor(random_state=42)
        grid_search_xgb = GridSearchCV(
            estimator=xgb, param_grid=xgb_param_grid, cv=tscv, 
            scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0
        )
        grid_search_xgb.fit(X, y, sample_weight=sample_weight)
        
        print_result(f"✅ XGBoost for {stat_target} Complete!", True)
        print(f"  Best MAE: {-grid_search_xgb.best_score_:.3f}")
        print(f"  Best Params: {grid_search_xgb.best_params_}\n")

        # --- Tune Ridge ---
        print(f"Tuning Ridge for {stat_target}...")
        ridge = Ridge()
        grid_search_ridge = GridSearchCV(
            estimator=ridge, param_grid=ridge_param_grid, cv=tscv,
            scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0
        )
        grid_search_ridge.fit(X, y, sample_weight=sample_weight)

        print_result(f"✅ Ridge for {stat_target} Complete!", True)
        print(f"  Best MAE: {-grid_search_ridge.best_score_:.3f}")
        print(f"  Best Params: {grid_search_ridge.best_params_}")

    print_section("ALL TUNING COMPLETE")
    print("ACTION: To use these parameters, update the `XGBOOST_PARAMS` dictionary in `src/config.py` and the Ridge instantiation in `src/model.py`.")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning script for NBA prediction models.")
    parser.add_argument(
        '--player-name',
        type=str,
        default=DEFAULT_PLAYER_NAME,
        help="The player to use for data fetching (e.g., \"LeBron James\")."
    )
    args = parser.parse_args()
    tune_all_models(player_name=args.player_name)

if __name__ == "__main__":
    main()
