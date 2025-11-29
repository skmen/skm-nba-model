"""
NBA Player Performance Prediction Pipeline

Main orchestration module that coordinates data fetching, feature engineering,
model training, and prediction.

Updates:
- Implements Level 4 Simulation Logic (Rate -> Total)
- Ensures compatibility with bet_analyzer.py by saving clean keys ('PTS', 'REB')
- Converts Model MAE (Rate) to Projected MAE (Total) for accurate betting edges
"""

import logging
import json
import pandas as pd
from typing import Optional

from src.config import (
    DEFAULT_PLAYER_NAME,
    DEFAULT_SEASON,
    FEATURES,
    TARGETS,
    POSSESSION_BASE,
)
from src.utils import (
    setup_logger,
    print_section,
    print_model_results,
    print_prediction,
)
from src.data_fetcher import acquire_all_data, get_advanced_stats_cache
from src.feature_engineer import engineer_features
from src.model import (
    train_model,
    evaluate_model,
    plot_model_results,
    predict_next_game,
    prepare_prediction_data,
    get_feature_importance,
)

# Setup logger
logger = setup_logger(__name__)


# ============================================================================
# HELPER: POSITION INFERENCE
# ============================================================================

def infer_position_group(df: pd.DataFrame) -> str:
    """
    Simple heuristic to guess position group (G/F/C) from stats 
    if we don't have the roster data.
    """
    if df.empty: return 'G'
    
    avg_ast = df['AST'].mean()
    avg_reb = df['REB'].mean()
    
    if avg_ast > 5.0: return 'G'
    if avg_reb > 8.0: return 'C'
    return 'F'

# ============================================================================
# MAIN PIPELINE ORCHESTRATION
# ============================================================================

def run_prediction_pipeline(
    player_name: str = DEFAULT_PLAYER_NAME,
    season: str = DEFAULT_SEASON,
) -> None:
    """
    Run the complete NBA prediction pipeline.
    """
    try:
        logger.info("\n" + "=" * 60)
        logger.info(f"NBA PREDICTION PIPELINE: {player_name.upper()}")
        logger.info("=" * 60)

        # ====================================================================
        # STEP 1: ACQUIRE DATA
        # ====================================================================
        try:
            game_log_df, opponent_defense, player_id, usage_rate = (
                acquire_all_data(player_name, season)
            )
            
            # Fetch Advanced Stats Cache (Level 4 Requirement)
            advanced_stats = get_advanced_stats_cache(season)
            
        except Exception as e:
            logger.error(f"Data acquisition failed: {e}")
            raise

        # ====================================================================
        # STEP 2: ENGINEER FEATURES
        # ====================================================================
        try:
            # Infer Position for DvP Context
            pos_group = infer_position_group(game_log_df)
            logger.info(f"Inferred Position Group: {pos_group}")

            engineered_df = engineer_features(
                game_log_df,
                opponent_defense,
                usage_rate,
                position_group=pos_group,
                advanced_stats=advanced_stats
            )
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise

        all_predictions = {}
        mae_scores = {}  # Store MAEs for bet_analyzer
        simulation_context = {}

        # ====================================================================
        # PREPARE SIMULATION CONTEXT
        # ====================================================================
        # We need to calculate these ONCE to apply to all targets
        latest_row = engineered_df.iloc[-1]
        
        # 1. Predict Minutes (Weighted Avg)
        avg_min = game_log_df['MIN'].mean()
        l5_min = game_log_df['MIN'].tail(5).mean()
        pred_minutes = (l5_min * 0.7) + (avg_min * 0.3)
        
        # 2. Get Pace
        game_pace = latest_row.get('OPP_PACE', 100.0)
        
        # 3. Calculate Personal Possessions
        # (Minutes/48) * Pace * Usage
        usg = usage_rate if usage_rate and usage_rate > 0 else 20.0
        if usg > 1.0: usg = usg / 100.0
        
        personal_possessions = (pred_minutes / 48.0) * game_pace * usg
        
        simulation_context = {
            'MINUTES': round(pred_minutes, 1),
            'PACE': round(game_pace, 1),
            'USAGE': round(usg * 100, 1),
            'POSS': round(personal_possessions, 1)
        }
        
        logger.info("-" * 30)
        logger.info(f"SIMULATION CONTEXT: {simulation_context}")
        logger.info("-" * 30)

        # ====================================================================
        # MAIN LOOP: TRAIN & PREDICT
        # ====================================================================
        
        # We iterate over the RATE targets (PTS_PER_100, etc.)
        for target in TARGETS:
            clean_name = target.replace('_PER_100', '') # e.g., 'PTS'
            print_section(f"PROCESSING TARGET: {clean_name}")
            
            # STEP 3: TRAIN MODEL
            try:
                # Note: This trains a fresh Level 3 Stack on the fly
                model, X_train, X_test, y_train, y_test = train_model(
                    engineered_df,
                    FEATURES,
                    target,
                )
            except Exception as e:
                logger.error(f"Model training failed for {target}: {e}")
                continue

            # STEP 4: EVALUATE MODEL
            try:
                # xgb_mae is in "Per 100 Possessions" units
                xgb_mae, naive_mae, success = evaluate_model(model, X_test, y_test, target)
                
                # CONVERT RATE MAE TO TOTAL MAE (Crucial for Bet Analyzer)
                # If error is 5 pts/100 poss, and player has 20 poss, error is 1 pt
                projected_mae = (xgb_mae / POSSESSION_BASE) * personal_possessions
                
                # Store for JSON output
                mae_scores[clean_name] = round(projected_mae, 2)
                
                # Print result for user
                logger.info(f"   Model Rate MAE: {xgb_mae:.2f}")
                logger.info(f"   Projected Total MAE: {projected_mae:.2f} (Used for betting edge)")

            except Exception as e:
                logger.error(f"Model evaluation failed for {target}: {e}")
                continue

            # STEP 6: FEATURE IMPORTANCE
            try:
                importance_df = get_feature_importance(model, FEATURES)
                if not importance_df.empty:
                    logger.info(f"Top Feature: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['importance']:.3f})")
            except Exception as e:
                logger.debug(f"Could not get feature importance for {target}: {e}")

            # STEP 7: MAKE PREDICTION (SIMULATION)
            try:
                # A. Get Rate Prediction (e.g., 35.5 Pts/100)
                prediction_features = prepare_prediction_data(engineered_df)
                predicted_rate = predict_next_game(model, prediction_features, FEATURES)
                
                # B. Convert to Total (The Simulation)
                # Formula: (Rate / 100) * Personal_Possessions
                predicted_total = (predicted_rate / POSSESSION_BASE) * personal_possessions
                
                # Store CLEAN NAME for Bet Analyzer (e.g., 'PTS': 24.5)
                all_predictions[clean_name] = round(predicted_total, 1)
                
                # Store RATE name for debugging
                all_predictions[target] = round(predicted_rate, 1) 
                
                logger.info(f"👉 Rate: {predicted_rate:.1f} | Total: {predicted_total:.1f}")
                
            except Exception as e:
                logger.error(f"Prediction failed for {target}: {e}")
                continue

        # ====================================================================
        # FINAL OUTPUT
        # ====================================================================
        print_prediction(all_predictions)

        try:
            results_data={
                'player': player_name,
                'season': season,
                'context': simulation_context,
                'predictions': all_predictions,
                'mae_scores': mae_scores # NOW POPULATED
            }

            output_file = "nba_predictions.json"
            with open(output_file, 'w') as f:
                json.dump(results_data, f, indent=4)
            logger.info(f"Results saved to '{output_file}' for bet analyzer")

        except Exception as e:
            logger.error(f"Failed to save JSON results: {e}")

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

# --- Main Execution ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the NBA player performance prediction pipeline.")
    parser.add_argument(
        "--player",
        type=str,
        default=DEFAULT_PLAYER_NAME,
        help=f"Full name of the player to predict for (default: {DEFAULT_PLAYER_NAME})",
    )
    args = parser.parse_args()

    run_prediction_pipeline(player_name=args.player)