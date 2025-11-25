"""
NBA Player Performance Prediction Pipeline

Main orchestration module that coordinates data fetching, feature engineering,
model training, and prediction.

Usage:
    python prediction_pipeline.py
"""

import logging
from typing import Optional

from config import (
    DEFAULT_PLAYER_NAME,
    DEFAULT_SEASON,
    FEATURES,
    TARGET,
)
from utils import (
    setup_logger,
    print_section,
    print_model_results,
    print_prediction,
)
from data_fetcher import acquire_all_data
from feature_engineer import engineer_features
from model import (
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
# MAIN PIPELINE ORCHESTRATION
# ============================================================================

def run_prediction_pipeline(
    player_name: str = DEFAULT_PLAYER_NAME,
    season: str = DEFAULT_SEASON,
) -> None:
    """
    Run the complete NBA prediction pipeline.

    Steps:
    1. Acquire data (game logs, opponent stats)
    2. Engineer features
    3. Train model
    4. Evaluate performance
    5. Make prediction

    Args:
        player_name: Full name of the player to predict for
        season: NBA season (e.g., "2023-24")
    """
    try:
        logger.info("\n" + "=" * 60)
        logger.info("NBA PLAYER PERFORMANCE PREDICTION PIPELINE")
        logger.info("=" * 60)

        # ====================================================================
        # STEP 1: ACQUIRE DATA
        # ====================================================================
        try:
            game_log_df, opponent_defense, player_id, usage_rate = (
                acquire_all_data(player_name, season)
            )
        except Exception as e:
            logger.error(f"Data acquisition failed: {e}")
            raise

        # ====================================================================
        # STEP 2: ENGINEER FEATURES
        # ====================================================================
        try:
            engineered_df = engineer_features(
                game_log_df,
                opponent_defense,
                usage_rate,
            )
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise

        # ====================================================================
        # STEP 3: TRAIN MODEL
        # ====================================================================
        try:
            model, X_train, X_test, y_train, y_test = train_model(
                engineered_df,
                FEATURES,
                TARGET,
            )
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise

        # ====================================================================
        # STEP 4: EVALUATE MODEL
        # ====================================================================
        try:
            xgb_mae, naive_mae, success = evaluate_model(model, X_test, y_test)
            print_model_results(xgb_mae, naive_mae)
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise

        # ====================================================================
        # STEP 5: VISUALIZE RESULTS
        # ====================================================================
        try:
            predictions = model.predict(X_test)
            plot_model_results(y_test, predictions)
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
            # Don't raise - visualization is optional

        # ====================================================================
        # STEP 6: FEATURE IMPORTANCE
        # ====================================================================
        try:
            importance_df = get_feature_importance(model, FEATURES)
            if not importance_df.empty:
                logger.info("\n" + "=" * 60)
                logger.info("TOP 10 MOST IMPORTANT FEATURES")
                logger.info("=" * 60)
                for _, row in importance_df.iterrows():
                    logger.info(f"{row['feature']:.<40} {row['importance']:.0f}")
        except Exception as e:
            logger.debug(f"Could not get feature importance: {e}")

        # ====================================================================
        # STEP 7: MAKE PREDICTION
        # ====================================================================
        try:
            prediction_features = prepare_prediction_data(engineered_df)
            predicted_score = predict_next_game(model, prediction_features, FEATURES)
            print_prediction(predicted_score)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

# --- Main Execution ---
if __name__ == "__main__":
    run_prediction_pipeline()
