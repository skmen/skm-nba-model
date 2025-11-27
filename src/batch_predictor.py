"""
Batch Prediction Module

Predict statistics (PTS, REB, AST, STL, BLK, PRA) for multiple players at once.
Optimized to run the 'Model Tournament' (Ridge vs XGBoost) for every player/stat.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import time
import numpy as np

# Import shared config and modules
from src.config import FEATURES, TARGETS, API_DELAY
from src.data_fetcher import get_player_gamelog, get_opponent_defense_metrics, get_player_usage_rate
from src.feature_engineer import engineer_features
from src.model import train_model, predict_next_game, prepare_prediction_data
from src.utils import setup_logger

logger = setup_logger(__name__)

class BatchPredictor:
    """Batch prediction engine."""
    
    def __init__(self):
        self.opponent_defense = None
    
    def predict_player_today(
        self,
        player_name: str,
        team_name: str,
        is_home_game: bool = True,
        season: str = "2024-25",
        raw_position: str = "G"
    ) -> Optional[Dict[str, float]]:
        """
        Run the full prediction pipeline for a single player.
        """
        try:
            logger.info(f"Processing {player_name} ({team_name})...")
            
            # 1. Fetch Data
            game_log = get_player_gamelog(player_name, season)
            if game_log is None or game_log.empty:
                return None

            # --- FILTER: CHECK RECENT FORM ---
            # Calculate Last 5 Games averages
            avg_min = game_log['MIN'].head(5).mean()
            avg_pts = game_log['PTS'].head(5).mean()

            # If player averages less than 5 minutes OR less than 1 point, SKIP THEM.
            if pd.isna(avg_min) or avg_min < 5.0 or pd.isna(avg_pts) or avg_pts < 1.0:
                logger.info(f"  -> Skipping {player_name} (Low Activity: {avg_min:.1f} min, {avg_pts:.1f} pts)")
                return None
            # ---------------------------------

            if 'G' in raw_position:
                simple_pos = 'G'
            elif 'F' in raw_position:
                simple_pos = 'F'
            else:
                simple_pos = 'C'
            
            # Cache opponent defense to save API calls
            if self.opponent_defense is None:
                self.opponent_defense = get_opponent_defense_metrics(season)
            
            # Get Usage Rate
            try:
                # Handle case where player ID access might fail
                pid = int(game_log.iloc[-1]['PLAYER_ID'])
                usage_rate = get_player_usage_rate(pid, season)
            except:
                usage_rate = 0.0
            
            # 2. Engineer Features (Includes PRA, Per Minute, etc.)
            engineered = engineer_features(game_log, self.opponent_defense, usage_rate)
            
            if engineered is None or engineered.empty:
                return None
            
            # 3. Prepare Input for Prediction
            prediction_input = prepare_prediction_data(engineered, is_home_game)
            
            # --- EXTRACT METRICS FOR REPORTING ---
            latest_row = engineered.iloc[-1]
            opp_pace = latest_row.get('OPP_PACE', 0.0)
            dvp_multiplier = latest_row.get('DVP_MULTIPLIER', 1.0)
            latest_row = engineered.iloc[-1]

            # Extract the new raw DvP stats (default to 0 if missing)
            opp_pts_allowed = latest_row.get('OPP_ALLOW_PTS', 0.0)
            opp_reb_allowed = latest_row.get('OPP_ALLOW_REB', 0.0)
            
            # 4. Construct Output Dictionary
            # (Matches user requirements: Add Usage/Pace/DvP, Remove Time/ID)
            player_preds = {
                'PLAYER_NAME': player_name,
                'TEAM_NAME': team_name,
                'IS_HOME': is_home_game,
                'USAGE_RATE': usage_rate if usage_rate else 0.0,
                'MINUTES': avg_min,
                'PACE': latest_row.get('OPP_PACE', 0.0),
                'OPP_DvP': latest_row.get('DVP_MULTIPLIER', 1.0),
                
                # NEW COLUMNS FROM dvp_stats.csv
                'OPP_ALLOW_PTS': opp_pts_allowed,
                'OPP_ALLOW_REB': opp_reb_allowed,
                
                'POSITION_GROUP': simple_pos
            }

            # 5. Run Tournament for EVERY Target
            for target in TARGETS:
                try:
                    # Train Model (Runs Ridge vs XGBoost Tournament)
                    model, _, _, _, _ = train_model(engineered, FEATURES, target)
                    
                    # Predict
                    val = predict_next_game(model, prediction_input, FEATURES)
                    player_preds[target] = val
                    
                    logger.debug(f"  -> {target}: {val:.1f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to predict {target} for {player_name}: {e}")
                    player_preds[target] = 0.0

            return player_preds
            
        except Exception as e:
            logger.error(f"Error processing {player_name}: {e}")
            return None
    
    def predict_multiple_players(
        self,
        players_data: List[Dict],
        season: str = "2024-25"
    ) -> pd.DataFrame:
        """
        Loop through a list of players and predict stats.
        """
        results = []
        total = len(players_data)
        
        # Prefetch defense once
        if self.opponent_defense is None:
            self.opponent_defense = get_opponent_defense_metrics(season)

        for i, p in enumerate(players_data):
            logger.info(f"[{i+1}/{total}] Starting Batch for {p['PLAYER_NAME']}")
            
            # Respect API Rate Limits
            time.sleep(API_DELAY)
            
            pred = self.predict_player_today(
                player_name=p['PLAYER_NAME'],
                team_name=p['TEAM_NAME'],
                is_home_game=p.get('IS_HOME', True),
                season=season,
                raw_position=p.get('POSITION', 'G')
            )
            
            if pred:
                results.append(pred)
        
        df = pd.DataFrame(results)

        # --- ROUNDING LOGIC ---
        if not df.empty:
            # Select all float columns
            cols_to_round = df.select_dtypes(include=['float', 'float64']).columns
            # Round to 2 decimal places
            df[cols_to_round] = df[cols_to_round].round(2)
        
        return df

    def get_predictions_for_today(self, playing_today_df: pd.DataFrame, season="2024-25"):
        """Convenience wrapper for the dataframe format."""
        # Convert DataFrame to List of Dicts for the loop
        players_list = playing_today_df.to_dict('records')
        return self.predict_multiple_players(players_list, season)

def predict_all_players_today(playing_today, output_csv, season="2024-25"):
    """Entry point used by automated_predictions.py"""
    predictor = BatchPredictor()
    df = predictor.get_predictions_for_today(playing_today, season)
    
    if not df.empty:
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved batch predictions to {output_csv}")
    else:
        logger.warning("No predictions generated (all players filtered out or errors).")
        
    return df