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
import os

# Import shared config and modules
from src.config import FEATURES, TARGETS, API_DELAY
from src.data_fetcher import get_player_gamelog, get_opponent_defense_metrics, get_player_usage_rate
from src.feature_engineer import engineer_features
from src.model import train_model, predict_next_game, prepare_prediction_data
# Updated import to include DataAcquisitionError
from src.utils import setup_logger, DataAcquisitionError

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
            try:
                game_log = get_player_gamelog(player_name, season)
            except DataAcquisitionError:
                # Re-raise specifically to handle timeouts in batch loop
                raise
            except Exception as e:
                # Handle other fetch errors gracefully
                logger.warning(f"Error fetching data for {player_name}: {e}")
                return None

            if game_log is None or game_log.empty:
                return None

            # --- FILTER: CHECK RECENT FORM ---
            avg_min = game_log['MIN'].head(5).mean()
            avg_pts = game_log['PTS'].head(5).mean()

            if pd.isna(avg_min) or avg_min < 5.0 or pd.isna(avg_pts) or avg_pts < 1.0:
                logger.info(f"  -> Skipping {player_name} (Low Activity: {avg_min:.1f} min, {avg_pts:.1f} pts)")
                return None
            
            # Cache opponent defense
            if self.opponent_defense is None:
                self.opponent_defense = get_opponent_defense_metrics(season)
            
            # Get Usage Rate
            try:
                pid = int(game_log.iloc[-1]['PLAYER_ID'])
                usage_rate = get_player_usage_rate(pid, season)
            except:
                usage_rate = 0.0
            
            # 2. Engineer Features
            engineered = engineer_features(game_log, self.opponent_defense, usage_rate)
            
            if engineered is None or engineered.empty:
                return None
            
            # 3. Prepare Input
            prediction_input = prepare_prediction_data(engineered, is_home_game)
            
            # --- EXTRACT METRICS ---
            latest_row = engineered.iloc[-1]
            opp_pace = latest_row.get('OPP_PACE', 0.0)
            
            # --- NORMALIZE POSITION ---
            if 'G' in raw_position:
                simple_pos = 'G'
            elif 'F' in raw_position:
                simple_pos = 'F'
            else:
                simple_pos = 'C'

            # 4. Construct Output
            player_preds = {
                'PLAYER_NAME': player_name,
                'TEAM_NAME': team_name,
                'IS_HOME': is_home_game,
                'USAGE_RATE': usage_rate if usage_rate else 0.0,
                'MINUTES': avg_min,
                'PACE': opp_pace,
                'OPP_DvP': latest_row.get('DVP_MULTIPLIER', 1.0),
                'OPP_ALLOW_PTS': latest_row.get('OPP_ALLOW_PTS', 0.0),
                'OPP_ALLOW_REB': latest_row.get('OPP_ALLOW_REB', 0.0),
                'POSITION_GROUP': simple_pos
            }

            # 5. Run Tournament
            for target in TARGETS:
                try:
                    model, _, _, _, _ = train_model(engineered, FEATURES, target)
                    val = predict_next_game(model, prediction_input, FEATURES)
                    player_preds[target] = val
                    logger.debug(f"  -> {target}: {val:.1f}")
                except Exception as e:
                    logger.warning(f"Failed to predict {target} for {player_name}: {e}")
                    player_preds[target] = 0.0

            return player_preds
            
        except DataAcquisitionError:
            raise
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
        Includes timeout protection and team tracking.
        """
        results = []
        total = len(players_data)
        
        # Track Teams
        all_teams = set(p['TEAM_NAME'] for p in players_data)
        processed_teams = set()
        
        # Timeout Logic
        timeout_count = 0
        MAX_TIMEOUTS = 10
        
        # Prefetch defense once
        if self.opponent_defense is None:
            try:
                self.opponent_defense = get_opponent_defense_metrics(season)
            except Exception as e:
                logger.warning(f"Failed to prefetch defense metrics: {e}")

        for i, p in enumerate(players_data):
            # Check Timeout Threshold
            if timeout_count >= MAX_TIMEOUTS:
                logger.error(f"❌ Aborting: Exceeded {MAX_TIMEOUTS} timeouts from stats.nba.com")
                break

            logger.info(f"[{i+1}/{total}] Starting Batch for {p['PLAYER_NAME']}")
            time.sleep(API_DELAY)
            
            try:
                pred = self.predict_player_today(
                    player_name=p['PLAYER_NAME'],
                    team_name=p['TEAM_NAME'],
                    is_home_game=p.get('IS_HOME', True),
                    season=season,
                    raw_position=p.get('POSITION', 'G')
                )
                
                # Mark team as processed (even if player was skipped due to low minutes, 
                # we successfully accessed their data)
                processed_teams.add(p['TEAM_NAME'])
                
                if pred:
                    results.append(pred)
                    
            except DataAcquisitionError:
                timeout_count += 1
                logger.warning(f"⚠️ Timeout/Data Error for {p['PLAYER_NAME']} ({p['TEAM_NAME']}). "
                               f"Consecutive Errors: {timeout_count}/{MAX_TIMEOUTS}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error for {p['PLAYER_NAME']}: {e}")
                continue
        
        # Report Unprocessed Teams
        unprocessed_teams = all_teams - processed_teams
        if unprocessed_teams:
            logger.warning("\n" + "!"*60)
            logger.warning("THE FOLLOWING TEAMS WERE NOT PROCESSED DUE TO TIMEOUTS:")
            for team in sorted(unprocessed_teams):
                logger.warning(f"  - {team}")
            logger.warning("!"*60 + "\n")
        
        df = pd.DataFrame(results)

        if not df.empty:
            cols_to_round = df.select_dtypes(include=['float', 'float64']).columns
            df[cols_to_round] = df[cols_to_round].round(2)
        
        return df

    def get_predictions_for_today(self, playing_today_df: pd.DataFrame, season="2024-25"):
        players_list = playing_today_df.to_dict('records')
        return self.predict_multiple_players(players_list, season)

def predict_all_players_today(playing_today, output_csv, season="2024-25"):
    """
    Entry point used by automated_predictions.py.
    Handles 'Update vs Create' logic for the CSV file.
    """
    predictor = BatchPredictor()
    new_df = predictor.get_predictions_for_today(playing_today, season)
    
    if new_df.empty:
        logger.warning("No new predictions generated.")
        return new_df

    # --- UPDATE OR APPEND LOGIC ---
    if os.path.exists(output_csv):
        try:
            logger.info(f"Existing file found: {output_csv}. Merging new data...")
            existing_df = pd.read_csv(output_csv)
            
            # Combine old and new data
            combined_df = pd.concat([existing_df, new_df])
            
            # Drop duplicates based on Player and Team.
            combined_df = combined_df.drop_duplicates(subset=['PLAYER_NAME', 'TEAM_NAME'], keep='last')
            
            # Sort
            combined_df = combined_df.sort_values(by=['TEAM_NAME', 'PLAYER_NAME'])
            
            combined_df.to_csv(output_csv, index=False)
            logger.info(f"Updated {output_csv} (Total Players: {len(combined_df)})")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error merging with existing file: {e}. Overwriting instead.")
            new_df.to_csv(output_csv, index=False)
            return new_df
    else:
        # File does not exist, create fresh
        new_df.to_csv(output_csv, index=False)
        logger.info(f"Created new file: {output_csv}")
        return new_df