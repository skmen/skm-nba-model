"""
Batch Prediction Module

Predict statistics for multiple players.
Supports Level 3 Architecture:
1. Tries to use Global Cached Models (Fastest)
2. Falls back to Individual Stacked Training (Slower, but works without cache)
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import time
import numpy as np
import os

# Import shared config and modules
from src.config import FEATURES, TARGETS, API_DELAY, PREDICTIONS_DIR, MODELS_DIR
from src.data_fetcher import get_player_gamelog, get_opponent_defense_metrics, get_player_usage_rate, get_advanced_stats_cache
from src.feature_engineer import engineer_features
from src.model import train_model, predict_next_game, prepare_prediction_data, NBAPlayerStack
from src.utils import setup_logger, DataAcquisitionError

logger = setup_logger(__name__)

class BatchPredictor:
    """Batch prediction engine."""
    
    def __init__(self):
        self.opponent_defense = None
        self.advanced_stats_cache = None
        # Cache for global models
        self.global_models = {}
        self._load_global_models()
    
    def _load_global_models(self):
        """Attempts to load pre-trained global models for each stat/position."""
        # This structure assumes we might have specific models like 'Guard_PTS', 'Wing_REB'
        # Or generic bucket models. adapting to your future naming convention.
        # For now, we will just prepare the dictionary.
        pass 

    def _predict_minutes(self, game_log: pd.DataFrame, season_avg: float) -> float:
        """
        Predict minutes based on weighted L5 and Season Average.
        Minutes are the most critical multiplier in a rate-based model.
        """
        if 'MIN' not in game_log.columns:
            return season_avg
            
        l5_min = game_log['MIN'].tail(5).mean()
        
        # Weight L5 heavily (70%) as rotation changes are recent
        pred_min = (l5_min * 0.7) + (season_avg * 0.3)
        return pred_min

    def predict_player_today(
        self,
        player_name: str,
        team_name: str,
        is_home_game: bool = True,
        season: str = "2024-25",
        raw_position: str = "G",
        advanced_stats: Optional[Dict] = None
    ) -> Optional[Dict[str, float]]:
        """
        Run the full prediction pipeline for a single player.
        """
        
        # --- 1. NORMALIZE POSITION ---
        if 'G' in raw_position:
            simple_pos = 'G'
        elif 'F' in raw_position:
            simple_pos = 'F'
        else:
            simple_pos = 'C'

        try:
            logger.info(f"Processing {player_name} ({team_name})...")
            
            # --- 2. FETCH DATA ---
            try:
                game_log = get_player_gamelog(player_name, season)
            except DataAcquisitionError:
                raise 
            except Exception as e:
                logger.warning(f"Error fetching data for {player_name}: {e}")
                return None

            if game_log is None or game_log.empty:
                return None

            # --- 3. CHECK RECENT FORM ---
            if len(game_log) < 5:
                avg_min = game_log['MIN'].mean()
            else:
                avg_min = game_log['MIN'].head(5).mean()

            if pd.isna(avg_min) or avg_min < 5.0:
                logger.info(f"  -> Skipping {player_name} (Low Activity: {avg_min} min)")
                return None
            
            # --- 4. FETCH CONTEXT ---
            if self.opponent_defense is None:
                self.opponent_defense = get_opponent_defense_metrics(season)
            
            # Get Usage Rate
            try:
                pid = int(game_log.iloc[-1]['PLAYER_ID'])
                usage_rate = get_player_usage_rate(pid, season)
            except:
                usage_rate = 0.0

            # --- CRITICAL FIX: HANDLE NONE TYPE ---
            # If API returns None, force it to 0.0 so comparisons don't crash
            if usage_rate is None:
                usage_rate = 0.0
            # --------------------------------------
            
            # --- 5. ENGINEER FEATURES ---
            engineered = engineer_features(
                game_log, 
                self.opponent_defense, 
                usage_rate,
                position_group=simple_pos,
                advanced_stats=advanced_stats
            )
            
            if engineered is None or engineered.empty:
                return None
            
            # --- 6. PREPARE INPUT ---
            prediction_input = prepare_prediction_data(engineered, is_home_game)
            latest_row = engineered.iloc[-1]
            
            # --- 7. CALCULATE SIMULATION VARIABLES ---
            
            # A. Predict Minutes
            season_avg_min = game_log['MIN'].mean() if 'MIN' in game_log.columns else avg_min
            pred_minutes = (avg_min * 0.7) + (season_avg_min * 0.3)
            
            # B. Estimate Game Pace
            game_pace = latest_row.get('OPP_PACE', 100.0)
            
            # C. Calculate TEAM Possessions
            team_possessions = (pred_minutes / 48.0) * game_pace

            # D. Calculate PERSONAL Possessions
            # Safe comparison now because usage_rate is guaranteed to be float
            usg = usage_rate if usage_rate > 0 else 20.0
            
            # Handle percentage vs decimal (e.g. 25.0 vs 0.25)
            if usg > 1.0: 
                usg = usg / 100.0
                
            personal_possessions = team_possessions * usg
            
            # Store for CSV
            player_preds = {
                'PLAYER_NAME': player_name,
                'TEAM_NAME': team_name,
                'IS_HOME': is_home_game,
                'USAGE_RATE': usage_rate if usage_rate > 0 else 0.0,
                'MINUTES': round(pred_minutes, 1),
                'PACE': round(game_pace, 1),
                'USAGE_PCT': round(usg, 3),
                'PROJ_POSS': round(personal_possessions, 1),
                'OPP_DvP': latest_row.get('DVP_MULTIPLIER', 1.0),
                'OPP_ALLOW_PTS': latest_row.get('OPP_ALLOW_PTS', 0.0),
                'OPP_ALLOW_REB': latest_row.get('OPP_ALLOW_REB', 0.0),
                'POSITION_GROUP': simple_pos
            }

            # --- 9. RUN PREDICTIONS ---
            for target in TARGETS:
                try:
                    # Model Inference
                    model, _, _, _, _ = train_model(engineered, FEATURES, target)
                    rate_val = predict_next_game(model, prediction_input, FEATURES)
                    
                    # Convert Rate to Total
                    final_total = (rate_val / 100.0) * personal_possessions
                    
                    clean_name = target.replace('_PER_100', '')
                    player_preds[clean_name] = final_total
                    player_preds[target] = rate_val
                    
                    logger.debug(f"  -> {clean_name}: {final_total:.1f} (Rate: {rate_val:.1f})")
                    
                except Exception as e:
                    logger.warning(f"Failed to predict {target} for {player_name}: {e}")
                    clean_name = target.replace('_PER_100', '')
                    player_preds[clean_name] = 0.0
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
        MAX_TIMEOUTS = 5
        
        # Prefetch defense once
        if self.opponent_defense is None:
            try:
                self.opponent_defense = get_opponent_defense_metrics(season)
            except Exception as e:
                logger.warning(f"Failed to prefetch defense metrics: {e}")
        
        if self.advanced_stats_cache is None:
            self.advanced_stats_cache = get_advanced_stats_cache(season)

        for i, p in enumerate(players_data):
            # Check Timeout Threshold
            if timeout_count >= MAX_TIMEOUTS:
                logger.error(f" Aborting: Exceeded {MAX_TIMEOUTS} timeouts from stats.nba.com")
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
                
                # Mark team as processed
                processed_teams.add(p['TEAM_NAME'])
                
                if pred:
                    results.append(pred)
                    
            except DataAcquisitionError:
                timeout_count += 1
                logger.warning(f"[TIMEOUT] Data Error for {p['PLAYER_NAME']} ({p['TEAM_NAME']}). Consecutive Errors: {timeout_count}/{MAX_TIMEOUTS}")
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
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
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