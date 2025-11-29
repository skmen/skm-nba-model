"""
Batch Prediction Module

Predict statistics for multiple players.
Supports Level 3 Architecture (Global Models) & Level 4 Simulation (Rate * Opportunity).
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import time
import sys
import numpy as np
import os
import requests
from datetime import datetime, timedelta

# Import shared config and modules
from src.config import FEATURES, TARGETS, API_DELAY, PREDICTIONS_DIR, MODELS_DIR, POSSESSION_BASE
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
        self.pos_map = {'G': 'guards', 'F': 'wings', 'C': 'bigs'}
        self._load_global_models()
    
    def _load_global_models(self):
        """Loads pre-trained global models into memory."""
        logger.info("⚡ Loading Global Models into Memory...")
        groups = ['guards', 'wings', 'bigs']
        count = 0
        
        if not os.path.exists(MODELS_DIR):
            logger.warning(f"Models directory {MODELS_DIR} not found. Skipping load.")
            return

        for group in groups:
            for target in TARGETS:
                filename = f"model_{group}_{target.lower()}.pkl"
                filepath = os.path.join(MODELS_DIR, filename)
                
                if os.path.exists(filepath):
                    try:
                        model_wrapper = NBAPlayerStack()
                        model_wrapper.load(filename)
                        key = f"{group}_{target}"
                        self.global_models[key] = model_wrapper
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load {filename}: {e}")
        
        if count > 0:
            logger.info(f"✅ Successfully loaded {count} global models.")
        else:
            logger.warning("⚠️ No global models found.")

    def _get_previous_season(self, season_str: str) -> str:
        """Calculates the previous season string (e.g. '2024-25' -> '2023-24')."""
        try:
            start_year = int(season_str.split('-')[0])
            prev_start = start_year - 1
            prev_end = str(start_year)[-2:] # The end year of prev season is the start year of current
            return f"{prev_start}-{prev_end}"
        except:
            return "2023-24" # Fallback

    def _is_player_active_recently(self, game_log: pd.DataFrame, player_name: str, target_date: datetime, days_threshold: int = 14) -> bool:
        """
        Checks if the player has played a game within the threshold days RELATIVE TO TARGET DATE.
        This prevents 'Time Travel' bugs where a player is marked inactive because we are simulating a past date.
        """
        if game_log.empty or 'GAME_DATE' not in game_log.columns:
            return False
            
        try:
            # Ensure dates are datetime objects
            game_dates = pd.to_datetime(game_log['GAME_DATE'])
            
            # Get last played game (game_log is already filtered < target_date by caller)
            if game_dates.empty:
                return False
                
            last_played = game_dates.max()
            
            # Calculate diff relative to the SIMULATION date, not today
            diff = target_date - last_played
            
            if diff.days > days_threshold:
                logger.info(f"   -> {player_name} Inactive: Last played {diff.days} days before {target_date.strftime('%Y-%m-%d')} (Game: {last_played.strftime('%Y-%m-%d')})")
                return False
                
            return True
        except Exception as e:
            logger.warning(f"Date check failed for {player_name}: {e}")
            return True

    def predict_player_today(
        self,
        player_name: str,
        team_name: str,
        is_home_game: bool = True,
        season: str = "2024-25",
        raw_position: str = "G",
        advanced_stats: Optional[Dict] = None,
        target_date: Optional[datetime] = None 
    ) -> Optional[Dict[str, float]]:
        
        # Default to today if not provided (fallback)
        if target_date is None:
            target_date = datetime.now()

        # 1. NORMALIZE POSITION
        if 'G' in raw_position: simple_pos = 'G'
        elif 'F' in raw_position: simple_pos = 'F'
        else: simple_pos = 'C'
        
        group_name = self.pos_map.get(simple_pos, 'wings')

        try:
            # 2. FETCH DATA (CURRENT SEASON)
            try:
                game_log = get_player_gamelog(player_name, season)
            except DataAcquisitionError: raise 
            except Exception: return None

            if game_log is None: game_log = pd.DataFrame()

            # --- CRITICAL FIX: DATA WARM-UP (PREVIOUS SEASON) ---
            # If we have very few games (< 10), fetch the previous season to fill the rolling window.
            # This fixes the "All rows dropped" error for players with few games in the new season.
            if len(game_log) < 10:
                try:
                    prev_season = self._get_previous_season(season)
                    logger.debug(f"Fetching history ({prev_season}) for {player_name}...")
                    prev_log = get_player_gamelog(player_name, prev_season)
                    
                    if prev_log is not None and not prev_log.empty:
                        # Safety check for GAME_ID before concatenation
                        if 'GAME_ID' not in prev_log.columns:
                             # Try to be lenient if columns mismatch slightly or just warn and skip
                             logger.debug(f"Previous season log for {player_name} missing GAME_ID. Skipping history.")
                        else:
                             game_log = pd.concat([prev_log, game_log], ignore_index=True)
                             
                             # Ensure no duplicates if seasons overlap in data source
                             # FIX: Check if GAME_ID exists before dropping to prevent KeyError
                             if 'GAME_ID' in game_log.columns:
                                 game_log = game_log.drop_duplicates(subset=['GAME_ID'])
                             else:
                                 logger.warning(f"Combined game log for {player_name} missing GAME_ID column.")

                except Exception as e:
                    logger.warning(f"Could not fetch history for {player_name}: {e}")
            
            if game_log.empty: return None

            # --- CRITICAL FIX: DATA LEAKAGE PREVENTION ---
            # 1. Convert dates
            game_log['GAME_DATE'] = pd.to_datetime(game_log['GAME_DATE'])
            
            # 2. Filter out future games (Data Leakage)
            # We strictly want games BEFORE the target date
            game_log = game_log[game_log['GAME_DATE'] < target_date].copy()
            
            if game_log.empty:
                logger.info(f"  -> Skipping {player_name} (No games found prior to {target_date.date()})")
                return None

            # 3. Sort Descending (Newest First) so .head(5) grabs the correct 'Last 5' relative to target_date
            game_log = game_log.sort_values('GAME_DATE', ascending=False)
            # ------------------------------------------------------

            # 3. CHECK STATUS (RECENCY & MINUTES)
            # Pass target_date to check recency relative to simulation date
            if not self._is_player_active_recently(game_log, player_name, target_date, days_threshold=14):
                return None

            if len(game_log) < 5: avg_min = game_log['MIN'].mean()
            else: avg_min = game_log['MIN'].head(5).mean()

            if pd.isna(avg_min) or avg_min < 10.0:
                logger.info(f"  -> Skipping {player_name} (Low Activity: {avg_min:.1f} min)")
                return None
            
            # 4. FETCH CONTEXT
            if self.opponent_defense is None:
                self.opponent_defense = get_opponent_defense_metrics(season)
            
            try:
                # Use iloc[0] because we sorted descending (Newest)
                pid = int(game_log.iloc[0]['PLAYER_ID'])
                usage_rate = get_player_usage_rate(pid, season)
            except: usage_rate = 0.0
            
            # 5. ENGINEER FEATURES
            # engineer_features sorts Ascending internally, which is fine for rolling calculations
            engineered = engineer_features(
                game_log, self.opponent_defense, usage_rate,
                position_group=simple_pos, advanced_stats=advanced_stats
            )
            if engineered is None or engineered.empty: return None
            
            prediction_input = prepare_prediction_data(engineered, is_home_game)
            latest_row = engineered.iloc[-1]
            
            # 7. CALCULATE SIMULATION VARIABLES
            season_avg_min = game_log['MIN'].mean() if 'MIN' in game_log.columns else avg_min
            pred_minutes = (avg_min * 0.7) + (season_avg_min * 0.3)
            game_pace = latest_row.get('OPP_PACE', 100.0)
            
            team_possessions = (pred_minutes / 48.0) * game_pace
            usg = usage_rate if usage_rate and usage_rate > 0 else 20.0
            if usg > 1.0: usg = usg / 100.0
            
            personal_possessions = team_possessions * usg
            
            player_preds = {
                'PLAYER_NAME': player_name,
                'TEAM_NAME': team_name,
                'IS_HOME': is_home_game,
                'USAGE_RATE': usage_rate if usage_rate and usage_rate > 0 else 0.0,
                'MINUTES': round(pred_minutes, 1),
                'PACE': round(game_pace, 1),
                'USAGE_PCT': round(usg, 3),
                'PROJ_POSS': round(personal_possessions, 1),
                'OPP_DvP': latest_row.get('DVP_MULTIPLIER', 1.0),
                'POSITION_GROUP': simple_pos
            }

            # 9. RUN PREDICTIONS
            for target in TARGETS:
                try:
                    lookup_key = f"{group_name}_{target}"
                    if lookup_key in self.global_models:
                        model = self.global_models[lookup_key]
                        rate_val = predict_next_game(model, prediction_input, FEATURES)
                    else:
                        model, _, _, _, _ = train_model(engineered, FEATURES, target)
                        rate_val = predict_next_game(model, prediction_input, FEATURES)
                    
                    final_total = (rate_val / POSSESSION_BASE) * personal_possessions
                    clean_name = target.replace('_PER_100', '')
                    player_preds[clean_name] = final_total
                    player_preds[target] = rate_val
                    
                except Exception as e:
                    clean_name = target.replace('_PER_100', '')
                    player_preds[clean_name] = 0.0
                    player_preds[target] = 0.0

            return player_preds
            
        except Exception as e:
            # --- IMPROVED ERROR DETECTION (TIMEOUTS) ---
            error_str = str(e).lower()
            is_timeout = (
                "timed out" in error_str or 
                "timeout" in error_str or 
                "connectionpool" in error_str or
                isinstance(e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError))
            )
            
            if is_timeout:
                logger.error(f"⏳ TIMEOUT for {player_name}: {e}")
                # Re-raise specifically as a ReadTimeout to be caught by the batch loop
                raise requests.exceptions.ReadTimeout(f"Timeout processing {player_name}")
            
            logger.error(f"Error processing {player_name}: {e}")
            return None
    
    def predict_multiple_players(self, players_data: List[Dict], season: str = "2024-25") -> pd.DataFrame:
        """Batch processing with status bar, Cool Down, Abort, and Date handling."""
        results = []
        total = len(players_data)
        
        # --- CIRCUIT BREAKER VARS ---
        timeout_streak = 0
        MAX_TIMEOUTS = 5
        COOL_DOWN_SECONDS = 60
        # ----------------------------

        if self.opponent_defense is None:
            try: self.opponent_defense = get_opponent_defense_metrics(season)
            except: pass
        if self.advanced_stats_cache is None:
            self.advanced_stats_cache = get_advanced_stats_cache(season)

        print(f"\nProcessing {total} players...")
        
        # Extract target_date if passed in dict (see get_predictions_for_today)
        # Default to Now if not found
        target_date_obj = datetime.now()
        if players_data and 'TARGET_DATE' in players_data[0]:
             target_date_obj = players_data[0]['TARGET_DATE']

        for i, p in enumerate(players_data):
            msg = f"[{i+1}/{total}] Processing {p['PLAYER_NAME']} ({p['TEAM_NAME']})"
            sys.stdout.write(f"\r{msg:<60}")
            sys.stdout.flush()
            
            time.sleep(API_DELAY)
            
            try:
                pred = self.predict_player_today(
                    player_name=p['PLAYER_NAME'],
                    team_name=p['TEAM_NAME'],
                    is_home_game=p.get('IS_HOME', True),
                    season=season,
                    raw_position=p.get('POSITION', 'G'),
                    advanced_stats=self.advanced_stats_cache,
                    target_date=target_date_obj # PASS THE DATE DOWN
                )
                
                if pred: 
                    results.append(pred)
                    timeout_streak = 0 
                else:
                    # Non-timeout error (data missing, etc) -> reset streak
                    timeout_streak = 0 

            except (requests.exceptions.RequestException, requests.exceptions.ReadTimeout) as e:
                timeout_streak += 1
                
                if timeout_streak < MAX_TIMEOUTS:
                    wait_time = COOL_DOWN_SECONDS
                    msg = f"⚠️ Timeout ({timeout_streak}/{MAX_TIMEOUTS}). Cooling down for {wait_time}s..."
                    logger.warning(msg)
                    sys.stdout.write(f"\r{msg:<60}")
                    sys.stdout.flush()
                    time.sleep(wait_time)
                else:
                    print(f"\n\n❌ MAX TIMEOUTS ({MAX_TIMEOUTS}) REACHED. Aborting to save progress.")
                    logger.error("Max consecutive timeouts reached. Aborting batch.")
                    
                    remaining_players = players_data[i+1:]
                    if remaining_players:
                        skipped_teams = sorted(list(set(pl['TEAM_NAME'] for pl in remaining_players)))
                        print(f"⚠️  SKIPPED TEAMS: {', '.join(skipped_teams)}")
                        logger.info(f"Skipped Teams: {', '.join(skipped_teams)}")
                    
                    break 

            except Exception as e:
                logger.error(f"Unexpected error in batch loop: {e}")
                continue

        print("\n") 
        return pd.DataFrame(results)

    def get_predictions_for_today(self, playing_today_df: pd.DataFrame, season="2024-25"):
        # Convert DataFrame to list of dicts
        players_list = playing_today_df.to_dict('records')
        
        # INJECT TARGET_DATE into the list so predict_multiple_players can find it
        # automated_predictions.py passes playing_today which usually has a GAME_DATE
        if not playing_today_df.empty and 'GAME_DATE' in playing_today_df.columns:
             try:
                 # Take the first date found (batch is usually for one day)
                 first_val = playing_today_df['GAME_DATE'].iloc[0]
                 target = pd.to_datetime(first_val)
             except:
                 target = datetime.now()
        else:
             target = datetime.now()

        # Inject into every player dict
        for p in players_list: 
            p['TARGET_DATE'] = target

        return self.predict_multiple_players(players_list, season)

def normalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces a realistic NBA rotation (Strict 240 Team Minutes).
    UPDATED: Increased Rotation Size to 13 to avoid cutting valid role players.
    """
    if df.empty or 'TEAM_NAME' not in df.columns:
        return df
        
    logger.info("⚖️  Applying Rotation Logic (Top 13 Trim & 240 Min Normalization)...")
    
    normalized_rows = []
    teams = df['TEAM_NAME'].unique()
    
    for team in teams:
        team_df = df[df['TEAM_NAME'] == team].copy()
        
        # 1. SORT by Predicted Minutes (Highest to Lowest)
        team_df = team_df.sort_values('MINUTES', ascending=False)
        
        # 2. TRIM ROTATION (Simulate DNP-CDs)
        # CHANGED: 10 -> 13 to include more bench players
        ROTATION_SIZE = 13
        
        if len(team_df) > ROTATION_SIZE:
            # Identify bench warmers
            bench_warmers = team_df.iloc[ROTATION_SIZE:]
            
            # Zero out their stats
            cols_to_zero = ['MINUTES', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'PRA', 'PROJ_POSS']
            valid_cols = [c for c in cols_to_zero if c in team_df.columns]
            
            # We use the index to target specific rows
            team_df.loc[bench_warmers.index, valid_cols] = 0.0

        # 3. CALCULATE SCALE FACTOR FOR ROTATION PLAYERS
        # Only sum minutes of the Top 13 (who are actually playing)
        rotation_players = team_df.head(ROTATION_SIZE)
        total_min = rotation_players['MINUTES'].sum()
        
        # Scale to exactly 240 minutes (48 mins * 5 positions)
        if total_min > 0:
            scale_factor = 240.0 / total_min
            
            # Only apply scaling to the active rotation
            cols_to_scale = ['MINUTES', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'PRA', 'PROJ_POSS']
            valid_cols = [c for c in cols_to_scale if c in team_df.columns]
            
            # Update only the active rows
            team_df.iloc[:ROTATION_SIZE, team_df.columns.get_indexer(valid_cols)] *= scale_factor
        
        normalized_rows.append(team_df)
        
    return pd.concat(normalized_rows)

def predict_all_players_today(playing_today, output_csv, season="2024-25"):
    """Entry point used by automated_predictions.py."""
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    # Ensure playing_today has consistent data types
    if 'GAME_DATE' in playing_today.columns:
        playing_today['GAME_DATE'] = pd.to_datetime(playing_today['GAME_DATE'])
        
    predictor = BatchPredictor()
    new_df = predictor.get_predictions_for_today(playing_today, season)
    
    if new_df.empty:
        logger.warning("No new predictions generated.")
        return new_df

    if os.path.exists(output_csv):
        try:
            logger.info(f"Merging with existing file: {output_csv}")
            existing_df = pd.read_csv(output_csv)
            combined_df = pd.concat([existing_df, new_df])
            combined_df = combined_df.drop_duplicates(subset=['PLAYER_NAME', 'TEAM_NAME'], keep='last')
        except Exception:
            combined_df = new_df
    else:
        combined_df = new_df

    final_df = normalize_predictions(combined_df)
    final_df = final_df.sort_values(by=['TEAM_NAME', 'PLAYER_NAME'])
    final_df.to_csv(output_csv, index=False)
    
    logger.info(f"Saved Normalized Predictions to {output_csv}")
    return final_df