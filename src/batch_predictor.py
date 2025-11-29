"""
Batch Prediction Module

Predict statistics for multiple players.
Supports Level 3 Architecture (Global Models) & Level 4 Simulation (Rate * Opportunity).

Features:
- Fast Inference: Loads pre-trained models to skip daily retraining.
- Normalization: Scales team minutes to 240 to prevent inflated totals.
- Robustness: Handles missing data, inactive players, and low-minute outliers.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import time
import numpy as np
import os
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
        """
        Loads pre-trained global models into memory for Fast Inference.
        """
        logger.info("⚡ Loading Global Models into Memory...")
        groups = ['guards', 'wings', 'bigs']
        count = 0
        
        # Check if directory exists
        if not os.path.exists(MODELS_DIR):
            logger.warning(f"Models directory {MODELS_DIR} not found. Skipping load.")
            return

        for group in groups:
            for target in TARGETS:
                # Expects filenames like: model_guards_pts_per_100.pkl
                filename = f"model_{group}_{target.lower()}.pkl"
                filepath = os.path.join(MODELS_DIR, filename)
                
                if os.path.exists(filepath):
                    try:
                        # Wrapper class handles the loading
                        model_wrapper = NBAPlayerStack()
                        model_wrapper.load(filename)
                        
                        # Store with key like "guards_PTS_PER_100"
                        key = f"{group}_{target}"
                        self.global_models[key] = model_wrapper
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load {filename}: {e}")
        
        if count > 0:
            logger.info(f"✅ Successfully loaded {count} global models. Fast Inference Mode: ON")
        else:
            logger.warning("⚠️ No global models found. Falling back to slow training mode (Run train_global_models.py).")

    def _is_player_active_recently(self, game_log: pd.DataFrame, days_threshold: int = 14) -> bool:
        """
        Checks if the player has played a game within the threshold days.
        Used to filter out injured players who are technically on the roster.
        """
        if game_log.empty or 'GAME_DATE' not in game_log.columns:
            return False
            
        try:
            # Ensure datetime format
            dates = pd.to_datetime(game_log['GAME_DATE'])
            last_played = dates.max()
            
            # Check difference from today
            diff = datetime.now() - last_played
            
            if diff.days > days_threshold:
                logger.info(f"   -> Inactive: Last played {diff.days} days ago ({last_played.strftime('%Y-%m-%d')})")
                return False
                
            return True
        except Exception as e:
            logger.warning(f"Date check failed: {e}")
            return True # Assume active if check fails to avoid false negatives

    def predict_player_today(
        self,
        player_name: str,
        team_name: str,
        is_home_game: bool = True,
        season: str = "2024-25",
        raw_position: str = "G",
        advanced_stats: Optional[Dict] = None
    ) -> Optional[Dict[str, float]]:
        
        # --- 1. NORMALIZE POSITION ---
        if 'G' in raw_position: simple_pos = 'G'
        elif 'F' in raw_position: simple_pos = 'F'
        else: simple_pos = 'C'
        
        group_name = self.pos_map.get(simple_pos, 'wings')

        try:
            logger.info(f"Processing {player_name} ({team_name})...")
            
            # --- 2. FETCH DATA ---
            try:
                game_log = get_player_gamelog(player_name, season)
            except DataAcquisitionError: raise 
            except Exception: return None

            if game_log is None or game_log.empty: return None

            # --- 3. CHECK STATUS (RECENCY & MINUTES) ---
            
            # A. Check Recency (Injured/Inactive Filter)
            if not self._is_player_active_recently(game_log, days_threshold=14):
                return None

            # B. Check Minutes (Garbage Time Filter)
            # Increased threshold to 10 minutes to filter out outliers
            if len(game_log) < 5: avg_min = game_log['MIN'].mean()
            else: avg_min = game_log['MIN'].head(5).mean()

            if pd.isna(avg_min) or avg_min < 10.0:
                logger.info(f"  -> Skipping {player_name} (Low Activity: {avg_min:.1f} min)")
                return None
            
            # --- 4. FETCH CONTEXT ---
            if self.opponent_defense is None:
                self.opponent_defense = get_opponent_defense_metrics(season)
            
            # Get Usage Rate (With Safety Check)
            try:
                pid = int(game_log.iloc[-1]['PLAYER_ID'])
                usage_rate = get_player_usage_rate(pid, season)
            except: usage_rate = 0.0
            
            if usage_rate is None: usage_rate = 0.0

            # --- 5. ENGINEER FEATURES ---
            engineered = engineer_features(
                game_log, self.opponent_defense, usage_rate,
                position_group=simple_pos, advanced_stats=advanced_stats
            )
            if engineered is None or engineered.empty: return None
            
            prediction_input = prepare_prediction_data(engineered, is_home_game)
            latest_row = engineered.iloc[-1]
            
            # --- 7. CALCULATE SIMULATION VARIABLES ---
            season_avg_min = game_log['MIN'].mean() if 'MIN' in game_log.columns else avg_min
            
            # Weighted Average for Minutes Prediction
            pred_minutes = (avg_min * 0.7) + (season_avg_min * 0.3)
            
            # Game Pace
            game_pace = latest_row.get('OPP_PACE', 100.0)
            
            # TEAM Possessions
            team_possessions = (pred_minutes / 48.0) * game_pace

            # PERSONAL Possessions (Usage Adjusted)
            usg = usage_rate if usage_rate > 0 else 20.0
            if usg > 1.0: usg = usg / 100.0 # Convert 25.0 to 0.25
            
            personal_possessions = team_possessions * usg
            
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
                'POSITION_GROUP': simple_pos
            }

            # --- 9. RUN PREDICTIONS ---
            for target in TARGETS:
                try:
                    # OPTION A: FAST LOOKUP (Preferred)
                    lookup_key = f"{group_name}_{target}"
                    if lookup_key in self.global_models:
                        model = self.global_models[lookup_key]
                        rate_val = predict_next_game(model, prediction_input, FEATURES)
                    
                    # OPTION B: SLOW TRAINING (Fallback)
                    else:
                        model, _, _, _, _ = train_model(engineered, FEATURES, target)
                        rate_val = predict_next_game(model, prediction_input, FEATURES)
                    
                    # SIMULATION MATH (Efficiency Rate -> Total Output)
                    # Formula: (Rate / 100) * Personal Possessions
                    final_total = (rate_val / POSSESSION_BASE) * personal_possessions
                    
                    clean_name = target.replace('_PER_100', '')
                    player_preds[clean_name] = final_total
                    player_preds[target] = rate_val
                    
                except Exception as e:
                    logger.warning(f"Failed to predict {target} for {player_name}: {e}")
                    clean_name = target.replace('_PER_100', '')
                    player_preds[clean_name] = 0.0
                    player_preds[target] = 0.0

            return player_preds
            
        except Exception as e:
            logger.error(f"Error processing {player_name}: {e}")
            return None
    
    def predict_multiple_players(self, players_data: List[Dict], season: str = "2024-25") -> pd.DataFrame:
        """Batch processing with pre-fetched context."""
        results = []
        total = len(players_data)
        
        # Pre-fetch context once per batch
        if self.opponent_defense is None:
            try: self.opponent_defense = get_opponent_defense_metrics(season)
            except: pass
        if self.advanced_stats_cache is None:
            self.advanced_stats_cache = get_advanced_stats_cache(season)

        for i, p in enumerate(players_data):
            time.sleep(API_DELAY)
            try:
                pred = self.predict_player_today(
                    player_name=p['PLAYER_NAME'],
                    team_name=p['TEAM_NAME'],
                    is_home_game=p.get('IS_HOME', True),
                    season=season,
                    raw_position=p.get('POSITION', 'G'),
                    advanced_stats=self.advanced_stats_cache
                )
                if pred: results.append(pred)
            except DataAcquisitionError:
                continue
            except Exception:
                continue
        
        return pd.DataFrame(results)

    def get_predictions_for_today(self, playing_today_df: pd.DataFrame, season="2024-25"):
        players_list = playing_today_df.to_dict('records')
        return self.predict_multiple_players(players_list, season)

# ============================================================================
# HELPER: NORMALIZATION LOGIC
# ============================================================================

def normalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces a realistic NBA rotation (Strict 240 Team Minutes).
    
    Logic:
    1. Groups by Team.
    2. Keeps only the Top 10 players by projected minutes (Rotation Cutoff).
    3. Sets minutes/stats to 0 for deep bench players (11th+ man).
    4. Scales the Top 10 players so their minutes sum to exactly 240.
    """
    if df.empty or 'TEAM_NAME' not in df.columns:
        return df
        
    logger.info("⚖️  Applying Rotation Logic (Top 10 Trim & 240 Min Normalization)...")
    
    normalized_rows = []
    teams = df['TEAM_NAME'].unique()
    
    for team in teams:
        team_df = df[df['TEAM_NAME'] == team].copy()
        
        # 1. SORT by Predicted Minutes (Highest to Lowest)
        team_df = team_df.sort_values('MINUTES', ascending=False)
        
        # 2. TRIM ROTATION (Simulate DNP-CDs)
        # Most NBA teams play 9-10 guys. We cut off anyone below rank 10.
        ROTATION_SIZE = 10
        if len(team_df) > ROTATION_SIZE:
            # Identify bench warmers
            bench_warmers = team_df.iloc[ROTATION_SIZE:]
            
            # Zero out their stats
            cols_to_zero = ['MINUTES', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'PRA', 'PROJ_POSS']
            valid_cols = [c for c in cols_to_zero if c in team_df.columns]
            
            # We use the index to target specific rows
            team_df.loc[bench_warmers.index, valid_cols] = 0.0

        # 3. CALCULATE SCALE FACTOR FOR ROTATION PLAYERS
        # Only sum minutes of the Top 10 (who are actually playing)
        rotation_players = team_df.head(ROTATION_SIZE)
        total_min = rotation_players['MINUTES'].sum()
        
        # Scale to exactly 240 minutes (48 mins * 5 positions)
        if total_min > 0:
            scale_factor = 240.0 / total_min
            
            # Only apply scaling to the active rotation
            cols_to_scale = ['MINUTES', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'PRA', 'PROJ_POSS']
            valid_cols = [c for c in cols_to_scale if c in team_df.columns]
            
            # Update only the top 10 rows
            team_df.iloc[:ROTATION_SIZE, team_df.columns.get_indexer(valid_cols)] *= scale_factor
            
            logger.info(f"   -> {team}: Trimmed to 10 & Scaled (Factor: {scale_factor:.2f})")
        
        normalized_rows.append(team_df)
        
    return pd.concat(normalized_rows)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def predict_all_players_today(playing_today, output_csv, season="2024-25"):
    """
    Entry point used by automated_predictions.py.
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
            logger.info(f"Merging with existing file: {output_csv}")
            existing_df = pd.read_csv(output_csv)
            combined_df = pd.concat([existing_df, new_df])
            # Keep latest prediction if duplicate
            combined_df = combined_df.drop_duplicates(subset=['PLAYER_NAME', 'TEAM_NAME'], keep='last')
        except Exception:
            combined_df = new_df
    else:
        combined_df = new_df

    # --- CRITICAL FIX: NORMALIZE BEFORE SAVING ---
    # This ensures the raw CSV used by betting sheets is already corrected
    final_df = normalize_predictions(combined_df)
    
    # Sort and Save
    final_df = final_df.sort_values(by=['TEAM_NAME', 'PLAYER_NAME'])
    final_df.to_csv(output_csv, index=False)
    
    logger.info(f"Saved Normalized Predictions to {output_csv}")
    return final_df