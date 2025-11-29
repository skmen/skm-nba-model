"""
Feature engineering module for NBA prediction pipeline.

Handles all feature creation from raw game logs including lag features,
EWMA, volatility, home/away splits, opponent context, travel distance, 
rest days, usage rate, and efficiency metrics.
"""

import logging
from math import radians, cos, sin, asin, sqrt
from typing import Dict, Optional, Tuple
import os

import pandas as pd
import numpy as np

from src.config import (
    ARENA_COORDINATES,
    LAG_STATS,
    LAG_WINDOW,
    DAYS_REST_MIN,
    DAYS_REST_MAX,
    DATA_DIR,
    SEASON_WEIGHTS,
)
from src.utils import (
    get_data_filepath,
    FeatureEngineeringError,
    validate_dataframe,
    logger,
)

# ============================================================================
# SAMPLE WEIGHTING (Data Decay)
# ============================================================================

def assign_weight(season: str) -> float:
    """
    Assign sample weight to a season based on recency (data decay).
    Handles both 'YYYY-YY' format and NBA API '2YYYY' format.
    """
    season_str = str(season)

    # CASE 1: Direct Match (e.g., "2024-25")
    if season_str in SEASON_WEIGHTS:
        return SEASON_WEIGHTS[season_str]
    
    # CASE 2: Handle NBA API SEASON_ID format (e.g., "22024" -> "2024-25")
    if len(season_str) == 5 and season_str.startswith('2'):
        try:
            start_year = int(season_str[1:])
            next_year_suffix = (start_year + 1) % 100
            formatted_season = f"{start_year}-{next_year_suffix:02d}"
            
            if formatted_season in SEASON_WEIGHTS:
                return SEASON_WEIGHTS[formatted_season]
        except ValueError:
            pass
   
    return 0.2  # Default weight for ancient history


# ============================================================================
# DISTANCE CALCULATION
# ============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on Earth."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 3959 # Radius of earth in miles
    return c * r


# ============================================================================
# OPPONENT EXTRACTION
# ============================================================================

def extract_opponent_name(matchup: str) -> str:
    """Extract opponent team abbreviation from MATCHUP string."""
    if not isinstance(matchup, str) or not matchup:
        return None
    # MATCHUP formats: "GSW vs. LAL" or "GSW @ LAL"
    try:
        if 'vs.' in matchup: return matchup.split('vs.')[1].strip()
        if '@' in matchup: return matchup.split('@')[1].strip()
        parts = matchup.split()
        return parts[-1] if parts else None
    except:
        return None


# ============================================================================
# FEATURE CREATION HELPERS
# ============================================================================

def create_advanced_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates sophisticated form features:
    1. Standard Lag-5 Rolling Averages
    2. Exponential Weighted Moving Averages (EWMA)
    3. Volatility (Standard Deviation)
    4. Home/Away Performance Splits
    """
    try:
        logger.debug("Creating advanced form features (EWMA, Volatility, Splits)...")
        
        # 1. Standard Rolling Averages (Lag-5)
        # Shift(1) prevents data leakage (using today's stats to predict today)
        for stat in LAG_STATS:
            if stat in df.columns:
                df[f'{stat}_L5'] = df[stat].shift(1).rolling(window=LAG_WINDOW, min_periods=1).mean()
        # 2. Exponential Weighted Moving Average (EWMA)
        # Responds faster to recent changes in form than simple rolling avg
        ewm_stats = ['PTS', 'REB', 'AST']
        for stat in ewm_stats:
            if stat in df.columns:
                df[f'{stat}_EWMA_5'] = df[stat].shift(1).ewm(span=5, adjust=False).mean()

        # 3. Volatility (Risk Metric)
        # Standard deviation over last 10 games
        vol_stats = ['PTS', 'REB']
        for stat in vol_stats:
            if stat in df.columns:
                df[f'{stat}_STD_10'] = df[stat].shift(1).rolling(window=10, min_periods=3).std()

        # 4. Home/Away Splits (Contextual Form)
        # "How does this player perform at Home vs Away over their last 10 games?"
        if 'HOME_GAME' in df.columns and 'PTS' in df.columns:
            is_home = df['HOME_GAME'] == 1
            
            # Create streams for Home and Away games
            home_pts = df.loc[is_home, 'PTS'].shift(1).rolling(window=10, min_periods=1).mean()
            away_pts = df.loc[~is_home, 'PTS'].shift(1).rolling(window=10, min_periods=1).mean()
            
            # Map back to main dataframe
            df['PTS_L10_HOME'] = np.nan
            df['PTS_L10_AWAY'] = np.nan
            
            df.loc[is_home, 'PTS_L10_HOME'] = home_pts
            df.loc[~is_home, 'PTS_L10_AWAY'] = away_pts
            
            # Forward fill to ensure we have a value even if switching venues
            # (Use the last known average for that venue type)
            df['PTS_L10_HOME'] = df['PTS_L10_HOME'].ffill()
            df['PTS_L10_AWAY'] = df['PTS_L10_AWAY'].ffill()
            
            # Fill remaining NaNs (e.g. start of season) with general L5
            if 'PTS_L5' in df.columns:
                df['PTS_L10_HOME'] = df['PTS_L10_HOME'].fillna(df['PTS_L5'])
                df['PTS_L10_AWAY'] = df['PTS_L10_AWAY'].fillna(df['PTS_L5'])

        return df

    except Exception as e:
        logger.error(f"Error creating advanced form features: {e}")
        # Fallback: Ensure critical Lag columns exist at minimum
        for stat in LAG_STATS:
            if stat in df.columns and f'{stat}_L{LAG_WINDOW}' not in df.columns:
                 df[f'{stat}_L{LAG_WINDOW}'] = df[stat].shift(1).rolling(window=LAG_WINDOW).mean()
        return df


def create_home_away_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Create HOME_GAME binary feature."""
    # Logic: 'vs.' usually implies Home, '@' implies Away in NBA data
    df['HOME_GAME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in str(x) else 0)
    return df


def create_granular_opponent_features(df: pd.DataFrame, opponent_defense: Optional[Dict] = None) -> pd.DataFrame:
    """
    Maps Granular Opponent Defense Metrics (Pts, Reb, Ast multipliers).
    Replaces the old 'OPP_DEF_RATING' single-metric approach.
    """
    try:
        df['OPP_NAME'] = df['MATCHUP'].apply(extract_opponent_name)
        
        # Default neutral values
        defaults = {
            'OPP_DEF_RATING': 110.0,
            'OPP_PACE': 100.0,
            'OPP_PTS_MULT': 1.0,
            'OPP_REB_MULT': 1.0,
            'OPP_AST_MULT': 1.0
        }

        if opponent_defense is None:
            # Apply defaults if no data
            for col, val in defaults.items():
                df[col] = val
            return df

        # Helper to map a row to the dict
        def get_opp_stats(opp_name):
            if opp_name in opponent_defense:
                stats = opponent_defense[opp_name]
                return pd.Series([
                    stats.get('OPP_DEF_RATING', defaults['OPP_DEF_RATING']),
                    stats.get('OPP_PACE', defaults['OPP_PACE']),
                    stats.get('OPP_PTS_MULT', defaults['OPP_PTS_MULT']),
                    stats.get('OPP_REB_MULT', defaults['OPP_REB_MULT']),
                    stats.get('OPP_AST_MULT', defaults['OPP_AST_MULT'])
                ])
            else:
                return pd.Series(list(defaults.values()))

        # Apply mapping
        cols = ['OPP_DEF_RATING', 'OPP_PACE', 'OPP_PTS_MULT', 'OPP_REB_MULT', 'OPP_AST_MULT']
        df[cols] = df['OPP_NAME'].apply(get_opp_stats)
        
        return df

    except Exception as e:
        logger.error(f"Error creating granular opponent features: {e}")
        # Fallback defaults
        df['OPP_DEF_RATING'] = 110.0
        df['OPP_PACE'] = 100.0
        return df


def create_travel_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Create TRAVEL_DISTANCE feature using Haversine formula."""
    try:
        df['PREV_OPP_NAME'] = df['OPP_NAME'].shift(1)
        df['TRAVEL_DISTANCE'] = 0.0
        
        # Optimization: Create a mapping of Arena Coords for fast lookup
        # (Assuming ARENA_COORDINATES is imported from config)
        
        for idx in range(1, len(df)):
            # We need to know where they played PREVIOUSLY vs CURRENTLY
            # Logic: 
            # If Prev Game was HOME -> Prev Loc is Home City
            # If Prev Game was AWAY -> Prev Loc is Prev Opponent City
            # If Curr Game is HOME -> Curr Loc is Home City
            # If Curr Game is AWAY -> Curr Loc is Curr Opponent City
            
            # Simplified Logic (Tracking just opponents):
            # This is an approximation. A robust system tracks the team's actual schedule.
            # Using the simpler logic from the original file for stability:
            prev_opp = df.loc[idx, 'PREV_OPP_NAME']
            curr_opp = df.loc[idx, 'OPP_NAME']
            
            if prev_opp in ARENA_COORDINATES and curr_opp in ARENA_COORDINATES:
                lat1, lon1 = ARENA_COORDINATES[prev_opp]
                lat2, lon2 = ARENA_COORDINATES[curr_opp]
                df.loc[idx, 'TRAVEL_DISTANCE'] = haversine_distance(lat1, lon1, lat2, lon2)
                
        return df
    except Exception as e:
        logger.error(f"Error creating travel distance feature: {e}")
        df['TRAVEL_DISTANCE'] = 0.0
        return df


def create_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create DAYS_REST and BACK_TO_BACK features."""
    try:
        if 'GAME_DATE' in df.columns:
            df['DAYS_REST'] = df['GAME_DATE'].diff().dt.days - 1
            df['DAYS_REST'] = df['DAYS_REST'].fillna(3).clip(lower=DAYS_REST_MIN, upper=DAYS_REST_MAX)
            df['BACK_TO_BACK'] = (df['DAYS_REST'] == 0).astype(int)
        else:
            df['DAYS_REST'] = 3
            df['BACK_TO_BACK'] = 0
        return df
    except Exception as e:
        logger.error(f"Error creating rest features: {e}")
        df['DAYS_REST'] = 3
        return df


def create_usage_rate_feature(df: pd.DataFrame, usage_rate: Optional[float]) -> pd.DataFrame:
    """Add USAGE_RATE feature."""
    try:
        if usage_rate is not None and not np.isnan(usage_rate):
            df['USAGE_RATE'] = usage_rate
        else:
            # Fallback: Calculate approximate usage from L5
            # USG% approx = (FGA + 0.44*FTA + TOV) / (TmFGA + ...)
            # Simplified: Just fill with L5 average if available or default
            if 'USG_PCT_L5' in df.columns:
                df['USAGE_RATE'] = df['USG_PCT_L5']
            else:
                df['USAGE_RATE'] = 20.0 # League average
        return df
    except Exception as e:
        logger.error(f"Error creating usage rate feature: {e}")
        df['USAGE_RATE'] = 20.0
        return df
    
def add_advanced_stats(df: pd.DataFrame, advanced_cache: Dict) -> pd.DataFrame:
    """
    Merges season-long advanced stats into the daily dataframe.
    """
    from src.config import ADVANCED_FEATURES
    
    # Initialize cols
    for col in ADVANCED_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    if not advanced_cache:
        return df

    # Map values
    # In a real pipeline, this would merge on PLAYER_ID + SEASON
    # Here we assume cache is for the current season being processed
    
    # We can iterate and map, or convert cache to DF and merge. 
    # Since this function is called per-player-dataframe in the pipeline:
    if 'PLAYER_ID' in df.columns:
        player_id = df['PLAYER_ID'].iloc[0]
        if player_id in advanced_cache:
            stats = advanced_cache[player_id]
            for col in ADVANCED_FEATURES:
                # Handle alias mapping if needed
                key = col
                if col == 'SAST' and 'SECONDARY_AST' in stats: key = 'SECONDARY_AST'
                
                if key in stats:
                    df[col] = stats[key]
    
    return df

def create_possession_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Estimates Possessions: FGA + 0.44*FTA - OREB + TOV
    2. Creates Per-100-Possession Target Columns
    """
    try:
        # Ensure we have required columns (fill 0 to avoid crash)
        required = ['FGA', 'FTA', 'OREB', 'TOV', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'PRA']
        for col in required:
            if col not in df.columns:
                df[col] = 0.0
        
        # 1. Calculate Possessions (Basic NBA Formula)
        df['POSS_EST'] = (df['FGA'] + (0.44 * df['FTA']) - df['OREB'] + df['TOV'])
        df['POSS_EST'] = df['POSS_EST'].replace(0, 1) # Avoid div/0

        # 2. Create Rate Targets
        targets_map = {
            'PTS': 'PTS_PER_100',
            'REB': 'REB_PER_100',
            'AST': 'AST_PER_100',
            'STL': 'STL_PER_100',
            'BLK': 'BLK_PER_100',
            'PRA': 'PRA_PER_100',
            'FG3M': 'FG3M_PER_100'
        }
        
        for raw, rate in targets_map.items():
            df[rate] = (df[raw] / df['POSS_EST']) * 100
            
        return df

    except Exception as e:
        logger.warning(f"Error creating possession metrics: {e}")
        return df


# ============================================================================
# DATA CLEANING & PREPARATION
# ============================================================================

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe for feature engineering."""
    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        # Sort ASCENDING for rolling calculations
        df = df.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)
    return df


def clean_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean engineered features by removing rows with missing critical values."""
    # Critical columns that MUST exist for the model to work
    critical_columns = [
        'PTS_L5', 'MIN_L5', 'OPP_DEF_RATING', 'OPP_PACE'
    ]
    
    existing_critical = [col for col in critical_columns if col in df.columns]
    
    rows_before = len(df)
    df_cleaned = df.dropna(subset=existing_critical).reset_index(drop=True)
    
    dropped_rows = rows_before - len(df_cleaned)
    if dropped_rows > 0:
        logger.debug(f"Dropped {dropped_rows} rows with missing history (Warm-up period)")

    return df_cleaned


# ============================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# ============================================================================

def engineer_features(
    raw_df: pd.DataFrame,
    opponent_defense: Dict[str, Dict[str, float]],
    usage_rate: Optional[float] = None,
    position_group: str = "G",
    advanced_stats: Dict=None
) -> pd.DataFrame:
    """
    Engineer all features from raw game log data.
    Orchestrates the entire feature generation pipeline.
    """
    try:
        # 1. Preparation
        df = prepare_dataframe(raw_df)
        if df.empty: return df

        # 2. Contextual Features
        df = create_home_away_feature(df)
        df = create_rest_features(df)
        
        # 3. Opponent Context (New Granular Logic)
        df = create_granular_opponent_features(df, opponent_defense)
        
        # 4. Travel Distance
        df = create_travel_distance_feature(df)
        
        # 5. Advanced Form (Lags, EWMA, Volatility, Splits)
        df = create_advanced_form_features(df)
        
        # 6. Roster Context
        df = create_usage_rate_feature(df, usage_rate)
        df = add_advanced_stats(df, advanced_stats)
        
        # 7. PRA Creation (Points + Rebounds + Assists)
        if {'PTS', 'REB', 'AST'}.issubset(df.columns):
            df['PRA'] = df['PTS'] + df['REB'] + df['AST']

        # 8. Possession Normalization (Targets)
        df = create_possession_metrics(df)

        # 9. Lagged PRA (Robust method)
        if {'PTS_L5', 'REB_L5', 'AST_L5'}.issubset(df.columns):
            df['PRA_L5'] = df['PTS_L5'] + df['REB_L5'] + df['AST_L5']
        
        # 10. Efficiency Features (Per Minute)
        df['PTS_PER_MIN'] = np.nan
        df['REB_PER_MIN'] = np.nan
        df['PRA_PER_MIN'] = np.nan

        if 'MIN_L5' in df.columns:
            # Replace 0 minutes with 1 to avoid Inf
            safe_min = df['MIN_L5'].replace(0, 1)
            if 'PTS_L5' in df.columns: df['PTS_PER_MIN'] = df['PTS_L5'] / safe_min
            if 'REB_L5' in df.columns: df['REB_PER_MIN'] = df['REB_L5'] / safe_min
            if 'PRA_L5' in df.columns: df['PRA_PER_MIN'] = df['PRA_L5'] / safe_min

        # 11. Clean and Weight
        df_cleaned = clean_engineered_features(df)

        if 'SEASON_ID' in df_cleaned.columns:
            df_cleaned['SAMPLE_WEIGHT'] = df_cleaned['SEASON_ID'].apply(assign_weight)
        else:
            df_cleaned['SAMPLE_WEIGHT'] = 1.0

        return df_cleaned

    except Exception as e:
        logger.error(f"Unexpected error in feature engineering: {e}")
        raise FeatureEngineeringError(f"Feature engineering failed: {e}")