"""
Feature engineering module for NBA prediction pipeline.

Handles all feature creation from raw game logs including lag features,
opponent context, travel distance, rest days, usage rate, and efficiency metrics.
"""

import logging
from math import radians, cos, sin, asin, sqrt
from typing import Dict, Optional, Tuple
import os

import pandas as pd
import numpy as np

from .config import (
    ARENA_COORDINATES,
    LAG_STATS,
    LAG_WINDOW,
    DAYS_REST_MIN,
    DAYS_REST_MAX,
    DATA_DIR,
    SEASON_WEIGHTS,
)
from .utils import (
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
# OPPONENT EXTRACTION & MAPPING
# ============================================================================

def extract_opponent_name(matchup: str) -> str:
    """Extract opponent team abbreviation from MATCHUP string."""
    if not isinstance(matchup, str) or not matchup:
        return None
    parts = matchup.split()
    return parts[-1] if parts else None


def extract_opponent_stats(opponent_name: str, opponent_defense: Dict[str, Dict[str, float]]) -> Tuple[Optional[float], Optional[float]]:
    """Get opponent defense stats for a given opponent."""
    if opponent_name not in opponent_defense:
        return np.nan, np.nan
    metrics = opponent_defense[opponent_name]
    return metrics.get('DEF_RATING', np.nan), metrics.get('PACE', np.nan)


# ============================================================================
# FEATURE CREATION HELPERS
# ============================================================================

def create_lag_features(df: pd.DataFrame, lag_window: int = LAG_WINDOW) -> pd.DataFrame:
    """Create lag features (rolling averages) for specified stats."""
    try:
        logger.debug(f"Creating {lag_window}-game lag features...")
        for stat in LAG_STATS:
            if stat in df.columns:
                df[f'{stat}_L{lag_window}'] = df[stat].shift(1).rolling(window=lag_window).mean()
        return df
    except Exception as e:
        logger.error(f"Error creating lag features: {e}")
        raise FeatureEngineeringError(f"Failed to create lag features: {e}")


def create_home_away_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Create HOME_GAME binary feature."""
    df['HOME_GAME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in str(x) else 0)
    return df


def create_opponent_context_features(df: pd.DataFrame, opponent_defense: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Create opponent defense rating and pace features."""
    try:
        df['OPP_NAME'] = df['MATCHUP'].apply(extract_opponent_name)
        opponent_stats = df['OPP_NAME'].apply(lambda opp: extract_opponent_stats(opp, opponent_defense))
        df['OPP_DEF_RATING'] = opponent_stats.apply(lambda x: x[0])
        df['OPP_PACE'] = opponent_stats.apply(lambda x: x[1])
        return df
    except Exception as e:
        logger.error(f"Error creating opponent context features: {e}")
        raise FeatureEngineeringError(f"Failed to create opponent context: {e}")
    
def apply_dvp_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges local DvP (Defense vs Position) stats into the dataframe.
    If the DvP stats file is not available, it creates a neutral
    'DVP_MULTIPLIER' column with a default value of 1.0.
    """
    dvp_path = "data/dvp_stats.csv"
    
    try:
        if not os.path.exists(dvp_path):
            logger.warning(f"DvP file not found at '{dvp_path}'. Creating neutral DVP_MULTIPLIER.")
            df['DVP_MULTIPLIER'] = 1.0
            return df

        dvp_df = pd.read_csv(dvp_path)
        
        # Ensure OPP_NAME exists
        if 'OPP_NAME' not in df.columns:
            logger.warning("Cannot apply DvP context without 'OPP_NAME' column.")
            df['DVP_MULTIPLIER'] = 1.0
            return df

        # Use a default position if not present.
        if 'POSITION_GROUP' not in df.columns:
            df['POSITION_GROUP'] = 'Guard'  # Default position
            logger.debug("No 'POSITION_GROUP' found, defaulting to 'Guard' for DvP merge.")
        
        # Merge DvP stats
        df = df.merge(
            dvp_df,
            left_on=['OPP_NAME', 'POSITION_GROUP'],
            right_on=['OPP_TEAM', 'POSITION_GROUP'],
            how='left'
        )
        
        # Fill missing values with a neutral 1.0 and clean up
        df['DVP_MULTIPLIER'] = df['DVP_MULTIPLIER'].fillna(1.0)
        
        if 'OPP_TEAM' in df.columns:
            df.drop(columns=['OPP_TEAM'], inplace=True)
            
    except Exception as e:
        logger.error(f"Error applying DvP context: {e}. Creating neutral DVP_MULTIPLIER.")
        df['DVP_MULTIPLIER'] = 1.0
        
    return df


def create_travel_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Create TRAVEL_DISTANCE feature using Haversine formula."""
    try:
        df['PREV_OPP_NAME'] = df['OPP_NAME'].shift(1)
        df['TRAVEL_DISTANCE'] = 0.0
        for idx in range(1, len(df)):
            prev_opp = df.loc[idx, 'PREV_OPP_NAME']
            curr_opp = df.loc[idx, 'OPP_NAME']
            if prev_opp in ARENA_COORDINATES and curr_opp in ARENA_COORDINATES:
                lat1, lon1 = ARENA_COORDINATES[prev_opp]
                lat2, lon2 = ARENA_COORDINATES[curr_opp]
                df.loc[idx, 'TRAVEL_DISTANCE'] = haversine_distance(lat1, lon1, lat2, lon2)
        return df
    except Exception as e:
        logger.error(f"Error creating travel distance feature: {e}")
        raise FeatureEngineeringError(f"Failed to create travel distance: {e}")


def create_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create DAYS_REST and BACK_TO_BACK features."""
    try:
        df['DAYS_REST'] = df['GAME_DATE'].diff().dt.days - 1
        df['DAYS_REST'] = df['DAYS_REST'].fillna(0).clip(lower=DAYS_REST_MIN, upper=DAYS_REST_MAX)
        df['BACK_TO_BACK'] = (df['DAYS_REST'] == 0).astype(int)
        return df
    except Exception as e:
        logger.error(f"Error creating rest features: {e}")
        raise FeatureEngineeringError(f"Failed to create rest features: {e}")


def create_usage_rate_feature(df: pd.DataFrame, usage_rate: Optional[float]) -> pd.DataFrame:
    """Add USAGE_RATE feature."""
    try:
        if usage_rate is not None and not np.isnan(usage_rate):
            df['USAGE_RATE'] = usage_rate
        else:
            df['USAGE_RATE'] = df['PTS_L5'].fillna(0) / df['MIN_L5'].fillna(1)
        return df
    except Exception as e:
        logger.error(f"Error creating usage rate feature: {e}")
        raise FeatureEngineeringError(f"Failed to create usage rate: {e}")


# ============================================================================
# DATA CLEANING & PREPARATION
# ============================================================================

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe for feature engineering."""
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE').reset_index(drop=True)
    return df


def clean_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean engineered features by removing rows with missing critical values."""
    critical_columns = [
        'PTS_L5', 'MIN_L5', 'REB_L5', 'AST_L5', 'STL_L5', 'BLK_L5',
        'FG3M_L5', 'OPP_DEF_RATING', 'OPP_PACE'
    ]
    # Add PRA columns to critical list if they exist in config
    if 'PRA_L5' in df.columns:
        critical_columns.append('PRA_L5')

    existing_critical = [col for col in critical_columns if col in df.columns]
    
    rows_before = len(df)
    df_cleaned = df.dropna(subset=existing_critical).reset_index(drop=True)
    
    dropped_rows = rows_before - len(df_cleaned)
    if dropped_rows > 0:
        logger.info(f"Dropped {dropped_rows} rows with missing values")

    if df_cleaned.empty:
        raise FeatureEngineeringError("All rows dropped after cleaning - insufficient data")

    return df_cleaned


# ============================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# ============================================================================

def engineer_features(
    raw_df: pd.DataFrame,
    opponent_defense: Dict[str, Dict[str, float]],
    usage_rate: Optional[float] = None,
) -> pd.DataFrame:
    """Engineer all features from raw game log data."""
    try:
        logger.info("=" * 60)
        logger.info("ENGINEERING FEATURES")
        logger.info("=" * 60)

        validate_dataframe(raw_df)
        df = prepare_dataframe(raw_df)

        # 1. Base Features
        df = create_lag_features(df)
        df = create_home_away_feature(df)
        df = create_opponent_context_features(df, opponent_defense)
        df = create_travel_distance_feature(df)
        df = create_rest_features(df)
        df = create_usage_rate_feature(df, usage_rate)
        df = apply_dvp_context(df)

        # 2. PRA Creation (Points + Rebounds + Assists)
        # Must be done BEFORE efficiency metrics
        logger.debug("Creating PRA features...")
        if all(col in df.columns for col in ['PTS', 'REB', 'AST']):
            df['PRA'] = df['PTS'] + df['REB'] + df['AST']
            # Create PRA Lag Feature
            df['PRA_L5'] = df['PRA'].shift(1).rolling(window=LAG_WINDOW).mean()
        else:
            logger.warning("Could not create PRA feature (Missing PTS, REB, or AST)")

        # 3. Efficiency Features (Per Minute)
        logger.debug("Creating Efficiency Metrics...")
        # Avoid division by zero
        safe_min = df['MIN_L5'].replace(0, 1)
        
        if 'PTS_L5' in df.columns:
            df['PTS_PER_MIN'] = df['PTS_L5'] / safe_min
        
        if 'REB_L5' in df.columns:
            df['REB_PER_MIN'] = df['REB_L5'] / safe_min
            
        if 'PRA_L5' in df.columns:
            df['PRA_PER_MIN'] = df['PRA_L5'] / safe_min

        # 4. Clean and Weight
        df_cleaned = clean_engineered_features(df)

        if 'SEASON_ID' in df_cleaned.columns:
            df_cleaned['SAMPLE_WEIGHT'] = df_cleaned['SEASON_ID'].apply(assign_weight)
            logger.info("Sample weights assigned")
        else:
            df_cleaned['SAMPLE_WEIGHT'] = 1.0

        # Save
        output_path = get_data_filepath("engineered_player_features.parquet")
        df_cleaned.to_parquet(output_path, index=False)
        logger.info(f"Engineered features saved to '{output_path}'")

        return df_cleaned

    except FeatureEngineeringError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in feature engineering: {e}")
        raise FeatureEngineeringError(f"Feature engineering failed: {e}")