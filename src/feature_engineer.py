"""
Feature engineering module for NBA prediction pipeline.

Handles all feature creation from raw game logs including lag features,
opponent context, travel distance, rest days, and usage rate.
"""

import logging
from math import radians, cos, sin, asin, sqrt
from typing import Dict, Optional, Tuple

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

    Recent seasons are weighted higher to prioritize more recent player
    performance patterns and playstyle evolution.

    Args:
        season: NBA season string (e.g., "2024-25", "2023-24")

    Returns:
        Sample weight (float) between 0.0 and 1.0
        
    Examples:
        >>> assign_weight("2024-25")
        1.0
        >>> assign_weight("2023-24")
        0.8
        >>> assign_weight("2022-23")
        0.5
        >>> assign_weight("2021-22")
        0.2
    """
    if season in SEASON_WEIGHTS:
        return SEASON_WEIGHTS[season]
    return 0.2  # Default weight for ancient history


# ============================================================================
# DISTANCE CALCULATION
# ============================================================================

def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calculate the great circle distance between two points on Earth.

    Uses the Haversine formula to compute the shortest distance between
    two points specified in decimal degrees.

    Args:
        lat1, lon1: Latitude and longitude of first point (degrees)
        lat2, lon2: Latitude and longitude of second point (degrees)

    Returns:
        Distance in miles
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    # Radius of earth in miles
    r = 3959
    return c * r


# ============================================================================
# OPPONENT EXTRACTION & MAPPING
# ============================================================================

def extract_opponent_name(matchup: str) -> str:
    """
    Extract opponent team abbreviation from MATCHUP string.

    Examples:
        "LAL vs. BOS" → "BOS"
        "GSW @ LAC" → "LAC"

    Args:
        matchup: MATCHUP string from game log

    Returns:
        Opponent team abbreviation (last word)
    """
    if not isinstance(matchup, str) or not matchup:
        return None

    parts = matchup.split()
    return parts[-1] if parts else None


def extract_opponent_stats(
    opponent_name: str,
    opponent_defense: Dict[str, Dict[str, float]],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Get opponent defense stats for a given opponent.

    Args:
        opponent_name: Opponent team abbreviation
        opponent_defense: Dictionary of opponent metrics

    Returns:
        Tuple of (DEF_RATING, PACE) or (np.nan, np.nan) if not found
    """
    if opponent_name not in opponent_defense:
        return np.nan, np.nan

    metrics = opponent_defense[opponent_name]
    return metrics.get('DEF_RATING', np.nan), metrics.get('PACE', np.nan)


# ============================================================================
# LAG FEATURE CREATION
# ============================================================================

def create_lag_features(df: pd.DataFrame, lag_window: int = LAG_WINDOW) -> pd.DataFrame:
    """
    Create lag features (rolling averages) for specified stats.

    Args:
        df: DataFrame with game logs
        lag_window: Number of games for rolling average (default: 5)

    Returns:
        DataFrame with new lag feature columns

    Raises:
        FeatureEngineeringError: If feature creation fails
    """
    try:
        logger.debug(f"Creating {lag_window}-game lag features for {len(LAG_STATS)} stats...")

        for stat in LAG_STATS:
            if stat not in df.columns:
                logger.warning(f"Stat '{stat}' not found in dataframe")
                continue

            # Create lag feature: shift by 1, then rolling average
            df[f'{stat}_L{lag_window}'] = (
                df[stat].shift(1).rolling(window=lag_window).mean()
            )

        logger.debug("Lag features created successfully")
        return df

    except Exception as e:
        logger.error(f"Error creating lag features: {e}")
        raise FeatureEngineeringError(f"Failed to create lag features: {e}")


# ============================================================================
# HOME/AWAY FEATURE
# ============================================================================

def create_home_away_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create HOME_GAME binary feature.

    Args:
        df: DataFrame with MATCHUP column

    Returns:
        DataFrame with HOME_GAME feature (1=Home, 0=Away)
    """
    logger.debug("Creating HOME_GAME feature...")

    df['HOME_GAME'] = df['MATCHUP'].apply(
        lambda x: 1 if 'vs.' in str(x) else 0
    )

    return df


# ============================================================================
# OPPONENT CONTEXT FEATURES
# ============================================================================

def create_opponent_context_features(
    df: pd.DataFrame,
    opponent_defense: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Create opponent defense rating and pace features.

    Args:
        df: DataFrame with MATCHUP column
        opponent_defense: Dictionary of opponent stats

    Returns:
        DataFrame with OPP_DEF_RATING and OPP_PACE features
    """
    try:
        logger.debug("Creating opponent context features...")

        # Extract opponent names
        df['OPP_NAME'] = df['MATCHUP'].apply(extract_opponent_name)

        # Extract opponent stats
        opponent_stats = df['OPP_NAME'].apply(
            lambda opp: extract_opponent_stats(opp, opponent_defense)
        )

        df['OPP_DEF_RATING'] = opponent_stats.apply(lambda x: x[0])
        df['OPP_PACE'] = opponent_stats.apply(lambda x: x[1])

        logger.debug("Opponent context features created")
        return df

    except Exception as e:
        logger.error(f"Error creating opponent context features: {e}")
        raise FeatureEngineeringError(f"Failed to create opponent context: {e}")


# ============================================================================
# TRAVEL DISTANCE FEATURE
# ============================================================================

def create_travel_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create TRAVEL_DISTANCE feature using Haversine formula.

    Calculates distance between consecutive game opponents based on arena
    coordinates. First game has distance = 0.

    Args:
        df: DataFrame with OPP_NAME column

    Returns:
        DataFrame with TRAVEL_DISTANCE feature (in miles)
    """
    try:
        logger.debug("Creating TRAVEL_DISTANCE feature...")

        df['PREV_OPP_NAME'] = df['OPP_NAME'].shift(1)
        df['TRAVEL_DISTANCE'] = 0.0

        for idx in range(1, len(df)):
            prev_opp = df.loc[idx, 'PREV_OPP_NAME']
            curr_opp = df.loc[idx, 'OPP_NAME']

            # Check if both opponents have coordinates
            if (prev_opp in ARENA_COORDINATES and
                    curr_opp in ARENA_COORDINATES):
                lat1, lon1 = ARENA_COORDINATES[prev_opp]
                lat2, lon2 = ARENA_COORDINATES[curr_opp]
                distance = haversine_distance(lat1, lon1, lat2, lon2)
                df.loc[idx, 'TRAVEL_DISTANCE'] = distance

        logger.debug("TRAVEL_DISTANCE feature created")
        return df

    except Exception as e:
        logger.error(f"Error creating travel distance feature: {e}")
        raise FeatureEngineeringError(f"Failed to create travel distance: {e}")


# ============================================================================
# REST DAYS & BACK-TO-BACK FEATURES
# ============================================================================

def create_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create DAYS_REST and BACK_TO_BACK features.

    Args:
        df: DataFrame with GAME_DATE column

    Returns:
        DataFrame with DAYS_REST and BACK_TO_BACK features
    """
    try:
        logger.debug("Creating rest features...")

        # Calculate days between games
        df['DAYS_REST'] = df['GAME_DATE'].diff().dt.days - 1

        # Handle first game (no rest data)
        df['DAYS_REST'] = df['DAYS_REST'].fillna(0)

        # Cap at max value
        df['DAYS_REST'] = df['DAYS_REST'].clip(
            lower=DAYS_REST_MIN,
            upper=DAYS_REST_MAX
        )

        # Create back-to-back flag
        df['BACK_TO_BACK'] = (df['DAYS_REST'] == 0).astype(int)

        logger.debug("Rest features created")
        return df

    except Exception as e:
        logger.error(f"Error creating rest features: {e}")
        raise FeatureEngineeringError(f"Failed to create rest features: {e}")


# ============================================================================
# USAGE RATE FEATURE
# ============================================================================

def create_usage_rate_feature(
    df: pd.DataFrame,
    usage_rate: Optional[float],
) -> pd.DataFrame:
    """
    Add USAGE_RATE feature to DataFrame.

    If usage_rate is provided, uses that value. Otherwise, computes it
    from basic stats (PTS_L5 / MIN_L5).

    Args:
        df: DataFrame with lag features
        usage_rate: Player's usage rate from API, or None

    Returns:
        DataFrame with USAGE_RATE feature
    """
    try:
        logger.debug("Creating USAGE_RATE feature...")

        if usage_rate is not None and not np.isnan(usage_rate):
            df['USAGE_RATE'] = usage_rate
            logger.debug(f"Using API usage rate: {usage_rate:.2f}%")
        else:
            # Fallback: compute from basic stats
            df['USAGE_RATE'] = (
                df['PTS_L5'].fillna(0) / df['MIN_L5'].fillna(1)
            )
            logger.debug("Using computed usage rate from basic stats")

        return df

    except Exception as e:
        logger.error(f"Error creating usage rate feature: {e}")
        raise FeatureEngineeringError(f"Failed to create usage rate: {e}")


# ============================================================================
# DATA CLEANING & PREPARATION
# ============================================================================

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for feature engineering.

    Args:
        df: Raw game log DataFrame

    Returns:
        Prepared DataFrame (sorted by date)
    """
    logger.debug("Preparing dataframe...")

    # Parse dates
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    # Sort by date
    df = df.sort_values('GAME_DATE').reset_index(drop=True)

    return df


def clean_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean engineered features by removing rows with missing critical values.

    Args:
        df: DataFrame with engineered features

    Returns:
        Cleaned DataFrame

    Raises:
        FeatureEngineeringError: If all rows are dropped
    """
    logger.debug("Cleaning engineered features...")

    # Critical features that must be non-null
    critical_columns = [
        'PTS_L5', 'MIN_L5', 'REB_L5', 'AST_L5', 'STL_L5', 'BLK_L5',
        'FG3M_L5', 'OPP_DEF_RATING', 'OPP_PACE'
    ]

    # Find columns that exist in dataframe
    existing_critical = [col for col in critical_columns if col in df.columns]

    # Drop rows with any missing critical values
    rows_before = len(df)
    df_cleaned = df.dropna(subset=existing_critical).reset_index(drop=True)
    rows_after = len(df_cleaned)

    dropped_rows = rows_before - rows_after
    logger.info(f"Dropped {dropped_rows} rows with missing values "
                f"({rows_after} rows remaining)")

    if df_cleaned.empty:
        raise FeatureEngineeringError(
            "All rows dropped after cleaning - insufficient data"
        )

    return df_cleaned


# ============================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# ============================================================================

def engineer_features(
    raw_df: pd.DataFrame,
    opponent_defense: Dict[str, Dict[str, float]],
    usage_rate: Optional[float] = None,
) -> pd.DataFrame:
    """
    Engineer all features from raw game log data.

    Creates:
    - Lag features (5-game rolling averages)
    - Home/away indicator
    - Opponent context (DEF_RATING, PACE)
    - Travel distance
    - Rest days and back-to-back flag
    - Usage rate

    Args:
        raw_df: Raw game log DataFrame
        opponent_defense: Dictionary of opponent stats
        usage_rate: Player's usage rate (optional)

    Returns:
        Engineered DataFrame with all features

    Raises:
        FeatureEngineeringError: If feature engineering fails
    """
    try:
        logger.info("=" * 60)
        logger.info("ENGINEERING FEATURES")
        logger.info("=" * 60)

        # Validate input
        validate_dataframe(raw_df)

        # Prepare dataframe
        df = prepare_dataframe(raw_df)

        # Create features in logical order
        df = create_lag_features(df)
        df = create_home_away_feature(df)
        df = create_opponent_context_features(df, opponent_defense)
        df = create_travel_distance_feature(df)
        df = create_rest_features(df)
        df = create_usage_rate_feature(df, usage_rate)

        # Clean engineered features
        df_cleaned = clean_engineered_features(df)

        # Apply sample weights based on season (data decay)
        if 'SEASON_ID' in df_cleaned.columns:
            df_cleaned['SAMPLE_WEIGHT'] = df_cleaned['SEASON_ID'].apply(assign_weight)
            logger.info("Sample weights assigned based on season recency")
        else:
            df_cleaned['SAMPLE_WEIGHT'] = 1.0
            logger.debug("SEASON_ID not found; using uniform sample weights")

        # Save to parquet
        output_path = get_data_filepath("engineered_player_features.parquet")
        df_cleaned.to_parquet(output_path, index=False)
        logger.info(f"Engineered features saved to '{output_path}'")

        logger.info("Feature engineering complete!")

        return df_cleaned

    except FeatureEngineeringError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in feature engineering: {e}")
        raise FeatureEngineeringError(f"Feature engineering failed: {e}")
