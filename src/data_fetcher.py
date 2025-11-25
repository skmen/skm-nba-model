"""
Data acquisition module for NBA prediction pipeline.

Handles fetching player game logs, opponent stats, and player usage rates
from nba_api.
"""

import logging
import time
from typing import Dict, Optional, List

import pandas as pd
import numpy as np
from nba_api.stats.static import players
from nba_api.stats.endpoints import (
    playergamelog,
    leaguedashteamstats,
    leaguedashplayerstats,
)

from config import DATA_DIR, API_DELAY, SEASONS_TO_FETCH, GAME_TYPE_FILTER
from utils import (
    get_data_filepath,
    DataAcquisitionError,
    sanitize_filename,
    logger,
)

# ============================================================================
# GAME LOG ACQUISITION
# ============================================================================

def get_player_gamelog(player_name: str, season: str) -> Optional[pd.DataFrame]:
    """
    Fetch the game log for a specific player and season.

    Saves raw data to the data/ directory as CSV.
    Only includes regular season games.

    Args:
        player_name: Full name of the player (e.g., "James Harden")
        season: NBA season (e.g., "2023-24")

    Returns:
        DataFrame with game log data, or None if player not found

    Raises:
        DataAcquisitionError: If data acquisition fails
    """
    try:
        logger.info(f"Fetching game log for {player_name} ({season})...")

        # Find player by name
        player_info = players.find_players_by_full_name(player_name)
        if not player_info:
            logger.error(f"Player '{player_name}' not found.")
            raise DataAcquisitionError(f"Player '{player_name}' not found in nba_api")

        player_id = player_info[0]['id']
        logger.debug(f"Found player ID: {player_id}")

        # Fetch game log
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = gamelog.get_data_frames()[0]

        if df.empty:
            logger.warning(f"No game log data found for {player_name} in {season}")
            return None

        # Filter for regular season games only
        initial_count = len(df)
        df = df[df['GAME_TYPE'] == GAME_TYPE_FILTER].reset_index(drop=True)
        filtered_count = len(df)
        
        if initial_count > filtered_count:
            logger.info(f"Filtered from {initial_count} to {filtered_count} "
                       f"regular season games")

        if df.empty:
            logger.warning(f"No regular season games found for {player_name} in {season}")
            return None

        # Save to CSV
        sanitized_name = sanitize_filename(player_name)
        raw_data_filename = get_data_filepath(
            f"{sanitized_name}_{season}_raw_gamelog.csv"
        )
        df.to_csv(raw_data_filename, index=False)
        logger.info(f"Raw data saved to '{raw_data_filename}'")

        return df

    except DataAcquisitionError:
        raise
    except Exception as e:
        logger.error(f"Error acquiring game log: {e}")
        raise DataAcquisitionError(f"Failed to acquire game log: {e}")


# ============================================================================
# MULTI-SEASON DATA ACQUISITION
# ============================================================================

def get_player_gamelog_multiple_seasons(
    player_name: str,
    seasons: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch and combine game logs for a player across multiple seasons.

    Fetches regular season games from each season and combines them into
    a single DataFrame. Includes SEASON_ID column for sample weighting.

    Args:
        player_name: Full name of the player (e.g., "James Harden")
        seasons: List of seasons to fetch (e.g., ["2024-25", "2023-24", "2022-23"])
                If None, uses SEASONS_TO_FETCH from config

    Returns:
        Combined DataFrame with game logs from all seasons, or None if no data

    Raises:
        DataAcquisitionError: If data acquisition fails for all seasons
    """
    if seasons is None:
        seasons = SEASONS_TO_FETCH

    logger.info(f"Fetching game logs for {player_name} across {len(seasons)} seasons:")
    for s in seasons:
        logger.info(f"  - {s}")

    combined_df = None
    successful_seasons = 0

    for season in seasons:
        try:
            # Add rate limiting between requests
            time.sleep(API_DELAY)

            season_df = get_player_gamelog(player_name, season)

            if season_df is not None and not season_df.empty:
                # Ensure SEASON_ID column exists for weighting
                if 'SEASON_ID' not in season_df.columns:
                    season_df['SEASON_ID'] = season

                if combined_df is None:
                    combined_df = season_df.copy()
                else:
                    combined_df = pd.concat([combined_df, season_df], ignore_index=True)

                successful_seasons += 1
                logger.info(f"Successfully fetched {len(season_df)} games from {season}")
            else:
                logger.warning(f"No data for {player_name} in {season}")

        except Exception as e:
            logger.warning(f"Could not fetch {season}: {e}")
            continue

    if combined_df is None or combined_df.empty:
        raise DataAcquisitionError(
            f"Could not fetch game logs for {player_name} from any season"
        )

    logger.info(f"Combined {successful_seasons}/{len(seasons)} seasons "
                f"({len(combined_df)} total games)")

    # Sort by date to maintain chronological order
    if 'GAME_DATE' in combined_df.columns:
        combined_df = combined_df.sort_values('GAME_DATE').reset_index(drop=True)

    # Save combined data
    sanitized_name = sanitize_filename(player_name)
    combined_filename = get_data_filepath(
        f"{sanitized_name}_multi_season_gamelog.csv"
    )
    combined_df.to_csv(combined_filename, index=False)
    logger.info(f"Combined data saved to '{combined_filename}'")

    return combined_df



# ============================================================================
# OPPONENT DEFENSE METRICS
# ============================================================================

def get_opponent_defense_metrics(season: str) -> Dict[str, Dict[str, float]]:
    """
    Fetch opponent defensive metrics (DEF_RATING, PACE) from LeagueDashTeamStats.

    Args:
        season: NBA season (e.g., "2023-24")

    Returns:
        Dictionary mapping team names to {'DEF_RATING': value, 'PACE': value}

    Raises:
        DataAcquisitionError: If API call fails
    """
    try:
        logger.info(f"Fetching opponent defense metrics for {season}...")

        stats = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_nullable='Advanced',
            season=season
        )
        df = stats.get_data_frames()[0]

        opponent_defense = {}
        for _, row in df.iterrows():
            team_name = row['TEAM_NAME']
            opponent_defense[team_name] = {
                'DEF_RATING': row.get('DEF_RATING', np.nan),
                'PACE': row.get('PACE', np.nan)
            }

        logger.info(f"Retrieved defense metrics for {len(opponent_defense)} teams in {season}")
        return opponent_defense

    except Exception as e:
        logger.error(f"Error fetching opponent defense metrics: {e}")
        raise DataAcquisitionError(f"Failed to fetch opponent defense metrics: {e}")


def get_opponent_defense_metrics_multiple_seasons(
    seasons: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Fetch opponent defense metrics across multiple seasons and average them.

    Fetches defense metrics for each season and averages them to provide
    robust opponent strength estimates across multiple years.

    Args:
        seasons: List of seasons to fetch (e.g., ["2024-25", "2023-24", "2022-23"])
                If None, uses SEASONS_TO_FETCH from config

    Returns:
        Dictionary mapping team names to averaged {'DEF_RATING': value, 'PACE': value}

    Raises:
        DataAcquisitionError: If API call fails for all seasons
    """
    if seasons is None:
        seasons = SEASONS_TO_FETCH

    logger.info(f"Fetching opponent defense metrics for {len(seasons)} seasons")

    all_metrics = {}
    season_count = {}

    for season in seasons:
        try:
            time.sleep(API_DELAY)
            season_metrics = get_opponent_defense_metrics(season)

            for team, metrics in season_metrics.items():
                if team not in all_metrics:
                    all_metrics[team] = {'DEF_RATING': [], 'PACE': []}
                    season_count[team] = 0

                if not np.isnan(metrics['DEF_RATING']):
                    all_metrics[team]['DEF_RATING'].append(metrics['DEF_RATING'])
                if not np.isnan(metrics['PACE']):
                    all_metrics[team]['PACE'].append(metrics['PACE'])

                season_count[team] += 1

        except Exception as e:
            logger.warning(f"Could not fetch defense metrics for {season}: {e}")
            continue

    # Average the metrics
    averaged_metrics = {}
    for team, metrics in all_metrics.items():
        averaged_metrics[team] = {
            'DEF_RATING': np.mean(metrics['DEF_RATING']) if metrics['DEF_RATING'] else np.nan,
            'PACE': np.mean(metrics['PACE']) if metrics['PACE'] else np.nan
        }

    logger.info(f"Averaged defense metrics for {len(averaged_metrics)} teams")
    return averaged_metrics



# ============================================================================
# PLAYER USAGE RATE
# ============================================================================

def get_player_usage_rate(player_id: int, season: str) -> Optional[float]:
    """
    Fetch the player's recent Usage Rate (Last 5 games).

    Args:
        player_id: NBA player ID
        season: NBA season (e.g., "2023-24")

    Returns:
        Usage rate (percentage) or None if not available

    Raises:
        DataAcquisitionError: If API call fails
    """
    try:
        logger.debug(f"Fetching usage rate for player {player_id}...")

        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            player_id_nullable=player_id,
            season=season,
            last_n_games=5,
            measure_type_nullable='Advanced'
        )
        df = stats.get_data_frames()[0]

        if df.empty:
            logger.warning(f"No usage rate data for player {player_id}")
            return None

        usage_rate = df['USG_PCT'].iloc[0]
        logger.debug(f"Usage rate: {usage_rate:.2f}%")
        return usage_rate

    except Exception as e:
        logger.error(f"Error fetching usage rate: {e}")
        return None


# ============================================================================
# BATCH DATA ACQUISITION
# ============================================================================

def acquire_all_data(
    player_name: str,
    season: str = None,
    use_multi_season: bool = True,
) -> tuple:
    """
    Acquire all required data for the pipeline.

    Fetches player game logs from multiple seasons, opponent defense metrics,
    and player usage rate.

    Args:
        player_name: Full name of the player
        season: NBA season (deprecated - kept for backward compatibility)
        use_multi_season: If True, fetch from multiple seasons. If False, use single season.

    Returns:
        Tuple of (game_log_df, opponent_defense_dict, player_id, usage_rate)

    Raises:
        DataAcquisitionError: If any data acquisition fails
    """
    logger.info("=" * 60)
    logger.info("ACQUIRING DATA")
    logger.info("=" * 60)

    # Use multi-season approach by default
    if use_multi_season:
        logger.info("Multi-season acquisition mode: ON")
        game_log_df = get_player_gamelog_multiple_seasons(player_name)
        opponent_defense = get_opponent_defense_metrics_multiple_seasons()
    else:
        # Fallback to single season if specified
        if season is None:
            season = SEASONS_TO_FETCH[0]
        logger.info(f"Single-season acquisition mode: {season}")
        game_log_df = get_player_gamelog(player_name, season)
        opponent_defense = get_opponent_defense_metrics(season)

    if game_log_df is None:
        raise DataAcquisitionError(f"Could not fetch game log for {player_name}")

    # Get player ID for usage rate fetch
    player_info = players.find_players_by_full_name(player_name)
    player_id = player_info[0]['id']

    # Rate limiting
    time.sleep(API_DELAY)

    # Get player usage rate (from most recent season)
    recent_season = SEASONS_TO_FETCH[0]
    usage_rate = get_player_usage_rate(player_id, recent_season)

    logger.info("Data acquisition complete!")

    return game_log_df, opponent_defense, player_id, usage_rate

