"""
Generates Defense vs. Position (DvP) statistics.

This script fetches game logs for all players in the league, calculates the
average stats (PTS, REB, AST, STL, BLK) allowed by each team to each
position (Guard, Forward, Center), and saves the results to a CSV file.
"""

import logging
import time
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonteamroster, playergamelog
from src.config import API_DELAY, SEASONS_TO_FETCH, DATA_DIR
from src.utils import setup_logger, get_data_filepath, DataAcquisitionError

logger = setup_logger(__name__)

def get_all_players_by_position():
    """
    Fetches all players from all teams and categorizes their position.

    Returns:
        A DataFrame with PLAYER_ID, PLAYER_NAME, and a simplified POSITION ('G', 'F', 'C').
    """
    all_teams = teams.get_teams()
    player_list = []
    
    logger.info(f"Fetching rosters for all {len(all_teams)} teams...")
    for team in all_teams:
        try:
            logger.debug(f"Fetching roster for {team['full_name']}...")
            roster = commonteamroster.CommonTeamRoster(team_id=team['id'], season=SEASONS_TO_FETCH[0])
            roster_df = roster.get_data_frames()[0]
            
            for _, player in roster_df.iterrows():
                position = player['POSITION']
                if 'G' in position:
                    simple_pos = 'G'
                elif 'F' in position:
                    simple_pos = 'F'
                else:
                    simple_pos = 'C'
                
                player_list.append({
                    'PLAYER_ID': player['PLAYER_ID'],
                    'PLAYER_NAME': player['PLAYER'],
                    'POSITION': simple_pos
                })
            time.sleep(API_DELAY)
        except Exception as e:
            logger.warning(f"Could not fetch roster for {team['full_name']}: {e}")
            
    if not player_list:
        raise DataAcquisitionError("Failed to fetch any team rosters.")
        
    logger.info(f"Successfully fetched {len(player_list)} players.")
    return pd.DataFrame(player_list).drop_duplicates(subset=['PLAYER_ID'])

def get_dvp_stats(players_df):
    """
    Calculates DvP stats for all teams.

    Args:
        players_df: DataFrame of players with their positions.

    Returns:
        A DataFrame with DvP stats.
    """
    all_teams_abbr = {team['abbreviation'] for team in teams.get_teams()}
    dvp_data = []

    for index, player in players_df.iterrows():
        player_id = player['PLAYER_ID']
        player_pos = player['POSITION']
        player_name = player['PLAYER_NAME']
        
        logger.debug(f"Fetching game log for {player_name} ({player_id})...")
        try:
            gamelogs = playergamelog.PlayerGameLog(player_id=player_id, season=SEASONS_TO_FETCH[0])
            gamelogs_df = gamelogs.get_data_frames()[0]

            if gamelogs_df.empty:
                continue

            for _, game in gamelogs_df.iterrows():
                matchup = game['MATCHUP']
                opponent = matchup.split(' ')[-1]
                
                if opponent not in all_teams_abbr:
                    continue

                dvp_data.append({
                    'OPPONENT_TEAM': opponent,
                    'POSITION': player_pos,
                    'PTS': game['PTS'],
                    'REB': game['REB'],
                    'AST': game['AST'],
                    'STL': game['STL'],
                    'BLK': game['BLK'],
                })
            time.sleep(API_DELAY)
        except Exception as e:
            logger.warning(f"Could not fetch game log for {player_name} ({player_id}): {e}")
    
    if not dvp_data:
        raise DataAcquisitionError("Failed to collect any game log data.")
        
    dvp_df = pd.DataFrame(dvp_data)
    
    # Calculate the average stats allowed
    dvp_agg = dvp_df.groupby(['OPPONENT_TEAM', 'POSITION']).mean().reset_index()
    
    return dvp_agg

def main():
    """
    Main function to generate and save DvP stats.
    """
    logger.info("Starting DvP stats generation...")
    try:
        players_df = get_all_players_by_position()
        dvp_stats_df = get_dvp_stats(players_df)
        
        filepath = get_data_filepath('dvp_stats.csv')
        dvp_stats_df.to_csv(filepath, index=False)
        
        logger.info(f"Successfully generated and saved DvP stats to {filepath}")
        
    except DataAcquisitionError as e:
        logger.error(f"A data acquisition error occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
