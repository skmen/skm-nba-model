"""
Game Schedule and Lineup Fetcher Module

Fetches today's NBA games and retrieves active roster + starting lineup information.
Provides utility functions to check if players are playing today.

Author: NBA Prediction Model
Date: 2025
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

import pandas as pd
from nba_api.stats.endpoints import scoreboardv2

from src.config import API_DELAY
from src.utils import setup_logger, DataAcquisitionError

logger = setup_logger(__name__)
class GameFetcher:
    """Fetches today's NBA games and player information."""
    
    def __init__(self):
        """Initialize the game fetcher with NBA API client."""
        pass  # Initialize if needed
    
    def get_today_games(self, date: str = None) -> Optional[pd.DataFrame]:
        """
        Fetch all NBA games scheduled for a specific date.
        
        Args:
            date: Optional date string in YYYY-MM-DD format.
        
        Returns:
            DataFrame with columns: GAME_ID, HOME_TEAM, AWAY_TEAM, GAME_STATUS
            None if error occurs
            
        Raises:
            DataAcquisitionError: If API call fails
        """
        try:
            game_date = date if date else datetime.now().strftime('%Y-%m-%d')
            logger.info(f"Fetching NBA games for {game_date}...")

            # Fetch scoreboard data for the specified date
            scoreboard_data = scoreboardv2.ScoreboardV2(game_date=game_date)
            games_df = scoreboard_data.game_header.get_data_frame()

            if games_df.empty:
                logger.info(f"No games scheduled for {game_date}")
                return None

            logger.info(f"Found {len(games_df)} game(s) for {game_date}")

            # Extract relevant columns
            result = games_df[[
                'GAME_ID',
                'HOME_TEAM_ID',
                'VISITOR_TEAM_ID',
                'GAME_STATUS_ID'
            ]].copy()

            # Rename columns for consistency
            result.rename(columns={
                'VISITOR_TEAM_ID': 'AWAY_TEAM_ID'
            }, inplace=True)

            # Add team names mapping
            result['HOME_TEAM'] = result['HOME_TEAM_ID'].apply(self._get_team_name)
            result['AWAY_TEAM'] = result['AWAY_TEAM_ID'].apply(self._get_team_name)

            logger.debug(f"Today's games:\n{result}")
            return result

        except Exception as e:
            logger.error(f"Error fetching games for {game_date}: {e}")
            raise DataAcquisitionError(f"Failed to fetch games for {game_date}: {e}")
    
    def get_team_roster(self, team_id: int) -> Optional[pd.DataFrame]:
        """
        Get roster for a specific team.
        
        Args:
            team_id: NBA team ID
            
        Returns:
            DataFrame with columns: PLAYER_ID, PLAYER_NAME, POSITION, JERSEY_NUM
            None if error occurs
        """
        try:
            from nba_api.stats.endpoints import commonteamroster
            
            logger.debug(f"Fetching roster for team_id={team_id}...")
            time.sleep(API_DELAY)
            
            roster = commonteamroster.CommonTeamRoster(team_id=team_id)
            roster_df = roster.get_data_frames()[0]
            
            if roster_df.empty:
                logger.warning(f"No roster found for team_id={team_id}")
                return None
            
            # Extract relevant columns
            result = roster_df[[
                'PLAYER_ID',
                'PLAYER',
                'POSITION',
                'NUM'
            ]].copy()

            # Rename columns for consistency
            result.rename(columns={
                'PLAYER': 'PLAYER_NAME',
                'NUM': 'JERSEY_NUM'
            }, inplace=True)
            
            logger.debug(f"Found {len(result)} players in roster for team_id={team_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching roster for team_id={team_id}: {e}")
            return None
    
    def get_team_lineups(self, game_id: str) -> Optional[Dict[str, List[Dict]]]:
        """
        Get starting lineups for a game.
        
        Args:
            game_id: NBA game ID
            
        Returns:
            Dict with 'HOME_LINEUP' and 'AWAY_LINEUP', each containing list of player info
            None if error occurs
        """
        try:
            from nba_api.stats.endpoints import boxscoretraditionalv2
            
            logger.debug(f"Fetching lineups for game_id={game_id}...")
            time.sleep(API_DELAY)
            
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            player_stats = boxscore.get_data_frames()[0]
            
            if player_stats.empty:
                logger.warning(f"No lineups found for game_id={game_id}")
                return None
            
            # Get home and away team IDs from the first two rows
            home_team_id = player_stats.iloc[0]['TEAM_ID']
            away_team_id = player_stats[player_stats['TEAM_ID'] != home_team_id].iloc[0]['TEAM_ID']

            # Separate home and away (START_POSITION will indicate starters)
            home_lineup = player_stats[
                (player_stats['TEAM_ID'] == home_team_id) &
                (player_stats['START_POSITION'].notna()) &
                (player_stats['START_POSITION'] != '')
            ][['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'START_POSITION']].to_dict('records')
            
            away_lineup = player_stats[
                (player_stats['TEAM_ID'] == away_team_id) &
                (player_stats['START_POSITION'].notna()) &
                (player_stats['START_POSITION'] != '')
            ][['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'START_POSITION']].to_dict('records')
            
            logger.debug(f"Found {len(home_lineup)} starters for home team and {len(away_lineup)} for away team")
            
            return {
                'HOME_LINEUP': home_lineup,
                'AWAY_LINEUP': away_lineup,
                'GAME_ID': game_id
            }
            
        except Exception as e:
            logger.error(f"Error fetching lineups for game_id={game_id}: {e}")
            return None
    
    def get_playing_today(
        self,
        team_name: str,
        date: str = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a team is playing today.
        
        Args:
            team_name: NBA team name (e.g., 'Los Angeles Lakers')
            date: Optional date string in DD-MM-YYYY format.
            
        Returns:
            Tuple of (is_playing, game_id)
            (False, None) if team is not playing
        """
        try:
            today_games = self.get_today_games(date)
            
            if today_games is None or today_games.empty:
                return False, None
            
            # Check if team is in any game
            matching_games = today_games[
                (today_games['HOME_TEAM'] == team_name) |
                (today_games['AWAY_TEAM'] == team_name)
            ]
            
            if matching_games.empty:
                logger.debug(f"Team {team_name} is not playing today")
                return False, None
            
            game_id = matching_games.iloc[0]['GAME_ID']
            logger.debug(f"Team {team_name} is playing today (game_id={game_id})")
            return True, game_id
            
        except Exception as e:
            logger.error(f"Error checking if {team_name} is playing: {e}")
            return False, None
    
    def is_in_starting_lineup(
        self,
        player_name: str,
        team_name: str,
        game_id: str
    ) -> bool:
        """
        Check if player is in starting lineup for today's game.
        
        Args:
            player_name: NBA player name
            team_name: NBA team name
            game_id: NBA game ID
            
        Returns:
            True if player is in starting lineup, False otherwise
        """
        try:
            lineups = self.get_team_lineups(game_id)
            
            if lineups is None:
                logger.warning(f"Could not fetch lineups for {player_name}")
                return False
            
            # Check both home and away lineups
            all_starters = lineups.get('HOME_LINEUP', []) + lineups.get('AWAY_LINEUP', [])
            
            is_starter = any(
                starter['PLAYER_NAME'].lower() == player_name.lower()
                for starter in all_starters
            )
            
            if is_starter:
                logger.debug(f"{player_name} is in starting lineup")
            else:
                logger.debug(f"{player_name} is NOT in starting lineup")
            
            return is_starter
            
        except Exception as e:
            logger.error(f"Error checking if {player_name} is in starting lineup: {e}")
            return False
    
    def is_active(
        self,
        player_name: str,
        team_id: int
    ) -> bool:
        """
        Check if player is on active roster for team.
        
        Args:
            player_name: NBA player name
            team_id: NBA team ID
            
        Returns:
            True if player is active, False otherwise
        """
        try:
            roster = self.get_team_roster(team_id)
            
            if roster is None or roster.empty:
                logger.warning(f"Could not fetch roster for team_id={team_id}")
                return False
            
            is_on_roster = any(
                row['PLAYER_NAME'].lower() == player_name.lower()
                for _, row in roster.iterrows()
            )
            
            if is_on_roster:
                logger.debug(f"{player_name} is on active roster")
            else:
                logger.debug(f"{player_name} is NOT on active roster")
            
            return is_on_roster
            
        except Exception as e:
            logger.error(f"Error checking if {player_name} is active: {e}")
            return False
    
    def get_all_playing_today(self, date: str = None) -> pd.DataFrame:
        """
        Get all players playing in today's games (starters + active roster).
        
        Args:
            date: Optional date string in DD-MM-YYYY format.
        
        Returns:
            DataFrame with columns: PLAYER_NAME, TEAM_NAME, GAME_ID, IS_STARTER
        """
        try:
            today_games = self.get_today_games(date)
            
            if today_games is None or today_games.empty:
                logger.info("No games today, returning empty result")
                return pd.DataFrame(columns=[
                    'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_NAME', 
                    'GAME_ID', 'IS_STARTER'
                ])
            
            all_players = []
            
            # Process each game
            for _, game in today_games.iterrows():
                game_id = game['GAME_ID']
                home_team_id = game['HOME_TEAM_ID']
                away_team_id = game['AWAY_TEAM_ID']
                home_team = game['HOME_TEAM']
                away_team = game['AWAY_TEAM']

                # Get starting lineups
                lineups = self.get_team_lineups(game_id)
                starters = []
                if lineups:
                    starters.extend(player['PLAYER_ID'] for player in lineups.get('HOME_LINEUP', []))
                    starters.extend(player['PLAYER_ID'] for player in lineups.get('AWAY_LINEUP', []))

                # Get full rosters
                home_roster = self.get_team_roster(home_team_id)
                away_roster = self.get_team_roster(away_team_id)
                
                # Process home team roster
                if home_roster is not None:
                    for _, player in home_roster.iterrows():
                        all_players.append({
                            'PLAYER_ID': player['PLAYER_ID'],
                            'PLAYER_NAME': player['PLAYER_NAME'],
                            'TEAM_ID': home_team_id,
                            'TEAM_NAME': home_team,
                            'GAME_ID': game_id,
                            'IS_STARTER': player['PLAYER_ID'] in starters,
                            'IS_HOME': True
                        })

                # Process away team roster
                if away_roster is not None:
                    for _, player in away_roster.iterrows():
                        all_players.append({
                            'PLAYER_ID': player['PLAYER_ID'],
                            'PLAYER_NAME': player['PLAYER_NAME'],
                            'TEAM_ID': away_team_id,
                            'TEAM_NAME': away_team,
                            'GAME_ID': game_id,
                            'IS_STARTER': player['PLAYER_ID'] in starters,
                            'IS_HOME': False
                        })

            result_df = pd.DataFrame(all_players)
            logger.info(f"Found {len(result_df)} players playing today")
            return result_df
            
        except Exception as e:
            logger.error(f"Error getting all playing today: {e}")
            return pd.DataFrame(columns=[
                'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_NAME', 
                'GAME_ID', 'IS_STARTER'
            ])
    
    @staticmethod
    def _get_team_name(team_id: int) -> str:
        """Convert team ID to team name."""
        team_mapping = {
            1610612737: 'Atlanta Hawks',
            1610612738: 'Boston Celtics',
            1610612739: 'Cleveland Cavaliers',
            1610612740: 'New Orleans Pelicans',
            1610612741: 'Chicago Bulls',
            1610612742: 'Dallas Mavericks',
            1610612743: 'Denver Nuggets',
            1610612744: 'Golden State Warriors',
            1610612745: 'Houston Rockets',
            1610612746: 'Los Angeles Clippers',
            1610612747: 'Los Angeles Lakers',
            1610612748: 'Miami Heat',
            1610612749: 'Milwaukee Bucks',
            1610612750: 'Minnesota Timberwolves',
            1610612751: 'Brooklyn Nets',
            1610612752: 'New York Knicks',
            1610612753: 'Orlando Magic',
            1610612754: 'Indiana Pacers',
            1610612755: 'Philadelphia 76ers',
            1610612756: 'Phoenix Suns',
            1610612757: 'Portland Trail Blazers',
            1610612758: 'Sacramento Kings',
            1610612759: 'San Antonio Spurs',
            1610612760: 'Oklahoma City Thunder',
            1610612761: 'Toronto Raptors',
            1610612762: 'Utah Jazz',
            1610612763: 'Memphis Grizzlies',
            1610612764: 'Washington Wizards',
            1610612765: 'Detroit Pistons',
            1610612766: 'Charlotte Hornets',
        }
        return team_mapping.get(team_id, f"Team {team_id}")
