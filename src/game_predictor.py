import sys
import os
import pandas as pd
import re
from datetime import datetime
import numpy as np

# Add project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.game_fetcher import GameFetcher
from src.data_fetcher import get_opponent_defense_metrics
from src.utils import setup_logger
from nba_api.stats.static import teams

logger = setup_logger(__name__)

def find_latest_prediction_file():
    """
    Finds the most recent prediction file in the data/predictions directory.
    Returns the path to the latest file, or None if no files are found.
    """
    search_dir = 'data/predictions'
    prediction_files = []
    file_pattern = re.compile(r'predictions_(\d{8})\.csv')

    if not os.path.isdir(search_dir):
        logger.error(f"Predictions directory not found at '{search_dir}'")
        return None

    for filename in os.listdir(search_dir):
        match = file_pattern.match(filename)
        if match:
            date_str = match.group(1)
            file_path = os.path.join(search_dir, filename)
            prediction_files.append((date_str, file_path))

    if not prediction_files:
        logger.error(f"No prediction files found in '{search_dir}' with pattern 'predictions_YYYYMMDD.csv'.")
        return None

    # Sort by date descending and return the path of the newest file
    latest_file = sorted(prediction_files, key=lambda x: x[0], reverse=True)[0]
    return latest_file[1]

def get_season_from_date(game_date: datetime):
    """Determines the NBA season string from a date (e.g., 2023-24)."""
    if game_date.month >= 10:
        return f"{game_date.year}-{(game_date.year + 1) % 100:02d}"
    else:
        return f"{game_date.year - 1}-{game_date.year % 100:02d}"

def run_game_predictions(predictions_file: str):
    """
    Analyzes a prediction sheet and combines it with game schedule data
    to produce more realistic, adjusted game-level predictions.
    """
    # 1. Parse date and determine season
    match = re.search(r'_(\d{8})\.csv$', predictions_file)
    if not match:
        logger.error(f"Could not parse date from filename: {predictions_file}")
        return

    date_str = match.group(1)
    try:
        game_date = datetime.strptime(date_str, '%Y%m%d')
        game_date_iso = game_date.strftime('%Y-%m-%d')
        season = get_season_from_date(game_date)
    except ValueError:
        logger.error(f"Invalid date format '{date_str}' in filename.")
        return

    logger.info(f"Processing predictions for date: {game_date_iso} (Season: {season})")

    # 2. Load all required data
    try:
        predictions_df = pd.read_csv(predictions_file)
        dvp_df = pd.read_csv('data/dvp_stats.csv')
        defense_metrics = get_opponent_defense_metrics(season)
        
        required_cols = ['TEAM_NAME', 'PTS', 'USAGE_RATE', 'MIN', 'POSITION_GROUP']
        if not all(col in predictions_df.columns for col in required_cols):
            logger.error(f"Prediction file must contain: {', '.join(required_cols)}")
            return
    except FileNotFoundError as e:
        logger.error(f"Required data file not found: {e}. Make sure 'dvp_stats.csv' exists.")
        return
    except Exception as e:
        logger.error(f"Error loading auxiliary data: {e}")
        return

    # 3. Fetch game schedule
    try:
        game_fetcher = GameFetcher()
        games_on_date = game_fetcher.get_today_games(date=game_date_iso)
        if games_on_date is None or games_on_date.empty:
            logger.info(f"No games found for {game_date_iso}. Exiting.")
            return
    except Exception as e:
        logger.error(f"Failed to fetch games: {e}")
        return

    # 4. Prepare data for adjustments
    team_map = {team['full_name']: team['abbreviation'] for team in teams.get_teams()}
    all_paces = [v['PACE'] for v in defense_metrics.values() if v and not np.isnan(v['PACE'])]
    league_avg_pace = np.mean(all_paces) if all_paces else 100.0

    games_dict = {
        game['HOME_TEAM']: {'OPPONENT': game['AWAY_TEAM']} for _, game in games_on_date.iterrows()
    }
    games_dict.update({
        game['AWAY_TEAM']: {'OPPONENT': game['HOME_TEAM']} for _, game in games_on_date.iterrows()
    })
    
    predictions_df['OPPONENT'] = predictions_df['TEAM_NAME'].map(lambda x: games_dict.get(x, {}).get('OPPONENT'))
    predictions_df.dropna(subset=['OPPONENT'], inplace=True)

    # 5. Apply Adjustments
    # DvP Adjustment
    merged_df = predictions_df.merge(dvp_df, left_on=['OPPONENT', 'POSITION_GROUP'], right_on=['OPP_TEAM', 'POSITION_GROUP'], how='left')
    merged_df['DVP_MULTIPLIER'].fillna(1.0, inplace=True)
    merged_df['ADJUSTED_PTS'] = merged_df['PTS'] * merged_df['DVP_MULTIPLIER']

    # Usage and Minutes Weighting
    team_stats = merged_df.groupby('TEAM_NAME').agg(TEAM_TOTAL_MIN=('MIN', 'sum')).reset_index()
    merged_df = merged_df.merge(team_stats, on='TEAM_NAME', how='left')
    merged_df['MINUTES_WEIGHT'] = (merged_df['MIN'] / merged_df['TEAM_TOTAL_MIN']) * len(merged_df[merged_df['MIN'] > 0]) # Crude team contribution
    
    # Apply weights, capping the effect
    merged_df['ADJUSTED_PTS'] *= merged_df['MINUTES_WEIGHT'].clip(0.75, 1.25)

    # 6. Aggregate scores and apply pace adjustment
    team_scores = merged_df.groupby('TEAM_NAME')['ADJUSTED_PTS'].sum().to_dict()

    print("\n" + "="*50)
    print(f"ADJUSTED GAME PREDICTIONS FOR {game_date_iso}")
    print("="*50 + "\n")

    for _, game in games_on_date.iterrows():
        home_team, away_team = game['HOME_TEAM'], game['AWAY_TEAM']
        home_abbr, away_abbr = team_map.get(home_team), team_map.get(away_team)

        home_base_score = team_scores.get(home_team, 0)
        away_base_score = team_scores.get(away_team, 0)
        
        # Pace Adjustment
        home_pace = defense_metrics.get(home_abbr, {}).get('PACE', league_avg_pace) if home_abbr else league_avg_pace
        away_pace = defense_metrics.get(away_abbr, {}).get('PACE', league_avg_pace) if away_abbr else league_avg_pace
        game_pace = (home_pace + away_pace) / 2
        pace_adjustment = game_pace / league_avg_pace if league_avg_pace > 0 else 1.0

        home_final_score = home_base_score * pace_adjustment
        away_final_score = away_base_score * pace_adjustment

        game_total = home_final_score + away_final_score
        point_spread = home_final_score - away_final_score
        winner = home_team if point_spread > 0 else away_team

        print(f"{away_team} vs {home_team}")
        print(f"Predicted Score: {away_final_score:.1f} vs {home_final_score:.1f}")
        print(f"Predicted Winner: {winner}")
        print(f"Game Total: {game_total:.1f}, Point Spread: {point_spread:+.1f} (Home)")
        print("-" * 30 + "\n")

def main():
    """Main entry point for the script."""
    latest_predictions_file = find_latest_prediction_file()
    
    if not latest_predictions_file:
        logger.error("No prediction files found in '.' or 'data/predictions/'.")
        logger.error("Please ensure a file like 'betting_sheet_predictions_YYYYMMDD.csv' exists.")
        return
        
    logger.info(f"Using latest prediction file: {latest_predictions_file}")
    run_game_predictions(latest_predictions_file)

if __name__ == "__main__":
    main()
