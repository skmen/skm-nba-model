import sys
import os
import pandas as pd
import re
import numpy as np
from datetime import datetime

# Add project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.game_fetcher import GameFetcher
from src.utils import setup_logger
from nba_api.stats.static import teams

logger = setup_logger(__name__)

def find_latest_prediction_file():
    """Finds the most recent prediction file."""
    search_dir = 'data/predictions/'
    file_pattern = re.compile(r'predictions_(\d{8})\.csv')

    if not os.path.isdir(search_dir):
        logger.error(f"Predictions directory not found at '{search_dir}'")
        return None

    prediction_files = []
    for filename in os.listdir(search_dir):
        match = file_pattern.match(filename)
        if match:
            date_str = match.group(1)
            prediction_files.append((date_str, os.path.join(search_dir, filename)))

    if not prediction_files:
        logger.error("No prediction files found.")
        return None

    # Sort by date descending
    return sorted(prediction_files, key=lambda x: x[0], reverse=True)[0][1]

def run_game_predictions(predictions_file: str):
    """
    Generates game-level predictions using player stats, Pace, Usage, and DvP.
    """
    logger.info(f"Processing Game Predictions from: {predictions_file}")

    # 1. Load Data
    try:
        df = pd.read_csv(predictions_file)
        
        # Check for required new columns
        required = ['TEAM_NAME', 'PTS', 'PACE', 'OPP_DvP', 'USAGE_RATE', 'MINUTES']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            logger.warning(f"Missing columns {missing}. Falling back to basic aggregation.")
            # Create dummy cols if missing to prevent crash
            for col in missing:
                if col == 'PACE': df[col] = 100.0
                elif col == 'OPP_DvP': df[col] = 1.0
                elif col == 'USAGE_RATE': df[col] = 0.2
                elif col == 'MINUTES': df[col] = 20.0
                
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        return

    # 2. Get Game Schedule (to identify matchups)
    try:
        # Extract date from filename for game fetching
        date_match = re.search(r'predictions_(\d{8})', predictions_file)
        if date_match:
            date_str = date_match.group(1)
            date_iso = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        else:
            date_iso = datetime.now().strftime('%Y-%m-%d')

        game_fetcher = GameFetcher()
        games_df = game_fetcher.get_today_games(date_iso)
        
        if games_df is None or games_df.empty:
            logger.error("No games found for this date.")
            return

    except Exception as e:
        logger.error(f"Failed to fetch schedule: {e}")
        return

    print("\n" + "="*60)
    print(f"🏀 GAME PREDICTIONS FOR {date_iso}")
    print("="*60 + "\n")

    # 3. Calculate Team-Level Metrics
    # We aggregate player stats to get team totals
    
    # Filter out low-impact players for team aggregates (optional, but cleaner)
    active_df = df[df['MINUTES'] > 5].copy()
    
    team_stats = active_df.groupby('TEAM_NAME').agg({
        'PTS': 'sum',                # Sum of Model Predictions
        'OPP_ALLOW_PTS': 'sum',      # Sum of Points Opponent Allows to these positions
        'PACE': 'mean',              # This is the Average Opponent Pace faced by players
        'OPP_DvP': 'mean',           # Average Difficulty of Matchup ( >1.0 is easy)
        'USAGE_RATE': 'sum'          # Total Usage (Should be around 1.0 or 100%)
    }).reset_index()

    # Convert to dictionary for fast lookup
    team_data = team_stats.set_index('TEAM_NAME').to_dict('index')

    # League Average Pace (Estimate if not calculated)
    league_avg_pace = active_df['PACE'].mean() if not active_df.empty else 100.0

    # 4. Process Each Game
    for _, game in games_df.iterrows():
        home_team = game['HOME_TEAM']
        away_team = game['AWAY_TEAM']

        if home_team not in team_data or away_team not in team_data:
            logger.warning(f"Skipping {away_team} @ {home_team} (Data missing)")
            continue

        h_stats = team_data[home_team]
        a_stats = team_data[away_team]

        # --- PACE CALCULATION ---
        # Player's 'PACE' column = The Pace of the Opponent they are facing.
        # So Home Team Pace = Away Players' Average 'PACE'
        # And Away Team Pace = Home Players' Average 'PACE'
        
        # However, relying on the 'PACE' column in the CSV (which is Opponent Pace)
        # means h_stats['PACE'] is actually the Away Team's Pace.
        estimated_pace = (h_stats['PACE'] + a_stats['PACE']) / 2
        pace_factor = estimated_pace / league_avg_pace

        # --- SCORE PREDICTION ---
        
        # Method A: Model Sum (Pure aggregation of your XGBoost/Ridge models)
        h_model_score = h_stats['PTS']
        a_model_score = a_stats['PTS']

        # Method B: Implied Defense Score (What opponents usually allow)
        # If available, we use OPP_ALLOW_PTS
        h_implied = h_stats.get('OPP_ALLOW_PTS', h_model_score)
        a_implied = a_stats.get('OPP_ALLOW_PTS', a_model_score)
        
        # BLEND: 70% Model, 30% Defensive History
        # We also apply the Pace Factor to the final result
        h_final = ((h_model_score * 0.7) + (h_implied * 0.3)) * pace_factor
        a_final = ((a_model_score * 0.7) + (a_implied * 0.3)) * pace_factor
        
        # Total & Spread
        total = h_final + a_final
        spread = h_final - a_final
        winner = home_team if spread > 0 else away_team

        # --- OUTPUT ---
        print(f"{away_team} ({a_final:.1f}) @ {home_team} ({h_final:.1f})")
        print(f"  > Winner: {winner} ({abs(spread):.1f})")
        print(f"  > Total:  {total:.1f}")
        print(f"  > Pace:   {estimated_pace:.1f}")
        print("-" * 40)

def main():
    latest = find_latest_prediction_file()
    if latest:
        run_game_predictions(latest)

if __name__ == "__main__":
    main()