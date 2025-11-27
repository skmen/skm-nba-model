import sys
import os
import pandas as pd
import re
import numpy as np
import json
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
    Generates game-level predictions using player stats normalized to 240 minutes.
    """
    logger.info(f"Processing Game Predictions from: {predictions_file}")

    # 1. Load Data
    try:
        df = pd.read_csv(predictions_file)
        
        # Check for required new columns
        required = ['TEAM_NAME', 'PTS', 'PACE', 'MINUTES']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            logger.warning(f"Missing columns {missing}. Totals may be inaccurate.")
            # Create dummy cols if missing
            for col in missing:
                if col == 'PACE': df[col] = 100.0
                elif col == 'MINUTES': df[col] = 20.0
                
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        return

    # 2. Get Game Schedule
    try:
        date_match = re.search(r'predictions_(\d{8})', predictions_file)
        if date_match:
            date_str_raw = date_match.group(1)
            date_iso = f"{date_str_raw[:4]}-{date_str_raw[4:6]}-{date_str_raw[6:]}"
        else:
            date_iso = datetime.now().strftime('%Y-%m-%d')
            date_str_raw = datetime.now().strftime('%Y%m%d')

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

    # 3. Calculate Team-Level Metrics (Normalized)
    # Filter out invalid rows first
    active_df = df[df['MINUTES'] > 0].copy()
    
    team_stats = active_df.groupby('TEAM_NAME').agg({
        'PTS': 'sum',
        'MINUTES': 'sum',
        'PACE': 'mean' # Average Opponent Pace
    }).reset_index()
    
    # --- NORMALIZATION LOGIC ---
    # We calculate Points Per Minute (PPM) and scale to 240 team minutes.
    # Formula: (Total Predicted PTS / Total Predicted Minutes) * 240
    team_stats['NORMALIZED_PTS'] = (team_stats['PTS'] / team_stats['MINUTES']) * 240
    
    team_data = team_stats.set_index('TEAM_NAME').to_dict('index')
    league_avg_pace = active_df['PACE'].mean() if not active_df.empty else 100.0

    game_predictions_output = []

    # 4. Process Each Game
    for _, game in games_df.iterrows():
        home_team = game['HOME_TEAM']
        away_team = game['AWAY_TEAM']

        if home_team not in team_data or away_team not in team_data:
            logger.warning(f"Skipping {away_team} @ {home_team} (Data missing)")
            continue

        h_stats = team_data[home_team]
        a_stats = team_data[away_team]

        # Pace Calculation
        # The 'PACE' column represents the PACE of the OPPONENT the player is facing.
        # So Home Players' Pace = Away Team's Pace.
        estimated_pace = (h_stats['PACE'] + a_stats['PACE']) / 2
        pace_factor = estimated_pace / league_avg_pace

        # Score Prediction (Using Normalized Points)
        # We removed the 'Implied Defense' sum because summing "Points Allowed Per Game" 
        # across 15 players leads to massive inflation.
        h_final = h_stats['NORMALIZED_PTS'] * pace_factor
        a_final = a_stats['NORMALIZED_PTS'] * pace_factor
        
        total = h_final + a_final
        spread = h_final - a_final
        winner = home_team if spread > 0 else away_team

        # CLI Output
        print(f"{away_team} ({a_final:.1f}) @ {home_team} ({h_final:.1f})")
        print(f"  > Winner: {winner} ({abs(spread):.1f})")
        print(f"  > Total:  {total:.1f}")
        print("-" * 40)

        # JSON Data Collection
        game_predictions_output.append({
            "away_team": away_team,
            "away_score": clean_for_json(a_final),
            "home_team": home_team,
            "home_score": clean_for_json(h_final),
            "winner": winner,
            "spread": clean_for_json(abs(spread)),
            "total": clean_for_json(total),
            "pace": clean_for_json(estimated_pace)
        })

    # 5. Save JSON
    output_filename = f"data/predictions/game_predictions_{date_str_raw}.json"
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(game_predictions_output, f, indent=4, ensure_ascii=False)
        logger.info(f"Game predictions saved to {output_filename}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")

# Helper to sanitize JSON values
def clean_for_json(val):
    if pd.isna(val) or np.isnan(val) or np.isinf(val):
        return 0.0
    return float(round(val, 1))

def main():
    latest = find_latest_prediction_file()
    if latest:
        run_game_predictions(latest)

if __name__ == "__main__":
    main()