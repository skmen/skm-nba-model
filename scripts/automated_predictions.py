#!/usr/bin/env python3
"""
Automated Daily NBA Predictions and Actuals Retrieval

Updates:
- Auto-detects season from date if not manually specified.
- Enforced output to 'data/predictions/'
- Added --exclude-file argument to filter injured players
"""

import argparse
import sys
import os
from typing import Optional
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logger, print_section, print_result
from src.game_fetcher import GameFetcher
from src.batch_predictor import predict_all_players_today
from src.scheduler import get_scheduler
from src.config import PREDICTIONS_DIR

import logging
logger = setup_logger(__name__)


def ensure_directories() -> None:
    """Ensure all necessary directories exist."""
    dirs = ["data", "logs", PREDICTIONS_DIR]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")

def get_season_from_date(date_obj: datetime) -> str:
    """
    Infers the NBA season from a date.
    NBA Season typically runs Oct (Month 10) to June (Month 6).
    """
    year = date_obj.year
    month = date_obj.month
    
    # If Month is Oct(10), Nov(11), Dec(12) -> Season starts this year
    # Example: Nov 2025 -> Season 2025-26
    if month >= 10:
        start_year = year
    # If Month is Jan(1) to Sept(9) -> Season started previous year
    # Example: Feb 2026 -> Season 2025-26
    else:
        start_year = year - 1
        
    next_year_suffix = str(start_year + 1)[-2:]
    return f"{start_year}-{next_year_suffix}"

def get_actuals_for_date(target_date: datetime, output_filename: str, season: str) -> bool:
    """Fetches actual game stats for ALL players on a given date."""
    try:
        date_str_api = target_date.strftime('%m/%d/%Y')
        date_str_iso = target_date.strftime('%Y-%m-%d')
        
        print_section(f"GETTING ACTUALS FOR {date_str_iso} (Season: {season})")
        print("Fetching league-wide stats (1 Batch Request)...")

        logs = leaguegamelog.LeagueGameLog(
            player_or_team_abbreviation='P',
            season=season,
            date_from_nullable=date_str_api,
            date_to_nullable=date_str_api
        )
        
        df = logs.get_data_frames()[0]

        if df.empty:
            print_result(f"No games found for {date_str_iso}", False)
            return False

        actuals_list = []
        for _, row in df.iterrows():
            player_actuals = {
                'PLAYER_NAME': row['PLAYER_NAME'],
                'TEAM_NAME': row['TEAM_NAME'],
                'PTS': row['PTS'],
                'REB': row['REB'],
                'AST': row['AST'],
                'STL': row['STL'],
                'BLK': row['BLK'],
                'PRA': row['PTS'] + row['REB'] + row['AST'] 
            }
            actuals_list.append(player_actuals)

        actuals_df = pd.DataFrame(actuals_list)
        full_path = os.path.join(PREDICTIONS_DIR, output_filename)
        actuals_df.to_csv(full_path, index=False)
        
        print_result(f"✅ Success", f"Retrieved actuals for {len(actuals_df)} players")
        print_result(f"Saved to", full_path)
        return True

    except Exception as e:
        logger.error(f"Fatal error in get_actuals_for_date: {e}", exc_info=True)
        print_result("Error getting actuals", str(e)) 
        return False


def run_predictions_for_date(
    target_date: datetime, 
    output_filename: str, 
    season: str,
    team_filter: Optional[str] = None,
    exclude_file: Optional[str] = None
) -> bool:
    """Run complete prediction workflow for a given date."""
    try:
        date_str_api = target_date.strftime('%Y-%m-%d')
        
        start_time = datetime.now()
        print_section(f"NBA PREDICTIONS FOR {target_date.strftime('%Y-%m-%d')} (Season: {season})")
        
        fetcher = GameFetcher()
        games = fetcher.get_today_games(date_str_api)
        
        if games is None or games.empty:
            print_result(f"No games scheduled for {date_str_api}", True)
            return True
        
        print_result(f"Found {len(games)} game(s)", True)
        
        # Get all players
        playing_today = fetcher.get_all_playing_today(date_str_api)
        
        if team_filter:
            print_result("Filtering by Team", team_filter)
            playing_today = playing_today[
                playing_today['TEAM_NAME'].str.contains(team_filter, case=False, na=False)
            ]

        # --- EXCLUSION LOGIC (NEW) ---
        if exclude_file:
            if os.path.exists(exclude_file):
                print_result("Applying Exclusion File", exclude_file)
                try:
                    injuries_df = pd.read_csv(exclude_file)
                    
                    # Filter for players who are definitively OUT
                    # Keywords: "Out" (e.g. "Out", "Expected to be out"), "Season" (Out for Season)
                    mask_out = injuries_df['STATUS'].str.contains('Out|Season', case=False, na=False)
                    out_players = injuries_df.loc[mask_out, 'PLAYER_NAME'].tolist()
                    
                    if out_players:
                        # Normalize names for matching (remove dots, lower case)
                        # e.g., "P.J. Washington" -> "pj washington"
                        out_players_norm = [str(p).replace('.', '').lower().strip() for p in out_players]
                        
                        initial_count = len(playing_today)
                        
                        # Create temporary normalized column for filtering
                        playing_today['NAME_NORM'] = playing_today['PLAYER_NAME'].str.replace('.', '').str.lower().str.strip()
                        
                        # Filter out matches
                        playing_today = playing_today[~playing_today['NAME_NORM'].isin(out_players_norm)].copy()
                        
                        # Cleanup
                        playing_today.drop(columns=['NAME_NORM'], inplace=True)
                        
                        filtered_count = initial_count - len(playing_today)
                        print_result("Players Removed (Injury)", filtered_count)
                    else:
                        print_result("Exclusion File Check", "No 'Out' players found in file")
                        
                except Exception as e:
                    logger.error(f"Error reading exclusion file: {e}")
                    print_result("Exclusion File Error", "Skipping filter")
            else:
                print_result("Exclusion File Warning", f"File not found: {exclude_file}")

        if playing_today.empty:
            print_result("No players found matching criteria", True)
            return True
        
        print_section("Generating Predictions")
        
        full_path = os.path.join(PREDICTIONS_DIR, output_filename)
        
        predictions = predict_all_players_today(
            playing_today,
            output_csv=full_path,
            season=season
        )
        
        if predictions.empty:
            print_result("No predictions generated", False)
            return False
        
        print_result(f"Generated predictions for {len(predictions)} players", True)
        print_result(f"Saved predictions to {full_path}", True)
        
        duration = (datetime.now() - start_time).total_seconds()
        print_section("PREDICTIONS COMPLETE")
        print(f"Duration: {duration:.1f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"Fatal error in daily predictions: {e}", exc_info=True)
        print_result(f"Error in prediction pipeline: {e}", False)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Automated NBA predictions and actuals retrieval.")
    
    parser.add_argument('--run-once', action='store_true', help='Run predictions once for today and exit')
    parser.add_argument('--get-actuals', type=str, help='Get actual results for a given date (YYYY-MM-DD)')
    parser.add_argument('--date', type=str, help='Date for predictions (YYYY-MM-DD), defaults to today')
    parser.add_argument('--time', type=str, default='09:00', help='Time for daily scheduler (HH:MM)')
    parser.add_argument('--setup-cron', action='store_true', help='Show cron setup instructions')
    parser.add_argument('--season', type=str, help='NBA season (e.g. 2024-25). Defaults to auto-detect based on date.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--team', type=str, help='Filter predictions to a specific team')
    parser.add_argument('--exclude-file', type=str, help='CSV file with list of injured players to exclude')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger('src').setLevel(logging.DEBUG)
    
    ensure_directories()
    
    # DETERMINE DATE AND SEASON
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
    elif args.get_actuals:
        target_date = datetime.strptime(args.get_actuals, '%Y-%m-%d')
    else:
        target_date = datetime.now()

    # Logic: Use provided season, OR auto-detect from date
    if args.season:
        season = args.season
    else:
        season = get_season_from_date(target_date)

    if args.setup_cron:
        # Cron logic omitted for brevity
        return

    if args.run_once:
        date_str = target_date.strftime('%Y-%m-%d')
        output_file = f"predictions_{date_str}.csv"
        # Pass exclude_file here
        success = run_predictions_for_date(target_date, output_file, season, team_filter=args.team, exclude_file=args.exclude_file)
        sys.exit(0 if success else 1)

    if args.get_actuals:
        date_str = target_date.strftime('%Y-%m-%d')
        output_file = f"actuals_{date_str}.csv"
        success = get_actuals_for_date(target_date, output_file, season)
        sys.exit(0 if success else 1)

    # Scheduler logic
    try:
        print_section("STARTING SCHEDULER")
        print(f"Running daily predictions at {args.time} (Season: {season})")
        
        def scheduled_job():
            curr_date = datetime.now()
            curr_season = get_season_from_date(curr_date)
            date_str = curr_date.strftime('%Y-%m-%d')
            output_file = f"predictions_{date_str}.csv"
            # Scheduler doesn't support --exclude-file yet as that's a manual override
            run_predictions_for_date(curr_date, output_file, curr_season)

        scheduler = get_scheduler(args.time, use_apscheduler=False) 
        scheduler.schedule_daily(scheduled_job)
        scheduler.start()
        
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    main()