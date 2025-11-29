#!/usr/bin/env python3
"""
Automated Daily NBA Predictions and Actuals Retrieval

Updates:
- Enforced output to 'data/predictions/'
- Fixed filename to 'predictions_YYYY-MM-DD.csv'
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
from src.data_fetcher import get_player_gamelog
from src.scheduler import get_scheduler, print_cron_setup
from src.config import TARGETS, PREDICTIONS_DIR

import logging
logger = setup_logger(__name__)


def ensure_directories() -> None:
    """Ensure all necessary directories exist."""
    # We use PREDICTIONS_DIR from config to be safe
    dirs = ["data", "logs", PREDICTIONS_DIR]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")


def get_actuals_for_date(target_date: datetime, output_filename: str, season: str = "2024-25") -> bool:
    """
    Fetches actual game stats for ALL players on a given date.
    """
    try:
        date_str_api = target_date.strftime('%m/%d/%Y')
        date_str_iso = target_date.strftime('%Y-%m-%d')
        
        print_section(f"GETTING ACTUALS FOR {date_str_iso}")
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

        # ... (Processing logic remains same) ...
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
        
        # FIX: Ensure full path is used
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
    season: str = "2024-25",
    team_filter: Optional[str] = None
) -> bool:
    """
    Run complete prediction workflow for a given date.
    """
    try:
        date_str_api = target_date.strftime('%Y-%m-%d')
        
        start_time = datetime.now()
        print_section(f"NBA PREDICTIONS FOR {target_date.strftime('%Y-%m-%d')}")
        
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

        if playing_today.empty:
            print_result("No players found matching criteria", True)
            return True
        
        print_section("Generating Predictions")
        
        # FIX: We pass the FULL PATH to predict_all_players_today
        full_path = os.path.join(PREDICTIONS_DIR, output_filename)
        
        predictions = predict_all_players_today(
            playing_today,
            output_csv=full_path, # Pass the full path here
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
    parser.add_argument('--season', type=str, default='2024-25', help='NBA season')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--team', type=str, help='Filter predictions to a specific team')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger('src').setLevel(logging.DEBUG)
    
    ensure_directories()
    
    if args.setup_cron:
        # ... (cron logic)
        return

    if args.run_once:
        target_date = datetime.strptime(args.date, '%Y-%m-%d') if args.date else datetime.now()
        date_str = target_date.strftime('%Y-%m-%d')
        
        # FIX: Restore original naming convention
        output_file = f"predictions_{date_str}.csv"
        
        # Note: We do NOT join path here, we pass filename to run_predictions_for_date
        # which now joins it safely.
        
        success = run_predictions_for_date(target_date, output_file, args.season, team_filter=args.team)
        sys.exit(0 if success else 1)

    if args.get_actuals:
        try:
            target_date = datetime.strptime(args.get_actuals, '%Y-%m-%d')
            date_str = target_date.strftime('%Y-%m-%d')
            
            # FIX: Restore naming convention
            output_file = f"actuals_{date_str}.csv"
            
            success = get_actuals_for_date(target_date, output_file, args.season)
            sys.exit(0 if success else 1)
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
            sys.exit(1)

    # Scheduler logic
    try:
        print_section("STARTING SCHEDULER")
        print(f"Running daily predictions at {args.time}")
        print("Press Ctrl+C to stop\n")
        
        def scheduled_job():
            target_date = datetime.now()
            date_str = target_date.strftime('%Y-%m-%d')
            output_file = f"predictions_{date_str}.csv"
            run_predictions_for_date(target_date, output_file, args.season)

        scheduler = get_scheduler(args.time, use_apscheduler=False) 
        scheduler.schedule_daily(scheduled_job)
        scheduler.start()
        
    except KeyboardInterrupt:
        print("\n\nScheduler stopped by user")
        sys.exit(0)


if __name__ == "__main__":
    main()