#!/usr/bin/env python3
"""
Automated Daily NBA Predictions and Actuals Retrieval

This script automates the entire prediction workflow and can also retrieve
actual results for a given day.

Usage:
    # Run predictions for today
    python scripts/automated_predictions.py --run-once

    # Get actuals for yesterday
    python scripts/automated_predictions.py --get-actuals YYYY-MM-DD

Author: NBA Prediction Model
Date: 2025
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logger, print_section, print_result
from src.game_fetcher import GameFetcher
from src.batch_predictor import predict_all_players_today
from src.data_fetcher import get_player_gamelog
from src.scheduler import get_scheduler, print_cron_setup
from src.config import TARGETS

import logging
logger = setup_logger(__name__)


def ensure_directories() -> None:
    """Ensure all necessary directories exist."""
    dirs = ["data", "logs", "data/predictions"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")


def get_actuals_for_date(target_date: datetime, output_filename: str, season: str = "2024-25") -> bool:
    """
    Fetches the actual game stats for all players who played on a given date.
    """
    try:
        date_str = target_date.strftime('%Y-%m-%d')
        print_section(f"📊 GETTING ACTUALS FOR {date_str}")
        
        fetcher = GameFetcher()
        playing_that_day = fetcher.get_all_playing_today(target_date.strftime('%d-%m-%Y'))

        if playing_that_day.empty:
            print_result(f"❌ No players found for {date_str}", True)
            return True

        print_result(f"✅ Found {len(playing_that_day)} players scheduled on {date_str}", True)
        
        actuals_list = []
        for _, player in playing_that_day.iterrows():
            player_name = player['PLAYER_NAME']
            print(f"Fetching actuals for {player_name}...")
            
            # Fetch game log, which will include the game from target_date
            game_log = get_player_gamelog(player_name, season)
            
            if game_log is None or game_log.empty:
                logger.warning(f"Could not get game log for {player_name}")
                continue

            # Filter for the specific game date
            game_log['GAME_DATE'] = pd.to_datetime(game_log['GAME_DATE']).dt.date
            target_date_date = target_date.date()
            
            actual_game = game_log[game_log['GAME_DATE'] == target_date_date]

            if not actual_game.empty:
                actual_stats = actual_game.iloc[0]
                player_actuals = {
                    'PLAYER_NAME': player_name,
                    'TEAM_NAME': player['TEAM_NAME'],
                }
                for target in TARGETS:
                    player_actuals[target] = actual_stats.get(target, 0)
                actuals_list.append(player_actuals)
                logger.info(f"Found actuals for {player_name}: {player_actuals}")
            else:
                logger.warning(f"No game found on {date_str} for {player_name}")

        if not actuals_list:
            print_result("❌ No actuals could be retrieved.", False)
            return False

        actuals_df = pd.DataFrame(actuals_list)
        actuals_df.to_csv(output_filename, index=False)
        print_result(f"✅ Saved actuals for {len(actuals_df)} players to {output_filename}", True)
        return True

    except Exception as e:
        logger.error(f"Fatal error in get_actuals_for_date: {e}", exc_info=True)
        print_result(f"❌ Error getting actuals: {e}", False)
        return False


def run_predictions_for_date(target_date: datetime, output_filename: str, season: str = "2024-25") -> bool:
    """
    Run complete prediction workflow for a given date.
    """
    try:
        # 1. Format the date correctly (ISO format)
        date_str_api = target_date.strftime('%Y-%m-%d')
        
        start_time = datetime.now()
        print_section(f"NBA PREDICTIONS FOR {target_date.strftime('%Y-%m-%d')}")
        
        # 2. ADD THIS MISSING LINE HERE 👇
        fetcher = GameFetcher()
        
        # 3. Now you can use it
        games = fetcher.get_today_games(date_str_api)
        
        if games is None or games.empty:
            print_result(f"No games scheduled for {date_str_api}", True)
            return True
        
        print_result(f"✅ Found {len(games)} game(s)", True)
        
        playing_today = fetcher.get_all_playing_today(date_str_api)
        
        if playing_today.empty:
            print_result("❌ No starting players found", True)
            return True
        
        print_result(f"✅ Found {len(playing_today)} starting players", True)
        
        print_section("Generating Predictions")
        print("This may take a few minutes...\n")
        
        predictions = predict_all_players_today(
            playing_today,
            output_csv=output_filename,
            season=season
        )
        
        if predictions.empty:
            print_result("❌ No predictions generated", False)
            return False
        
        print_result(f"✅ Generated predictions for {len(predictions)} players", True)
        print_result(f"Saved predictions to {output_filename}", True)
        
        duration = (datetime.now() - start_time).total_seconds()
        print_section("✅ PREDICTIONS COMPLETE")
        print(f"Duration: {duration:.1f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"Fatal error in daily predictions: {e}", exc_info=True)
        print_result(f"❌ Error in prediction pipeline: {e}", False)
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
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger('src').setLevel(logging.DEBUG)
    
    ensure_directories()
    
    if args.setup_cron:
        # ... (cron setup logic remains the same)
        return

    if args.run_once:
        target_date = datetime.strptime(args.date, '%Y-%m-%d') if args.date else datetime.now()
        output_file = f"data/predictions/predictions_{target_date.strftime('%Y%m%d')}.csv"
        success = run_predictions_for_date(target_date, output_file, args.season)
        sys.exit(0 if success else 1)

    if args.get_actuals:
        try:
            target_date = datetime.strptime(args.get_actuals, '%Y-%m-%d')
            output_file = f"data/predictions/predictions_{target_date.strftime('%Y%m%d')}_ACTUAL.csv"
            success = get_actuals_for_date(target_date, output_file, args.season)
            sys.exit(0 if success else 1)
        except ValueError:
            print("❌ Invalid date format. Please use YYYY-MM-DD.")
            sys.exit(1)

    # Scheduler logic
    try:
        print_section("🎯 STARTING SCHEDULER")
        print(f"Running daily predictions at {args.time}")
        print("Press Ctrl+C to stop\n")
        
        def scheduled_job():
            target_date = datetime.now()
            output_file = f"data/predictions/predictions_{target_date.strftime('%Y%m%d')}.csv"
            run_predictions_for_date(target_date, output_file, args.season)

        scheduler = get_scheduler(args.time, use_apscheduler=False) # Keep it simple
        scheduler.schedule_daily(scheduled_job)
        scheduler.start()
        
    except KeyboardInterrupt:
        print("\n\nScheduler stopped by user")
        sys.exit(0)


if __name__ == "__main__":
    main()