#!/usr/bin/env python3
"""
Automated Daily NBA Predictions

This script automates the entire prediction workflow:
1. Fetches today's NBA games
2. Identifies players in starting lineups and active rosters
3. Gets predictions for PTS, REB, AST, STL, BLK for each player
4. Saves results to CSV and database
5. Can be scheduled to run daily at a specific time

Usage:
    # Run once for today's games
    python scripts/automated_predictions.py --run-once

    # Run daily at 9:00 AM
    python scripts/automated_predictions.py --time 09:00

    # Setup system cron job
    python scripts/automated_predictions.py --setup-cron

    # Run with verbose logging
    python scripts/automated_predictions.py --run-once --verbose

Author: NBA Prediction Model
Date: 2025
"""

import argparse
import sys
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logger, print_section, print_result
from src.game_fetcher import GameFetcher
from src.batch_predictor import predict_all_players_today
from src.scheduler import get_scheduler, print_cron_setup

import logging
logger = setup_logger(__name__)


def ensure_directories() -> None:
    """Ensure all necessary directories exist."""
    dirs = [
        "data",
        "logs",
        "data/predictions"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")


def run_daily_predictions(season: str = "2024-25") -> bool:
    """
    Run complete prediction workflow for today.
    
    Args:
        season: NBA season (e.g., "2024-25")
        
    Returns:
        True if successful, False otherwise
    """
    try:
        start_time = datetime.now()
        print_section("🏀 NBA DAILY PREDICTIONS")
        logger.info(f"Starting daily prediction run at {start_time}")
        
        # Step 1: Fetch today's games
        print_section("Step 1: Fetching Today's Games")
        fetcher = GameFetcher()
        today_games = fetcher.get_today_games()
        
        if today_games is None or today_games.empty:
            print_result("❌ No games scheduled for today", success=True)
            logger.info("No games today - exiting")
            return True  # Not a failure, just no games
        
        print_result(f"✅ Found {len(today_games)} game(s) today", success=True)
        logger.info(f"Today's games:\n{today_games}")
        
        # Step 2: Get all playing players
        print_section("Step 2: Fetching Players Playing Today")
        playing_today = fetcher.get_all_playing_today()
        
        if playing_today.empty:
            print_result("❌ No starting players found", success=True)
            logger.info("No players in starting lineups")
            return True
        
        print_result(
            f"✅ Found {len(playing_today)} starting players",
            success=True
        )
        logger.info(f"Players playing today:\n{playing_today}")
        
        # Step 3: Filter for starters only (as requested)
        starters_only = playing_today[playing_today['IS_STARTER'] == True].copy()
        print_result(
            f"✅ Filtered to {len(starters_only)} starters",
            success=True
        )
        
        # Step 4: Make predictions
        print_section("Step 3: Generating Predictions")
        print("This may take a few minutes...\n")
        
        predictions = predict_all_players_today(
            starters_only,
            output_csv=f"data/predictions/predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            season=season
        )
        
        if predictions.empty:
            print_result("❌ No predictions generated", success=False)
            logger.error("Prediction failed")
            return False
        
        print_result(
            f"✅ Generated predictions for {len(predictions)} players",
            success=True
        )
        
        # Step 5: Summary
        print_section("📊 PREDICTION SUMMARY")
        print(f"Games today:           {len(today_games)}")
        print(f"Players starting:      {len(starters_only)}")
        print(f"Predictions made:      {len(predictions)}")
        print(f"Success rate:          {len(predictions)/len(starters_only)*100:.1f}%")
        
        if not predictions.empty:
            print(f"\nTop 5 predicted scores:")
            top_5 = predictions.nlargest(5, 'PTS')[['PLAYER_NAME', 'TEAM_NAME', 'PTS']]
            for idx, (_, row) in enumerate(top_5.iterrows(), 1):
                print(f"  {idx}. {row['PLAYER_NAME']:25} ({row['TEAM_NAME']:20}) - {row['PTS']:.1f} pts")
        
        # Step 6: Save summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print_section("✅ PREDICTIONS COMPLETE")
        print(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Ended:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration:.1f} seconds")
        
        logger.info(f"Daily prediction run completed in {duration:.1f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"Fatal error in daily predictions: {e}", exc_info=True)
        print_result("❌ Error in prediction pipeline", success=False)
        print(f"Error: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Automated NBA daily predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run once for today
  python scripts/automated_predictions.py --run-once

  # Start daily scheduler at 9:00 AM
  python scripts/automated_predictions.py --time 09:00

  # Setup system cron job
  python scripts/automated_predictions.py --setup-cron

  # Show cron setup instructions
  python scripts/automated_predictions.py --show-cron
        """
    )
    
    parser.add_argument(
        '--run-once',
        action='store_true',
        help='Run predictions once and exit'
    )
    
    parser.add_argument(
        '--time',
        type=str,
        default='09:00',
        help='Time to run daily predictions (HH:MM format, default: 09:00)'
    )
    
    parser.add_argument(
        '--setup-cron',
        action='store_true',
        help='Setup system cron job'
    )
    
    parser.add_argument(
        '--show-cron',
        action='store_true',
        help='Show cron setup instructions'
    )
    
    parser.add_argument(
        '--season',
        type=str,
        default='2024-25',
        help='NBA season (default: 2024-25)'
    )
    
    parser.add_argument(
        '--use-apscheduler',
        action='store_true',
        help='Use APScheduler instead of simple scheduler'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger('src').setLevel(logging.DEBUG)
    
    # Ensure directories
    ensure_directories()
    
    # Show cron setup
    if args.show_cron:
        project_dir = os.path.abspath(os.path.dirname(__file__))
        script_path = os.path.join(project_dir, "scripts/automated_predictions.py")
        print_cron_setup(project_dir, script_path, args.time)
        return
    
    # Setup cron
    if args.setup_cron:
        project_dir = os.path.abspath(os.path.dirname(__file__))
        script_path = os.path.join(project_dir, "scripts/automated_predictions.py")
        print_cron_setup(project_dir, script_path, args.time)
        print("\n⚠️  To add to crontab, copy the line above and run: crontab -e")
        return
    
    # Run once
    if args.run_once:
        success = run_daily_predictions(args.season)
        sys.exit(0 if success else 1)
    
    # Start scheduler
    try:
        print_section("🎯 STARTING SCHEDULER")
        print(f"Running daily predictions at {args.time}")
        print("Press Ctrl+C to stop\n")
        
        scheduler = get_scheduler(args.time, use_apscheduler=args.use_apscheduler)
        scheduler.schedule_daily(
            lambda: run_daily_predictions(args.season)
        )
        scheduler.start()
        
    except KeyboardInterrupt:
        print("\n\nScheduler stopped by user")
        logger.info("Scheduler stopped")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Scheduler error: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
