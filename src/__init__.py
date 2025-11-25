"""
NBA Player Performance Prediction Package

A modular, scalable pipeline for predicting NBA player performance using
machine learning (XGBoost) with advanced features including opponent context,
travel distance, rest days, and player usage rates.

Core Modules:
    config: Configuration and constants
    utils: Utility functions and logging
    data_fetcher: NBA API data acquisition
    feature_engineer: Feature engineering and computation
    model: Model training, evaluation, and prediction
    prediction_pipeline: Main orchestration module

Automation Modules:
    game_fetcher: Fetch today's games and lineups
    batch_predictor: Batch predictions for multiple players
    scheduler: Time-based scheduling for daily runs

Example:
    Run the complete pipeline:
    
    $ python prediction_pipeline.py
    
    Run automated daily predictions:
    
    $ python scripts/automated_predictions.py --run-once
    $ python scripts/automated_predictions.py --time 09:00
"""

__version__ = "2.1.0"
__author__ = "Your Name"

from config import FEATURES, TARGET, DEFAULT_PLAYER_NAME, DEFAULT_SEASON
from prediction_pipeline import run_prediction_pipeline
from game_fetcher import GameFetcher
from batch_predictor import BatchPredictor, predict_all_players_today
from scheduler import get_scheduler, print_cron_setup

__all__ = [
    # Prediction
    'run_prediction_pipeline',
    # Automation
    'GameFetcher',
    'BatchPredictor',
    'predict_all_players_today',
    'get_scheduler',
    'print_cron_setup',
    # Configuration
    'FEATURES',
    'TARGET',
    'DEFAULT_PLAYER_NAME',
    'DEFAULT_SEASON',
]
