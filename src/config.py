"""
Configuration module for NBA prediction pipeline.

Contains all constants, settings, and configurable parameters.
"""

import os
from typing import Dict, Tuple

# ============================================================================
# DATA DIRECTORY CONFIGURATION
# ============================================================================

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================================
# NBA ARENA COORDINATES (for travel distance calculation)
# ============================================================================

ARENA_COORDINATES: Dict[str, Tuple[float, float]] = {
    'ATL': (33.7490, -84.3880),   # State Farm Arena
    'BOS': (42.3661, -71.0096),   # TD Garden
    'BRK': (40.6826, -73.9754),   # Barclays Center
    'CHI': (41.8807, -87.6742),   # United Center
    'CLE': (41.4965, -81.6882),   # Quicken Loans Arena
    'DAL': (32.7905, -96.8101),   # American Airlines Center
    'DEN': (39.7487, -104.9957),  # Ball Arena
    'DET': (42.6416, -83.0554),   # Little Caesars Arena
    'GSW': (37.7683, -122.3969),  # Chase Center
    'HOU': (29.7589, -95.3677),   # Toyota Center
    'LAC': (34.0430, -118.2673),  # Intuit Dome
    'LAL': (34.0430, -118.2673),  # Crypto.com Arena
    'MEM': (35.1381, -90.0076),   # FedEx Forum
    'MIA': (25.7814, -80.1899),   # Kaseya Center
    'MIL': (43.0449, -87.9171),   # Fiserv Forum
    'MIN': (44.9795, -93.2789),   # Target Center
    'NOP': (29.9487, -90.0821),   # Smoothie King Center
    'NYK': (40.7505, -73.9934),   # Madison Square Garden
    'OKC': (35.4033, -97.5161),   # Paycom Center
    'ORL': (28.5394, -81.3839),   # Amway Center
    'PHI': (39.9012, -75.1720),   # Wells Fargo Center
    'PHX': (33.3764, -112.0697),  # Footprint Center
    'POR': (45.2308, -122.7121),  # Moda Center
    'SAC': (38.5816, -121.4944),  # Golden 1 Center
    'SAS': (29.4269, -98.4375),   # Frost Bank Center
    'TOR': (43.6426, -79.3957),   # Scotiabank Arena
    'UTA': (40.7683, -111.9011),  # Delta Center
    'WAS': (38.8980, -77.0209),   # Capital One Arena
}

# ============================================================================
# MODEL FEATURES & TARGET
# ============================================================================

FEATURES = [
    'PTS_L5', 'MIN_L5', 'REB_L5', 'AST_L5',
    'STL_L5', 'BLK_L5', 'FG3M_L5', 'HOME_GAME',
    'OPP_DEF_RATING', 'OPP_PACE',           # Opponent Context
    'TRAVEL_DISTANCE', 'DAYS_REST', 'BACK_TO_BACK',  # Fatigue & Travel
    'USAGE_RATE'                             # Roster Context
]

TARGET = 'PTS'

# ============================================================================
# LAG STATISTICS (for feature engineering)
# ============================================================================

LAG_STATS = ['PTS', 'MIN', 'REB', 'AST', 'STL', 'BLK', 'FG3M']
LAG_WINDOW = 5  # 5-game rolling average

# ============================================================================
# REST DAYS CONFIGURATION
# ============================================================================

DAYS_REST_MIN = 0
DAYS_REST_MAX = 5  # Cap maximum rest days at 5

# ============================================================================
# XGBOOST HYPERPARAMETERS
# ============================================================================

XGBOOST_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 3,
    'early_stopping_rounds': 50,
    'random_state': 42,
}

# ============================================================================
# TRAIN/TEST SPLIT RATIO
# ============================================================================

TRAIN_TEST_RATIO = 0.8  # 80% train, 20% test

# ============================================================================
# DEFAULT PLAYER & SEASON (for quick testing)
# ============================================================================

DEFAULT_PLAYER_NAME = "James Harden"
DEFAULT_SEASON = "2023-24"

# ============================================================================
# MULTI-SEASON CONFIGURATION
# ============================================================================

SEASONS_TO_FETCH = [
    "2024-25",  # Current season
    "2023-24",  # Last season
    "2022-23",  # 2 seasons ago
]

SEASON_WEIGHTS = {
    "2024-25": 1.0,   # Current Season (Most important)
    "2023-24": 0.8,   # Last Season
    "2022-23": 0.5,   # 2 Seasons ago
}

# ============================================================================
# GAME TYPE FILTERING
# ============================================================================

GAME_TYPE_FILTER = "Regular Season"  # Only regular season games

# ============================================================================
# API RATE LIMITING
# ============================================================================

API_DELAY = 0.5  # seconds between API calls

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

PLOT_FIGSIZE = (10, 6)
PLOT_ALPHA = 0.5

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
