"""
Batch Prediction Module

Predict statistics (PTS, REB, AST, STL, BLK) for multiple players at once.
Optimized for daily predictions of all players playing today.

Author: NBA Prediction Model
Date: 2025
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.config import FEATURES, TARGET
from src.data_fetcher import get_player_gamelog, get_opponent_defense_metrics, get_player_usage_rate
from src.feature_engineer import engineer_features
from src.model import predict_next_game, prepare_prediction_data
from src.utils import setup_logger, FeatureEngineeringError
import xgboost as xgb

logger = setup_logger(__name__)


class BatchPredictor:
    """Batch prediction for multiple players and stats."""
    
    # Primary stat predictions
    STAT_FEATURES = ['PTS', 'REB', 'AST', 'STL', 'BLK']
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize batch predictor.
        
        Args:
            model_path: Optional path to pre-trained XGBoost model.
                       If None, will train on-the-fly for each player.
        """
        self.model_path = model_path
        self.model = None
        self.opponent_defense = None
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str) -> bool:
        """Load pre-trained model from disk."""
        try:
            logger.info(f"Loading model from {model_path}...")
            self.model = xgb.XGBRegressor()
            self.model.load_model(model_path)
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Will train on-the-fly.")
            return False
    
    def predict_player_today(
        self,
        player_name: str,
        team_name: str,
        is_home_game: bool = True,
        season: str = "2024-25"
    ) -> Optional[Dict[str, float]]:
        """
        Predict stats for a player today.
        
        Args:
            player_name: NBA player name
            team_name: NBA team name
            is_home_game: Whether game is home or away
            season: NBA season (e.g., "2024-25")
            
        Returns:
            Dict with predicted stats: {PTS, REB, AST, STL, BLK, PREDICTION_TIME}
            None if prediction fails
        """
        try:
            logger.info(f"Predicting stats for {player_name} ({team_name})...")
            
            # Step 1: Fetch player game log
            game_log = get_player_gamelog(player_name, season)
            if game_log is None or game_log.empty:
                logger.warning(f"No game log found for {player_name}")
                return None
            
            # Step 2: Fetch opponent defense metrics
            if self.opponent_defense is None:
                self.opponent_defense = get_opponent_defense_metrics(season)
            
            # Step 3: Fetch player usage rate
            usage_rate = get_player_usage_rate(
                int(game_log.iloc[-1]['PLAYER_ID']),
                season
            )
            
            # Step 4: Engineer features
            engineered = engineer_features(
                game_log,
                self.opponent_defense,
                usage_rate
            )
            
            if engineered is None or engineered.empty:
                logger.warning(f"Could not engineer features for {player_name}")
                return None
            
            # Step 5: Prepare prediction data
            prediction_data = prepare_prediction_data(engineered, is_home_game)
            
            # Step 6: Make prediction
            prediction = predict_next_game(self.model, prediction_data, FEATURES)
            
            logger.info(f"Predicted {prediction:.2f} PTS for {player_name}")
            
            return {
                'PLAYER_NAME': player_name,
                'TEAM_NAME': team_name,
                'IS_HOME': is_home_game,
                'PTS': prediction,
                'REB': None,  # Would require separate models
                'AST': None,
                'STL': None,
                'BLK': None,
                'PREDICTION_TIME': pd.Timestamp.now()
            }
            
        except Exception as e:
            logger.error(f"Error predicting for {player_name}: {e}")
            return None
    
    def predict_multiple_players(
        self,
        players_data: List[Dict],
        season: str = "2024-25"
    ) -> pd.DataFrame:
        """
        Predict stats for multiple players.
        
        Args:
            players_data: List of dicts with keys: PLAYER_NAME, TEAM_NAME, IS_HOME, GAME_ID
            season: NBA season
            
        Returns:
            DataFrame with predictions for all players
        """
        try:
            logger.info(f"Making predictions for {len(players_data)} players...")
            
            # Fetch opponent defense once for efficiency
            if self.opponent_defense is None:
                self.opponent_defense = get_opponent_defense_metrics(season)
            
            predictions = []
            
            for i, player in enumerate(players_data):
                logger.info(f"[{i+1}/{len(players_data)}] Predicting {player['PLAYER_NAME']}...")
                
                prediction = self.predict_player_today(
                    player_name=player['PLAYER_NAME'],
                    team_name=player['TEAM_NAME'],
                    is_home_game=player.get('IS_HOME', True),
                    season=season
                )
                
                if prediction:
                    prediction['GAME_ID'] = player.get('GAME_ID', '')
                    predictions.append(prediction)
            
            result_df = pd.DataFrame(predictions)
            logger.info(f"Successfully predicted for {len(result_df)} players")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            return pd.DataFrame()
    
    def get_predictions_for_today(
        self,
        playing_today: pd.DataFrame,
        season: str = "2024-25"
    ) -> pd.DataFrame:
        """
        Get predictions for all players playing today.
        
        Args:
            playing_today: DataFrame from GameFetcher.get_all_playing_today()
            season: NBA season
            
        Returns:
            DataFrame with predictions
        """
        try:
            # Prepare player data list
            players_list = []
            for _, row in playing_today.iterrows():
                players_list.append({
                    'PLAYER_NAME': row['PLAYER_NAME'],
                    'TEAM_NAME': row['TEAM_NAME'],
                    'GAME_ID': row['GAME_ID'],
                    'IS_STARTER': row['IS_STARTER'],
                    'IS_HOME': None  # Would need to determine from game info
                })
            
            # Get predictions
            predictions = self.predict_multiple_players(players_list, season)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting predictions for today: {e}")
            return pd.DataFrame()
    
    def save_model(self, output_path: str) -> bool:
        """
        Save trained model to disk.
        
        Args:
            output_path: Path to save model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.model is None:
                logger.error("No model to save")
                return False
            
            logger.info(f"Saving model to {output_path}...")
            self.model.save_model(output_path)
            logger.info("Model saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False


def predict_all_players_today(
    playing_today: pd.DataFrame,
    output_csv: str = "data/predictions_today.csv",
    season: str = "2024-25"
) -> pd.DataFrame:
    """
    Convenience function to predict all players playing today.
    
    Args:
        playing_today: DataFrame from GameFetcher.get_all_playing_today()
        output_csv: Path to save CSV results
        season: NBA season
        
    Returns:
        DataFrame with predictions
    """
    try:
        predictor = BatchPredictor()
        predictions = predictor.get_predictions_for_today(playing_today, season)
        
        if not predictions.empty:
            predictions.to_csv(output_csv, index=False)
            logger.info(f"Predictions saved to {output_csv}")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error in predict_all_players_today: {e}")
        return pd.DataFrame()
