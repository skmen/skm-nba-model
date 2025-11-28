"""
Performance Tracker Module

Aggregates predictions and actual results into a persistent JSON history file.
Tracks both Player Props and Game Outcomes.

Usage:
    python scripts/track_performance.py --date YYYY-MM-DD
"""

import argparse
import json
import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path
from nba_api.stats.endpoints import scoreboardv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logger, print_section, print_result

logger = setup_logger(__name__)

HISTORY_FILE = "data/history.json"

def get_actual_game_scores(date_str_iso: str):
    """
    Fetch actual game scores for a specific date using NBA API.
    Returns a dictionary keyed by matchup string.
    """
    try:
        # Convert YYYY-MM-DD to MM/DD/YYYY for API
        dt = datetime.strptime(date_str_iso, "%Y-%m-%d")
        date_str_api = dt.strftime("%m/%d/%Y")
        
        board = scoreboardv2.ScoreboardV2(game_date=date_str_api)
        games_df = board.game_header.get_data_frame()
        line_score = board.line_score.get_data_frame()
        
        if games_df.empty:
            return {}

        actual_scores = {}
        
        # Merge header with line score to get points
        # We need to map TEAM_ID to Points
        # Line Score has columns: GAME_ID, TEAM_ID, PTS
        
        for _, game in games_df.iterrows():
            game_id = game['GAME_ID']
            
            # Filter line scores for this game
            game_scores = line_score[line_score['GAME_ID'] == game_id]
            if len(game_scores) < 2:
                continue
                
            # Get Home and Visitor info
            # GAME_HEADER has HOME_TEAM_ID and VISITOR_TEAM_ID
            home_id = game['HOME_TEAM_ID']
            visitor_id = game['VISITOR_TEAM_ID']
            
            try:
                home_pts = int(game_scores[game_scores['TEAM_ID'] == home_id]['PTS'].iloc[0])
                visitor_pts = int(game_scores[game_scores['TEAM_ID'] == visitor_id]['PTS'].iloc[0])
            except IndexError:
                continue # Skip if stats aren't ready
                
            # We need Team Names to match your prediction file
            # We rely on the game_fetcher mapping or simple lookup if available
            # For robustness, we will try to match based on the prediction file's expected format later
            # For now, store by ID and we'll map names if we can, or just store raw
            
            # To match your existing pipeline, we need full names (e.g. "Los Angeles Lakers")
            # We can use the static teams module
            from nba_api.stats.static import teams
            try:
                home_team_info = teams.find_team_name_by_id(home_id)
                visitor_team_info = teams.find_team_name_by_id(visitor_id)
                
                home_name = home_team_info['full_name']
                visitor_name = visitor_team_info['full_name']
                
                matchup_key = f"{visitor_name} @ {home_name}"
                
                actual_scores[matchup_key] = {
                    "home_team": home_name,
                    "home_score": home_pts,
                    "away_team": visitor_name,
                    "away_score": visitor_pts,
                    "total": home_pts + visitor_pts,
                    "spread": abs(home_pts - visitor_pts),
                    "winner": home_name if home_pts > visitor_pts else visitor_name
                }
            except:
                continue

        return actual_scores

    except Exception as e:
        logger.error(f"Failed to fetch actual scores: {e}")
        return {}

def load_json(filepath):
    if not os.path.exists(filepath): return None
    try:
        with open(filepath, 'r') as f: return json.load(f)
    except: return None

def track_performance(target_date: str):
    print_section(f"TRACKING PERFORMANCE FOR {target_date}")
    
    # 1. File Paths
    date_clean = target_date.replace("-", "")
    pred_csv = f"data/predictions/predictions_{date_clean}.csv"
    actual_csv = f"data/predictions/predictions_{date_clean}_ACTUAL.csv"
    game_json = f"data/predictions/game_predictions_{date_clean}.json"
    
    # 2. Validation
    if not os.path.exists(pred_csv):
        print(f"❌ Predictions file not found: {pred_csv}")
        return
    if not os.path.exists(actual_csv):
        print(f"❌ Actuals file not found: {actual_csv}")
        print("   (Run automated_predictions.py --get-actuals first)")
        return
    
    # 3. Load Player Data
    try:
        df_pred = pd.read_csv(pred_csv)
        df_actual = pd.read_csv(actual_csv)
        
        # Merge on Player + Team
        merged = pd.merge(df_pred, df_actual, on=['PLAYER_NAME', 'TEAM_NAME'], suffixes=('_pred', '_act'))
        
        player_predictions = {}
        
        for _, row in merged.iterrows():
            pname = row['PLAYER_NAME']
            
            # Calculate PRA if missing in actuals
            act_pts = row.get('PTS_act', 0)
            act_reb = row.get('REB_act', 0)
            act_ast = row.get('AST_act', 0)
            act_pra = row.get('PRA_act', act_pts + act_reb + act_ast)
            
            player_predictions[pname] = {
                "predictedPoints": round(row.get('PTS_pred', 0), 2),
                "actualPoints": int(act_pts),
                "predictedRebounds": round(row.get('REB_pred', 0), 2),
                "actualRebounds": int(act_reb),
                "predictedAssists": round(row.get('AST_pred', 0), 2),
                "actualAssists": int(act_ast),
                "predictedPRA": round(row.get('PRA_pred', 0), 2),
                "actualPRA": int(act_pra),
                # Add others if needed
            }
            
        print_result("Player Props Processed", len(player_predictions))
        
    except Exception as e:
        logger.error(f"Error processing player data: {e}")
        return

    # 4. Process Game Data
    game_predictions_map = {}
    
    if os.path.exists(game_json):
        try:
            # Load your predictions
            with open(game_json, 'r') as f:
                preds_list = json.load(f)
            
            # Fetch actuals from API
            actual_scores = get_actual_game_scores(target_date)
            
            for pred in preds_list:
                home = pred['home_team']
                away = pred['away_team']
                matchup_key = f"{away} @ {home}"
                
                # Check if we have actuals
                if matchup_key in actual_scores:
                    actual = actual_scores[matchup_key]
                    
                    game_entry = {
                        "predictedWinner": pred['winner'],
                        "actualWinner": actual['winner'],
                        "predictedSpread": pred['spread'],
                        "actualSpread": actual['spread'],
                        "predictedTotal": pred['total'],
                        "actualTotal": actual['total'],
                        "predictionCorrect": pred['winner'] == actual['winner']
                    }
                    
                    game_predictions_map[matchup_key] = game_entry
            
            print_result("Games Verified", len(game_predictions_map))
            
        except Exception as e:
            logger.error(f"Error processing game data: {e}")
    else:
        print("⚠️ Game predictions JSON not found. Skipping game verification.")

    # 5. Build Final Record
    daily_record = {
        "dateOfGames": target_date,
        "gameData": {
            "gamePredictions": game_predictions_map,
            "playerPredictions": player_predictions
        }
    }

    # 6. Update History File
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except:
            history = []

    # Check if date exists and update, otherwise append
    updated = False
    for i, entry in enumerate(history):
        if entry['dateOfGames'] == target_date:
            history[i] = daily_record
            updated = True
            break
    
    if not updated:
        history.append(daily_record)
        
    # Sort history by date
    history.sort(key=lambda x: x['dateOfGames'], reverse=True)

    # Save
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)
        
    print_section("HISTORY UPDATED")
    print(f"✅ Saved results for {target_date} to {HISTORY_FILE}")
    print(f"Total days tracked: {len(history)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    
    track_performance(args.date)