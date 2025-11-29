import sys
import os
import pandas as pd
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PREDICTIONS_DIR
from src.game_fetcher import GameFetcher
from nba_api.stats.static import teams as nba_teams

def get_team_name_by_id(team_id):
    """Helper to convert NBA Team ID to Full Name"""
    try:
        # Convert to int if it's a string/float
        tid = int(team_id)
        team_info = nba_teams.find_team_name_by_id(tid)
        return team_info['full_name'] if team_info else None
    except:
        return None

def normalize_team_minutes(df, target_minutes=240.0):
    """Scales all player stats so the team total minutes equals 240."""
    team_total_min = df['MINUTES'].sum()
    if team_total_min > target_minutes:
        scale_factor = target_minutes / team_total_min
        cols_to_scale = ['MINUTES', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M']
        for col in cols_to_scale:
            if col in df.columns:
                df[col] = df[col] * scale_factor
    return df

def run_daily_simulation():
    # 1. Determine the filename
    target_date = datetime.now()
    date_str = target_date.strftime('%Y-%m-%d')
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
        target_date = datetime.strptime(date_str, '%Y-%m-%d')

    filename = f"predictions_{date_str}.csv"
    filepath = os.path.join(PREDICTIONS_DIR, filename)
    
    print(f"\n🔍 SIMULATING GAMES FOR: {date_str}")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    # 2. Load Predictions
    df = pd.read_csv(filepath)
    
    # 3. Aggregate Team Stats
    teams = df['TEAM_NAME'].unique()
    team_stats_map = {}
    
    # --- RESTORED: List to collect normalized player data ---
    all_players_normalized = [] 
    # --------------------------------------------------------

    print(f"   - Processing {len(teams)} teams...")

    for team in teams:
        team_df = df[df['TEAM_NAME'] == team].copy()
        
        # A. Normalize Minutes
        team_df = normalize_team_minutes(team_df)
        
        # --- RESTORED: Collect the normalized dataframe ---
        all_players_normalized.append(team_df)
        # --------------------------------------------------

        # B. Store Aggregates for Matchups
        team_stats_map[team] = {
            'PTS': team_df['PTS'].sum(),
            'PACE': team_df['PACE'].mean()
        }

    # 4. Fetch Schedule
    print("   - Fetching schedule to build matchups...")
    fetcher = GameFetcher()
    games_df = fetcher.get_today_games(date_str)
    
    game_json_output = []

    if games_df is not None and not games_df.empty:
        for _, game in games_df.iterrows():
            # --- ROBUST ID EXTRACTION ---
            h_id = game.get('HOME_TEAM_ID') or game.get('TEAM_ID_HOME')
            a_id = (game.get('VISITOR_TEAM_ID') or 
                   game.get('TEAM_ID_AWAY') or 
                   game.get('AWAY_TEAM_ID'))

            home_team = get_team_name_by_id(h_id)
            away_team = get_team_name_by_id(a_id)
            
            if not home_team or not away_team:
                print(f"⚠️ Could not map IDs: {h_id} vs {a_id}")
                continue

            # Check if we have predictions
            if home_team in team_stats_map and away_team in team_stats_map:
                h_stats = team_stats_map[home_team]
                a_stats = team_stats_map[away_team]
                
                h_score = h_stats['PTS']
                a_score = a_stats['PTS']
                
                total = h_score + a_score
                spread = abs(h_score - a_score)
                winner = home_team if h_score > a_score else away_team
                game_pace = (h_stats['PACE'] + a_stats['PACE']) / 2
                
                game_obj = {
                    "away_team": away_team,
                    "away_score": round(a_score, 1),
                    "home_team": home_team,
                    "home_score": round(h_score, 1),
                    "winner": winner,
                    "spread": round(spread, 1),
                    "total": round(total, 1),
                    "pace": round(game_pace, 1)
                }
                game_json_output.append(game_obj)
            else:
                if home_team not in team_stats_map:
                    print(f"⚠️ Missing predictions for: {home_team}")
                if away_team not in team_stats_map:
                    print(f"⚠️ Missing predictions for: {away_team}")

    # 5. Save Outputs
    
    # A. Save JSON (Matchups)
    json_filename = f"game_predictions_{date_str}.json"
    json_path = os.path.join(PREDICTIONS_DIR, json_filename)
    
    with open(json_path, 'w') as f:
        json.dump(game_json_output, f, indent=4)
        
    print(f"✅ Game JSON saved to: {json_path}")
    
    # B. --- RESTORED: Save Player CSV (Simulation Viewer) ---
    if all_players_normalized:
        full_box_score = pd.concat(all_players_normalized)
        
        # Select and Reorder columns for the viewer
        # Ensure we look for 'USAGE_PCT' or 'USAGE_RATE' depending on what batch_predictor saved
        # Usually it is 'USAGE_PCT' after the Level 4 update.
        cols_to_save = ['PLAYER_NAME', 'TEAM_NAME', 'MINUTES', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'USAGE_PCT']
        
        # Filter to ensure columns exist (avoid KeyErrors if older data format)
        final_cols = [c for c in cols_to_save if c in full_box_score.columns]
        
        player_output = os.path.join(PREDICTIONS_DIR, f"game_sims_players_{date_str}.csv")
        full_box_score[final_cols].round(2).to_csv(player_output, index=False)
        print(f"✅ Player Stats CSV saved to: {player_output}")
    # --------------------------------------------------------

    # C. Save Team CSV (Summary)
    if game_json_output:
        flat_data = []
        for g in game_json_output:
            flat_data.append({'TEAM': g['away_team'], 'PTS': g['away_score'], 'PACE': g['pace']})
            flat_data.append({'TEAM': g['home_team'], 'PTS': g['home_score'], 'PACE': g['pace']})
            
        csv_df = pd.DataFrame(flat_data).sort_values('PTS', ascending=False)
        csv_path = os.path.join(PREDICTIONS_DIR, f"game_sims_teams_{date_str}.csv")
        csv_df.to_csv(csv_path, index=False)
        print(f"✅ Team CSV saved to: {csv_path}")

if __name__ == "__main__":
    run_daily_simulation()