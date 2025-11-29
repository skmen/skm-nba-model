"""
NBA Bet Analyzer (Universal)

Interactive tool that compares model predictions against Sportsbook lines.
Supports both:
1. Single Player JSON (nba_predictions.json)
2. Daily Betting Sheet CSV (betting_sheet_predictions_YYYY-MM-DD.csv)
"""

import json
import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def get_float_input(prompt):
    while True:
        try:
            val = input(prompt).strip()
            if not val: continue
            return float(val)
        except ValueError:
            print(f"{RED}Invalid number.{RESET}")

def analyze_bet(target, prediction, mae, line, vegas_odds):
    decimal_odds = (1 + (vegas_odds / 100)) if vegas_odds > 0 else (1 + (100 / abs(vegas_odds)))
    implied_prob = (1 / decimal_odds) * 100
    
    edge = prediction - line
    abs_edge = abs(edge)
    direction = "OVER" if edge > 0 else "UNDER"
    
    print(f"\n{CYAN}--- ANALYZING {target} ---{RESET}")
    print(f"Model Prediction: {GREEN}{prediction:.1f}{RESET}")
    print(f"Model Error (MAE): {YELLOW}{mae:.2f}{RESET}")
    print(f"Vegas Line:       {CYAN}{line:.1f}{RESET} ({vegas_odds})")
    print(f"Break-Even Win %: {YELLOW}{implied_prob:.1f}%{RESET}")
    print("-" * 30)
    
    if abs_edge > mae:
        print(f"{GREEN}✅ STRONG BET {direction} {line}{RESET}")
    elif abs_edge > (mae * 0.8):
        print(f"{YELLOW}⚠️ LEAN {direction} {line}{RESET}")
    else:
        print(f"{RED}❌ PASS{RESET}")
    print("-" * 30)

def load_from_csv(file_path):
    if not os.path.exists(file_path):
        print(f"{RED}File not found: {file_path}{RESET}")
        return None
        
    df = pd.read_csv(file_path)
    print(f"\n📂 Loaded {len(df)} lines from {file_path}")
    
    while True:
        search = input(f"\n{CYAN}Enter Player Name to Analyze (part or full): {RESET}").strip().lower()
        if not search: continue
        if search == 'q': sys.exit(0)
        
        matches = df[df['Player'].str.lower().str.contains(search)]
        
        if matches.empty:
            print(f"{RED}No player found matching '{search}'{RESET}")
            continue
            
        # If multiple matches, list them
        players = matches['Player'].unique()
        if len(players) > 1:
            print(f"{YELLOW}Multiple matches found:{RESET}")
            for i, p in enumerate(players):
                print(f"  {i+1}. {p}")
            try:
                idx = int(input("Select Player #: ")) - 1
                selected_player = players[idx]
            except:
                continue
        else:
            selected_player = players[0]
            
        print(f"\n{GREEN}Selected: {selected_player}{RESET}")
        player_rows = df[df['Player'] == selected_player]
        
        # Extract predictions for this player
        preds = {}
        maes = {}
        
        for _, row in player_rows.iterrows():
            stat = row['Stat']
            preds[stat] = row['Prediction']
            maes[stat] = row['MAE']
            
        return {'player': selected_player, 'predictions': preds, 'mae_scores': maes}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, help="YYYY-MM-DD to load daily betting sheet")
    parser.add_argument('--file', type=str, help="Direct path to CSV or JSON")
    args = parser.parse_args()

    data = {}
    
    # 1. Try Loading CSV (Batch Mode)
    if args.date:
        fpath = f"betting_sheet_predictions_{args.date}.csv"
        # Try current dir or data/predictions
        if not os.path.exists(fpath):
            fpath = os.path.join("data", "predictions", fpath)
        data = load_from_csv(fpath)
    elif args.file and args.file.endswith('.csv'):
        data = load_from_csv(args.file)
        
    # 2. Default to Single Player JSON
    else:
        json_path = args.file if args.file else "nba_predictions.json"
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
        else:
            print(f"{RED}No input found.{RESET}")
            print("Usage:")
            print("  python scripts/bet_analyzer.py --date 2025-11-28  (Analyze Daily Sheet)")
            print("  python scripts/bet_analyzer.py                    (Analyze Single Run)")
            sys.exit(1)

    if not data: sys.exit(1)

    # Interactive Loop
    preds = data.get('predictions', {})
    maes = data.get('mae_scores', {})
    
    while True:
        print(f"\nTargets: {', '.join(preds.keys())}")
        target = input("Target (e.g. PTS) or 'q': ").upper()
        if target == 'Q': break
        if target not in preds: continue
        
        analyze_bet(target, preds[target], maes.get(target, 0), 
                    get_float_input("Line: "), get_float_input("Odds: "))

if __name__ == "__main__":
    main()