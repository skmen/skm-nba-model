import pandas as pd
import argparse
import os
import sys
import numpy as np

def generate_full_report(input_file):
    """
    Generates a Master Betting Sheet from a raw predictions CSV.
    Includes advanced context: Usage, Minutes, Pace, and DvP.
    """
    # 1. Validation
    if not os.path.exists(input_file):
        print(f"❌ Error: The file '{input_file}' was not found.")
        print("   Please check the path and try again.")
        sys.exit(1)

    # 2. Load Data
    try:
        df = pd.read_csv(input_file)
        print(f"📂 Loaded: {input_file}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        sys.exit(1)

    # 3. Configuration (Your Model's Trust Levels)
    stat_config = {
        'STL': {'trust': '1_ELITE',    'mae': 0.56, 'label': '🟢 ELITE'},
        'REB': {'trust': '2_HIGH',     'mae': 3.10, 'label': '🟡 HIGH'},
        'BLK': {'trust': '3_MEDIUM',   'mae': 0.67, 'label': '🟠 MEDIUM'},
        'PRA': {'trust': '3_MEDIUM',   'mae': 6.88, 'label': '🟠 MEDIUM'},
        'PTS': {'trust': '4_VOLATILE', 'mae': 5.73, 'label': '🔴 VOLATILE'},
        'AST': {'trust': '4_VOLATILE', 'mae': 1.88, 'label': '🔴 VOLATILE'}
    }

    report_data = []

    # 4. Processing Loop
    for _, row in df.iterrows():
        player = row.get('PLAYER_NAME', 'Unknown')
        team = row.get('TEAM_NAME', 'Unknown')
        is_home = row.get('IS_HOME', True)
        
        # --- NEW CONTEXT METRICS ---
        # Extract new data if available, with safe defaults
        minutes = row.get('MINUTES', 0)
        usage = row.get('USAGE_RATE', 0)
        pace = row.get('PACE', 0)
        dvp = row.get('OPP_DvP', 1.0)
        
        # Format Usage (if it's a decimal like 0.30, convert to 30.0)
        if usage < 1.0 and usage > 0:
            usage = usage * 100

        for stat, config in stat_config.items():
            if stat not in row:
                continue

            prediction = row[stat]
            mae = config['mae']

            # Calculate Target Lines (The "Safe Edge")
            target_over = prediction - mae
            target_under = prediction + mae

            # --- OPPONENT ALLOWED CONTEXT ---
            # Dynamically look for 'OPP_ALLOW_PTS', 'OPP_ALLOW_REB', etc.
            opp_allow_col = f"OPP_ALLOW_{stat}"
            opp_avg = row.get(opp_allow_col, np.nan)
            
            # Format Opp Avg (Show 'N/A' if missing)
            opp_avg_display = round(opp_avg, 1) if pd.notna(opp_avg) and opp_avg > 0 else "N/A"

            report_data.append({
                'Player': player,
                'Team': team,
                'Loc': '🏠' if is_home else '✈️',
                'Stat': stat,
                'Prediction': round(prediction, 2),
                'Opp_Avg': opp_avg_display,     # New: What opponent allows
                'Diff': round(prediction - opp_avg, 1) if isinstance(opp_avg_display, float) else 0,
                'Trust_Rank': config['trust'],  # Hidden sort key
                'Trust': config['label'],
                'MAE': mae,
                'Line_Over': round(target_over, 1),
                'Line_Under': round(target_under, 1),
                # Context Columns
                'Mins': round(minutes, 1),
                'Usg%': round(usage, 1),
                'Pace': round(pace, 1),
                'DvP': round(dvp, 2)
            })

    # 5. Create DataFrame & Sort
    report_df = pd.DataFrame(report_data)
    
    if report_df.empty:
        print("⚠️ No valid data found in file.")
        sys.exit(0)

    # Sort by Trust (Elite first), then by Prediction value
    report_df.sort_values(by=['Trust_Rank', 'Prediction'], ascending=[True, False], inplace=True)
    
    # Reorder columns for readability
    cols_order = [
        'Player', 'Team', 'Loc', 'Stat', 
        'Prediction', 'Line_Over', 'Line_Under', 'Trust', 
        'Opp_Avg', 'Mins', 'Usg%', 'Pace', 'DvP'
    ]
    # Filter to only existing cols (in case some are missing)
    cols_order = [c for c in cols_order if c in report_df.columns]
    report_df = report_df[cols_order]

    # 6. Save Report
    input_filename = os.path.basename(input_file)
    output_filename = f"betting_sheet_{input_filename}"
    
    report_df.to_csv(output_filename, index=False)
    
    print(f"✅ Master Betting Sheet Generated: {output_filename}")
    print(f"   Contains {len(report_df)} betting lines with full context.")
    print("-" * 80)
    print(report_df.head(5).to_string(index=False))
    print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Master Betting Sheet from a predictions CSV file.")
    parser.add_argument(
        "file_path", 
        type=str, 
        help="Path to the predictions CSV (e.g. data/predictions/predictions_20251126.csv)"
    )

    args = parser.parse_args()
    generate_full_report(args.file_path)