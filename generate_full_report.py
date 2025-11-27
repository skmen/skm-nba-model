import pandas as pd
import argparse
import os
import sys

def generate_full_report(input_file):
    """
    Generates a Master Betting Sheet from a raw predictions CSV.
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

        for stat, config in stat_config.items():
            if stat not in row:
                continue

            prediction = row[stat]
            mae = config['mae']

            # Calculate Target Lines (The "Safe Edge")
            # Over: Line must be lower than (Pred - Error)
            # Under: Line must be higher than (Pred + Error)
            target_over = prediction - mae
            target_under = prediction + mae

            report_data.append({
                'Player': player,
                'Team': team,
                'Stat': stat,
                'Prediction': round(prediction, 2),
                'Trust_Rank': config['trust'],  # Hidden sort key
                'Trust': config['label'],
                'MAE': mae,
                'Target_Line_Over': round(target_over, 1),
                'Target_Line_Under': round(target_under, 1)
            })

    # 5. Create DataFrame & Sort
    report_df = pd.DataFrame(report_data)
    
    if report_df.empty:
        print("⚠️ No valid data found in file.")
        sys.exit(0)

    # Sort by Trust (Elite first), then by highest Prediction
    report_df.sort_values(by=['Trust_Rank', 'Prediction'], ascending=[True, False], inplace=True)
    report_df.drop(columns=['Trust_Rank'], inplace=True)

    # 6. Save Report
    # We save it to the same folder as the input, but with a new name
    # e.g. "predictions_20251126.csv" -> "betting_sheet_20251126.csv"
    input_filename = os.path.basename(input_file)
    output_filename = f"betting_sheet_{input_filename}"
    
    report_df.to_csv(output_filename, index=False)
    
    print(f"✅ Report Generated: {output_filename}")
    print(f"   Contains {len(report_df)} betting lines.")
    print("-" * 60)
    print(report_df.head(5).to_string(index=False))
    print("-" * 60)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate a Master Betting Sheet from a predictions CSV file.")
    
    # Add the file argument (Required)
    parser.add_argument(
        "file_path", 
        type=str, 
        help="Path to the predictions CSV (e.g. data/predictions/predictions_20251126.csv)"
    )

    args = parser.parse_args()

    # Run the function
    generate_full_report(args.file_path)