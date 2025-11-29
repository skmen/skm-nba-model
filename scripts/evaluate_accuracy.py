import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PREDICTIONS_DIR

def evaluate_performance(date_str):
    # 1. Define File Paths
    pred_file = os.path.join(PREDICTIONS_DIR, f"predictions_{date_str}.csv")
    actual_file = os.path.join(PREDICTIONS_DIR, f"actuals_{date_str}.csv")
    
    # 2. Validation
    if not os.path.exists(pred_file):
        print(f"❌ Predictions file not found: {pred_file}")
        return
    if not os.path.exists(actual_file):
        print(f"❌ Actuals file not found: {actual_file}")
        print(f"   Run: python scripts/automated_predictions.py --get-actuals {date_str}")
        return

    # 3. Load Data
    print(f"📊 Evaluating Performance for {date_str}...")
    preds_df = pd.read_csv(pred_file)
    actuals_df = pd.read_csv(actual_file)
    
    # 4. Merge Data (Inner Join on Player Name)
    # We rename columns in actuals to distinguish them
    actuals_df = actuals_df.rename(columns={
        'PTS': 'PTS_ACTUAL',
        'REB': 'REB_ACTUAL',
        'AST': 'AST_ACTUAL',
        'STL': 'STL_ACTUAL',
        'BLK': 'BLK_ACTUAL',
        'PRA': 'PRA_ACTUAL'
    })
    
    # Select only necessary columns from Actuals
    cols_to_use = ['PLAYER_NAME', 'PTS_ACTUAL', 'REB_ACTUAL', 'AST_ACTUAL', 'STL_ACTUAL', 'BLK_ACTUAL', 'PRA_ACTUAL']
    
    merged = pd.merge(preds_df, actuals_df[cols_to_use], on='PLAYER_NAME', how='inner')
    
    if merged.empty:
        print("⚠️ No matching players found between predictions and actuals.")
        return

    # 5. Calculate Errors
    stats = ['PTS', 'REB', 'AST', 'PRA']
    error_summary = []

    print("\n" + "="*60)
    print(f"{'STAT':<10} {'MAE (Avg Error)':<20} {'Bias (Over/Under)':<20} {'Win Rate'}")
    print("="*60)

    for stat in stats:
        pred_col = stat
        act_col = f"{stat}_ACTUAL"
        
        if pred_col not in merged.columns or act_col not in merged.columns:
            continue
            
        # Calculate Difference
        merged[f'{stat}_DIFF'] = merged[pred_col] - merged[act_col]
        merged[f'{stat}_ABS_ERR'] = merged[f'{stat}_DIFF'].abs()
        
        # Metrics
        mae = merged[f'{stat}_ABS_ERR'].mean()
        bias = merged[f'{stat}_DIFF'].mean() # Positive = Overpredicted, Negative = Underpredicted
        
        # "Win Rate" (How often was the prediction closer than the Vegas margin? e.g. 5 pts)
        # Using a standard margin of error for "Good Prediction"
        margin = {'PTS': 4, 'REB': 2, 'AST': 1.5, 'PRA': 5}
        wins = merged[merged[f'{stat}_ABS_ERR'] <= margin.get(stat, 2)]
        win_rate = (len(wins) / len(merged)) * 100
        
        print(f"{stat:<10} {mae:<20.2f} {bias:<20.2f} {win_rate:.1f}%")
        
        error_summary.append({'Stat': stat, 'MAE': mae, 'Bias': bias})

    # 6. Save Detailed Report
    output_file = os.path.join(PREDICTIONS_DIR, f"accuracy_report_{date_str}.csv")
    merged.to_csv(output_file, index=False)
    
    print("-" * 60)
    print(f"📝 Detailed Accuracy Report saved to: {output_file}")
    
    # 7. Identify Top Misses (Debugging)
    print("\n🔍 BIGGEST MISSES (PTS):")
    merged['PTS_ABS_ERR'] = merged['PTS_DIFF'].abs()
    misses = merged.sort_values('PTS_ABS_ERR', ascending=False).head(3)
    for _, row in misses.iterrows():
        print(f"   {row['PLAYER_NAME']}: Pred {row['PTS']:.1f} vs Actual {row['PTS_ACTUAL']} (Diff: {row['PTS_DIFF']:.1f})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        date_arg = sys.argv[1]
    else:
        # Default to yesterday
        from datetime import timedelta
        date_arg = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
    evaluate_performance(date_arg)