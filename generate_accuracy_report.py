"""
NBA Prediction Accuracy Report Generator

This script generates a daily HTML report to validate the model's performance
by comparing the previous day's predictions against the actual game results.
"""
import os
import sys
import subprocess
import pandas as pd
from datetime import datetime, timedelta

# --- Configuration ---
# This configuration is replicated from generate_full_report.py to ensure consistency.
STAT_CONFIG = {
    'STL': {'trust': '1_ELITE',    'mae': 0.56, 'label': '🟢 ELITE'},
    'REB': {'trust': '2_HIGH',     'mae': 3.10, 'label': '🟡 HIGH'},
    'BLK': {'trust': '3_MEDIUM',   'mae': 0.67, 'label': '🟠 MEDIUM'},
    'PRA': {'trust': '3_MEDIUM',   'mae': 6.88, 'label': '🟠 MEDIUM'},
    'PTS': {'trust': '4_VOLATILE', 'mae': 5.73, 'label': '🔴 VOLATILE'},
    'AST': {'trust': '4_VOLATILE', 'mae': 1.88, 'label': '🔴 VOLATILE'}
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Accuracy Report for {report_date}</title>
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <header class="header">
        <div class="logo">🏀 NBA Model</div>
        <nav class="nav">
            <a href="index.html">Prediction Viewer</a>
            <a href="master_sheet.html">Betting Sheet</a>
            <a href="accuracy_report.html">Accuracy Report</a>
        </nav>
    </header>
    <div class="container">
        <h1>NBA Accuracy Report</h1>
        <p style="text-align: center; font-size: 1.2em;">Analysis Date: {report_date}</p>

        <h2>🎯 Trust Scorecard</h2>
        <p>This table shows the model's performance based on its confidence level. A "Win" is defined as a prediction where the actual score was within the model's Mean Absolute Error (MAE).</p>
        {trust_scorecard_table}

        <h2>📈 Bias Analysis (Mean Signed Error)</h2>
        <p>This metric shows if the model is consistently predicting too high (positive bias) or too low (negative bias) for each stat. A value near zero is ideal.</p>
        {bias_analysis_table}

        <h2>🚨 Bad Beats</h2>
        <p>These are the Top 10 predictions where the model's prediction was furthest from the actual result, highlighting potential areas for investigation.</p>
        {bad_beats_table}
    </div>
</body>
</html>
"""

def generate_report_for_date(target_date: datetime):
    """Generates the full accuracy report for the given date."""
    date_str = target_date.strftime('%Y%m%d')
    date_str_iso = target_date.strftime('%Y-%m-%d')
    
    print(f"Generating accuracy report for {date_str_iso}...")

    # --- File Paths ---
    base_path = os.path.join('data', 'predictions')
    predictions_file = os.path.join(base_path, f'predictions_{date_str}.csv')
    actuals_file = os.path.join(base_path, f'predictions_{date_str}_ACTUAL.csv')
    report_output_file = os.path.join('docs', 'accuracy_report.html')
    
    # --- 1. Ensure Actuals Exist ---
    if not os.path.exists(actuals_file):
        print(f"'{actuals_file}' not found. Fetching actuals from the day before...")
        cmd = [sys.executable, 'scripts/automated_predictions.py', '--get-actuals', date_str_iso]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Successfully fetched actuals.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error fetching actuals for {date_str_iso}.")
            print(f"   STDOUT: {e.stdout}")
            print(f"   STDERR: {e.stderr}")
            sys.exit(1)
            
    # --- 2. Load Data ---
    if not os.path.exists(predictions_file):
        print(f"❌ Error: Prediction file '{predictions_file}' not found. Cannot generate report.")
        sys.exit(1)
        
    preds_df = pd.read_csv(predictions_file)
    actuals_df = pd.read_csv(actuals_file)
    
    # --- 3. Process & Merge Data ---
    # Unpivot (melt) dataframes to a long format
    id_vars = ['PLAYER_NAME', 'TEAM_NAME']
    preds_long = preds_df.melt(id_vars=id_vars, var_name='Stat', value_name='Prediction')
    actuals_long = actuals_df.melt(id_vars=id_vars, var_name='Stat', value_name='Actual')

    # Filter for stats we have config for
    stats_to_analyze = list(STAT_CONFIG.keys())
    preds_long = preds_long[preds_long['Stat'].isin(stats_to_analyze)]
    actuals_long = actuals_long[actuals_long['Stat'].isin(stats_to_analyze)]

    # Merge predictions and actuals
    df = pd.merge(preds_long, actuals_long, on=['PLAYER_NAME', 'TEAM_NAME', 'Stat'], how='inner')
    
    if df.empty:
        print("⚠️ No matching player data found between predictions and actuals. Cannot generate report.")
        sys.exit(0)

    # Add config data (MAE, Trust)
    df['Trust'] = df['Stat'].apply(lambda x: STAT_CONFIG[x]['label'])
    df['MAE'] = df['Stat'].apply(lambda x: STAT_CONFIG[x]['mae'])

    # --- 4. Perform Analysis ---
    # a) Calibration (Trust Scorecard)
    df['Error'] = df['Actual'] - df['Prediction']
    df['Abs_Error'] = df['Error'].abs()
    df['Is_Win'] = df['Abs_Error'] < df['MAE']
    
    trust_scorecard = df.groupby('Trust')['Is_Win'].value_counts(normalize=True).unstack(fill_value=0)
    trust_scorecard['Win_Rate_%'] = trust_scorecard.get(True, 0) * 100
    trust_scorecard = trust_scorecard[['Win_Rate_%']].sort_index()

    # b) Bias Analysis
    bias_analysis = df.groupby('Stat')['Error'].mean().reset_index()
    bias_analysis.rename(columns={'Error': 'Mean_Signed_Error'}, inplace=True)
    
    # c) Bad Beats
    bad_beats = df.sort_values(by='Abs_Error', ascending=False).head(10)
    bad_beats = bad_beats[['PLAYER_NAME', 'Stat', 'Prediction', 'Actual', 'Error', 'Abs_Error', 'Trust']]

    # --- 5. Generate HTML Report ---
    def format_bias(val):
        color = 'positive-bias' if val > 0 else 'negative-bias' if val < 0 else 'black'
        return f'<span class="{color}">{val:.2f}</span>'

    bias_analysis_html = bias_analysis.to_html(index=False, classes='small-table', formatters={'Mean_Signed_Error': format_bias}, escape=False)
    
    html_content = HTML_TEMPLATE.format(
        report_date=date_str_iso,
        trust_scorecard_table=trust_scorecard.to_html(classes='small-table', formatters={'Win_Rate_%': '{:,.1f}%'.format}),
        bias_analysis_table=bias_analysis_html,
        bad_beats_table=bad_beats.to_html(index=False)
    )
    
    # --- 6. Save File ---
    with open(report_output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"✅ Successfully generated HTML report: {report_output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate a model accuracy report for a given date.")
    parser.add_argument(
        '--date', 
        type=str, 
        help='Date to analyze in YYYY-MM-DD format. Defaults to yesterday.'
    )
    
    args = parser.parse_args()
    
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print("❌ Invalid date format. Please use YYYY-MM-DD.")
            sys.exit(1)
    else:
        # Default to yesterday
        target_date = datetime.now() - timedelta(days=1)
        
    generate_report_for_date(target_date)

if __name__ == "__main__":
    main()