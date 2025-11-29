import sys
import os
import subprocess
import datetime
from gooey import Gooey, GooeyParser

# --- CONFIGURATION ---
PYTHON_EXE = sys.executable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Map script names to their paths
SCRIPTS = {
    "auto_pred": os.path.join(SCRIPT_DIR, "scripts", "automated_predictions.py"),
    "game_pred": os.path.join(SCRIPT_DIR, "src", "game_predictor.py"),
    "full_report": os.path.join(SCRIPT_DIR, "src", "generate_full_report.py"),
    "track_perf": os.path.join(SCRIPT_DIR, "scripts", "track_performance.py"),
    "acc_report": os.path.join(SCRIPT_DIR, "generate_accuracy_report.py"),
    "dvp_stats": os.path.join(SCRIPT_DIR, "generate_dvp_stats.py"),
    "bet_analyzer": os.path.join(SCRIPT_DIR, "bet_analyzer.py"),
    "single_pred": os.path.join(SCRIPT_DIR, "src", "prediction_pipeline.py"),
}

# --- DEPENDENCY CHECKERS ---
def check_file(filepath, description):
    if not os.path.exists(filepath):
        print(f"❌ ERROR: Missing Dependency.")
        print(f"   Could not find {description}:")
        print(f"   {filepath}")
        print("   Please run the necessary prerequisite script first.")
        return False
    return True

def get_date_str(args):
    """Extract date string from args, handling default today if needed."""
    if hasattr(args, 'date') and args.date:
        return args.date
    return datetime.datetime.now().strftime('%Y-%m-%d')

# --- EXECUTION FUNCTIONS ---

def run_automated_predictions(args):
    cmd = [PYTHON_EXE, SCRIPTS["auto_pred"], "--run-once"]
    
    if args.date:
        cmd.extend(["--date", args.date])
    
    if args.team:
        cmd.extend(["--team", args.team])
        
    if args.get_actuals:
        # Override to just get actuals
        cmd = [PYTHON_EXE, SCRIPTS["auto_pred"], "--get-actuals", args.date]

    print("🏀 Starting Automated Predictions...")
    subprocess.run(cmd)

def run_game_predictor(args):
    # Dependency: Predictions CSV
    # Since game_predictor finds the latest file automatically, we trust it,
    # but we can warn if the folder is empty.
    pred_dir = os.path.join(SCRIPT_DIR, "data", "predictions")
    if not os.path.exists(pred_dir) or not os.listdir(pred_dir):
        print("❌ ERROR: No prediction files found in data/predictions/")
        print("   Please run 'Automated Predictions' first.")
        return

    print("🎮 Running Game Simulator...")
    subprocess.run([PYTHON_EXE, SCRIPTS["game_pred"]])

def run_full_report(args):
    # Dependency: The input CSV
    if not check_file(args.input_file, "Input Prediction CSV"):
        return

    print("Pp Generating Betting Sheet...")
    subprocess.run([PYTHON_EXE, SCRIPTS["full_report"], args.input_file])

def run_track_performance(args):
    target_date = args.date
    date_clean = target_date.replace("-", "")
    
    # Check Dependencies
    pred_file = os.path.join(SCRIPT_DIR, "data", "predictions", f"predictions_{date_clean}.csv")
    actual_file = os.path.join(SCRIPT_DIR, "data", "predictions", f"predictions_{date_clean}_ACTUAL.csv")
    
    if not check_file(pred_file, f"Predictions for {target_date}"): return
    if not check_file(actual_file, f"Actual Results for {target_date}"): return

    print("📈 Tracking Performance...")
    subprocess.run([PYTHON_EXE, SCRIPTS["track_perf"], "--date", target_date])

def run_accuracy_report(args):
    target_date = args.date
    # Note: Accuracy report logic might need update to accept date arg, 
    # passing it just in case or relying on its internal default
    print("🎯 Generating Accuracy HTML Report...")
    subprocess.run([PYTHON_EXE, SCRIPTS["acc_report"], "--date", target_date])

def run_dvp_stats(args):
    print("🛡️ Generating DvP Stats (This takes a while)...")
    subprocess.run([PYTHON_EXE, SCRIPTS["dvp_stats"]])

def run_bet_analyzer(args):
    # Special Case: Interactive Script
    # We must open a NEW terminal window because Gooey captures stdout and can't handle input()
    print("🎰 Launching Bet Analyzer in new window...")
    
    if sys.platform == 'win32':
        subprocess.Popen(f'start cmd /k "{PYTHON_EXE} {SCRIPTS["bet_analyzer"]}"', shell=True)
    else:
        # Mac/Linux fallback (basic attempt)
        print("   Please run 'python bet_analyzer.py' in your terminal.")

def run_single_pipeline(args):
    print(f"👤 Running Single Player Pipeline for {args.player}...")
    subprocess.run([PYTHON_EXE, SCRIPTS["single_pred"], "--player", args.player])

# --- GOOEY GUI CONFIGURATION ---

@Gooey(
    program_name="NBA Model Commander",
    program_description="Master Control Panel for NBA Machine Learning Pipeline",
    default_size=(900, 700),
    navigation='Tabbed',
    header_bg_color="#1D428A", # NBA Blue
    body_bg_color="#F9F9F9"
)
def main():
    parser = GooeyParser(description="Select a task to run")
    subs = parser.add_subparsers(help="commands", dest="command")

    # TAB 1: DAILY WORKFLOW
    # ---------------------
    daily_parser = subs.add_parser('Daily_Workflow', help='Daily Routine')
    
    daily_subs = daily_parser.add_subparsers(help="Daily Tasks", dest="subcommand")

    # 1. Automated Predictions
    ap_parser = daily_subs.add_parser('1_Run_Predictions', help='Fetch games & predict props')
    ap_parser.add_argument('--date', widget='DateChooser', help='Date to predict (Default: Today)')
    ap_parser.add_argument('--team', help='Filter by Team Name (Optional)')
    ap_parser.add_argument('--get_actuals', action='store_true', help='Fetch ACTUAL results instead of predicting')

    # 2. Game Predictor
    gp_parser = daily_subs.add_parser('2_Game_Simulator', help='Calculate Game Scores & Spreads')
    
    # 3. Betting Sheet
    bs_parser = daily_subs.add_parser('3_Create_Betting_Sheet', help='Generate Master CSV')
    bs_parser.add_argument('input_file', widget='FileChooser', help='Select the raw predictions_[date].csv')


    # TAB 2: ANALYSIS & TRACKING
    # --------------------------
    analysis_parser = subs.add_parser('Analysis', help='Performance & Tracking')
    analysis_subs = analysis_parser.add_subparsers(dest="subcommand")

    # 4. Track Performance
    tp_parser = analysis_subs.add_parser('Update_History', help='Add results to history.json')
    tp_parser.add_argument('date', widget='DateChooser', help='Date to verify')

    # 5. Accuracy Report
    ar_parser = analysis_subs.add_parser('Html_Report', help='Generate Accuracy HTML')
    ar_parser.add_argument('--date', widget='DateChooser', help='Date to analyze')

    # 6. Bet Analyzer
    ba_parser = analysis_subs.add_parser('Bet_Analyzer', help='Interactive Edge Calculator')


    # TAB 3: UTILITIES
    # ----------------
    util_parser = subs.add_parser('Utilities', help='Setup & Debug')
    util_subs = util_parser.add_subparsers(dest="subcommand")

    # 7. DvP Stats
    dvp_parser = util_subs.add_parser('Update_DvP_Data', help='Refetch Defense vs Position stats')

    # 8. Single Player
    sp_parser = util_subs.add_parser('Debug_Player', help='Run pipeline for one player')
    sp_parser.add_argument('player', help='Full Player Name (e.g. LeBron James)')


    args = parser.parse_args()

    # Routing Logic
    if args.subcommand == '1_Run_Predictions':
        run_automated_predictions(args)
    elif args.subcommand == '2_Game_Simulator':
        run_game_predictor(args)
    elif args.subcommand == '3_Create_Betting_Sheet':
        run_full_report(args)
    elif args.subcommand == 'Update_History':
        run_track_performance(args)
    elif args.subcommand == 'Html_Report':
        run_accuracy_report(args)
    elif args.subcommand == 'Bet_Analyzer':
        run_bet_analyzer(args)
    elif args.subcommand == 'Update_DvP_Data':
        run_dvp_stats(args)
    elif args.subcommand == 'Debug_Player':
        run_single_pipeline(args)

if __name__ == '__main__':
    main()