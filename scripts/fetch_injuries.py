#!/usr/bin/env python3
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.injury_fetcher import fetch_injuries, save_injuries_to_csv
from src.utils import print_section, print_result

def main():
    parser = argparse.ArgumentParser(description="Fetch NBA Daily Injury Report")
    parser.add_argument('--out', type=str, help="Custom output path")
    args = parser.parse_args()

    print_section("FETCHING DAILY INJURIES")
    
    df = fetch_injuries()
    
    if df.empty:
        print_result("Status", "Failed (No data found)")
        sys.exit(1)
        
    # Filter for 'Out' or 'Doubtful' status if desired, 
    # but usually we save raw and filter in the prediction step.
    # We will save the RAW report.
    
    saved_path = save_injuries_to_csv(df)
    
    print_result("Total Players Listed", len(df))
    print_result("Saved To", saved_path)
    
    # Optional: Quick Preview of OUT players
    out_players = df[df['STATUS'].str.contains('Out|Season', case=False, na=False)]
    print(f"\nPreview of OUT players ({len(out_players)}):")
    for _, row in out_players.head(5).iterrows():
        print(f" - {row['PLAYER_NAME']}: {row['STATUS']}")

if __name__ == "__main__":
    main()