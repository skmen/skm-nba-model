import os
import pandas as pd
import numpy as np
from src.config import RAW_DATA_DIR, FEATURES
from src.utils import logger
from src.feature_engineer import engineer_features
from src.data_fetcher import get_opponent_defense_metrics

# Simple heuristics to classify players
KNOWN_CENTERS = ['Nikola Jokic', 'Joel Embiid', 'Rudy Gobert', 'Bam Adebayo', 'Domantas Sabonis']
KNOWN_GUARDS = ['Luka Doncic', 'Stephen Curry', 'Trae Young', 'Shai Gilgeous-Alexander', 'Tyrese Haliburton']

def get_player_bucket(player_name, row_data):
    """Determines if a player is a Guard, Wing, or Big."""
    if player_name in KNOWN_CENTERS: return 'Big'
    if player_name in KNOWN_GUARDS: return 'Guard'
    
    avg_reb = row_data['REB'].mean()
    avg_ast = row_data['AST'].mean()
    
    if avg_reb > 7.0 and avg_ast < 4.0:
        return 'Big'
    elif avg_ast > 4.5:
        return 'Guard'
    else:
        return 'Wing'

def load_and_bucket_data(season="2024-25"):
    """
    Iterates through all CSVs, ENGINEERS FEATURES, buckets them, and returns 3 DataFrames.
    """
    logger.info("📚 Loading, Engineering, and Bucketing historical data...")
    
    # 1. Fetch Context Data Once (Efficiency)
    try:
        opp_defense = get_opponent_defense_metrics(season)
    except Exception:
        logger.warning("Could not fetch opponent defense. using defaults.")
        opp_defense = None

    guards_data = []
    wings_data = []
    bigs_data = []
    
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]
    
    if not files:
        raise FileNotFoundError(f"No data found in {RAW_DATA_DIR}. Run fetch operations first.")

    count = 0
    skipped = 0
    
    for filename in files:
        try:
            filepath = os.path.join(RAW_DATA_DIR, filename)
            df = pd.read_csv(filepath)
            
            # Basic validation
            if df.empty or 'PTS' not in df.columns or len(df) < 5:
                skipped += 1
                continue
                
            # --- CRITICAL: ENGINEER FEATURES HERE ---
            # We must engineer BEFORE merging, otherwise lag stats bleed between players
            # We use a default usage rate (20%) for training to save API calls
            engineered_df = engineer_features(df, opp_defense, usage_rate=20.0)
            
            if engineered_df is None or engineered_df.empty:
                skipped += 1
                continue

            # Identify Player & Bucket
            player_name = df['PLAYER_NAME'].iloc[0] if 'PLAYER_NAME' in df.columns else "Unknown"
            bucket = get_player_bucket(player_name, df)

            # Map Bucket to Position Group for Feature Engineering
            # 'Guard' -> 'G', 'Big' -> 'C', 'Wing' -> 'F'
            pos_map = {'Guard': 'G', 'Big': 'C', 'Wing': 'F'}
            pos_group = pos_map.get(bucket, 'F')

            # Engineer Features
            # We pass 'None' for opponent_defense if we don't have historical data.
            # The updated feature_engineer.py will handle this by using defaults.
            engineered_df = engineer_features(
                df, 
                opponent_defense=opp_defense, 
                usage_rate=20.0,
                position_group=pos_group # <--- CRITICAL ADDITION
            )
            
            if engineered_df is None or engineered_df.empty:
                skipped += 1
                continue
            
            # Add to list
            if bucket == 'Guard':
                guards_data.append(engineered_df)
            elif bucket == 'Big':
                bigs_data.append(engineered_df)
            else:
                wings_data.append(engineered_df)
                
            count += 1
            if count % 20 == 0:
                logger.info(f"Processed {count} players...")
                
        except Exception as e:
            logger.warning(f"Skipping {filename}: {e}")
            skipped += 1

    # Concatenate
    df_guards = pd.concat(guards_data, ignore_index=True) if guards_data else pd.DataFrame()
    df_wings = pd.concat(wings_data, ignore_index=True) if wings_data else pd.DataFrame()
    df_bigs = pd.concat(bigs_data, ignore_index=True) if bigs_data else pd.DataFrame()
    
    logger.info(f"✅ Data Ready: Guards({len(df_guards)}), Wings({len(df_wings)}), Bigs({len(df_bigs)})")
    
    return df_guards, df_wings, df_bigs