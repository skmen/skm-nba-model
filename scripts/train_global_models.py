import sys
import os
import pandas as pd

# Fix path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_and_bucket_data
from src.model import NBAPlayerStack  # UPDATED CLASS NAME
from src.config import FEATURES, TARGETS
from src.utils import logger

def train_all_buckets():
    # 1. Load Data (Features are now engineered inside the loader)
    df_guards, df_wings, df_bigs = load_and_bucket_data()
    
    buckets = [
        ('Guards', df_guards),
        ('Wings', df_wings),
        ('Bigs', df_bigs)
    ]
    
    # 2. Train and Save
    for name, df in buckets:
        if df.empty:
            logger.warning(f"Skipping {name} (No Data)")
            continue
            
        logger.info(f"--- Training {name} Models ---")
        
        # We now need to train a separate model for EACH target (PTS, REB, AST, etc.)
        for target in TARGETS:
            try:
                # Filter for training columns
                # We need features + the specific target we are training for
                cols_needed = FEATURES + [target]
                
                # Drop rows where this specific target is missing (NaN)
                train_data = df[cols_needed].dropna()
                
                if train_data.empty:
                    logger.warning(f"No valid data for {name} - {target}")
                    continue

                X = train_data[FEATURES]
                y = train_data[target]
                
                # Initialize Stack
                model_name = f"{name}_{target}" # e.g., Guards_PTS
                model = NBAPlayerStack(name=model_name)
                
                # Train
                logger.info(f"Fitting {model_name} on {len(X)} rows...")
                model.fit(X, y)
                
                # Save
                save_name = f"model_{name.lower()}_{target.lower()}.pkl"
                model.save(save_name)
                
            except Exception as e:
                logger.error(f"Failed to train {name} - {target}: {e}")

if __name__ == "__main__":
    train_all_buckets()