"""
Injury Fetcher Module

Scrapes NBA injury reports from CBS Sports.
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import os
from src.utils import setup_logger, print_section

logger = setup_logger(__name__)

CBS_INJURY_URL = "https://www.cbssports.com/nba/injuries/"

def fetch_injuries() -> pd.DataFrame:
    """
    Scrapes CBS Sports NBA injuries page and returns a DataFrame.
    """
    try:
        logger.info(f"Fetching injuries from {CBS_INJURY_URL}...")
        
        # Add headers to mimic a browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(CBS_INJURY_URL, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # CBS Sports organizes injuries by Team tables
        tables = soup.find_all('table', class_='TableBase-table')
        
        all_injuries = []
        
        if not tables:
            logger.warning("No injury tables found on CBS page.")
            return pd.DataFrame()

        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 4:
                    # Column structure: Player, Position, Updated, Injury, Status
                    # Note: Player name often has a span with the team or position inside, need to clean text
                    player_cell = cols[0].find('span', class_='CellPlayerName--long')
                    player_name = player_cell.get_text(strip=True) if player_cell else cols[0].get_text(strip=True)
                    
                    status = cols[4].get_text(strip=True)
                    
                    all_injuries.append({
                        'PLAYER_NAME': player_name,
                        'POSITION': cols[1].get_text(strip=True),
                        'INJURY': cols[3].get_text(strip=True),
                        'STATUS': status,
                        'FETCH_DATE': datetime.now().strftime('%Y-%m-%d')
                    })
        
        df = pd.DataFrame(all_injuries)
        logger.info(f"Successfully scraped {len(df)} injury reports.")
        return df

    except Exception as e:
        logger.error(f"Error fetching injuries: {e}")
        return pd.DataFrame()

def save_injuries_to_csv(df: pd.DataFrame, folder="data/injuries") -> str:
    """Saves injury dataframe to CSV."""
    if df.empty:
        return None
        
    os.makedirs(folder, exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d')
    filename = os.path.join(folder, f"injuries_{date_str}.csv")
    
    df.to_csv(filename, index=False)
    logger.info(f"Saved injury report to {filename}")
    return filename