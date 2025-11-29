import pandas as pd
from src.config import POSSESSION_BASE

class GameSimulator:
    def __init__(self, home_team, away_team, player_predictions):
        self.home_team = home_team
        self.away_team = away_team
        self.preds = player_predictions # DataFrame of all players in this game
        
    def calculate_game_pace(self):
        """
        Calculates projected possessions for the GAME (48 mins).
        Formula: (Home_Pace + Away_Pace) / 2
        """
        # You would fetch these from your team stats cache
        home_pace = self.preds[self.preds['TEAM_NAME'] == self.home_team]['PACE'].mean()
        away_pace = self.preds[self.preds['TEAM_NAME'] == self.away_team]['PACE'].mean()
        
        return (home_pace + away_pace) / 2

    def simulate(self) -> pd.DataFrame:
        if self.preds.empty:
            return pd.DataFrame()

        game_pace = self.calculate_game_pace()
        
        # 1. Normalize Minutes (Ensure Team Total Minutes == 240)
        # This prevents inflating scores by summing 15 players' minutes
        for team in [self.home_team, self.away_team]:
            team_mask = self.preds['TEAM_NAME'] == team
            total_pred_min = self.preds.loc[team_mask, 'MINUTES'].sum()
            
            if total_pred_min > 0:
                # e.g., if Total is 300, factor is 0.8. All minutes shrink to fit 240.
                scaling_factor = 240.0 / total_pred_min
                self.preds.loc[team_mask, 'MINUTES'] *= scaling_factor

        results = []
        
        for _, player in self.preds.iterrows():
            # 2. Recalculate Opportunity
            minutes = player.get('MINUTES', 0)
            usg = player.get('USAGE_PCT', 0.20) # Read the normalized usage we saved
            
            team_possessions = (minutes / 48.0) * game_pace
            personal_possessions = team_possessions * usg
            
            # 3. Apply Rates
            def get_total(stat_name):
                rate_col = f"{stat_name}_PER_100"
                if rate_col in player:
                    return (player[rate_col] / POSSESSION_BASE) * personal_possessions
                elif stat_name in player:
                    # Fallback for columns without rates
                    return player[stat_name] # Raw value might be inaccurate if pace changes
                return 0.0

            sim_stats = {
                'PLAYER_NAME': player['PLAYER_NAME'],
                'TEAM': player['TEAM_NAME'],
                'MINUTES': round(minutes, 1),
                'PROJ_PTS': get_total('PTS'),
                'PROJ_REB': get_total('REB'),
                'PROJ_AST': get_total('AST'),
                # ... other stats
            }
            results.append(sim_stats)
            
        return pd.DataFrame(results)

    def get_game_summary(self, simulated_df):
        """Aggregates players into team scores and spreads."""
        team_totals = simulated_df.groupby('TEAM')[['PROJ_PTS', 'PROJ_REB']].sum()
        
        home_score = team_totals.loc[self.home_team, 'PROJ_PTS']
        away_score = team_totals.loc[self.away_team, 'PROJ_PTS']
        
        spread = home_score - away_score
        total = home_score + away_score
        
        return {
            'HOME_SCORE': home_score,
            'AWAY_SCORE': away_score,
            'SPREAD': spread, # Negative means Away is favored (usually), Positive means Home favored
            'TOTAL': total,
            'WINNER': self.home_team if spread > 0 else self.away_team
        }