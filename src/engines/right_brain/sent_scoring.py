import numpy as np
from numba import njit
import pandas as pd

@njit
def compute_decay_numba(scores: np.ndarray, has_news_mask: np.ndarray, decay_factor: float = 0.85) -> np.ndarray:
    """
    Numba-accelerated sentiment decay and reset logic.
    Since current state depends on yesterday, this is not directly vectorizable.
    
    Args:
        scores (np.ndarray): Daily sentiment scores (e.g. averaged if multiple news).
        has_news_mask (np.ndarray): Boolean array where True means news exists for that day.
        decay_factor (float): Factor to decay the score when no news (default 0.85).
        
    Returns:
        np.ndarray: Smoothed/decayed daily scores.
    """
    n = len(scores)
    result = np.zeros(n, dtype=np.float64)
    last_score = 0.0
    
    for i in range(n):
        if has_news_mask[i]:
            # Reset: New news overrides decayed score
            current = scores[i]
        else:
            # Decay: Previous score decays
            current = last_score * decay_factor
        
        result[i] = current
        last_score = current
        
    return result

class SentimentProcessor:
    """
    Project Janus Phase 2.4: Sentiment Decay & Reset Logic
    Handles time-series smoothing and state-dependent updates.
    """
    
    def __init__(self, decay_factor: float = 0.85):
        self.decay_factor = decay_factor

    def process_sentiment_series(self, df: pd.DataFrame, date_col: str = 'date', score_col: str = 'score') -> pd.DataFrame:
        """
        Takes a DataFrame with daily scores, fills missing dates, and applies decay.
        Implements T+1 delay logic: news from day T impacts day T+1.
        
        Args:
            df (pd.DataFrame): DataFrame containing 'date' and 'score'.
            date_col (str): Name of the date column.
            score_col (str): Name of the score column.
            
        Returns:
            pd.DataFrame: A full-range daily DataFrame with original and decayed scores.
        """
        if df.empty:
            return pd.DataFrame()
            
        # Convert to datetime, normalize to date (T-trap defense), and sort
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
        
        # Apply T+1 Delay: News from day T impacts day T+1
        df[date_col] = df[date_col] + pd.Timedelta(days=1)
        
        df = df.sort_values(date_col)
        
        # Aggregate if there are multiple entries for the same date
        daily_avg = df.groupby(date_col)[score_col].mean().reset_index()
        
        if daily_avg.empty:
            return pd.DataFrame()
            
        # Create full date range
        min_date = daily_avg[date_col].min()
        max_date = daily_avg[date_col].max()
        full_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        # Reindex to fill gaps
        full_df = pd.DataFrame({date_col: full_range})
        full_df = full_df.merge(daily_avg, on=date_col, how='left')
        
        # Prepare arrays for Numba
        # Score is 0 if NaN, mask tracks where we actually had news
        scores_arr = full_df[score_col].fillna(0.0).values
        has_news_mask = full_df[score_col].notna().values
        
        # Run Numba-accelerated logic
        decayed_scores = compute_decay_numba(scores_arr, has_news_mask, self.decay_factor)
        
        full_df['decayed_score'] = decayed_scores
        
        return full_df

from typing import Optional, Union

def get_sent_score(df: pd.DataFrame, target_date: str = None) -> Union[pd.Series, float]:
    """
    Project Janus Phase 2 Facade: Unified Sentiment Scoring Interface.
    Includes T+1 delay and decay logic.
    
    Args:
        df: DataFrame containing 'date' and 'score'.
        target_date: Specific date for point-in-time score (YYYY-MM-DD). If None, returns full Series.
        
    Returns:
        Union[pd.Series, float]: Full time-series (decayed) or a single score for target_date.
    """
    processor = SentimentProcessor()
    full_df = processor.process_sentiment_series(df)
    
    if full_df.empty:
        return pd.Series() if target_date is None else 0.0
        
    # Re-index to use date for easier access
    full_df = full_df.set_index('date')
    scores = full_df['decayed_score']
    
    if target_date is not None:
        target_dt = pd.to_datetime(target_date).normalize()
        if target_dt in scores.index:
            return float(scores.loc[target_dt])
        else:
            # If target date is beyond the range, we might need to extend decay
            # But for simplicity, we return 0.0 or the last decayed value if it's within reason
            return 0.0
            
    return scores

if __name__ == "__main__":
    # Quick example test
    data = {
        'date': ['2024-01-01', '2024-01-03', '2024-01-06'],
        'score': [1.0, -0.5, 0.8]
    }
    df_test = pd.DataFrame(data)
    
    processor = SentimentProcessor(decay_factor=0.85)
    processed = processor.process_sentiment_series(df_test)
    
    print("Sentiment Decay Processed Results:")
    print(processed)
