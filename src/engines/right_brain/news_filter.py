import pandas as pd
import numpy as np
from typing import List, Dict

class NewsFilter:
    """
    Project Janus Phase 2.1: Smart News Filtering
    Responsible for noise reduction and reliability assessment of news data.
    """
    def __init__(self, keywords: List[str] = None):
        if keywords is None:
            # High-weight macro keywords
            self.keywords = [
                "Fed", "Federal Reserve", "inflation", "CPI", "PPI", "GDP",
                "interest rate", "FOMC", "unemployment", "non-farm payroll",
                "recession", "economic growth", "monetary policy", "central bank",
                "yield curve", "treasury", "debt ceiling", "PMI", "retail sales"
            ]
        else:
            self.keywords = keywords

    def filter_news(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters news based on high-weight keywords.
        Checks multiple possible columns: 'title', 'summary', 'description', 'content'.
        """
        if news_df.empty:
            return news_df
            
        # Possible columns that might contain news text
        text_columns = ['title', 'summary', 'description', 'content']
        existing_cols = [col for col in text_columns if col in news_df.columns]
        
        if not existing_cols:
            # If no text columns found, we can't filter, return empty to be safe
            return pd.DataFrame(columns=news_df.columns)
            
        pattern = '|'.join(self.keywords)
        
        # Search across all existing text columns
        # We use a helper series that combines all available text
        combined_text = news_df[existing_cols].fillna('').agg(' '.join, axis=1)
        
        # Case insensitive match
        is_relevant = combined_text.str.contains(pattern, case=False, na=False)
        return news_df[is_relevant].copy()

    def assess_reliability(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assesses daily news count and assigns a macro weight.
        If daily news count < 2, weight is reduced.
        Handles timestamps by truncating to pure date.
        """
        if news_df.empty:
            return pd.DataFrame(columns=['date', 'News_Count', 'macro_weight'])

        # Create a copy to avoid SettingWithCopyWarning
        df = news_df.copy()
        
        # Ensure date column is datetime and truncate to date
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Group by date and count
        daily_stats = df.groupby('date').size().reset_index(name='News_Count')
        
        # Rule: If News_Count < 2, reduce weight
        daily_stats['macro_weight'] = daily_stats['News_Count'].apply(lambda x: 1.0 if x >= 2 else 0.5)
        
        return daily_stats

    def run_pipeline(self, raw_news_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Full pipeline for Phase 2.1.
        Returns filtered news and daily reliability statistics.
        """
        filtered_news = self.filter_news(raw_news_df)
        reliability_stats = self.assess_reliability(filtered_news)
        
        return {
            "filtered_news": filtered_news,
            "reliability_stats": reliability_stats
        }

if __name__ == "__main__":
    # Simple test logic for Phase 2.1
    data = {
        'date': ['2024-01-01 10:00:00', '2024-01-01 15:00:00', '2024-01-02 09:00:00', '2024-01-03 12:00:00', '2024-01-03 18:00:00', '2024-01-04 11:00:00'],
        'title': [
            "Fed signals interest rate hold.", 
            "CPI inflation data released today.", 
            "Nothing important happened in the market.", # Should be filtered out
            "GDP growth exceeds expectations.",
            "Retail sales are up.",
            "Only one relevant news about Fed." 
        ],
        'summary': [
            "Details on Federal Reserve policy.",
            "Impact of CPI on markets.",
            "Market is quiet.",
            "Strong economic data.",
            "Consumer spending is strong.",
            "Brief mention of the Fed."
        ]
    }
    df = pd.DataFrame(data)
    nf = NewsFilter()
    results = nf.run_pipeline(df)
    
    print("Filtered News (should find keywords across title and summary):")
    print(results['filtered_news'][['date', 'title']])
    print("\nReliability Stats (should group same days correctly):")
    print(results['reliability_stats'])
