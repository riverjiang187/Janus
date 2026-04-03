import unittest
import pandas as pd
import numpy as np
from src.engines.right_brain.sentiment_processor import SentimentProcessor, compute_decay_numba

class TestSentimentProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = SentimentProcessor(decay_factor=0.5) # Use 0.5 for easy mental math

    def test_basic_decay(self):
        # 1.0 (news) -> 0.5 (no news) -> 0.25 (no news) -> -0.8 (news)
        scores = np.array([1.0, 0.0, 0.0, -0.8])
        mask = np.array([True, False, False, True])
        
        result = compute_decay_numba(scores, mask, decay_factor=0.5)
        
        expected = np.array([1.0, 0.5, 0.25, -0.8])
        np.testing.assert_array_almost_equal(result, expected)

    def test_process_sentiment_series(self):
        # 2.4 Fix: T+1 Delay Logic
        # 1.0 news from 01-01 should appear on 01-02
        data = {
            'date': ['2024-01-01', '2024-01-04'],
            'score': [1.0, 0.8]
        }
        df = pd.DataFrame(data)
        
        # Expectation (Shifted + Decay 0.5):
        # 2024-01-01 -> News 1.0
        # 2024-01-04 -> News 0.8
        
        # Shifted Dates (Internal):
        # 2024-01-02: 1.0 (from 01-01)
        # 2024-01-05: 0.8 (from 01-04)
        
        # Result DataFrame Range:
        # 2024-01-02: 1.0
        # 2024-01-03: 0.5
        # 2024-01-04: 0.25
        # 2024-01-05: 0.8
        
        processed = self.processor.process_sentiment_series(df)
        
        self.assertEqual(len(processed), 4)
        # Row 0 is now 2024-01-02
        self.assertEqual(processed.iloc[0]['date'].strftime('%Y-%m-%d'), '2024-01-02')
        self.assertEqual(processed.iloc[0]['decayed_score'], 1.0)
        self.assertEqual(processed.iloc[1]['decayed_score'], 0.5)
        self.assertEqual(processed.iloc[2]['decayed_score'], 0.25)
        self.assertEqual(processed.iloc[3]['decayed_score'], 0.8)
        self.assertEqual(processed.iloc[3]['date'].strftime('%Y-%m-%d'), '2024-01-05')

    def test_duplicate_dates(self):
        # Multiple scores on same day should be averaged and shifted
        data = {
            'date': ['2024-01-01', '2024-01-01', '2024-01-02'],
            'score': [1.0, 0.0, 0.5]
        }
        df = pd.DataFrame(data)
        
        processed = self.processor.process_sentiment_series(df)
        
        # 01-01 avg (0.5) shifted to 01-02
        # 01-02 (0.5) shifted to 01-03
        self.assertEqual(processed.iloc[0]['date'].strftime('%Y-%m-%d'), '2024-01-02')
        self.assertEqual(processed.iloc[0]['decayed_score'], 0.5)
        self.assertEqual(processed.iloc[1]['date'].strftime('%Y-%m-%d'), '2024-01-03')
        self.assertEqual(processed.iloc[1]['decayed_score'], 0.5)

    def test_empty_df(self):
        df = pd.DataFrame()
        processed = self.processor.process_sentiment_series(df)
        self.assertTrue(processed.empty)

    def test_timestamp_trap(self):
        # Different times on the same day should be merged then shifted
        data = {
            'date': ['2024-01-01 10:00:00', '2024-01-01 23:59:59'],
            'score': [1.0, 0.0]
        }
        df = pd.DataFrame(data)
        
        processed = self.processor.process_sentiment_series(df)
        
        # Should be averaged (0.5) and shifted to 2024-01-02
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed.iloc[0]['date'].strftime('%Y-%m-%d'), '2024-01-02')
        self.assertEqual(processed.iloc[0]['decayed_score'], 0.5)

if __name__ == '__main__':
    unittest.main()
