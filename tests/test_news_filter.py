import unittest
import pandas as pd
from src.engines.right_brain.news_filter import NewsFilter

class TestNewsFilter(unittest.TestCase):
    def setUp(self):
        self.nf = NewsFilter()

    def test_filtering_with_various_columns(self):
        # Testing if it searches in title and summary when content is missing
        data = {
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'title': ["Fed signals rate hold", "Normal news", "Market update"],
            'summary': ["Some details about Fed", "Inflation is rising", "Nothing important"]
        }
        df = pd.DataFrame(data)
        filtered = self.nf.filter_news(df)
        # Should catch "Fed" in title/summary and "Inflation" in summary
        self.assertEqual(len(filtered), 2)
        self.assertTrue(any(filtered['title'].str.contains("Fed")))
        self.assertTrue(any(filtered['summary'].str.contains("Inflation")))

    def test_reliability_timestamp_trap(self):
        # Testing different times on the same day
        data = {
            'date': ['2024-01-01 10:00:00', '2024-01-01 14:30:00'],
            'title': ["Fed news 1", "Fed news 2"]
        }
        df = pd.DataFrame(data)
        stats = self.nf.assess_reliability(df)
        # Should be grouped into 1 day with News_Count = 2
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats.iloc[0]['News_Count'], 2)
        self.assertEqual(stats.iloc[0]['macro_weight'], 1.0)

    def test_reliability_full_weight(self):
        data = {
            'date': ['2024-01-01', '2024-01-01'],
            'content': ["Fed news 1", "Fed news 2"]
        }
        df = pd.DataFrame(data)
        stats = self.nf.assess_reliability(df)
        self.assertEqual(stats.iloc[0]['News_Count'], 2)
        self.assertEqual(stats.iloc[0]['macro_weight'], 1.0)

    def test_reliability_low_weight(self):
        data = {
            'date': ['2024-01-01'],
            'content': ["Fed news 1"]
        }
        df = pd.DataFrame(data)
        stats = self.nf.assess_reliability(df)
        self.assertEqual(stats.iloc[0]['News_Count'], 1)
        self.assertEqual(stats.iloc[0]['macro_weight'], 0.5)

    def test_empty_input(self):
        df = pd.DataFrame(columns=['date', 'content'])
        results = self.nf.run_pipeline(df)
        self.assertTrue(results['filtered_news'].empty)
        self.assertTrue(results['reliability_stats'].empty)

if __name__ == '__main__':
    unittest.main()
