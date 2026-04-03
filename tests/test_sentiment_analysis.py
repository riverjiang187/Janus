import unittest
from unittest.mock import MagicMock
from src.engines.right_brain.sentiment_analysis import SentimentAnalyzer, get_sentiment_score

class TestSentimentAnalysis(unittest.TestCase):
    def setUp(self):
        self.analyzer = SentimentAnalyzer(api_key="mock_key")
        self.default_fallback = {
            "macro_sentiment": 0.0, "macro_relevance": 0.0,
            "asset_sentiment": 0.0, "asset_relevance": 0.0
        }

    def test_prompt_generation(self):
        news = "Test news content"
        target = "BTC"
        prompt = self.analyzer.generate_prompt(news, target_symbol=target)
        self.assertIn("You are a professional Macro Economist", prompt)
        self.assertIn("Test news content", prompt)
        self.assertIn(f"target asset: {target}", prompt)
        self.assertIn("JSON Output:", prompt)

    def test_json_parsing_success(self):
        response = '{"macro_sentiment": 0.5, "macro_relevance": 0.8, "asset_sentiment": -0.3, "asset_relevance": 0.9}'
        result = self.analyzer.parse_response(response)
        self.assertEqual(result['macro_sentiment'], 0.5)
        self.assertEqual(result['macro_relevance'], 0.8)
        self.assertEqual(result['asset_sentiment'], -0.3)
        self.assertEqual(result['asset_relevance'], 0.9)

    def test_json_parsing_with_markdown(self):
        # Testing if it can extract JSON from markdown blocks
        response = 'Sure! ```json\n{"macro_sentiment": 0.1, "macro_relevance": 0.2, "asset_sentiment": 0.3, "asset_relevance": 0.4}\n```'
        result = self.analyzer.parse_response(response)
        self.assertEqual(result['macro_sentiment'], 0.1)
        self.assertEqual(result['asset_relevance'], 0.4)

    def test_hallucination_defense_fallback(self):
        # If output is bad, should return 0.0s
        response = "The news is too vague."
        result = self.analyzer.parse_response(response)
        self.assertEqual(result, self.default_fallback)

    def test_partial_json_failure(self):
        # Missing fields should default to 0.0
        response = '{"macro_sentiment": 0.3}'
        result = self.analyzer.parse_response(response)
        self.assertEqual(result['macro_sentiment'], 0.3)
        self.assertEqual(result['macro_relevance'], 0.0)
        self.assertEqual(result['asset_sentiment'], 0.0)

    def test_get_sentiment_score_interface(self):
        # Ensure it returns a float after Phase 3.2 synthesis
        result = get_sentiment_score("Fake news", target_symbol="AAPL")
        self.assertIsInstance(result, float)

    def test_analyze_news_no_key(self):
        # Test behavior when API key is missing
        analyzer = SentimentAnalyzer(api_key="")
        result = analyzer.analyze_news("Test")
        self.assertEqual(result, self.default_fallback)

    def test_value_clamping(self):
        # Values outside [-1, 1] or [0, 1]
        response = '{"macro_sentiment": 1.5, "macro_relevance": -0.5, "asset_sentiment": -2.0, "asset_relevance": 2.0}'
        result = self.analyzer.parse_response(response)
        self.assertEqual(result['macro_sentiment'], 1.0)
        self.assertEqual(result['macro_relevance'], 0.0)
        self.assertEqual(result['asset_sentiment'], -1.0)
        self.assertEqual(result['asset_relevance'], 1.0)

    def test_short_circuit_empty_input(self):
        # Should return default for empty/whitespace/None
        self.assertEqual(self.analyzer.analyze_news(""), self.default_fallback)
        self.assertEqual(self.analyzer.analyze_news("   "), self.default_fallback)
        self.assertEqual(self.analyzer.analyze_news(None), self.default_fallback)

    def test_analyze_news_safety_block(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = None
        mock_client.models.generate_content.return_value = mock_response
        
        self.analyzer.client = mock_client
        result = self.analyzer.analyze_news("Risky content")
        self.assertEqual(result, self.default_fallback)

    def test_sentiment_synthesis_logic(self):
        from unittest.mock import patch
        mock_results = {
            "macro_sentiment": 0.5, "macro_relevance": 0.8, # adj_macro = 0.4
            "asset_sentiment": 0.2, "asset_relevance": 0.5  # adj_asset = 0.1
        }
        # Expected score: (0.3 * 0.4) + (0.7 * 0.1) = 0.12 + 0.07 = 0.19
        
        with patch('src.engines.right_brain.sentiment_analysis.SentimentAnalyzer.analyze_news', return_value=mock_results):
            score = get_sentiment_score("Some news", "AAPL")
            self.assertAlmostEqual(score, 0.19)

    def test_macro_veto_logic(self):
        from unittest.mock import patch
        # Case where adj_macro <= -0.8
        mock_results = {
            "macro_sentiment": -1.0, "macro_relevance": 0.9, # adj_macro = -0.9
            "asset_sentiment": 1.0, "asset_relevance": 1.0   # adj_asset = 1.0
        }
        # Even with positive asset sentiment, macro veto should trigger -1.0
        
        with patch('src.engines.right_brain.sentiment_analysis.SentimentAnalyzer.analyze_news', return_value=mock_results):
            score = get_sentiment_score("Economic collapse but AAPL is okay", "AAPL")
            self.assertEqual(score, -1.0)

if __name__ == '__main__':
    unittest.main()
