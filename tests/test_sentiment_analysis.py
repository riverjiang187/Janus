import unittest
from unittest.mock import MagicMock, PropertyMock
from src.engines.right_brain.sentiment_analysis import SentimentAnalyzer, get_sentiment_score

class TestSentimentAnalysis(unittest.TestCase):
    def setUp(self):
        self.analyzer = SentimentAnalyzer(api_key="mock_key")

    def test_prompt_generation(self):
        news = "Test news content"
        prompt = self.analyzer.generate_prompt(news)
        self.assertIn("You are a professional Macro Economist", prompt)
        self.assertIn("Test news content", prompt)
        self.assertIn("JSON Output:", prompt)

    def test_json_parsing_success(self):
        response = '{"sentiment": 0.5, "relevance": 0.8}'
        result = self.analyzer.parse_response(response)
        self.assertEqual(result['sentiment'], 0.5)
        self.assertEqual(result['relevance'], 0.8)

    def test_json_parsing_with_markdown(self):
        # Testing if it can extract JSON from markdown blocks
        response = 'Sure, here is the analysis: ```json\n{"sentiment": -0.2, "relevance": 0.5}\n``` hope it helps.'
        result = self.analyzer.parse_response(response)
        self.assertEqual(result['sentiment'], -0.2)
        self.assertEqual(result['relevance'], 0.5)

    def test_hallucination_defense_fallback(self):
        # If news is vague or output is bad, should return 0.0
        response = "The news is too vague to determine sentiment."
        result = self.analyzer.parse_response(response)
        self.assertEqual(result['sentiment'], 0.0)
        self.assertEqual(result['relevance'], 0.0)

    def test_partial_json_failure(self):
        # Missing one field or malformed
        response = '{"sentiment": 0.3}'
        result = self.analyzer.parse_response(response)
        self.assertEqual(result['sentiment'], 0.3)
        self.assertEqual(result['relevance'], 0.0) # Should default to 0.0 for missing field

    def test_get_sentiment_score_interface(self):
        # Ensure it returns a dictionary now
        result = get_sentiment_score("Fake news")
        self.assertIsInstance(result, dict)
        self.assertIn("sentiment", result)
        self.assertIn("relevance", result)

    def test_analyze_news_no_key(self):
        # Test behavior when API key is missing
        analyzer = SentimentAnalyzer(api_key="")
        result = analyzer.analyze_news("Test")
        self.assertEqual(result, {"sentiment": 0.0, "relevance": 0.0})

    def test_value_clamping(self):
        # Too high
        response = '{"sentiment": 1.5, "relevance": 2.0}'
        result = self.analyzer.parse_response(response)
        self.assertEqual(result['sentiment'], 1.0)
        self.assertEqual(result['relevance'], 1.0)
        
        # Too low
        response = '{"sentiment": -2.0, "relevance": -0.5}'
        result = self.analyzer.parse_response(response)
        self.assertEqual(result['sentiment'], -1.0)
        self.assertEqual(result['relevance'], 0.0)

    def test_short_circuit_empty_input(self):
        # Should return 0.0 for empty/whitespace/None
        self.assertEqual(self.analyzer.analyze_news(""), {"sentiment": 0.0, "relevance": 0.0})
        self.assertEqual(self.analyzer.analyze_news("   "), {"sentiment": 0.0, "relevance": 0.0})
        self.assertEqual(self.analyzer.analyze_news(None), {"sentiment": 0.0, "relevance": 0.0})

    def test_analyze_news_safety_block(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        # In new SDK, response.text might be None if blocked or error
        mock_response.text = None
        mock_client.models.generate_content.return_value = mock_response
        
        self.analyzer.client = mock_client
        result = self.analyzer.analyze_news("Risky content")
        self.assertEqual(result, {"sentiment": 0.0, "relevance": 0.0})

if __name__ == '__main__':
    unittest.main()
