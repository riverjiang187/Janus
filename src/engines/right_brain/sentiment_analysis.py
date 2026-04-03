import json
import os
import re
import logging
from typing import Dict, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv(override=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Project Janus Phase 2.2: Prompt Engineering & Sentiment Analysis
    Responsible for processing filtered news using Google Gemini API.
    Focuses on hallucination defense and strict format control.
    """
    
    SYSTEM_PROMPT = """
    You are a professional Macro Economist and Financial Analyst. 
    Your task is to analyze the sentiment of financial news and its impact on both the broad macro market and a specific asset.

    Instructions:
    1. Analyze the given news text for its sentiment towards the macro economy and the target asset: {target_symbol}.
    2. Assign the following 4 scores:
       - 'macro_sentiment': Impact on the overall broad market (-1.0 to 1.0).
       - 'macro_relevance': Relevance of the news to the macro economy (0.0 to 1.0).
       - 'asset_sentiment': Impact SPECIFICALLY on {target_symbol} (-1.0 to 1.0).
       - 'asset_relevance': Relevance of the news to {target_symbol} (0.0 to 1.0).
    
    Hallucination Defense:
    - If the news content is vague, ambiguous, or if you cannot determine the sentiment for either track, you MUST output 0.0 for that specific sentiment.
    - Do NOT guess or invent information.

    Output Format:
    - You MUST respond ONLY in JSON format.
    - Example: {{"macro_sentiment": 0.1, "macro_relevance": 0.8, "asset_sentiment": 0.4, "asset_relevance": 0.6}}
    - NO additional text, explanations, or code blocks.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            self.model_id = 'gemini-2.5-flash'
        else:
            logger.warning("GEMINI_API_KEY not found. API calls will fail.")
            self.client = None

    def generate_prompt(self, news_text: str, target_symbol: str = "General Market") -> str:
        """
        Combines the System Prompt with the news text, injecting the target symbol.
        """
        formatted_system_prompt = self.SYSTEM_PROMPT.format(target_symbol=target_symbol)
        return f"{formatted_system_prompt}\n\nNews to analyze:\n{news_text}\n\nJSON Output:"

    def parse_response(self, response_text: str) -> Dict[str, float]:
        """
        Parses the JSON response from the LLM. 
        Extracts and clamps macro_sentiment, macro_relevance, asset_sentiment, asset_relevance.
        """
        default_result = {
            "macro_sentiment": 0.0, "macro_relevance": 0.0,
            "asset_sentiment": 0.0, "asset_relevance": 0.0
        }
        try:
            # Attempt to find JSON-like structure if LLM adds markdown or chatter
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                clean_json = match.group(0)
                result = json.loads(clean_json)
                
                macro_s = float(result.get("macro_sentiment", 0.0))
                macro_r = float(result.get("macro_relevance", 0.0))
                asset_s = float(result.get("asset_sentiment", 0.0))
                asset_r = float(result.get("asset_relevance", 0.0))
                
                # Unsafe Value Bounds Protection: Clamp to strict ranges
                return {
                    "macro_sentiment": max(-1.0, min(1.0, macro_s)),
                    "macro_relevance": max(0.0, min(1.0, macro_r)),
                    "asset_sentiment": max(-1.0, min(1.0, asset_s)),
                    "asset_relevance": max(0.0, min(1.0, asset_r))
                }
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            
        # Default fallback in case of parsing failure or missing fields
        return default_result

    def analyze_news(self, news_text: str, target_symbol: str = "General Market") -> Dict[str, float]:
        """
        Performs actual Gemini API call with error handling and fallback.
        Now supports dual-track analysis with target_symbol.
        """
        default_val = {
            "macro_sentiment": 0.0, "macro_relevance": 0.0,
            "asset_sentiment": 0.0, "asset_relevance": 0.0
        }
        
        # Missing Short-Circuit (Credit Waste Protection)
        if not news_text or not str(news_text).strip():
            logger.info("Empty news text provided, skipping API call.")
            return default_val

        if not self.client:
            logger.error("Client not initialized (missing API key).")
            return default_val

        try:
            prompt = self.generate_prompt(news_text, target_symbol=target_symbol)
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=None # We already put instruction in prompt
                )
            )
            
            # Use response.text and handle potential blocks
            if response.text:
                return self.parse_response(response.text)
            else:
                logger.warning("Gemini API returned empty text (possibly blocked by safety filters).")
                return default_val
                
        except Exception as e:
            # Catching generic Exception including pydantic errors or API errors
            logger.error(f"Error during Gemini API call: {e}")
            return default_val

def get_sentiment_score(text: str, target_symbol: str = "General Market") -> float:
    """
    Project Janus Phase 3.2: Dual-Track Sentiment Synthesis.
    Performs relevance filtering, macro veto logic, and internal fusion.
    Returns a single float score for the decision center.
    """
    analyzer = SentimentAnalyzer()
    results = analyzer.analyze_news(text, target_symbol=target_symbol)
    
    # 1. Relevance Filtering (Confidence Noise Reduction)
    adj_macro = results["macro_sentiment"] * results["macro_relevance"]
    adj_asset = results["asset_sentiment"] * results["asset_relevance"]
    
    # 2. Macro Veto (Non-linear Risk Control)
    # If macro environment is extremely negative, ignore asset specifics.
    if adj_macro <= -0.8:
        logger.warning(f"MACRO VETO TRIGGERED: adj_macro={adj_macro:.2f}")
        return -1.0
        
    # 3. Internal Fusion (Weighted Synthesis)
    # Asset specifics have higher weight (0.7) for short-term price drive.
    right_brain_score = (0.3 * adj_macro) + (0.7 * adj_asset)
    
    return float(right_brain_score)

if __name__ == "__main__":
    # Example usage for Phase 2.2 Dual-Track
    # Note: Requires GEMINI_API_KEY environment variable for real API calls.
    analyzer = SentimentAnalyzer()
    sample_news = "Apple Inc. (AAPL) reports record-breaking quarterly earnings, far exceeding analyst expectations."
    target = "AAPL"
    
    print(f"--- 1. Prompt Generation for {target} ---")
    prompt = analyzer.generate_prompt(sample_news, target_symbol=target)
    print(prompt)
    
    print("\n--- 2. Dual-Track Parsing Logic Test ---")
    mock_llm_output = """
    {
        "macro_sentiment": 0.2, 
        "macro_relevance": 0.5, 
        "asset_sentiment": 0.9, 
        "asset_relevance": 1.0
    }
    """
    parsed = analyzer.parse_response(mock_llm_output)
    print(f"Parsed from JSON output: {parsed}")
    
    print("\n--- 3. Fallback Test (Missing Keys) ---")
    bad_output = '{"macro_sentiment": 0.5}'
    parsed_bad = analyzer.parse_response(bad_output)
    print(f"Parsed from partial output: {parsed_bad}")

    print("\n--- 4. Real Call Test (Will fail without key/quota) ---")
    result = get_sentiment_score(sample_news, target_symbol=target)
    print(f"Final score (likely fallback): {result}")
