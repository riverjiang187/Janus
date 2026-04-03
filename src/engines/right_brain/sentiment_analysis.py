import json
import os
import re
import logging
from typing import Dict, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

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
    Your task is to analyze the sentiment of financial news and its relevance to the overall macro market.

    Instructions:
    1. Analyze the given news text for its sentiment towards the macro economy (Bullish/Bearish).
    2. Assign a 'sentiment' score between -1.0 (extremely bearish) and 1.0 (extremely bullish).
    3. Assign a 'relevance' score between 0.0 and 1.0 indicating how much this news impacts the broad market.
    
    Hallucination Defense:
    - If the news content is vague, ambiguous, or if you cannot determine the sentiment with confidence, you MUST output a 'sentiment' of 0.0.
    - Do not guess or invent information.

    Output Format:
    - You MUST respond ONLY in JSON format.
    - Example: {"sentiment": 0.15, "relevance": 0.8}
    - No additional text, explanations, or code blocks.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            self.model_id = 'gemini-2.5-flash'
        else:
            logger.warning("GEMINI_API_KEY not found. API calls will fail.")
            self.client = None

    def generate_prompt(self, news_text: str) -> str:
        """
        Combines the System Prompt with the news text.
        """
        return f"{self.SYSTEM_PROMPT}\n\nNews to analyze:\n{news_text}\n\nJSON Output:"

    def parse_response(self, response_text: str) -> Dict[str, float]:
        """
        Parses the JSON response from the LLM. 
        Includes basic cleaning to handle common LLM output quirks.
        """
        try:
            # Attempt to find JSON-like structure if LLM adds markdown or chatter
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                clean_json = match.group(0)
                result = json.loads(clean_json)
                sentiment = float(result.get("sentiment", 0.0))
                relevance = float(result.get("relevance", 0.0))
                
                # Unsafe Value Bounds Protection: Clamp to strict ranges
                return {
                    "sentiment": max(-1.0, min(1.0, sentiment)),
                    "relevance": max(0.0, min(1.0, relevance))
                }
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            
        # Default fallback in case of parsing failure or missing fields
        return {"sentiment": 0.0, "relevance": 0.0}

    def analyze_news(self, news_text: str) -> Dict[str, float]:
        """
        Performs actual Gemini API call with error handling and fallback.
        """
        # Missing Short-Circuit (Credit Waste Protection)
        if not news_text or not str(news_text).strip():
            logger.info("Empty news text provided, skipping API call.")
            return {"sentiment": 0.0, "relevance": 0.0}

        if not self.client:
            logger.error("Client not initialized (missing API key).")
            return {"sentiment": 0.0, "relevance": 0.0}

        try:
            prompt = self.generate_prompt(news_text)
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
                return {"sentiment": 0.0, "relevance": 0.0}
                
        except Exception as e:
            # Catching generic Exception including pydantic errors or API errors
            logger.error(f"Error during Gemini API call: {e}")
            return {"sentiment": 0.0, "relevance": 0.0}

def get_sentiment_score(text: str) -> Dict[str, float]:
    """
    Maintains compatibility with previous placeholder.
    Now returns the full sentiment and relevance dictionary.
    """
    analyzer = SentimentAnalyzer()
    return analyzer.analyze_news(text)

if __name__ == "__main__":
    # Example usage for Phase 2.2
    # Note: Requires GEMINI_API_KEY environment variable for real API calls.
    analyzer = SentimentAnalyzer()
    sample_news = "The Federal Reserve signals it may hold interest rates steady as inflation shows signs of cooling."
    
    print("--- 1. Prompt Generation ---")
    prompt = analyzer.generate_prompt(sample_news)
    print(prompt)
    
    print("\n--- 2. Parsing Logic Test ---")
    mock_llm_output = 'Sure! Here is the analysis: {"sentiment": 0.4, "relevance": 0.9}'
    parsed = analyzer.parse_response(mock_llm_output)
    print(f"Parsed from noisy output: {parsed}")
    
    print("\n--- 3. Fallback Test (Bad JSON) ---")
    bad_output = "I'm not sure, let's say 0.5 sentiment."
    parsed_bad = analyzer.parse_response(bad_output)
    print(f"Parsed from invalid output: {parsed_bad}")

    print("\n--- 4. Real Call Test (Will fail without key) ---")
    result = get_sentiment_score(sample_news)
    print(f"Final score (likely fallback): {result}")
