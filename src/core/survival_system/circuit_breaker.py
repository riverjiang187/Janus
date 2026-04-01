"""
Survival System: Market Circuit Breaker
Responsible for macro risk management:
1. Detecting market crashes (e.g., SPY daily drop > 3%).
2. Detecting volatility spikes (VIX > 40).
Action: Switch to "Cash Only" mode when triggered.
"""

def detect_market_crash(market_returns: float, threshold: float = -0.03) -> bool:
    """
    Placeholder for market circuit breaker detection.
    """
    return market_returns <= threshold
