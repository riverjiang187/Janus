"""
Survival System: Risk Management
Responsible for execution-level safety checks, including:
1. Hard Stop-Loss (Individual Asset level).
2. Portfolio Volatility Control.
3. Max Drawdown Protection.
"""

def check_stop_loss(current_price: float, entry_price: float, threshold: float = 0.05) -> bool:
    """
    Placeholder for hard stop-loss logic.
    Returns: True if stop-loss is triggered.
    """
    return (entry_price - current_price) / entry_price >= threshold
