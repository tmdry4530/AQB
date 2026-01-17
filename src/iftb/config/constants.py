"""
Risk Management Constants for IFTB Trading Bot.

This module defines all risk management parameters, limits, and thresholds
used throughout the IFTB trading system. These constants govern position sizing,
leverage limits, capital protection, circuit breakers, and performance targets.

All constants are immutable (Final) to prevent accidental modification during runtime.
"""

from typing import Final, List


# =============================================================================
# Kelly Criterion Limits
# =============================================================================

KELLY_FRACTION: Final[float] = 0.25
"""
Quarter-Kelly sizing for conservative capital allocation.
Uses 25% of the Kelly Criterion recommendation to reduce volatility.
"""

MAX_POSITION_PCT: Final[float] = 0.10
"""
Maximum position size as percentage of total capital (10%).
Hard limit to prevent over-concentration in any single position.
"""

MIN_POSITION_PCT: Final[float] = 0.02
"""
Minimum position size as percentage of total capital (2%).
Below this threshold, positions are not worth the transaction costs.
"""


# =============================================================================
# Leverage Limits
# =============================================================================

MAX_LEVERAGE: Final[int] = 8
"""
Absolute maximum leverage allowed under any circumstances.
Hard safety limit to prevent catastrophic losses.
"""

DEFAULT_LEVERAGE: Final[int] = 5
"""
Default leverage for standard trades with normal confidence levels.
Balanced between capital efficiency and risk management.
"""

MIN_LEVERAGE: Final[int] = 2
"""
Minimum leverage for low-confidence or high-risk scenarios.
Conservative leverage when market conditions are uncertain.
"""

HIGH_CONFIDENCE_LEVERAGE: Final[int] = 7
"""
Leverage for high-confidence setups with strong technical and fundamental alignment.
Only used when all signal quality checks pass.
"""


# =============================================================================
# Capital Protection
# =============================================================================

MAX_DAILY_LOSS_PCT: Final[float] = 0.08
"""
Maximum daily loss threshold (8% of capital).
Trading halts for the day if this limit is breached.
"""

MAX_WEEKLY_LOSS_PCT: Final[float] = 0.15
"""
Maximum weekly loss threshold (15% of capital).
Trading halts for the week if this limit is breached.
"""

MARGIN_CALL_THRESHOLD: Final[float] = 0.20
"""
Margin threshold that triggers position review (20%).
When margin level drops to this percentage, risk assessment is performed.
"""

EMERGENCY_EXIT_MARGIN: Final[float] = 0.10
"""
Critical margin level that triggers immediate emergency exit (10%).
All positions are liquidated to prevent margin call.
"""


# =============================================================================
# Circuit Breaker Settings
# =============================================================================

CONSECUTIVE_LOSS_LIMIT: Final[int] = 5
"""
Maximum consecutive losses before trading is paused.
Prevents continuation during adverse market conditions or system malfunction.
"""

HOURLY_TRADE_LIMIT: Final[int] = 3
"""
Maximum number of trades allowed per hour.
Prevents over-trading and excessive transaction costs.
"""

COOLDOWN_AFTER_DAILY_LIMIT_HOURS: Final[int] = 24
"""
Cooldown period in hours after hitting daily loss limit.
System remains inactive to allow market conditions to normalize.
"""

COOLDOWN_AFTER_CONSECUTIVE_LOSS_HOURS: Final[int] = 12
"""
Cooldown period in hours after hitting consecutive loss limit.
Shorter recovery period than daily loss, focuses on immediate risk.
"""

RECOVERY_WIN_REQUIRED: Final[int] = 2
"""
Number of consecutive wins required to exit cooldown mode early.
Demonstrates system recovery and market favorability.
"""


# =============================================================================
# LLM Veto Thresholds
# =============================================================================

SENTIMENT_VETO_THRESHOLD: Final[float] = -0.5
"""
Sentiment score below which LLM will veto the trade (-0.5).
Strong negative sentiment triggers automatic trade rejection.
"""

SENTIMENT_CAUTION_THRESHOLD: Final[float] = -0.2
"""
Sentiment score that triggers position size reduction (-0.2).
Moderate negative sentiment reduces confidence and position sizing.
"""

CONFIDENCE_VETO_THRESHOLD: Final[float] = 0.3
"""
Confidence score below which LLM will veto the trade (0.3).
Low confidence indicates insufficient signal quality.
"""

CONFIDENCE_CAUTION_THRESHOLD: Final[float] = 0.5
"""
Confidence score that triggers conservative position sizing (0.5).
Moderate confidence reduces leverage and position size.
"""

NEWS_CONFLICT_PENALTY: Final[float] = 0.5
"""
Confidence penalty multiplier when news conflicts with technical signals (0.5).
Reduces position size by 50% when fundamental and technical analysis diverge.
"""


# =============================================================================
# Performance Targets
# =============================================================================

TARGET_WIN_RATE: Final[float] = 0.60
"""
Target win rate for the trading system (60%).
Expected percentage of profitable trades under normal conditions.
"""

MIN_SAMPLE_SIZE: Final[int] = 500
"""
Minimum number of trades required for statistical significance.
Performance metrics are not reliable below this threshold.
"""

CONFIDENCE_LEVEL: Final[float] = 0.95
"""
Statistical confidence level for performance analysis (95%).
Used in hypothesis testing and performance validation.
"""

TARGET_SHARPE_RATIO: Final[float] = 1.5
"""
Target Sharpe ratio for risk-adjusted returns.
Measures excess return per unit of total risk.
"""

TARGET_SORTINO_RATIO: Final[float] = 2.0
"""
Target Sortino ratio focusing on downside risk.
Measures excess return per unit of downside deviation.
"""

TARGET_PROFIT_FACTOR: Final[float] = 1.8
"""
Target profit factor (gross profit / gross loss).
Indicates system profitability; >1.0 means profitable overall.
"""

MAX_DRAWDOWN: Final[float] = 0.30
"""
Maximum acceptable drawdown from peak equity (30%).
System requires reoptimization if this threshold is exceeded.
"""


# =============================================================================
# Trading Parameters
# =============================================================================

DEFAULT_TIMEFRAME: Final[str] = "1h"
"""
Default timeframe for technical analysis and trading signals.
1-hour candles provide balance between noise reduction and responsiveness.
"""

SUPPORTED_SYMBOLS: Final[List[str]] = ["BTCUSDT", "ETHUSDT"]
"""
List of cryptocurrency trading pairs supported by the system.
Currently limited to Bitcoin and Ethereum against USDT.
"""

SUPPORTED_EXCHANGES: Final[List[str]] = ["binance"]
"""
List of supported cryptocurrency exchanges.
Currently only Binance is integrated.
"""
