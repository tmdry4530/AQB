"""
Trading module for IFTB trading bot.

Provides the core trading decision engine with integrated risk management,
position sizing, circuit breaker systems, and order execution.
"""

from .decision_engine import (
    CircuitBreaker,
    DecisionEngine,
    KillSwitch,
    RiskManager,
    TradeHistory,
    TradingDecision,
    create_decision_engine,
)
from .executor import (
    ExecutionRequest,
    LiveExecutor,
    Order,
    OrderExecutor,
    PaperTrader,
    PositionState,
    convert_decision_to_request,
)

__all__ = [
    # Decision Engine
    "DecisionEngine",
    "RiskManager",
    "CircuitBreaker",
    "KillSwitch",
    "TradingDecision",
    "TradeHistory",
    "create_decision_engine",
    # Order Execution
    "Order",
    "PositionState",
    "ExecutionRequest",
    "PaperTrader",
    "LiveExecutor",
    "OrderExecutor",
    "convert_decision_to_request",
]
