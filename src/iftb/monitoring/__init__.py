"""
Monitoring module for IFTB trading bot.

Provides Prometheus metrics collection and Grafana dashboard support.
"""

from .metrics import (
    CACHE_HITS,
    CACHE_MISSES,
    CIRCUIT_BREAKER_STATUS,
    CURRENT_DRAWDOWN,
    DB_QUERIES_TOTAL,
    DB_QUERY_LATENCY,
    DECISION_LATENCY,
    # Decision Metrics
    DECISIONS_TOTAL,
    DECISIONS_VETOED,
    ERRORS_TOTAL,
    # External Data Metrics
    EXTERNAL_API_REQUESTS,
    FEAR_GREED_INDEX,
    FUNDING_RATE,
    INDICATOR_VALUE,
    KILL_SWITCH_STATUS,
    LLM_LATENCY,
    LLM_REQUESTS_TOTAL,
    LLM_SENTIMENT,
    LLM_VETO_TOTAL,
    # Data Pipeline Metrics
    MARKET_DATA_LATENCY,
    MARKET_DATA_UPDATES,
    MAX_DRAWDOWN,
    ML_PREDICTION_CONFIDENCE,
    ML_PREDICTIONS_TOTAL,
    OPEN_INTEREST,
    OPEN_POSITIONS,
    ORDER_LATENCY,
    # Trading Metrics
    ORDERS_TOTAL,
    PORTFOLIO_CASH,
    PORTFOLIO_VALUE,
    POSITION_PNL,
    POSITION_SIZE,
    # System Metrics
    PROCESS_UPTIME,
    # Registry
    REGISTRY,
    SHARPE_RATIO,
    TECHNICAL_CONFIDENCE,
    # Analysis Metrics
    TECHNICAL_SIGNAL_STRENGTH,
    TRADE_PNL,
    TRADES_TOTAL,
    TRADING_LOOP_ITERATIONS,
    TRADING_LOOP_LATENCY,
    WARNINGS_TOTAL,
    WEBSOCKET_STATUS,
    WIN_RATE,
    # Metrics Manager
    MetricsManager,
    get_metrics_manager,
    measure_latency,
)

__all__ = [
    # Registry
    "REGISTRY",
    # Manager
    "MetricsManager",
    "get_metrics_manager",
    "measure_latency",
    # Trading
    "ORDERS_TOTAL",
    "ORDER_LATENCY",
    "OPEN_POSITIONS",
    "POSITION_SIZE",
    "POSITION_PNL",
    "PORTFOLIO_VALUE",
    "PORTFOLIO_CASH",
    "TRADES_TOTAL",
    "TRADE_PNL",
    "WIN_RATE",
    "SHARPE_RATIO",
    "MAX_DRAWDOWN",
    "CURRENT_DRAWDOWN",
    # Analysis
    "TECHNICAL_SIGNAL_STRENGTH",
    "TECHNICAL_CONFIDENCE",
    "INDICATOR_VALUE",
    "LLM_REQUESTS_TOTAL",
    "LLM_LATENCY",
    "LLM_SENTIMENT",
    "LLM_VETO_TOTAL",
    "ML_PREDICTIONS_TOTAL",
    "ML_PREDICTION_CONFIDENCE",
    # Decision
    "DECISIONS_TOTAL",
    "DECISION_LATENCY",
    "DECISIONS_VETOED",
    "CIRCUIT_BREAKER_STATUS",
    "KILL_SWITCH_STATUS",
    # Data Pipeline
    "MARKET_DATA_LATENCY",
    "MARKET_DATA_UPDATES",
    "WEBSOCKET_STATUS",
    "CACHE_HITS",
    "CACHE_MISSES",
    "DB_QUERIES_TOTAL",
    "DB_QUERY_LATENCY",
    # External
    "EXTERNAL_API_REQUESTS",
    "FEAR_GREED_INDEX",
    "FUNDING_RATE",
    "OPEN_INTEREST",
    # System
    "PROCESS_UPTIME",
    "TRADING_LOOP_ITERATIONS",
    "TRADING_LOOP_LATENCY",
    "ERRORS_TOTAL",
    "WARNINGS_TOTAL",
]
