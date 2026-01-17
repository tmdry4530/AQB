"""
Backtest module for IFTB Trading Bot.

This module provides comprehensive backtesting capabilities including:
- Event-driven backtest engine
- Walk-forward optimization
- Monte Carlo simulation
- Performance metrics calculation
- Report generation

Example Usage:
    ```python
    from iftb.backtest import BacktestEngine, BacktestConfig
    import pandas as pd

    # Load historical data
    data = pd.read_csv("btc_ohlcv.csv")

    # Configure backtest
    config = BacktestConfig(
        symbol="BTC/USDT",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        initial_capital=10000.0,
    )

    # Run backtest
    engine = BacktestEngine(config)
    result = await engine.run(data)

    # Print metrics
    print(f"Sharpe Ratio: {result.metrics.sharpe_ratio}")
    print(f"Max Drawdown: {result.metrics.max_drawdown_pct:.2%}")
    ```
"""

from iftb.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestMetrics,
    BacktestResult,
    BacktestTrade,
    MonteCarloSimulator,
    WalkForwardOptimizer,
)
from iftb.backtest.report import BacktestReporter

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestMetrics",
    "BacktestReporter",
    "BacktestResult",
    "BacktestTrade",
    "MonteCarloSimulator",
    "WalkForwardOptimizer",
]
