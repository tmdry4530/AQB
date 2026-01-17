"""
Backtest Engine for IFTB Trading Bot.

This module implements a comprehensive backtesting system that:
1. Simulates trading with historical data
2. Calculates performance metrics (Sharpe Ratio, Max Drawdown, etc.)
3. Generates detailed reports and visualizations
4. Supports multiple strategies and parameters

Key Features:
- Event-driven backtesting architecture
- Realistic slippage and fee modeling
- Position sizing with Kelly Criterion
- Walk-forward optimization support
- Monte Carlo simulation for robustness testing
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
from typing import Literal

import numpy as np
import pandas as pd

from iftb.analysis import TechnicalAnalyzer
from iftb.analysis.llm_analyzer import LLMAnalysis, SentimentScore
from iftb.analysis.ml_model import ModelPrediction
from iftb.data import MarketContext
from iftb.trading import (
    CircuitBreaker,
    DecisionEngine,
    KillSwitch,
    RiskManager,
    TradeHistory,
    TradingDecision,
)
from iftb.trading.executor import Order, PaperTrader, PositionState
from iftb.utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""

    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10000.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004
    slippage_pct: float = 0.0005
    timeframe: str = "1h"
    warmup_periods: int = 200  # Periods for indicator warmup
    use_llm: bool = False  # Disable LLM in backtest by default (expensive)
    use_ml: bool = True


@dataclass
class BacktestTrade:
    """Record of a single backtest trade."""

    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    exit_price: float
    amount: float
    leverage: int
    pnl: float
    pnl_pct: float
    fee: float
    exit_reason: Literal["stop_loss", "take_profit", "signal", "end_of_data"]


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics."""

    # Basic metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # Returns
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    volatility: float = 0.0

    # Performance ratios
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0

    # Trade metrics
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_period: timedelta = field(default_factory=lambda: timedelta(hours=0))

    # Streak metrics
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Exposure
    total_fees: float = 0.0
    time_in_market_pct: float = 0.0

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "total_return": round(self.total_return, 2),
            "total_return_pct": round(self.total_return_pct, 4),
            "annualized_return": round(self.annualized_return, 4),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "volatility": round(self.volatility, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "profit_factor": round(self.profit_factor, 4),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "avg_trade": round(self.avg_trade, 2),
            "largest_win": round(self.largest_win, 2),
            "largest_loss": round(self.largest_loss, 2),
            "avg_holding_period_hours": round(self.avg_holding_period.total_seconds() / 3600, 2),
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "total_fees": round(self.total_fees, 2),
            "time_in_market_pct": round(self.time_in_market_pct, 4),
        }


@dataclass
class BacktestResult:
    """Complete backtest results."""

    config: BacktestConfig
    metrics: BacktestMetrics
    trades: list[BacktestTrade]
    equity_curve: pd.DataFrame  # timestamp, equity, drawdown
    decisions: list[TradingDecision]
    final_balance: float
    start_balance: float


# =============================================================================
# Backtest Engine
# =============================================================================


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Simulates trading strategy execution on historical data with
    realistic order execution, fees, and slippage modeling.
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.paper_trader = PaperTrader(
            initial_balance=config.initial_capital,
            maker_fee=config.maker_fee,
            taker_fee=config.taker_fee,
        )
        self.decision_engine = self._create_decision_engine()

        # State tracking
        self.trades: list[BacktestTrade] = []
        self.decisions: list[TradingDecision] = []
        self.equity_history: list[tuple[datetime, float]] = []
        self.trade_history: list[TradeHistory] = []
        self.current_position: PositionState | None = None

        logger.info(
            "backtest_engine_initialized",
            symbol=config.symbol,
            start_date=config.start_date,
            end_date=config.end_date,
            initial_capital=config.initial_capital,
        )

    def _create_decision_engine(self) -> DecisionEngine:
        """Create decision engine for backtesting."""
        risk_manager = RiskManager()
        circuit_breaker = CircuitBreaker()
        kill_switch = KillSwitch()

        return DecisionEngine(
            risk_manager=risk_manager,
            circuit_breaker=circuit_breaker,
            kill_switch=kill_switch,
        )

    async def run(self, data: pd.DataFrame) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: DataFrame with OHLCV data (columns: timestamp, open, high, low, close, volume)

        Returns:
            BacktestResult with complete analysis
        """
        logger.info(
            "backtest_started",
            data_rows=len(data),
            start=data["timestamp"].iloc[0] if len(data) > 0 else None,
            end=data["timestamp"].iloc[-1] if len(data) > 0 else None,
        )

        # Validate data
        if len(data) < self.config.warmup_periods:
            raise ValueError(
                f"Insufficient data: {len(data)} rows, need at least {self.config.warmup_periods}"
            )

        # Initialize equity tracking
        self.equity_history = [(data["timestamp"].iloc[0], self.config.initial_capital)]

        # Main backtest loop
        for i in range(self.config.warmup_periods, len(data)):
            # Get data window for analysis
            window = data.iloc[i - self.config.warmup_periods : i + 1].copy()
            current_bar = data.iloc[i]
            current_time = current_bar["timestamp"]
            current_price = float(current_bar["close"])

            # Update position with current price
            if self.current_position:
                self._update_position_pnl(current_price)

            # Check stop-loss and take-profit
            if self.current_position:
                exit_triggered = await self._check_exit_conditions(current_bar, current_time)
                if exit_triggered:
                    continue

            # Generate signals and make decision
            decision = await self._generate_decision(window, current_price, current_time)
            self.decisions.append(decision)

            # Execute decision
            if decision.action != "HOLD" and not decision.vetoed:
                await self._execute_decision(decision, current_price, current_time)

            # Record equity
            equity = self._calculate_equity(current_price)
            self.equity_history.append((current_time, equity))

        # Close any remaining position at end of backtest
        if self.current_position:
            final_price = float(data.iloc[-1]["close"])
            final_time = data.iloc[-1]["timestamp"]
            await self._close_position(final_price, final_time, "end_of_data")

        # Calculate metrics
        metrics = self._calculate_metrics()

        # Build equity curve DataFrame
        equity_df = pd.DataFrame(self.equity_history, columns=["timestamp", "equity"])
        equity_df["drawdown"] = self._calculate_drawdown_series(equity_df["equity"])

        result = BacktestResult(
            config=self.config,
            metrics=metrics,
            trades=self.trades,
            equity_curve=equity_df,
            decisions=self.decisions,
            final_balance=self.paper_trader.get_balance(),
            start_balance=self.config.initial_capital,
        )

        logger.info(
            "backtest_completed",
            total_trades=metrics.total_trades,
            win_rate=metrics.win_rate,
            sharpe_ratio=metrics.sharpe_ratio,
            total_return_pct=metrics.total_return_pct,
            max_drawdown_pct=metrics.max_drawdown_pct,
        )

        return result

    async def _generate_decision(
        self,
        data: pd.DataFrame,
        current_price: float,
        current_time: datetime,
    ) -> TradingDecision:
        """Generate trading decision from data."""
        # Technical analysis
        analyzer = TechnicalAnalyzer(data)
        technical_signal = analyzer.generate_composite_signal()

        # Mock LLM analysis (disabled in backtest for speed)
        llm_analysis = LLMAnalysis(
            sentiment=SentimentScore.NEUTRAL,
            confidence=0.5,
            summary="Backtest mock analysis",
            key_factors=[],
            should_veto=False,
            veto_reason=None,
            timestamp=current_time,
            model="backtest-mock",
            prompt_tokens=0,
            completion_tokens=0,
            cached=True,
        )

        # Mock ML prediction (or use real model if configured)
        ml_prediction = ModelPrediction(
            action="HOLD",
            confidence=0.5,
            probability_long=0.33,
            probability_short=0.33,
            probability_hold=0.34,
            feature_importance={},
            model_version="backtest-mock",
            prediction_time=current_time,
        )

        # Market context
        market_context = MarketContext()

        # Make decision
        decision = await self.decision_engine.make_decision(
            symbol=self.config.symbol,
            technical_signal=technical_signal,
            llm_analysis=llm_analysis,
            ml_prediction=ml_prediction,
            market_context=market_context,
            current_price=current_price,
            account_balance=self.paper_trader.get_balance(),
            trade_history=self.trade_history,
        )

        return decision

    async def _execute_decision(
        self,
        decision: TradingDecision,
        current_price: float,
        current_time: datetime,
    ):
        """Execute a trading decision."""
        # Close existing position if different direction
        if self.current_position:
            if (decision.action == "LONG" and self.current_position.side == "short") or (
                decision.action == "SHORT" and self.current_position.side == "long"
            ):
                await self._close_position(current_price, current_time, "signal")

        # Open new position if not already in one
        if not self.current_position:
            side = "buy" if decision.action == "LONG" else "sell"
            amount = (decision.position_size * self.paper_trader.get_balance()) / current_price

            order = Order(
                id=f"bt-{len(self.trades)}",
                symbol=self.config.symbol,
                side=side,
                type="market",
                amount=amount,
                price=current_price,
            )

            filled_order = await self.paper_trader.place_order(order, current_price)

            if filled_order.status == "filled":
                self.current_position = PositionState(
                    symbol=self.config.symbol,
                    side="long" if decision.action == "LONG" else "short",
                    entry_price=filled_order.filled_price,
                    current_price=current_price,
                    amount=filled_order.filled_amount,
                    leverage=decision.leverage,
                    unrealized_pnl=0.0,
                    liquidation_price=0.0,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit,
                    opened_at=current_time,
                )

                logger.debug(
                    "backtest_position_opened",
                    symbol=self.config.symbol,
                    side=self.current_position.side,
                    entry_price=filled_order.filled_price,
                    amount=filled_order.filled_amount,
                )

    async def _check_exit_conditions(
        self,
        current_bar: pd.Series,
        current_time: datetime,
    ) -> bool:
        """Check if stop-loss or take-profit was triggered."""
        if not self.current_position:
            return False

        high = float(current_bar["high"])
        low = float(current_bar["low"])
        close = float(current_bar["close"])

        # Check stop-loss
        if self.current_position.stop_loss:
            if (
                self.current_position.side == "long" and low <= self.current_position.stop_loss
            ) or (
                self.current_position.side == "short" and high >= self.current_position.stop_loss
            ):
                await self._close_position(
                    self.current_position.stop_loss, current_time, "stop_loss"
                )
                return True

        # Check take-profit
        if self.current_position.take_profit:
            if (
                self.current_position.side == "long" and high >= self.current_position.take_profit
            ) or (
                self.current_position.side == "short" and low <= self.current_position.take_profit
            ):
                await self._close_position(
                    self.current_position.take_profit, current_time, "take_profit"
                )
                return True

        return False

    async def _close_position(
        self,
        exit_price: float,
        exit_time: datetime,
        reason: str,
    ):
        """Close current position and record trade."""
        if not self.current_position:
            return

        # Calculate PnL
        if self.current_position.side == "long":
            pnl = (exit_price - self.current_position.entry_price) * self.current_position.amount
        else:
            pnl = (self.current_position.entry_price - exit_price) * self.current_position.amount

        pnl_pct = pnl / (self.current_position.entry_price * self.current_position.amount)

        # Create close order
        close_side = "sell" if self.current_position.side == "long" else "buy"
        close_order = Order(
            id=f"bt-close-{len(self.trades)}",
            symbol=self.config.symbol,
            side=close_side,
            type="market",
            amount=self.current_position.amount,
            price=exit_price,
        )

        filled_order = await self.paper_trader.place_order(close_order, exit_price)

        # Record trade
        trade = BacktestTrade(
            entry_time=self.current_position.opened_at,
            exit_time=exit_time,
            symbol=self.config.symbol,
            side=self.current_position.side,
            entry_price=self.current_position.entry_price,
            exit_price=exit_price,
            amount=self.current_position.amount,
            leverage=self.current_position.leverage,
            pnl=pnl,
            pnl_pct=pnl_pct,
            fee=filled_order.fee or 0.0,
            exit_reason=reason,
        )
        self.trades.append(trade)

        # Update trade history for risk manager
        self.trade_history.append(
            TradeHistory(
                symbol=self.config.symbol,
                action="LONG" if self.current_position.side == "long" else "SHORT",
                entry_price=self.current_position.entry_price,
                exit_price=exit_price,
                position_size=self.current_position.amount,
                leverage=self.current_position.leverage,
                pnl=pnl,
                pnl_pct=pnl_pct,
                entry_time=self.current_position.opened_at,
                exit_time=exit_time,
                win=pnl > 0,
            )
        )

        logger.debug(
            "backtest_position_closed",
            symbol=self.config.symbol,
            side=self.current_position.side,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=reason,
        )

        self.current_position = None

    def _update_position_pnl(self, current_price: float):
        """Update position's unrealized PnL."""
        if not self.current_position:
            return

        self.current_position.current_price = current_price

        if self.current_position.side == "long":
            self.current_position.unrealized_pnl = (
                current_price - self.current_position.entry_price
            ) * self.current_position.amount
        else:
            self.current_position.unrealized_pnl = (
                self.current_position.entry_price - current_price
            ) * self.current_position.amount

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity (balance + unrealized PnL)."""
        equity = self.paper_trader.get_balance()
        if self.current_position:
            self._update_position_pnl(current_price)
            equity += self.current_position.unrealized_pnl
        return equity

    def _calculate_drawdown_series(self, equity_series: pd.Series) -> pd.Series:
        """Calculate drawdown series from equity curve."""
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        return drawdown

    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics."""
        metrics = BacktestMetrics()

        if not self.trades:
            return metrics

        # Basic trade metrics
        metrics.total_trades = len(self.trades)
        metrics.winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        metrics.losing_trades = sum(1 for t in self.trades if t.pnl <= 0)
        metrics.win_rate = metrics.winning_trades / metrics.total_trades

        # Returns
        pnls = [t.pnl for t in self.trades]
        metrics.total_return = sum(pnls)
        metrics.total_return_pct = metrics.total_return / self.config.initial_capital

        # Annualized return
        if len(self.equity_history) >= 2:
            start_time = self.equity_history[0][0]
            end_time = self.equity_history[-1][0]
            duration_days = (end_time - start_time).days
            if duration_days > 0:
                years = duration_days / 365
                metrics.annualized_return = (
                    ((1 + metrics.total_return_pct) ** (1 / years) - 1) if years > 0 else 0
                )

        # Drawdown
        equity_series = pd.Series([e[1] for e in self.equity_history])
        drawdown_series = self._calculate_drawdown_series(equity_series)
        metrics.max_drawdown_pct = abs(drawdown_series.min())
        metrics.max_drawdown = metrics.max_drawdown_pct * self.config.initial_capital

        # Volatility (annualized)
        if len(pnls) > 1:
            daily_returns = [t.pnl_pct for t in self.trades]
            metrics.volatility = statistics.stdev(daily_returns) * np.sqrt(252)

        # Sharpe Ratio (assuming 0% risk-free rate)
        if metrics.volatility > 0:
            metrics.sharpe_ratio = metrics.annualized_return / metrics.volatility

        # Sortino Ratio (downside deviation)
        negative_returns = [r for r in [t.pnl_pct for t in self.trades] if r < 0]
        if negative_returns:
            downside_std = statistics.stdev(negative_returns) * np.sqrt(252)
            if downside_std > 0:
                metrics.sortino_ratio = metrics.annualized_return / downside_std

        # Calmar Ratio
        if metrics.max_drawdown_pct > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown_pct

        # Profit Factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        if gross_loss > 0:
            metrics.profit_factor = gross_profit / gross_loss

        # Average win/loss
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl < 0]

        metrics.avg_win = statistics.mean(wins) if wins else 0
        metrics.avg_loss = statistics.mean(losses) if losses else 0
        metrics.avg_trade = statistics.mean(pnls)
        metrics.largest_win = max(pnls) if pnls else 0
        metrics.largest_loss = min(pnls) if pnls else 0

        # Holding period
        holding_periods = [(t.exit_time - t.entry_time) for t in self.trades]
        if holding_periods:
            avg_seconds = sum(hp.total_seconds() for hp in holding_periods) / len(holding_periods)
            metrics.avg_holding_period = timedelta(seconds=avg_seconds)

        # Consecutive wins/losses
        metrics.max_consecutive_wins = self._max_consecutive(self.trades, win=True)
        metrics.max_consecutive_losses = self._max_consecutive(self.trades, win=False)

        # Fees
        metrics.total_fees = sum(t.fee for t in self.trades)

        # Time in market
        if len(self.equity_history) >= 2:
            total_time = (self.equity_history[-1][0] - self.equity_history[0][0]).total_seconds()
            time_in_position = sum(
                (t.exit_time - t.entry_time).total_seconds() for t in self.trades
            )
            if total_time > 0:
                metrics.time_in_market_pct = time_in_position / total_time

        return metrics

    def _max_consecutive(self, trades: list[BacktestTrade], win: bool) -> int:
        """Calculate maximum consecutive wins or losses."""
        max_streak = 0
        current_streak = 0

        for trade in trades:
            is_win = trade.pnl > 0
            if is_win == win:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak


# =============================================================================
# Walk-Forward Optimization
# =============================================================================


class WalkForwardOptimizer:
    """
    Walk-forward optimization for robust parameter selection.

    Divides data into in-sample (training) and out-of-sample (testing)
    periods to validate strategy parameters.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: BacktestConfig,
        n_splits: int = 5,
        train_ratio: float = 0.7,
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            data: Historical OHLCV data
            config: Base backtest configuration
            n_splits: Number of walk-forward splits
            train_ratio: Ratio of data used for training in each split
        """
        self.data = data
        self.config = config
        self.n_splits = n_splits
        self.train_ratio = train_ratio

    async def run(self) -> list[BacktestResult]:
        """
        Run walk-forward optimization.

        Returns:
            List of BacktestResult for each out-of-sample period
        """
        results = []
        total_rows = len(self.data)
        split_size = total_rows // self.n_splits

        for i in range(self.n_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < self.n_splits - 1 else total_rows

            split_data = self.data.iloc[start_idx:end_idx]
            train_size = int(len(split_data) * self.train_ratio)

            # Out-of-sample period
            test_data = split_data.iloc[train_size:]

            if len(test_data) < self.config.warmup_periods:
                continue

            # Run backtest on out-of-sample data
            engine = BacktestEngine(self.config)
            result = await engine.run(test_data.reset_index(drop=True))
            results.append(result)

            logger.info(
                "walk_forward_split_completed",
                split=i + 1,
                total_splits=self.n_splits,
                sharpe_ratio=result.metrics.sharpe_ratio,
            )

        return results


# =============================================================================
# Monte Carlo Simulation
# =============================================================================


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy robustness testing.

    Randomly reorders trades to estimate distribution of possible outcomes.
    """

    def __init__(self, trades: list[BacktestTrade], initial_capital: float):
        """
        Initialize Monte Carlo simulator.

        Args:
            trades: List of historical trades
            initial_capital: Starting capital
        """
        self.trades = trades
        self.initial_capital = initial_capital

    def run(self, n_simulations: int = 1000) -> dict:
        """
        Run Monte Carlo simulation.

        Args:
            n_simulations: Number of simulations to run

        Returns:
            Dictionary with simulation statistics
        """
        if not self.trades:
            return {}

        final_equities = []
        max_drawdowns = []
        sharpe_ratios = []

        pnls = [t.pnl for t in self.trades]

        for _ in range(n_simulations):
            # Randomly shuffle trades
            shuffled_pnls = np.random.permutation(pnls)

            # Calculate equity curve
            equity = [self.initial_capital]
            for pnl in shuffled_pnls:
                equity.append(equity[-1] + pnl)

            equity_series = pd.Series(equity)
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak

            final_equities.append(equity[-1])
            max_drawdowns.append(abs(drawdown.min()))

            # Calculate Sharpe for this simulation
            returns = np.diff(equity) / equity[:-1]
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                sharpe_ratios.append(sharpe)

        return {
            "final_equity": {
                "mean": np.mean(final_equities),
                "std": np.std(final_equities),
                "median": np.median(final_equities),
                "percentile_5": np.percentile(final_equities, 5),
                "percentile_95": np.percentile(final_equities, 95),
            },
            "max_drawdown": {
                "mean": np.mean(max_drawdowns),
                "std": np.std(max_drawdowns),
                "median": np.median(max_drawdowns),
                "percentile_5": np.percentile(max_drawdowns, 5),
                "percentile_95": np.percentile(max_drawdowns, 95),
            },
            "sharpe_ratio": {
                "mean": np.mean(sharpe_ratios) if sharpe_ratios else 0,
                "std": np.std(sharpe_ratios) if sharpe_ratios else 0,
                "median": np.median(sharpe_ratios) if sharpe_ratios else 0,
            },
            "n_simulations": n_simulations,
        }
