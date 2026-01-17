"""
Prometheus metrics for IFTB trading bot.

Provides comprehensive metrics collection for monitoring trading performance,
system health, and operational metrics via Prometheus.
"""

from contextlib import asynccontextmanager
from datetime import UTC, datetime

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    start_http_server,
)

# Create a custom registry for IFTB metrics
REGISTRY = CollectorRegistry()


# =============================================================================
# System Info
# =============================================================================

SYSTEM_INFO = Info(
    "iftb_system",
    "IFTB trading bot system information",
    registry=REGISTRY,
)


# =============================================================================
# Trading Metrics
# =============================================================================

# Order execution metrics
ORDERS_TOTAL = Counter(
    "iftb_orders_total",
    "Total number of orders executed",
    ["symbol", "side", "order_type", "status"],
    registry=REGISTRY,
)

ORDER_LATENCY = Histogram(
    "iftb_order_latency_seconds",
    "Order execution latency in seconds",
    ["symbol", "side"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

# Position metrics
OPEN_POSITIONS = Gauge(
    "iftb_open_positions",
    "Number of open positions",
    ["symbol"],
    registry=REGISTRY,
)

POSITION_SIZE = Gauge(
    "iftb_position_size",
    "Current position size in base currency",
    ["symbol", "side"],
    registry=REGISTRY,
)

POSITION_PNL = Gauge(
    "iftb_position_pnl",
    "Unrealized P&L of current positions",
    ["symbol"],
    registry=REGISTRY,
)

# Portfolio metrics
PORTFOLIO_VALUE = Gauge(
    "iftb_portfolio_value",
    "Total portfolio value in quote currency",
    registry=REGISTRY,
)

PORTFOLIO_CASH = Gauge(
    "iftb_portfolio_cash",
    "Available cash balance",
    registry=REGISTRY,
)

PORTFOLIO_MARGIN_USED = Gauge(
    "iftb_portfolio_margin_used",
    "Total margin used across all positions",
    registry=REGISTRY,
)

# Trading performance metrics
TRADES_TOTAL = Counter(
    "iftb_trades_total",
    "Total number of completed trades",
    ["symbol", "side", "result"],  # result: win, loss, breakeven
    registry=REGISTRY,
)

TRADE_PNL = Histogram(
    "iftb_trade_pnl",
    "P&L distribution of completed trades",
    ["symbol"],
    buckets=(-1000, -500, -100, -50, -10, 0, 10, 50, 100, 500, 1000, 5000),
    registry=REGISTRY,
)

WIN_RATE = Gauge(
    "iftb_win_rate",
    "Current win rate (rolling window)",
    ["symbol"],
    registry=REGISTRY,
)

SHARPE_RATIO = Gauge(
    "iftb_sharpe_ratio",
    "Current Sharpe ratio (annualized)",
    registry=REGISTRY,
)

MAX_DRAWDOWN = Gauge(
    "iftb_max_drawdown",
    "Maximum drawdown percentage",
    registry=REGISTRY,
)

CURRENT_DRAWDOWN = Gauge(
    "iftb_current_drawdown",
    "Current drawdown percentage from peak",
    registry=REGISTRY,
)


# =============================================================================
# Analysis Metrics
# =============================================================================

# Technical analysis metrics
TECHNICAL_SIGNAL_STRENGTH = Gauge(
    "iftb_technical_signal_strength",
    "Current technical signal strength (-1 to 1)",
    ["symbol", "timeframe"],
    registry=REGISTRY,
)

TECHNICAL_CONFIDENCE = Gauge(
    "iftb_technical_confidence",
    "Technical analysis confidence (0 to 1)",
    ["symbol", "timeframe"],
    registry=REGISTRY,
)

INDICATOR_VALUE = Gauge(
    "iftb_indicator_value",
    "Current indicator value",
    ["symbol", "indicator"],
    registry=REGISTRY,
)

# LLM analysis metrics
LLM_REQUESTS_TOTAL = Counter(
    "iftb_llm_requests_total",
    "Total LLM API requests",
    ["status"],  # success, error, timeout
    registry=REGISTRY,
)

LLM_LATENCY = Histogram(
    "iftb_llm_latency_seconds",
    "LLM API response latency",
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0),
    registry=REGISTRY,
)

LLM_SENTIMENT = Gauge(
    "iftb_llm_sentiment",
    "LLM sentiment score (-1 to 1)",
    ["symbol"],
    registry=REGISTRY,
)

LLM_VETO_TOTAL = Counter(
    "iftb_llm_veto_total",
    "Total LLM veto decisions",
    ["symbol", "reason"],
    registry=REGISTRY,
)

LLM_TOKENS_USED = Counter(
    "iftb_llm_tokens_total",
    "Total LLM tokens used",
    ["type"],  # input, output
    registry=REGISTRY,
)

# ML model metrics
ML_PREDICTIONS_TOTAL = Counter(
    "iftb_ml_predictions_total",
    "Total ML model predictions",
    ["symbol", "prediction"],  # LONG, SHORT, HOLD
    registry=REGISTRY,
)

ML_PREDICTION_CONFIDENCE = Gauge(
    "iftb_ml_prediction_confidence",
    "ML model prediction confidence",
    ["symbol"],
    registry=REGISTRY,
)

ML_MODEL_ACCURACY = Gauge(
    "iftb_ml_model_accuracy",
    "ML model accuracy (rolling window)",
    registry=REGISTRY,
)


# =============================================================================
# Decision Engine Metrics
# =============================================================================

DECISIONS_TOTAL = Counter(
    "iftb_decisions_total",
    "Total trading decisions made",
    ["symbol", "action"],  # LONG, SHORT, HOLD
    registry=REGISTRY,
)

DECISION_LATENCY = Histogram(
    "iftb_decision_latency_seconds",
    "Decision engine processing latency",
    ["symbol"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    registry=REGISTRY,
)

DECISIONS_VETOED = Counter(
    "iftb_decisions_vetoed_total",
    "Total vetoed decisions",
    ["symbol", "veto_source"],  # llm, circuit_breaker, kill_switch, risk
    registry=REGISTRY,
)

CIRCUIT_BREAKER_STATUS = Gauge(
    "iftb_circuit_breaker_active",
    "Circuit breaker status (1=active, 0=inactive)",
    registry=REGISTRY,
)

KILL_SWITCH_STATUS = Gauge(
    "iftb_kill_switch_active",
    "Kill switch status (1=active, 0=inactive)",
    registry=REGISTRY,
)


# =============================================================================
# Data Pipeline Metrics
# =============================================================================

# Market data metrics
MARKET_DATA_LATENCY = Histogram(
    "iftb_market_data_latency_ms",
    "Market data latency in milliseconds",
    ["symbol", "source"],  # source: rest, websocket
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
    registry=REGISTRY,
)

MARKET_DATA_UPDATES = Counter(
    "iftb_market_data_updates_total",
    "Total market data updates received",
    ["symbol", "data_type"],  # ohlcv, ticker, orderbook
    registry=REGISTRY,
)

WEBSOCKET_RECONNECTS = Counter(
    "iftb_websocket_reconnects_total",
    "Total WebSocket reconnection attempts",
    ["symbol"],
    registry=REGISTRY,
)

WEBSOCKET_STATUS = Gauge(
    "iftb_websocket_connected",
    "WebSocket connection status (1=connected, 0=disconnected)",
    ["symbol"],
    registry=REGISTRY,
)

# Database metrics
DB_QUERIES_TOTAL = Counter(
    "iftb_db_queries_total",
    "Total database queries",
    ["operation", "table"],
    registry=REGISTRY,
)

DB_QUERY_LATENCY = Histogram(
    "iftb_db_query_latency_seconds",
    "Database query latency",
    ["operation"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=REGISTRY,
)

DB_CONNECTIONS_ACTIVE = Gauge(
    "iftb_db_connections_active",
    "Number of active database connections",
    registry=REGISTRY,
)

# Cache metrics
CACHE_HITS = Counter(
    "iftb_cache_hits_total",
    "Total cache hits",
    ["cache_type"],  # redis, local
    registry=REGISTRY,
)

CACHE_MISSES = Counter(
    "iftb_cache_misses_total",
    "Total cache misses",
    ["cache_type"],
    registry=REGISTRY,
)

CACHE_LATENCY = Histogram(
    "iftb_cache_latency_seconds",
    "Cache operation latency",
    ["operation"],  # get, set, delete
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1),
    registry=REGISTRY,
)


# =============================================================================
# External Data Metrics
# =============================================================================

EXTERNAL_API_REQUESTS = Counter(
    "iftb_external_api_requests_total",
    "Total external API requests",
    ["api", "status"],  # api: fear_greed, coinglass, etc.
    registry=REGISTRY,
)

EXTERNAL_API_LATENCY = Histogram(
    "iftb_external_api_latency_seconds",
    "External API response latency",
    ["api"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

FEAR_GREED_INDEX = Gauge(
    "iftb_fear_greed_index",
    "Current Fear & Greed Index value (0-100)",
    registry=REGISTRY,
)

FUNDING_RATE = Gauge(
    "iftb_funding_rate",
    "Current funding rate",
    ["symbol"],
    registry=REGISTRY,
)

OPEN_INTEREST = Gauge(
    "iftb_open_interest",
    "Current open interest",
    ["symbol"],
    registry=REGISTRY,
)

LONG_SHORT_RATIO = Gauge(
    "iftb_long_short_ratio",
    "Long/Short ratio",
    ["symbol"],
    registry=REGISTRY,
)


# =============================================================================
# System Health Metrics
# =============================================================================

PROCESS_UPTIME = Gauge(
    "iftb_process_uptime_seconds",
    "Process uptime in seconds",
    registry=REGISTRY,
)

TRADING_LOOP_ITERATIONS = Counter(
    "iftb_trading_loop_iterations_total",
    "Total trading loop iterations",
    registry=REGISTRY,
)

TRADING_LOOP_LATENCY = Histogram(
    "iftb_trading_loop_latency_seconds",
    "Trading loop iteration latency",
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=REGISTRY,
)

ERRORS_TOTAL = Counter(
    "iftb_errors_total",
    "Total errors by component",
    ["component", "error_type"],
    registry=REGISTRY,
)

WARNINGS_TOTAL = Counter(
    "iftb_warnings_total",
    "Total warnings by component",
    ["component"],
    registry=REGISTRY,
)


# =============================================================================
# Metrics Manager
# =============================================================================

class MetricsManager:
    """
    Centralized metrics management for IFTB.

    Provides helper methods for common metric operations and
    manages the Prometheus HTTP server.
    """

    def __init__(self, port: int = 8000):
        """
        Initialize the metrics manager.

        Args:
            port: Port for Prometheus metrics endpoint
        """
        self.port = port
        self._start_time = datetime.now(UTC)
        self._server_started = False

        # Set system info
        SYSTEM_INFO.info({
            "version": "1.0.0",
            "mode": "paper",  # Will be updated on start
        })

    def start_server(self, mode: str = "paper") -> None:
        """
        Start the Prometheus metrics HTTP server.

        Args:
            mode: Trading mode (paper/live)
        """
        if self._server_started:
            return

        SYSTEM_INFO.info({
            "version": "1.0.0",
            "mode": mode,
        })

        start_http_server(self.port, registry=REGISTRY)
        self._server_started = True

    def get_metrics(self) -> bytes:
        """
        Get all metrics in Prometheus format.

        Returns:
            Metrics in Prometheus exposition format
        """
        return generate_latest(REGISTRY)

    def update_uptime(self) -> None:
        """Update the process uptime metric."""
        uptime = (datetime.now(UTC) - self._start_time).total_seconds()
        PROCESS_UPTIME.set(uptime)

    # =========================================================================
    # Trading Metrics Helpers
    # =========================================================================

    def record_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        status: str,
        latency_seconds: float,
    ) -> None:
        """Record an order execution."""
        ORDERS_TOTAL.labels(
            symbol=symbol,
            side=side,
            order_type=order_type,
            status=status,
        ).inc()
        ORDER_LATENCY.labels(symbol=symbol, side=side).observe(latency_seconds)

    def record_trade(
        self,
        symbol: str,
        side: str,
        pnl: float,
    ) -> None:
        """Record a completed trade."""
        result = "win" if pnl > 0 else ("loss" if pnl < 0 else "breakeven")
        TRADES_TOTAL.labels(symbol=symbol, side=side, result=result).inc()
        TRADE_PNL.labels(symbol=symbol).observe(pnl)

    def update_position(
        self,
        symbol: str,
        side: str | None,
        size: float,
        pnl: float,
    ) -> None:
        """Update position metrics."""
        if size > 0 and side:
            OPEN_POSITIONS.labels(symbol=symbol).set(1)
            POSITION_SIZE.labels(symbol=symbol, side=side).set(size)
            POSITION_PNL.labels(symbol=symbol).set(pnl)
        else:
            OPEN_POSITIONS.labels(symbol=symbol).set(0)
            POSITION_SIZE.labels(symbol=symbol, side="long").set(0)
            POSITION_SIZE.labels(symbol=symbol, side="short").set(0)
            POSITION_PNL.labels(symbol=symbol).set(0)

    def update_portfolio(
        self,
        total_value: float,
        cash: float,
        margin_used: float,
    ) -> None:
        """Update portfolio metrics."""
        PORTFOLIO_VALUE.set(total_value)
        PORTFOLIO_CASH.set(cash)
        PORTFOLIO_MARGIN_USED.set(margin_used)

    def update_performance(
        self,
        win_rate: float,
        sharpe_ratio: float,
        max_drawdown: float,
        current_drawdown: float,
        symbol: str = "all",
    ) -> None:
        """Update performance metrics."""
        WIN_RATE.labels(symbol=symbol).set(win_rate)
        SHARPE_RATIO.set(sharpe_ratio)
        MAX_DRAWDOWN.set(max_drawdown)
        CURRENT_DRAWDOWN.set(current_drawdown)

    # =========================================================================
    # Analysis Metrics Helpers
    # =========================================================================

    def record_technical_signal(
        self,
        symbol: str,
        timeframe: str,
        strength: float,
        confidence: float,
        indicators: dict,
    ) -> None:
        """Record technical analysis signal."""
        TECHNICAL_SIGNAL_STRENGTH.labels(
            symbol=symbol,
            timeframe=timeframe,
        ).set(strength)
        TECHNICAL_CONFIDENCE.labels(
            symbol=symbol,
            timeframe=timeframe,
        ).set(confidence)

        for name, value in indicators.items():
            if isinstance(value, (int, float)):
                INDICATOR_VALUE.labels(symbol=symbol, indicator=name).set(value)

    def record_llm_request(
        self,
        status: str,
        latency_seconds: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record an LLM API request."""
        LLM_REQUESTS_TOTAL.labels(status=status).inc()
        LLM_LATENCY.observe(latency_seconds)

        if input_tokens > 0:
            LLM_TOKENS_USED.labels(type="input").inc(input_tokens)
        if output_tokens > 0:
            LLM_TOKENS_USED.labels(type="output").inc(output_tokens)

    def record_llm_sentiment(
        self,
        symbol: str,
        sentiment: float,
        veto: bool = False,
        veto_reason: str | None = None,
    ) -> None:
        """Record LLM sentiment analysis."""
        LLM_SENTIMENT.labels(symbol=symbol).set(sentiment)

        if veto and veto_reason:
            LLM_VETO_TOTAL.labels(symbol=symbol, reason=veto_reason).inc()

    def record_ml_prediction(
        self,
        symbol: str,
        prediction: str,
        confidence: float,
    ) -> None:
        """Record ML model prediction."""
        ML_PREDICTIONS_TOTAL.labels(symbol=symbol, prediction=prediction).inc()
        ML_PREDICTION_CONFIDENCE.labels(symbol=symbol).set(confidence)

    # =========================================================================
    # Decision Engine Helpers
    # =========================================================================

    def record_decision(
        self,
        symbol: str,
        action: str,
        latency_seconds: float,
        vetoed: bool = False,
        veto_source: str | None = None,
    ) -> None:
        """Record a trading decision."""
        DECISIONS_TOTAL.labels(symbol=symbol, action=action).inc()
        DECISION_LATENCY.labels(symbol=symbol).observe(latency_seconds)

        if vetoed and veto_source:
            DECISIONS_VETOED.labels(symbol=symbol, veto_source=veto_source).inc()

    def update_safety_status(
        self,
        circuit_breaker_active: bool,
        kill_switch_active: bool,
    ) -> None:
        """Update safety system status."""
        CIRCUIT_BREAKER_STATUS.set(1 if circuit_breaker_active else 0)
        KILL_SWITCH_STATUS.set(1 if kill_switch_active else 0)

    # =========================================================================
    # Data Pipeline Helpers
    # =========================================================================

    def record_market_data(
        self,
        symbol: str,
        data_type: str,
        source: str,
        latency_ms: float,
    ) -> None:
        """Record market data update."""
        MARKET_DATA_UPDATES.labels(symbol=symbol, data_type=data_type).inc()
        MARKET_DATA_LATENCY.labels(symbol=symbol, source=source).observe(latency_ms)

    def update_websocket_status(self, symbol: str, connected: bool) -> None:
        """Update WebSocket connection status."""
        WEBSOCKET_STATUS.labels(symbol=symbol).set(1 if connected else 0)
        if not connected:
            WEBSOCKET_RECONNECTS.labels(symbol=symbol).inc()

    def record_cache_operation(
        self,
        cache_type: str,
        hit: bool,
        operation: str,
        latency_seconds: float,
    ) -> None:
        """Record cache operation."""
        if hit:
            CACHE_HITS.labels(cache_type=cache_type).inc()
        else:
            CACHE_MISSES.labels(cache_type=cache_type).inc()
        CACHE_LATENCY.labels(operation=operation).observe(latency_seconds)

    def record_db_query(
        self,
        operation: str,
        table: str,
        latency_seconds: float,
    ) -> None:
        """Record database query."""
        DB_QUERIES_TOTAL.labels(operation=operation, table=table).inc()
        DB_QUERY_LATENCY.labels(operation=operation).observe(latency_seconds)

    # =========================================================================
    # External Data Helpers
    # =========================================================================

    def record_external_api(
        self,
        api: str,
        status: str,
        latency_seconds: float,
    ) -> None:
        """Record external API request."""
        EXTERNAL_API_REQUESTS.labels(api=api, status=status).inc()
        EXTERNAL_API_LATENCY.labels(api=api).observe(latency_seconds)

    def update_market_sentiment(
        self,
        fear_greed: int | None = None,
        funding_rate: dict | None = None,
        open_interest: dict | None = None,
        long_short_ratio: dict | None = None,
    ) -> None:
        """Update market sentiment metrics."""
        if fear_greed is not None:
            FEAR_GREED_INDEX.set(fear_greed)

        if funding_rate:
            for symbol, rate in funding_rate.items():
                FUNDING_RATE.labels(symbol=symbol).set(rate)

        if open_interest:
            for symbol, oi in open_interest.items():
                OPEN_INTEREST.labels(symbol=symbol).set(oi)

        if long_short_ratio:
            for symbol, ratio in long_short_ratio.items():
                LONG_SHORT_RATIO.labels(symbol=symbol).set(ratio)

    # =========================================================================
    # System Health Helpers
    # =========================================================================

    def record_trading_loop(self, latency_seconds: float) -> None:
        """Record trading loop iteration."""
        TRADING_LOOP_ITERATIONS.inc()
        TRADING_LOOP_LATENCY.observe(latency_seconds)

    def record_error(self, component: str, error_type: str) -> None:
        """Record an error."""
        ERRORS_TOTAL.labels(component=component, error_type=error_type).inc()

    def record_warning(self, component: str) -> None:
        """Record a warning."""
        WARNINGS_TOTAL.labels(component=component).inc()


# =============================================================================
# Context Managers
# =============================================================================

@asynccontextmanager
async def measure_latency(histogram: Histogram, labels: dict):
    """
    Context manager to measure operation latency.

    Usage:
        async with measure_latency(ORDER_LATENCY, {"symbol": "BTCUSDT", "side": "long"}):
            await execute_order()
    """
    import time
    start = time.perf_counter()
    try:
        yield
    finally:
        latency = time.perf_counter() - start
        histogram.labels(**labels).observe(latency)


# =============================================================================
# Singleton Instance
# =============================================================================

_metrics_manager: MetricsManager | None = None


def get_metrics_manager(port: int = 8000) -> MetricsManager:
    """
    Get or create the singleton MetricsManager instance.

    Args:
        port: Port for Prometheus metrics endpoint

    Returns:
        MetricsManager singleton instance
    """
    global _metrics_manager
    if _metrics_manager is None:
        _metrics_manager = MetricsManager(port=port)
    return _metrics_manager
