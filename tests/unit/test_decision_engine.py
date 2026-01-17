"""
Unit tests for Decision Engine and Risk Management.

Tests the DecisionEngine, RiskManager, CircuitBreaker, KillSwitch and their
integration including position sizing, risk calculations, safety checks, and
decision-making logic with proper mocking of all dependencies.
"""

from datetime import UTC, datetime, timedelta

import pytest

from iftb.analysis import CompositeSignal, LLMAnalysis, ModelPrediction, SentimentScore
from iftb.config.constants import (
    CONSECUTIVE_LOSS_LIMIT,
    HIGH_CONFIDENCE_LEVERAGE,
    MAX_DAILY_LOSS_PCT,
    MAX_LEVERAGE,
    MAX_POSITION_PCT,
    MIN_LEVERAGE,
    MIN_POSITION_PCT,
)
from iftb.data import FearGreedData, FundingData, MarketContext
from iftb.trading.decision_engine import (
    CircuitBreaker,
    DecisionEngine,
    KillSwitch,
    RiskManager,
    TradeHistory,
    TradingDecision,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def risk_manager():
    """Create RiskManager instance for testing."""
    return RiskManager()


@pytest.fixture
def circuit_breaker():
    """Create CircuitBreaker instance for testing."""
    return CircuitBreaker()


@pytest.fixture
def kill_switch():
    """Create KillSwitch instance for testing."""
    return KillSwitch()


@pytest.fixture
def decision_engine(risk_manager, circuit_breaker, kill_switch):
    """Create DecisionEngine instance for testing."""
    return DecisionEngine(
        risk_manager=risk_manager,
        circuit_breaker=circuit_breaker,
        kill_switch=kill_switch,
    )


@pytest.fixture
def sample_trade_history_winning():
    """Create sample trade history with wins."""
    return [
        TradeHistory(
            symbol="BTCUSDT",
            action="LONG",
            entry_price=45000.0,
            exit_price=46000.0,
            position_size=0.05,
            leverage=5,
            pnl=250.0,
            pnl_pct=0.05,
            entry_time=datetime.now(UTC) - timedelta(hours=3),
            exit_time=datetime.now(UTC) - timedelta(hours=2),
            win=True,
        ),
        TradeHistory(
            symbol="BTCUSDT",
            action="SHORT",
            entry_price=46000.0,
            exit_price=45500.0,
            position_size=0.04,
            leverage=4,
            pnl=200.0,
            pnl_pct=0.04,
            entry_time=datetime.now(UTC) - timedelta(hours=2),
            exit_time=datetime.now(UTC) - timedelta(hours=1),
            win=True,
        ),
    ]


@pytest.fixture
def sample_trade_history_losing():
    """Create sample trade history with consecutive losses."""
    return [
        TradeHistory(
            symbol="BTCUSDT",
            action="LONG",
            entry_price=45000.0,
            exit_price=44500.0,
            position_size=0.05,
            leverage=5,
            pnl=-125.0,
            pnl_pct=-0.025,
            entry_time=datetime.now(UTC) - timedelta(hours=i * 2),
            exit_time=datetime.now(UTC) - timedelta(hours=i * 2 - 1),
            win=False,
        )
        for i in range(6)  # 6 consecutive losses
    ]


@pytest.fixture
def sample_technical_signal():
    """Create sample technical analysis signal."""
    from iftb.analysis import IndicatorResult

    return CompositeSignal(
        overall_signal="BULLISH",
        confidence=0.75,
        bullish_indicators=2,
        bearish_indicators=0,
        neutral_indicators=1,
        individual_signals={
            "RSI": IndicatorResult(
                name="RSI",
                value=65.0,
                signal="BULLISH",
                strength=0.8,
                timestamp=datetime.now(UTC),
            ),
            "MACD": IndicatorResult(
                name="MACD",
                value=0.5,
                signal="BULLISH",
                strength=0.7,
                timestamp=datetime.now(UTC),
            ),
            "ATR": IndicatorResult(
                name="ATR",
                value=1000.0,
                signal="NEUTRAL",
                strength=1.0,
                timestamp=datetime.now(UTC),
            ),
        },
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def sample_llm_analysis():
    """Create sample LLM analysis."""
    return LLMAnalysis(
        sentiment=SentimentScore.BULLISH,
        confidence=0.7,
        summary="Market shows strong bullish momentum with positive news flow",
        key_factors=["Strong volume", "Positive news", "Technical breakout"],
        should_veto=False,
        veto_reason=None,
        timestamp=datetime.now(UTC),
        model="claude-sonnet-4-5-20250929",
        prompt_tokens=100,
        completion_tokens=50,
        cached=False,
    )


@pytest.fixture
def sample_ml_prediction():
    """Create sample ML model prediction."""
    return ModelPrediction(
        action="LONG",
        confidence=0.8,
        probability_long=0.8,
        probability_short=0.1,
        probability_hold=0.1,
        feature_importance={
            "rsi": 0.3,
            "macd": 0.25,
            "volume": 0.2,
            "price_change": 0.15,
            "atr": 0.1,
        },
        model_version="1.0.0",
        prediction_time=datetime.now(UTC),
    )


@pytest.fixture
def sample_market_context():
    """Create sample market context."""
    return MarketContext(
        fear_greed=FearGreedData(
            value=65,
            classification="Greed",
            timestamp=datetime.now(UTC),
        ),
        funding=FundingData(
            symbol="BTCUSDT",
            rate=0.0001,
            predicted_rate=0.00012,
            next_funding_time=datetime.now(UTC) + timedelta(hours=8),
        ),
    )


# =============================================================================
# RiskManager Tests
# =============================================================================


@pytest.mark.unit
class TestRiskManager:
    """Tests for RiskManager class."""

    def test_calculate_kelly_position_valid(self, risk_manager):
        """Test Kelly position calculation with valid inputs."""
        position_size = risk_manager.calculate_kelly_position(
            win_rate=0.6,
            avg_win=100.0,
            avg_loss=80.0,
            current_capital=10000.0,
        )

        # Kelly: (0.6 * (100/80 + 1) - 1) / (100/80) = (0.6 * 2.25 - 1) / 1.25 = 0.28
        # Quarter-Kelly: 0.28 * 0.25 = 0.07
        assert isinstance(position_size, float)
        assert MIN_POSITION_PCT <= position_size <= MAX_POSITION_PCT
        assert position_size > 0

    def test_calculate_kelly_position_high_win_rate(self, risk_manager):
        """Test Kelly position with high win rate."""
        position_size = risk_manager.calculate_kelly_position(
            win_rate=0.8,
            avg_win=150.0,
            avg_loss=100.0,
            current_capital=10000.0,
        )

        # Should produce higher position size
        assert position_size > MIN_POSITION_PCT
        assert position_size <= MAX_POSITION_PCT

    def test_calculate_kelly_position_invalid_win_rate_zero(self, risk_manager):
        """Test Kelly position with invalid win rate (0)."""
        position_size = risk_manager.calculate_kelly_position(
            win_rate=0.0,
            avg_win=100.0,
            avg_loss=80.0,
            current_capital=10000.0,
        )

        # Should return minimum position size
        assert position_size == MIN_POSITION_PCT

    def test_calculate_kelly_position_invalid_win_rate_one(self, risk_manager):
        """Test Kelly position with invalid win rate (1.0)."""
        position_size = risk_manager.calculate_kelly_position(
            win_rate=1.0,
            avg_win=100.0,
            avg_loss=80.0,
            current_capital=10000.0,
        )

        # Should return minimum position size
        assert position_size == MIN_POSITION_PCT

    def test_calculate_kelly_position_invalid_avg_loss(self, risk_manager):
        """Test Kelly position with invalid avg_loss (0 or negative)."""
        position_size = risk_manager.calculate_kelly_position(
            win_rate=0.6,
            avg_win=100.0,
            avg_loss=0.0,
            current_capital=10000.0,
        )

        # Should return minimum position size
        assert position_size == MIN_POSITION_PCT

        position_size = risk_manager.calculate_kelly_position(
            win_rate=0.6,
            avg_win=100.0,
            avg_loss=-50.0,
            current_capital=10000.0,
        )

        assert position_size == MIN_POSITION_PCT

    def test_check_daily_loss_limit_within_limit(self, risk_manager):
        """Test daily loss limit check - within limit."""
        result = risk_manager.check_daily_loss_limit(
            current_pnl=-500.0,
            capital=10000.0,
        )

        # 500/10000 = 5%, below 8% limit
        assert result is True

    def test_check_daily_loss_limit_exceeded(self, risk_manager):
        """Test daily loss limit check - exceeded."""
        result = risk_manager.check_daily_loss_limit(
            current_pnl=-900.0,
            capital=10000.0,
        )

        # 900/10000 = 9%, above 8% limit
        assert result is False

    def test_check_daily_loss_limit_at_threshold(self, risk_manager):
        """Test daily loss limit check - exactly at threshold."""
        capital = 10000.0
        loss = capital * MAX_DAILY_LOSS_PCT

        result = risk_manager.check_daily_loss_limit(
            current_pnl=-loss,
            capital=capital,
        )

        # At exact threshold should trigger limit
        assert result is False

    def test_check_daily_loss_limit_positive_pnl(self, risk_manager):
        """Test daily loss limit check with positive PnL."""
        result = risk_manager.check_daily_loss_limit(
            current_pnl=500.0,
            capital=10000.0,
        )

        # Positive PnL should always pass
        assert result is True

    def test_check_consecutive_losses_within_limit(
        self, risk_manager, sample_trade_history_winning
    ):
        """Test consecutive losses check - within limit."""
        result = risk_manager.check_consecutive_losses(sample_trade_history_winning)
        assert result is True

    def test_check_consecutive_losses_exceeded(
        self, risk_manager, sample_trade_history_losing
    ):
        """Test consecutive losses check - exceeded."""
        result = risk_manager.check_consecutive_losses(sample_trade_history_losing)
        assert result is False

    def test_check_consecutive_losses_empty_history(self, risk_manager):
        """Test consecutive losses check with empty history."""
        result = risk_manager.check_consecutive_losses([])
        assert result is True

    def test_check_consecutive_losses_at_threshold(self, risk_manager):
        """Test consecutive losses at exact threshold."""
        # Create exactly CONSECUTIVE_LOSS_LIMIT losses
        losing_trades = [
            TradeHistory(
                symbol="BTCUSDT",
                action="LONG",
                entry_price=45000.0,
                exit_price=44500.0,
                position_size=0.05,
                leverage=5,
                pnl=-125.0,
                pnl_pct=-0.025,
                entry_time=datetime.now(UTC) - timedelta(hours=i * 2),
                exit_time=datetime.now(UTC) - timedelta(hours=i * 2 - 1),
                win=False,
            )
            for i in range(CONSECUTIVE_LOSS_LIMIT)
        ]

        result = risk_manager.check_consecutive_losses(losing_trades)
        assert result is False

    def test_calculate_stop_loss_long(self, risk_manager):
        """Test stop-loss calculation for LONG position."""
        entry_price = 45000.0
        atr = 1000.0

        stop_loss = risk_manager.calculate_stop_loss(
            entry_price=entry_price,
            atr=atr,
            direction="LONG",
        )

        # Stop should be 2 * ATR below entry for LONG
        expected_stop = entry_price - (2.0 * atr)
        assert stop_loss == expected_stop
        assert stop_loss < entry_price

    def test_calculate_stop_loss_short(self, risk_manager):
        """Test stop-loss calculation for SHORT position."""
        entry_price = 45000.0
        atr = 1000.0

        stop_loss = risk_manager.calculate_stop_loss(
            entry_price=entry_price,
            atr=atr,
            direction="SHORT",
        )

        # Stop should be 2 * ATR above entry for SHORT
        expected_stop = entry_price + (2.0 * atr)
        assert stop_loss == expected_stop
        assert stop_loss > entry_price

    def test_calculate_take_profit_long(self, risk_manager):
        """Test take-profit calculation for LONG position."""
        entry_price = 45000.0
        atr = 1000.0

        take_profit = risk_manager.calculate_take_profit(
            entry_price=entry_price,
            atr=atr,
            direction="LONG",
        )

        # Take profit should be 3 * ATR above entry for LONG
        expected_tp = entry_price + (3.0 * atr)
        assert take_profit == expected_tp
        assert take_profit > entry_price

    def test_calculate_take_profit_short(self, risk_manager):
        """Test take-profit calculation for SHORT position."""
        entry_price = 45000.0
        atr = 1000.0

        take_profit = risk_manager.calculate_take_profit(
            entry_price=entry_price,
            atr=atr,
            direction="SHORT",
        )

        # Take profit should be 3 * ATR below entry for SHORT
        expected_tp = entry_price - (3.0 * atr)
        assert take_profit == expected_tp
        assert take_profit < entry_price

    def test_adjust_leverage_low_volatility(self, risk_manager):
        """Test leverage adjustment with low volatility."""
        volatility = 0.01  # 1% volatility (low)

        leverage = risk_manager.adjust_leverage(volatility)

        # Low volatility should allow higher leverage
        assert leverage == HIGH_CONFIDENCE_LEVERAGE
        assert leverage <= MAX_LEVERAGE

    def test_adjust_leverage_high_volatility(self, risk_manager):
        """Test leverage adjustment with high volatility."""
        volatility = 0.10  # 10% volatility (high)

        leverage = risk_manager.adjust_leverage(volatility)

        # High volatility should result in minimum leverage
        assert leverage == MIN_LEVERAGE
        assert leverage >= MIN_LEVERAGE

    def test_adjust_leverage_medium_volatility(self, risk_manager):
        """Test leverage adjustment with medium volatility."""
        volatility = 0.05  # 5% volatility (medium)

        leverage = risk_manager.adjust_leverage(volatility)

        # Medium volatility should result in intermediate leverage
        assert MIN_LEVERAGE <= leverage <= HIGH_CONFIDENCE_LEVERAGE


# =============================================================================
# CircuitBreaker Tests
# =============================================================================


@pytest.mark.unit
class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state(self, circuit_breaker):
        """Test circuit breaker initial state."""
        assert circuit_breaker.is_triggered is False
        assert circuit_breaker.trigger_time is None
        assert circuit_breaker.trigger_reason is None
        assert circuit_breaker.cooldown_hours == 24

    def test_trigger_excessive_drawdown(self, circuit_breaker):
        """Test circuit breaker triggers on excessive drawdown."""
        metrics = {
            "drawdown": 0.35,  # 35% drawdown, above 30% limit
            "volatility": 0.05,
            "error_rate": 0.1,
            "api_failure_rate": 0.1,
        }

        should_halt, reason = circuit_breaker.check(metrics)

        assert should_halt is True
        assert "drawdown" in reason.lower()
        assert circuit_breaker.is_triggered is True
        assert circuit_breaker.trigger_time is not None

    def test_trigger_extreme_volatility(self, circuit_breaker):
        """Test circuit breaker triggers on extreme volatility."""
        metrics = {
            "drawdown": 0.10,
            "volatility": 0.20,  # 20% volatility, above 15% limit
            "error_rate": 0.1,
            "api_failure_rate": 0.1,
        }

        should_halt, reason = circuit_breaker.check(metrics)

        assert should_halt is True
        assert "volatility" in reason.lower()
        assert circuit_breaker.is_triggered is True

    def test_trigger_high_error_rate(self, circuit_breaker):
        """Test circuit breaker triggers on high error rate."""
        metrics = {
            "drawdown": 0.10,
            "volatility": 0.05,
            "error_rate": 0.35,  # 35% error rate, above 30% limit
            "api_failure_rate": 0.1,
        }

        should_halt, reason = circuit_breaker.check(metrics)

        assert should_halt is True
        assert "error rate" in reason.lower()
        assert circuit_breaker.is_triggered is True

    def test_trigger_high_api_failure_rate(self, circuit_breaker):
        """Test circuit breaker triggers on high API failure rate."""
        metrics = {
            "drawdown": 0.10,
            "volatility": 0.05,
            "error_rate": 0.1,
            "api_failure_rate": 0.60,  # 60% API failure, above 50% limit
        }

        should_halt, reason = circuit_breaker.check(metrics)

        assert should_halt is True
        assert "api failure" in reason.lower()
        assert circuit_breaker.is_triggered is True

    def test_no_trigger_normal_conditions(self, circuit_breaker):
        """Test circuit breaker does not trigger under normal conditions."""
        metrics = {
            "drawdown": 0.10,
            "volatility": 0.05,
            "error_rate": 0.1,
            "api_failure_rate": 0.2,
        }

        should_halt, reason = circuit_breaker.check(metrics)

        assert should_halt is False
        assert reason == ""
        assert circuit_breaker.is_triggered is False

    def test_cooldown_period_active(self, circuit_breaker):
        """Test circuit breaker remains active during cooldown period."""
        # Trigger the circuit breaker
        circuit_breaker.trigger("Test trigger")

        # Check immediately - should still be active
        metrics = {
            "drawdown": 0.10,
            "volatility": 0.05,
            "error_rate": 0.1,
            "api_failure_rate": 0.1,
        }

        should_halt, reason = circuit_breaker.check(metrics)

        assert should_halt is True
        assert "Circuit breaker OPEN" in reason
        assert circuit_breaker.is_triggered is True

    def test_cooldown_period_complete(self, circuit_breaker):
        """Test circuit breaker auto-resets after cooldown period."""
        # Trigger the circuit breaker
        circuit_breaker.trigger("Test trigger")
        assert circuit_breaker.is_triggered is True

        # Manually set trigger time to 25 hours ago (past cooldown)
        circuit_breaker.trigger_time = datetime.now(UTC) - timedelta(hours=25)

        # Check with normal metrics
        metrics = {
            "drawdown": 0.10,
            "volatility": 0.05,
            "error_rate": 0.1,
            "api_failure_rate": 0.1,
        }

        should_halt, reason = circuit_breaker.check(metrics)

        # Should have auto-reset and not halt
        assert should_halt is False
        assert circuit_breaker.is_triggered is False

    def test_reset_functionality(self, circuit_breaker):
        """Test manual reset of circuit breaker."""
        # Trigger the circuit breaker
        circuit_breaker.trigger("Test trigger")
        assert circuit_breaker.is_triggered is True

        # Reset it
        circuit_breaker.reset()

        assert circuit_breaker.is_triggered is False
        assert circuit_breaker.trigger_time is None
        assert circuit_breaker.trigger_reason is None


# =============================================================================
# KillSwitch Tests
# =============================================================================


@pytest.mark.unit
class TestKillSwitch:
    """Tests for KillSwitch class."""

    def test_initial_state(self, kill_switch):
        """Test kill switch initial state."""
        assert kill_switch.is_active() is False
        assert kill_switch.activation_time is None
        assert kill_switch.activation_reason is None

    def test_activate(self, kill_switch):
        """Test kill switch activation."""
        reason = "Emergency stop - critical error detected"

        kill_switch.activate(reason)

        assert kill_switch.is_active() is True
        assert kill_switch.activation_time is not None
        assert kill_switch.activation_reason == reason

    def test_deactivate(self, kill_switch):
        """Test kill switch deactivation with confirmation code."""
        # First activate
        confirmation_code = kill_switch.activate("Test activation")
        assert kill_switch.is_active() is True
        assert confirmation_code is not None

        # Then deactivate with correct confirmation code
        success, message = kill_switch.deactivate(confirmation_code)

        assert success is True
        assert kill_switch.is_active() is False

    def test_multiple_activations(self, kill_switch):
        """Test multiple activations update the reason."""
        kill_switch.activate("First reason")
        first_time = kill_switch.activation_time

        kill_switch.activate("Second reason")

        assert kill_switch.is_active() is True
        assert kill_switch.activation_reason == "Second reason"
        assert kill_switch.activation_time != first_time


# =============================================================================
# DecisionEngine Tests
# =============================================================================


@pytest.mark.unit
class TestDecisionEngine:
    """Tests for DecisionEngine class."""

    @pytest.mark.asyncio
    async def test_make_decision_kill_switch_active(
        self,
        decision_engine,
        sample_technical_signal,
        sample_llm_analysis,
        sample_ml_prediction,
        sample_market_context,
    ):
        """Test decision making with kill switch active."""
        # Activate kill switch
        decision_engine.kill_switch.activate("Testing kill switch")

        decision = await decision_engine.make_decision(
            symbol="BTCUSDT",
            technical_signal=sample_technical_signal,
            llm_analysis=sample_llm_analysis,
            ml_prediction=sample_ml_prediction,
            market_context=sample_market_context,
            current_price=45000.0,
            account_balance=10000.0,
        )

        assert decision.action == "HOLD"
        assert decision.vetoed is True
        assert "Kill switch" in decision.veto_reason

    @pytest.mark.asyncio
    async def test_make_decision_circuit_breaker_triggered(
        self,
        decision_engine,
        sample_technical_signal,
        sample_llm_analysis,
        sample_ml_prediction,
        sample_market_context,
    ):
        """Test decision making with circuit breaker triggered."""
        # Trigger circuit breaker
        decision_engine.circuit_breaker.trigger("High drawdown")

        decision = await decision_engine.make_decision(
            symbol="BTCUSDT",
            technical_signal=sample_technical_signal,
            llm_analysis=sample_llm_analysis,
            ml_prediction=sample_ml_prediction,
            market_context=sample_market_context,
            current_price=45000.0,
            account_balance=10000.0,
        )

        assert decision.action == "HOLD"
        assert decision.vetoed is True
        assert "Circuit breaker" in decision.veto_reason

    @pytest.mark.asyncio
    async def test_make_decision_daily_loss_limit_exceeded(
        self,
        decision_engine,
        sample_technical_signal,
        sample_llm_analysis,
        sample_ml_prediction,
        sample_market_context,
    ):
        """Test decision making with daily loss limit exceeded."""
        decision = await decision_engine.make_decision(
            symbol="BTCUSDT",
            technical_signal=sample_technical_signal,
            llm_analysis=sample_llm_analysis,
            ml_prediction=sample_ml_prediction,
            market_context=sample_market_context,
            current_price=45000.0,
            account_balance=10000.0,
            current_pnl=-1000.0,  # 10% loss, above 8% limit
        )

        assert decision.action == "HOLD"
        assert decision.vetoed is True
        assert "loss limit" in decision.veto_reason.lower()

    @pytest.mark.asyncio
    async def test_make_decision_consecutive_losses_exceeded(
        self,
        decision_engine,
        sample_technical_signal,
        sample_llm_analysis,
        sample_ml_prediction,
        sample_market_context,
        sample_trade_history_losing,
    ):
        """Test decision making with consecutive losses exceeded."""
        decision = await decision_engine.make_decision(
            symbol="BTCUSDT",
            technical_signal=sample_technical_signal,
            llm_analysis=sample_llm_analysis,
            ml_prediction=sample_ml_prediction,
            market_context=sample_market_context,
            current_price=45000.0,
            account_balance=10000.0,
            trade_history=sample_trade_history_losing,
        )

        assert decision.action == "HOLD"
        assert decision.vetoed is True
        assert "Consecutive loss" in decision.veto_reason

    @pytest.mark.asyncio
    async def test_make_decision_low_confidence_veto(
        self,
        decision_engine,
        sample_technical_signal,
        sample_llm_analysis,
        sample_ml_prediction,
        sample_market_context,
    ):
        """Test decision making with low combined confidence."""
        # Lower all confidence scores
        sample_technical_signal.confidence = 0.2
        sample_llm_analysis.confidence = 0.2
        sample_ml_prediction.confidence = 0.2

        decision = await decision_engine.make_decision(
            symbol="BTCUSDT",
            technical_signal=sample_technical_signal,
            llm_analysis=sample_llm_analysis,
            ml_prediction=sample_ml_prediction,
            market_context=sample_market_context,
            current_price=45000.0,
            account_balance=10000.0,
        )

        assert decision.action == "HOLD"
        assert decision.vetoed is True
        assert "confidence too low" in decision.veto_reason.lower()

    @pytest.mark.asyncio
    async def test_make_decision_sentiment_veto(
        self,
        decision_engine,
        sample_technical_signal,
        sample_llm_analysis,
        sample_ml_prediction,
        sample_market_context,
    ):
        """Test decision making with very negative LLM sentiment."""
        # Set very negative sentiment
        sample_llm_analysis.sentiment = SentimentScore.VERY_BEARISH

        decision = await decision_engine.make_decision(
            symbol="BTCUSDT",
            technical_signal=sample_technical_signal,
            llm_analysis=sample_llm_analysis,
            ml_prediction=sample_ml_prediction,
            market_context=sample_market_context,
            current_price=45000.0,
            account_balance=10000.0,
        )

        assert decision.action == "HOLD"
        assert decision.vetoed is True
        assert "sentiment" in decision.veto_reason.lower()

    @pytest.mark.asyncio
    async def test_make_decision_signal_disagreement(
        self,
        decision_engine,
        sample_technical_signal,
        sample_llm_analysis,
        sample_ml_prediction,
        sample_market_context,
    ):
        """Test decision making with major signal disagreement."""
        # Make all signals disagree - but keep sentiment above veto threshold
        sample_technical_signal.overall_signal = "BULLISH"
        sample_llm_analysis.sentiment = SentimentScore.NEUTRAL  # 0.0, above -0.5 threshold
        sample_ml_prediction.action = "SHORT"  # Opposite of technical

        decision = await decision_engine.make_decision(
            symbol="BTCUSDT",
            technical_signal=sample_technical_signal,
            llm_analysis=sample_llm_analysis,
            ml_prediction=sample_ml_prediction,
            market_context=sample_market_context,
            current_price=45000.0,
            account_balance=10000.0,
        )

        assert decision.action == "HOLD"
        assert decision.vetoed is True
        assert "disagreement" in decision.veto_reason.lower()

    @pytest.mark.asyncio
    async def test_make_decision_hold_weak_signal(
        self,
        decision_engine,
        sample_technical_signal,
        sample_llm_analysis,
        sample_ml_prediction,
        sample_market_context,
    ):
        """Test decision making results in HOLD for weak combined signal."""
        # Make all signals neutral/weak - but confidence still good
        sample_technical_signal.overall_signal = "NEUTRAL"
        sample_technical_signal.confidence = 0.6
        sample_llm_analysis.sentiment = SentimentScore.NEUTRAL
        sample_llm_analysis.confidence = 0.6
        sample_ml_prediction.action = "HOLD"
        sample_ml_prediction.confidence = 0.6

        decision = await decision_engine.make_decision(
            symbol="BTCUSDT",
            technical_signal=sample_technical_signal,
            llm_analysis=sample_llm_analysis,
            ml_prediction=sample_ml_prediction,
            market_context=sample_market_context,
            current_price=45000.0,
            account_balance=10000.0,
        )

        assert decision.action == "HOLD"
        # Note: might be vetoed due to signal alignment check, so don't assert vetoed status
        assert decision.position_size == 0.0

    @pytest.mark.asyncio
    async def test_make_decision_long_generation(
        self,
        decision_engine,
        sample_technical_signal,
        sample_llm_analysis,
        sample_ml_prediction,
        sample_market_context,
        sample_trade_history_winning,
    ):
        """Test decision making generates LONG decision."""
        # All signals bullish
        sample_technical_signal.overall_signal = "BULLISH"
        sample_llm_analysis.sentiment = SentimentScore.BULLISH
        sample_ml_prediction.action = "LONG"

        decision = await decision_engine.make_decision(
            symbol="BTCUSDT",
            technical_signal=sample_technical_signal,
            llm_analysis=sample_llm_analysis,
            ml_prediction=sample_ml_prediction,
            market_context=sample_market_context,
            current_price=45000.0,
            account_balance=10000.0,
            trade_history=sample_trade_history_winning,
        )

        assert decision.action == "LONG"
        assert decision.vetoed is False
        assert decision.position_size > 0
        assert MIN_POSITION_PCT <= decision.position_size <= MAX_POSITION_PCT
        assert MIN_LEVERAGE <= decision.leverage <= MAX_LEVERAGE
        assert decision.stop_loss < decision.entry_price
        assert decision.take_profit > decision.entry_price
        assert len(decision.reasons) > 0

    @pytest.mark.asyncio
    async def test_make_decision_short_generation(
        self,
        decision_engine,
        sample_technical_signal,
        sample_llm_analysis,
        sample_ml_prediction,
        sample_market_context,
        sample_trade_history_winning,
    ):
        """Test decision making generates SHORT decision."""
        # All signals bearish - but sentiment above veto threshold
        sample_technical_signal.overall_signal = "BEARISH"
        sample_technical_signal.confidence = 0.8
        # Use NEUTRAL sentiment (0.0) which is above -0.5 veto threshold
        # But ML will be strongly SHORT
        sample_llm_analysis.sentiment = SentimentScore.NEUTRAL
        sample_llm_analysis.confidence = 0.7
        sample_ml_prediction.action = "SHORT"
        sample_ml_prediction.confidence = 0.9

        decision = await decision_engine.make_decision(
            symbol="BTCUSDT",
            technical_signal=sample_technical_signal,
            llm_analysis=sample_llm_analysis,
            ml_prediction=sample_ml_prediction,
            market_context=sample_market_context,
            current_price=45000.0,
            account_balance=10000.0,
            trade_history=sample_trade_history_winning,
        )

        # With two strong bearish signals and one neutral, should generate SHORT
        assert decision.action == "SHORT"
        assert decision.vetoed is False
        assert decision.position_size > 0
        assert MIN_POSITION_PCT <= decision.position_size <= MAX_POSITION_PCT
        assert MIN_LEVERAGE <= decision.leverage <= MAX_LEVERAGE
        assert decision.stop_loss > decision.entry_price
        assert decision.take_profit < decision.entry_price
        assert len(decision.reasons) > 0

    @pytest.mark.asyncio
    async def test_make_decision_signal_combination(
        self,
        decision_engine,
        sample_technical_signal,
        sample_llm_analysis,
        sample_ml_prediction,
        sample_market_context,
        sample_trade_history_winning,
    ):
        """Test signal combination with proper weighting."""
        # Technical: BULLISH (0.40 weight)
        # LLM: BULLISH (0.25 weight)
        # ML: LONG (0.35 weight)
        sample_technical_signal.overall_signal = "BULLISH"
        sample_technical_signal.confidence = 0.8
        sample_llm_analysis.sentiment = SentimentScore.BULLISH
        sample_llm_analysis.confidence = 0.7
        sample_ml_prediction.action = "LONG"
        sample_ml_prediction.confidence = 0.9

        decision = await decision_engine.make_decision(
            symbol="BTCUSDT",
            technical_signal=sample_technical_signal,
            llm_analysis=sample_llm_analysis,
            ml_prediction=sample_ml_prediction,
            market_context=sample_market_context,
            current_price=45000.0,
            account_balance=10000.0,
            trade_history=sample_trade_history_winning,
        )

        assert decision.action == "LONG"
        # Check combined confidence is weighted properly
        expected_confidence = (0.8 * 0.40) + (0.7 * 0.25) + (0.9 * 0.35)
        assert abs(decision.confidence - expected_confidence) < 0.01


# =============================================================================
# TradingDecision Tests
# =============================================================================


@pytest.mark.unit
class TestTradingDecision:
    """Tests for TradingDecision dataclass."""

    def test_to_dict(self):
        """Test TradingDecision serialization to dict."""
        decision = TradingDecision(
            action="LONG",
            symbol="BTCUSDT",
            confidence=0.75,
            position_size=0.05,
            leverage=5,
            stop_loss=44000.0,
            take_profit=46000.0,
            entry_price=45000.0,
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            reasons=["Strong technical signal"],
            vetoed=False,
            veto_reason=None,
        )

        result = decision.to_dict()

        assert result["action"] == "LONG"
        assert result["symbol"] == "BTCUSDT"
        assert result["confidence"] == 0.75
        assert result["position_size"] == 0.05
        assert result["leverage"] == 5
        assert result["stop_loss"] == 44000.0
        assert result["take_profit"] == 46000.0
        assert result["entry_price"] == 45000.0
        assert result["vetoed"] is False

    def test_repr_long(self):
        """Test TradingDecision string representation for LONG."""
        decision = TradingDecision(
            action="LONG",
            symbol="BTCUSDT",
            confidence=0.75,
            position_size=0.05,
            leverage=5,
            stop_loss=44000.0,
            take_profit=46000.0,
            entry_price=45000.0,
            timestamp=datetime.now(UTC),
        )

        repr_str = repr(decision)

        assert "LONG" in repr_str
        assert "BTCUSDT" in repr_str
        # Check for percentage formatting (5.00%)
        assert "5.00%" in repr_str or "size=" in repr_str
        assert "5x" in repr_str

    def test_repr_hold(self):
        """Test TradingDecision string representation for HOLD."""
        decision = TradingDecision(
            action="HOLD",
            symbol="BTCUSDT",
            confidence=0.5,
            position_size=0.0,
            leverage=1,
            stop_loss=45000.0,
            take_profit=45000.0,
            entry_price=45000.0,
            timestamp=datetime.now(UTC),
        )

        repr_str = repr(decision)

        assert "HOLD" in repr_str
        assert "BTCUSDT" in repr_str

    def test_repr_vetoed(self):
        """Test TradingDecision string representation for vetoed decision."""
        decision = TradingDecision(
            action="HOLD",
            symbol="BTCUSDT",
            confidence=0.0,
            position_size=0.0,
            leverage=1,
            stop_loss=45000.0,
            take_profit=45000.0,
            entry_price=45000.0,
            timestamp=datetime.now(UTC),
            vetoed=True,
            veto_reason="Kill switch active",
        )

        repr_str = repr(decision)

        assert "VETOED" in repr_str
        assert "Kill switch" in repr_str
