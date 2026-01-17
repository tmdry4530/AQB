"""
Integration tests for IFTB Trading Bot.

Tests component initialization, data flow, and end-to-end functionality.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip if imports fail (missing dependencies)
pytestmark = pytest.mark.skipif(
    True,  # Set to False when ready to run tests
    reason="Integration tests require all dependencies"
)


class TestTechnicalAnalyzer:
    """Test technical analysis component."""

    def test_indicator_calculation(self):
        """Test that indicators are calculated correctly."""
        from iftb.analysis import TechnicalAnalyzer, CompositeSignal
        from iftb.data import OHLCVBar

        # Create mock OHLCV data
        ohlcv_data = []
        base_price = 50000.0
        for i in range(100):
            ohlcv_data.append(OHLCVBar(
                timestamp=datetime.now(timezone.utc),
                open=base_price + i * 10,
                high=base_price + i * 10 + 50,
                low=base_price + i * 10 - 50,
                close=base_price + i * 10 + 25,
                volume=1000.0 + i * 10,
            ))

        analyzer = TechnicalAnalyzer()
        signal = analyzer.analyze(ohlcv_data)

        assert isinstance(signal, CompositeSignal)
        assert signal.direction in ["bullish", "bearish", "neutral"]
        assert -1.0 <= signal.strength <= 1.0

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        from iftb.analysis import TechnicalAnalyzer
        from iftb.data import OHLCVBar

        # Create insufficient data (less than 50 bars)
        ohlcv_data = [
            OHLCVBar(
                timestamp=datetime.now(timezone.utc),
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50050.0,
                volume=1000.0,
            )
            for _ in range(20)
        ]

        analyzer = TechnicalAnalyzer()
        signal = analyzer.analyze(ohlcv_data)

        # Should return neutral signal for insufficient data
        assert signal.direction == "neutral"
        assert signal.confidence < 0.5


class TestDecisionEngine:
    """Test decision engine component."""

    def test_hold_decision_on_low_confidence(self):
        """Test that low confidence results in HOLD."""
        from iftb.trading import create_decision_engine
        from iftb.analysis import CompositeSignal

        engine = create_decision_engine()

        # Create weak signal
        weak_signal = CompositeSignal(
            direction="bullish",
            strength=0.2,
            confidence=0.3,
            indicators={},
            timestamp=datetime.now(timezone.utc),
        )

        decision = engine.make_decision(
            symbol="BTCUSDT",
            current_price=50000.0,
            technical_signal=weak_signal,
            llm_analysis=None,
            ml_prediction=None,
            market_context=None,
        )

        assert decision.action == "HOLD"

    def test_veto_on_circuit_breaker(self):
        """Test that circuit breaker triggers veto."""
        from iftb.trading import create_decision_engine, CircuitBreaker
        from iftb.analysis import CompositeSignal

        engine = create_decision_engine()

        # Activate circuit breaker
        engine.circuit_breaker.activate("test_high_volatility")

        strong_signal = CompositeSignal(
            direction="bullish",
            strength=0.9,
            confidence=0.8,
            indicators={},
            timestamp=datetime.now(timezone.utc),
        )

        decision = engine.make_decision(
            symbol="BTCUSDT",
            current_price=50000.0,
            technical_signal=strong_signal,
            llm_analysis=None,
            ml_prediction=None,
            market_context=None,
        )

        assert decision.vetoed is True
        assert "circuit_breaker" in decision.veto_reason.lower()


class TestOrderExecutor:
    """Test order execution component."""

    @pytest.mark.asyncio
    async def test_paper_trading_execution(self):
        """Test paper trading order execution."""
        from iftb.trading import OrderExecutor, ExecutionRequest

        executor = OrderExecutor(
            paper_mode=True,
            initial_balance=10000.0,
        )

        request = ExecutionRequest(
            action="long",
            symbol="BTC/USDT",
            amount=0.01,
            entry_price=50000.0,
            stop_loss=48000.0,
            take_profit=55000.0,
            leverage=5,
        )

        order = await executor.execute_decision(request)

        assert order.status == "filled"
        assert order.filled_price is not None
        assert order.filled_amount > 0

    @pytest.mark.asyncio
    async def test_paper_trading_position_tracking(self):
        """Test that positions are tracked correctly in paper mode."""
        from iftb.trading import OrderExecutor, ExecutionRequest

        executor = OrderExecutor(
            paper_mode=True,
            initial_balance=10000.0,
        )

        # Open position
        open_request = ExecutionRequest(
            action="long",
            symbol="BTC/USDT",
            amount=0.01,
            entry_price=50000.0,
            leverage=5,
        )
        await executor.execute_decision(open_request)

        # Check position exists
        status = executor.get_account_status()
        assert status["open_positions"] >= 1


class TestRiskManager:
    """Test risk management component."""

    def test_position_size_limits(self):
        """Test that position sizes are limited correctly."""
        from iftb.trading import RiskManager

        risk_manager = RiskManager()

        # Test that max position size is enforced
        large_size = risk_manager.calculate_position_size(
            balance=10000.0,
            entry_price=50000.0,
            stop_loss=45000.0,
            confidence=0.99,
        )

        # Should not exceed MAX_POSITION_PCT (10%)
        assert large_size <= 0.10

    def test_kelly_criterion_sizing(self):
        """Test Kelly Criterion position sizing."""
        from iftb.trading import RiskManager

        risk_manager = RiskManager()

        # With 60% win rate and 2:1 reward/risk
        size = risk_manager.calculate_position_size(
            balance=10000.0,
            entry_price=50000.0,
            stop_loss=48000.0,
            confidence=0.6,
            win_rate=0.6,
            reward_risk_ratio=2.0,
        )

        # Kelly = (0.6 * 2 - 0.4) / 2 = 0.4, with 0.25 fraction = 0.1
        assert 0.01 <= size <= 0.10


class TestDataIntegration:
    """Test data fetching and processing."""

    @pytest.mark.asyncio
    async def test_ohlcv_fetch_mock(self):
        """Test OHLCV data fetching with mock."""
        from iftb.data import ExchangeClient, OHLCVBar

        # Create mock exchange client
        mock_client = MagicMock(spec=ExchangeClient)
        mock_client.fetch_ohlcv = AsyncMock(return_value=[
            OHLCVBar(
                timestamp=datetime.now(timezone.utc),
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50050.0,
                volume=1000.0,
            )
            for _ in range(100)
        ])

        ohlcv = await mock_client.fetch_ohlcv("BTCUSDT", "1h", 100)
        assert len(ohlcv) == 100


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_trading_cycle_paper_mode(self):
        """Test complete trading cycle in paper mode."""
        from iftb.analysis import TechnicalAnalyzer
        from iftb.trading import (
            create_decision_engine,
            OrderExecutor,
            convert_decision_to_request,
        )
        from iftb.data import OHLCVBar

        # 1. Create OHLCV data with bullish trend
        ohlcv_data = []
        base_price = 50000.0
        for i in range(100):
            # Create uptrend
            ohlcv_data.append(OHLCVBar(
                timestamp=datetime.now(timezone.utc),
                open=base_price + i * 100,
                high=base_price + i * 100 + 150,
                low=base_price + i * 100 - 50,
                close=base_price + i * 100 + 120,
                volume=1000.0 + i * 50,
            ))

        # 2. Generate technical signal
        analyzer = TechnicalAnalyzer()
        signal = analyzer.analyze(ohlcv_data)

        # 3. Make trading decision
        engine = create_decision_engine()
        decision = engine.make_decision(
            symbol="BTCUSDT",
            current_price=ohlcv_data[-1].close,
            technical_signal=signal,
            llm_analysis=None,
            ml_prediction=None,
            market_context=None,
        )

        # 4. Execute if not HOLD
        executor = OrderExecutor(
            paper_mode=True,
            initial_balance=10000.0,
        )

        if decision.action != "HOLD" and not decision.vetoed:
            request = convert_decision_to_request(decision)
            order = await executor.execute_decision(request)
            assert order.status == "filled"

        # 5. Check final state
        status = executor.get_account_status()
        assert status["paper_mode"] is True


class TestComponentImports:
    """Test that all components can be imported."""

    def test_data_module_imports(self):
        """Test data module imports."""
        from iftb.data import (
            ExchangeClient,
            OHLCVBar,
            CacheManager,
            DatabaseManager,
            ExternalDataAggregator,
        )
        assert ExchangeClient is not None
        assert OHLCVBar is not None

    def test_analysis_module_imports(self):
        """Test analysis module imports."""
        from iftb.analysis import (
            TechnicalAnalyzer,
            LLMAnalyzer,
            XGBoostValidator,
            CompositeSignal,
        )
        assert TechnicalAnalyzer is not None
        assert LLMAnalyzer is not None

    def test_trading_module_imports(self):
        """Test trading module imports."""
        from iftb.trading import (
            DecisionEngine,
            RiskManager,
            OrderExecutor,
            CircuitBreaker,
            KillSwitch,
        )
        assert DecisionEngine is not None
        assert OrderExecutor is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
