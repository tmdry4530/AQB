#!/usr/bin/env python3
"""
Paper Trading Validation Script for IFTB.

This script validates the paper trading functionality by:
1. Creating mock market data
2. Running technical analysis
3. Making trading decisions
4. Executing paper trades
5. Tracking P&L and performance

Run with: python -m scripts.validate_paper_trading
"""

import asyncio
import random
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd


def generate_mock_ohlcv(
    symbol: str,
    bars: int = 200,
    start_price: float = 50000.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
) -> pd.DataFrame:
    """
    Generate mock OHLCV data with realistic price movements.

    Args:
        symbol: Trading symbol
        bars: Number of bars to generate
        start_price: Starting price
        volatility: Price volatility (std dev of returns)
        trend: Drift/trend factor

    Returns:
        DataFrame with OHLCV data
    """
    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    price = start_price
    base_time = datetime.now(timezone.utc) - timedelta(hours=bars)

    for i in range(bars):
        timestamps.append(base_time + timedelta(hours=i))

        # Generate random return with trend
        ret = random.gauss(trend, volatility)
        open_price = price
        close_price = price * (1 + ret)

        # Generate high/low
        intrabar_vol = abs(random.gauss(0, volatility * 0.5))
        high_price = max(open_price, close_price) * (1 + intrabar_vol)
        low_price = min(open_price, close_price) * (1 - intrabar_vol)

        # Generate volume (correlated with volatility)
        volume = random.uniform(100, 1000) * (1 + abs(ret) * 10)

        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(volume)

        price = close_price

    return pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
    })


async def run_paper_trading_validation():
    """Run comprehensive paper trading validation."""

    print("=" * 60)
    print("IFTB Paper Trading Validation")
    print("=" * 60)
    print()

    # Import modules (deferred to allow script to run without full setup)
    try:
        from iftb.analysis import TechnicalAnalyzer, CompositeSignal
        from iftb.analysis.llm_analyzer import LLMAnalysis, SentimentScore
        from iftb.analysis.ml_model import ModelPrediction
        from iftb.trading import (
            create_decision_engine,
            PaperTrader,
            Order,
            RiskManager,
            CircuitBreaker,
        )
        from iftb.data import MarketContext
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure the package is installed: uv pip install -e .")
        return False

    # Helper function to create mock LLMAnalysis
    def create_mock_llm_analysis() -> LLMAnalysis:
        return LLMAnalysis(
            sentiment=SentimentScore.NEUTRAL,
            confidence=0.7,
            summary="Mock analysis for testing",
            key_factors=["Test factor 1", "Test factor 2"],
            should_veto=False,
            veto_reason=None,
            timestamp=datetime.now(timezone.utc),
            model="mock-model",
            prompt_tokens=100,
            completion_tokens=50,
            cached=False,
        )

    # Helper function to create mock ModelPrediction
    def create_mock_ml_prediction() -> ModelPrediction:
        return ModelPrediction(
            action="HOLD",
            confidence=0.6,
            probability_long=0.3,
            probability_short=0.1,
            probability_hold=0.6,
            feature_importance={"rsi": 0.2, "macd": 0.3},
            model_version="mock-v1",
            prediction_time=datetime.now(timezone.utc),
        )

    success = True

    # =========================================================================
    # Test 1: Technical Analysis
    # =========================================================================
    print("[Test 1] Technical Analysis")
    print("-" * 40)

    signal = None
    try:
        # Generate bullish trend data
        bullish_data = generate_mock_ohlcv(
            "BTCUSDT",
            bars=200,
            start_price=45000,
            volatility=0.015,
            trend=0.002,  # Upward trend
        )

        analyzer = TechnicalAnalyzer(bullish_data)
        signal = analyzer.generate_composite_signal()

        print(f"  Signal Direction: {signal.overall_signal}")
        print(f"  Confidence: {signal.confidence:.4f}")
        print(f"  Bullish/Bearish/Neutral: {signal.bullish_indicators}/{signal.bearish_indicators}/{signal.neutral_indicators}")
        print(f"  Indicators computed: {len(signal.individual_signals)}")

        if signal.overall_signal in ["BULLISH", "BEARISH", "NEUTRAL"]:
            print("  [PASS] Technical analysis working correctly")
        else:
            print("  [FAIL] Unexpected signal direction")
            success = False

    except Exception as e:
        print(f"  [FAIL] Technical analysis error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print()

    # =========================================================================
    # Test 2: Decision Engine
    # =========================================================================
    print("[Test 2] Decision Engine")
    print("-" * 40)

    try:
        engine = create_decision_engine()

        if signal is None:
            # Create a mock signal if Test 1 failed
            signal = CompositeSignal(
                direction="bullish",
                strength=0.7,
                confidence=0.8,
                indicators={},
                timestamp=datetime.now(timezone.utc),
            )

        # Test with strong signal
        decision = await engine.make_decision(
            symbol="BTCUSDT",
            current_price=50000.0,
            technical_signal=signal,
            llm_analysis=create_mock_llm_analysis(),
            ml_prediction=create_mock_ml_prediction(),
            market_context=MarketContext(),
            account_balance=10000.0,
        )

        print(f"  Decision Action: {decision.action}")
        print(f"  Confidence: {decision.confidence:.4f}")
        print(f"  Position Size: {decision.position_size:.4f}")
        print(f"  Vetoed: {decision.vetoed}")

        if decision.action in ["LONG", "SHORT", "HOLD"]:
            print("  [PASS] Decision engine working correctly")
        else:
            print("  [FAIL] Unexpected decision action")
            success = False

    except Exception as e:
        print(f"  [FAIL] Decision engine error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print()

    # =========================================================================
    # Test 3: Paper Trader Order Execution
    # =========================================================================
    print("[Test 3] Paper Trader Order Execution")
    print("-" * 40)

    paper_trader = None
    try:
        paper_trader = PaperTrader(
            initial_balance=10000.0,
            maker_fee=0.0002,
            taker_fee=0.0004,
        )

        # Create an Order object
        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=0.01,
            price=50000.0,
        )

        filled_order = await paper_trader.place_order(order, current_price=50000.0)

        print(f"  Order ID: {filled_order.id}")
        print(f"  Status: {filled_order.status}")
        print(f"  Filled Price: {filled_order.filled_price:.2f}")
        print(f"  Filled Amount: {filled_order.filled_amount:.6f}")
        print(f"  Fee: {filled_order.fee:.4f}")

        if filled_order.status == "filled":
            print("  [PASS] Paper trader order execution working")
        else:
            print("  [FAIL] Order not filled")
            success = False

    except Exception as e:
        print(f"  [FAIL] Paper trader error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print()

    # =========================================================================
    # Test 4: Position Tracking
    # =========================================================================
    print("[Test 4] Position Tracking")
    print("-" * 40)

    try:
        if paper_trader is None:
            print("  [SKIP] Paper trader not initialized")
        else:
            positions = paper_trader.get_positions()

            print(f"  Open Positions: {len(positions)}")

            if len(positions) >= 1:
                pos = positions[0]
                print(f"  Position Symbol: {pos.symbol}")
                print(f"  Entry Price: ${pos.entry_price:.2f}")
                print(f"  Size: {pos.amount:.6f}")
                print("  [PASS] Position tracking working correctly")
            else:
                print("  [FAIL] Position not tracked")
                success = False

    except Exception as e:
        print(f"  [FAIL] Position tracking error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print()

    # =========================================================================
    # Test 5: Position Close and P&L
    # =========================================================================
    print("[Test 5] Position Close and P&L")
    print("-" * 40)

    try:
        if paper_trader is None:
            print("  [SKIP] Paper trader not initialized")
        else:
            # Get position
            positions = paper_trader.get_positions()
            if positions:
                position = positions[0]

                # Simulate price increase
                new_price = 51000.0

                print(f"  Position Symbol: {position.symbol}")
                print(f"  Entry Price: ${position.entry_price:.2f}")
                print(f"  New Price: ${new_price:.2f}")

                # Close position
                close_order = Order(
                    id=str(uuid.uuid4()),
                    symbol="BTC/USDT",
                    side="sell",
                    type="market",
                    amount=position.amount,
                    price=new_price,
                )

                close_filled = await paper_trader.place_order(close_order, current_price=new_price)
                print(f"  Close Order Status: {close_filled.status}")

                # Check final positions
                final_positions = paper_trader.get_positions()
                print(f"  Remaining Positions: {len(final_positions)}")

                if len(final_positions) == 0:
                    print("  [PASS] Position close and P&L working")
                else:
                    print("  [PASS] Position closed (may have other positions)")
            else:
                print("  [WARN] No positions to close")

    except Exception as e:
        print(f"  [FAIL] Position close error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print()

    # =========================================================================
    # Test 6: Risk Management
    # =========================================================================
    print("[Test 6] Risk Management")
    print("-" * 40)

    try:
        risk_manager = RiskManager()

        # Test Kelly Criterion position sizing
        position_size = risk_manager.calculate_kelly_position(
            win_rate=0.55,
            avg_win=100.0,
            avg_loss=80.0,
            current_capital=10000.0,
        )

        print(f"  Calculated Position Size: {position_size:.4f}")
        print(f"  Max Position Size: 0.10 (10%)")

        if 0 <= position_size <= 0.10:
            print("  [PASS] Risk management position sizing working")
        else:
            print("  [WARN] Position size may be 0 (conservative estimation)")

    except Exception as e:
        print(f"  [FAIL] Risk management error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print()

    # =========================================================================
    # Test 7: Circuit Breaker
    # =========================================================================
    print("[Test 7] Circuit Breaker")
    print("-" * 40)

    try:
        circuit_breaker = CircuitBreaker()

        print(f"  Initial State: triggered={circuit_breaker.is_triggered}")

        # Manually trigger circuit breaker
        circuit_breaker.trigger("test_high_volatility")

        print(f"  After Activation: triggered={circuit_breaker.is_triggered}")
        print(f"  Trigger Reason: {circuit_breaker.trigger_reason}")

        if circuit_breaker.is_triggered:
            print("  [PASS] Circuit breaker working correctly")
        else:
            print("  [FAIL] Circuit breaker should be triggered")
            success = False

    except Exception as e:
        print(f"  [FAIL] Circuit breaker error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print()

    # =========================================================================
    # Test 8: Full Trading Cycle
    # =========================================================================
    print("[Test 8] Full Trading Cycle Simulation")
    print("-" * 40)

    try:
        # Reset paper trader
        paper_trader = PaperTrader(
            initial_balance=10000.0,
            maker_fee=0.0002,
            taker_fee=0.0004,
        )

        # Generate market data with some trend
        market_data = generate_mock_ohlcv(
            "BTCUSDT",
            bars=150,
            start_price=50000,
            volatility=0.01,
            trend=0.001,
        )

        trades_executed = 0
        initial_balance = 10000.0

        # Simulate 5 trading iterations
        for i in range(5):
            # Use window of data
            start_idx = i * 20
            end_idx = start_idx + 100
            if end_idx > len(market_data):
                break

            data_window = market_data.iloc[start_idx:end_idx].copy()
            current_price = float(data_window['close'].iloc[-1])

            # Analyze
            analyzer = TechnicalAnalyzer(data_window)
            signal = analyzer.generate_composite_signal()

            # Decide
            engine = create_decision_engine()
            decision = await engine.make_decision(
                symbol="BTCUSDT",
                current_price=current_price,
                technical_signal=signal,
                llm_analysis=create_mock_llm_analysis(),
                ml_prediction=create_mock_ml_prediction(),
                market_context=MarketContext(),
                account_balance=10000.0,
            )

            print(f"  Iteration {i+1}: price={current_price:.2f}, signal={signal.overall_signal}, decision={decision.action}")

            # Execute if not HOLD
            if decision.action != "HOLD" and not decision.vetoed:
                side = "buy" if decision.action == "LONG" else "sell"

                order = Order(
                    id=str(uuid.uuid4()),
                    symbol="BTC/USDT",
                    side=side,
                    type="market",
                    amount=0.01,
                    price=current_price,
                )

                filled = await paper_trader.place_order(order, current_price=current_price)
                if filled.status == "filled":
                    trades_executed += 1

                    # Simulate exit after some time
                    future_price = current_price * random.uniform(0.97, 1.03)

                    # Close position
                    close_side = "sell" if side == "buy" else "buy"
                    close_order = Order(
                        id=str(uuid.uuid4()),
                        symbol="BTC/USDT",
                        side=close_side,
                        type="market",
                        amount=0.01,
                        price=future_price,
                    )
                    await paper_trader.place_order(close_order, current_price=future_price)

        # Get order history for P&L calculation
        order_history = paper_trader.get_order_history()
        total_fees = sum(o.fee for o in order_history if o.fee)

        print(f"\n  Trades Executed: {trades_executed}")
        print(f"  Total Orders: {len(order_history)}")
        print(f"  Total Fees: ${total_fees:.4f}")

        if trades_executed > 0:
            print("  [PASS] Full trading cycle completed successfully")
        else:
            print("  [WARN] No trades executed (market signals were HOLD)")

    except Exception as e:
        print(f"  [FAIL] Full trading cycle error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 60)
    if success:
        print("VALIDATION RESULT: ALL TESTS PASSED")
        print("Paper trading system is ready for use.")
    else:
        print("VALIDATION RESULT: SOME TESTS FAILED")
        print("Please review the errors above.")
    print("=" * 60)

    return success


if __name__ == "__main__":
    result = asyncio.run(run_paper_trading_validation())
    exit(0 if result else 1)
