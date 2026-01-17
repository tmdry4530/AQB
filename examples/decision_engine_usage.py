"""
Example Usage of IFTB Trading Decision Engine

This example demonstrates how to use the DecisionEngine to make trading decisions
by integrating technical analysis, LLM analysis, and ML predictions with
comprehensive risk management.
"""

import asyncio
from datetime import datetime, timezone

import pandas as pd

from iftb.analysis import (
    CompositeSignal,
    IndicatorResult,
    LLMAnalysis,
    ModelPrediction,
    SentimentScore,
    TechnicalAnalyzer,
)
from iftb.data import MarketContext
from iftb.trading import TradeHistory, create_decision_engine


async def main():
    """Demonstrate complete decision-making workflow."""

    print("="*80)
    print("IFTB Trading Decision Engine - Example Usage")
    print("="*80)
    print()

    # =========================================================================
    # 1. Create Decision Engine
    # =========================================================================

    print("1. Creating Decision Engine...")
    engine = create_decision_engine()
    print(f"   Engine created: {engine}")
    print()

    # =========================================================================
    # 2. Prepare Mock Data (In production, this comes from real sources)
    # =========================================================================

    print("2. Preparing analysis inputs...")

    # Create mock OHLCV data for technical analysis
    ohlcv_data = pd.DataFrame({
        "timestamp": pd.date_range(end=datetime.now(), periods=100, freq="1h"),
        "open": [50000 + i * 10 for i in range(100)],
        "high": [50100 + i * 10 for i in range(100)],
        "low": [49900 + i * 10 for i in range(100)],
        "close": [50050 + i * 10 for i in range(100)],
        "volume": [1000000] * 100,
    })

    # Technical Analysis Signal
    technical_analyzer = TechnicalAnalyzer(ohlcv_data)
    technical_signal = technical_analyzer.get_composite_signal()
    print(f"   Technical Signal: {technical_signal.overall_signal}")
    print(f"   Technical Confidence: {technical_signal.confidence:.2f}")

    # LLM Analysis (mock data)
    llm_analysis = LLMAnalysis(
        sentiment=SentimentScore.BULLISH,
        confidence=0.75,
        reasoning="Strong positive sentiment from recent news...",
        key_factors=["Institutional adoption", "Technical breakout"],
        risks=["Regulatory uncertainty"],
        recommended_action="LONG",
        timestamp=datetime.now(timezone.utc),
        model="claude-3-7-sonnet-20250219",
    )
    print(f"   LLM Sentiment: {llm_analysis.sentiment}")
    print(f"   LLM Confidence: {llm_analysis.confidence:.2f}")

    # ML Model Prediction (mock data)
    ml_prediction = ModelPrediction(
        action="LONG",
        confidence=0.82,
        probability_long=0.82,
        probability_short=0.10,
        probability_hold=0.08,
        feature_importance={
            "rsi": 0.25,
            "macd": 0.20,
            "sentiment": 0.30,
            "volume": 0.15,
            "volatility": 0.10,
        },
        model_version="v1.0.0",
        prediction_time=datetime.now(timezone.utc),
    )
    print(f"   ML Prediction: {ml_prediction.action}")
    print(f"   ML Confidence: {ml_prediction.confidence:.2f}")

    # Market Context (mock data)
    market_context = MarketContext(
        fear_greed=None,
        funding=None,
        open_interest=None,
        long_short=None,
        fetch_time=datetime.now(timezone.utc),
        errors=[],
    )
    print(f"   Market Context: OK")
    print()

    # =========================================================================
    # 3. Make Trading Decision
    # =========================================================================

    print("3. Making trading decision...")

    symbol = "BTCUSDT"
    current_price = 51000.0
    account_balance = 10000.0
    current_pnl = -50.0  # Small loss today

    # Create some mock trade history
    trade_history = [
        TradeHistory(
            symbol="BTCUSDT",
            action="LONG",
            entry_price=50000.0,
            exit_price=50500.0,
            position_size=0.05,
            leverage=5,
            pnl=125.0,
            pnl_pct=2.5,
            entry_time=datetime.now(timezone.utc),
            exit_time=datetime.now(timezone.utc),
            win=True,
        ),
        TradeHistory(
            symbol="BTCUSDT",
            action="SHORT",
            entry_price=50500.0,
            exit_price=50600.0,
            position_size=0.05,
            leverage=5,
            pnl=-25.0,
            pnl_pct=-0.5,
            entry_time=datetime.now(timezone.utc),
            exit_time=datetime.now(timezone.utc),
            win=False,
        ),
    ]

    decision = await engine.make_decision(
        symbol=symbol,
        technical_signal=technical_signal,
        llm_analysis=llm_analysis,
        ml_prediction=ml_prediction,
        market_context=market_context,
        current_price=current_price,
        account_balance=account_balance,
        trade_history=trade_history,
        current_pnl=current_pnl,
    )

    print()
    print("="*80)
    print("TRADING DECISION")
    print("="*80)
    print(f"Symbol:         {decision.symbol}")
    print(f"Action:         {decision.action}")
    print(f"Confidence:     {decision.confidence:.2%}")
    print(f"Position Size:  {decision.position_size:.2%} of capital")
    print(f"Leverage:       {decision.leverage}x")
    print(f"Entry Price:    ${decision.entry_price:.2f}")
    print(f"Stop Loss:      ${decision.stop_loss:.2f}")
    print(f"Take Profit:    ${decision.take_profit:.2f}")
    print(f"Vetoed:         {decision.vetoed}")
    if decision.veto_reason:
        print(f"Veto Reason:    {decision.veto_reason}")
    print()
    print("Reasons:")
    for i, reason in enumerate(decision.reasons, 1):
        print(f"  {i}. {reason}")
    print()

    # =========================================================================
    # 4. Calculate Position Details
    # =========================================================================

    if decision.action != "HOLD" and not decision.vetoed:
        print("="*80)
        print("POSITION DETAILS")
        print("="*80)

        position_value = account_balance * decision.position_size
        position_with_leverage = position_value * decision.leverage
        quantity = position_with_leverage / decision.entry_price

        stop_loss_distance = abs(decision.entry_price - decision.stop_loss)
        stop_loss_pct = (stop_loss_distance / decision.entry_price) * 100

        take_profit_distance = abs(decision.take_profit - decision.entry_price)
        take_profit_pct = (take_profit_distance / decision.entry_price) * 100

        risk_amount = position_value * (stop_loss_distance / decision.entry_price) * decision.leverage
        reward_amount = position_value * (take_profit_distance / decision.entry_price) * decision.leverage
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0

        print(f"Capital Allocation:     ${position_value:.2f} ({decision.position_size:.2%})")
        print(f"Leverage:               {decision.leverage}x")
        print(f"Position Size:          ${position_with_leverage:.2f}")
        print(f"Quantity:               {quantity:.6f} {symbol[:3]}")
        print()
        print(f"Stop Loss Distance:     {stop_loss_pct:.2f}%")
        print(f"Take Profit Distance:   {take_profit_pct:.2f}%")
        print(f"Risk/Reward Ratio:      1:{risk_reward_ratio:.2f}")
        print()
        print(f"Risk Amount:            ${risk_amount:.2f}")
        print(f"Potential Reward:       ${reward_amount:.2f}")
        print()

    # =========================================================================
    # 5. Demonstrate Risk Controls
    # =========================================================================

    print("="*80)
    print("RISK CONTROLS STATUS")
    print("="*80)

    # Circuit Breaker
    print(f"Circuit Breaker:        {'ACTIVE' if engine.circuit_breaker.is_triggered else 'OK'}")
    if engine.circuit_breaker.is_triggered:
        print(f"  Reason:               {engine.circuit_breaker.trigger_reason}")
        print(f"  Triggered At:         {engine.circuit_breaker.trigger_time}")

    # Kill Switch
    print(f"Kill Switch:            {'ACTIVE' if engine.kill_switch.is_active() else 'OK'}")
    if engine.kill_switch.is_active():
        print(f"  Reason:               {engine.kill_switch.activation_reason}")
        print(f"  Activated At:         {engine.kill_switch.activation_time}")

    # Daily Loss Check
    daily_loss_ok = engine.risk_manager.check_daily_loss_limit(current_pnl, account_balance)
    print(f"Daily Loss Limit:       {'OK' if daily_loss_ok else 'EXCEEDED'}")
    print(f"  Current PnL:          ${current_pnl:.2f}")
    print(f"  Loss %:               {abs(current_pnl)/account_balance:.2%}")

    # Consecutive Losses
    consecutive_ok = engine.risk_manager.check_consecutive_losses(trade_history)
    consecutive_losses = sum(1 for t in trade_history if not t.win)
    print(f"Consecutive Losses:     {'OK' if consecutive_ok else 'EXCEEDED'}")
    print(f"  Current Streak:       {consecutive_losses}")

    print()

    # =========================================================================
    # 6. Demonstrate Safety Features
    # =========================================================================

    print("="*80)
    print("SAFETY FEATURE DEMONSTRATION")
    print("="*80)
    print()

    # Test Kill Switch
    print("Testing Kill Switch activation...")
    engine.kill_switch.activate("Manual test - demonstrating emergency stop")
    print(f"  Kill Switch Active: {engine.kill_switch.is_active()}")

    # Try to make a decision with kill switch active
    vetoed_decision = await engine.make_decision(
        symbol=symbol,
        technical_signal=technical_signal,
        llm_analysis=llm_analysis,
        ml_prediction=ml_prediction,
        market_context=market_context,
        current_price=current_price,
        account_balance=account_balance,
    )
    print(f"  Decision with Kill Switch: {vetoed_decision.action}")
    print(f"  Vetoed: {vetoed_decision.vetoed}")
    print(f"  Reason: {vetoed_decision.veto_reason}")
    print()

    # Deactivate kill switch
    engine.kill_switch.deactivate()
    print("Kill Switch deactivated.")
    print()

    # Test Circuit Breaker
    print("Testing Circuit Breaker with high drawdown...")
    bad_metrics = {
        "drawdown": 0.35,  # 35% drawdown (exceeds 30% limit)
        "volatility": 0.05,
        "error_rate": 0.1,
        "api_failure_rate": 0.2,
    }
    should_halt, reason = engine.circuit_breaker.check(bad_metrics)
    print(f"  Should Halt: {should_halt}")
    print(f"  Reason: {reason}")
    print()

    print("="*80)
    print("EXAMPLE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
