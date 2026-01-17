# Trading Decision Engine - Quick Reference

## Imports

```python
from iftb.trading import (
    create_decision_engine,
    TradingDecision,
    TradeHistory,
    RiskManager,
    CircuitBreaker,
    KillSwitch,
)
```

## Create Engine

```python
# Factory function (recommended)
engine = create_decision_engine()

# Manual instantiation
risk_manager = RiskManager()
circuit_breaker = CircuitBreaker()
kill_switch = KillSwitch()
engine = DecisionEngine(risk_manager, circuit_breaker, kill_switch)
```

## Make Trading Decision

```python
decision = await engine.make_decision(
    symbol="BTCUSDT",
    technical_signal=technical_signal,  # CompositeSignal
    llm_analysis=llm_analysis,          # LLMAnalysis
    ml_prediction=ml_prediction,        # ModelPrediction
    market_context=market_context,      # MarketContext
    current_price=50000.0,
    account_balance=10000.0,
    trade_history=recent_trades,        # Optional: list[TradeHistory]
    current_pnl=-50.0,                  # Optional: today's PnL
)
```

## Check Decision

```python
if decision.vetoed:
    print(f"Trade blocked: {decision.veto_reason}")
elif decision.action == "HOLD":
    print("No trade signal")
else:
    print(f"{decision.action} {decision.symbol}")
    print(f"Position: {decision.position_size:.2%} @ {decision.leverage}x")
    print(f"Entry: ${decision.entry_price:.2f}")
    print(f"Stop Loss: ${decision.stop_loss:.2f}")
    print(f"Take Profit: ${decision.take_profit:.2f}")
    print(f"Confidence: {decision.confidence:.2%}")
```

## TradingDecision Fields

```python
decision.action           # "LONG", "SHORT", or "HOLD"
decision.symbol           # Trading pair (e.g., "BTCUSDT")
decision.confidence       # 0-1 combined confidence
decision.position_size    # Fraction of capital (0.02-0.10)
decision.leverage         # 2-8x
decision.stop_loss        # Stop-loss price
decision.take_profit      # Take-profit price
decision.entry_price      # Expected entry price
decision.timestamp        # Decision timestamp
decision.reasons          # List of decision reasons
decision.vetoed           # True if decision was blocked
decision.veto_reason      # Reason for veto (if applicable)
```

## Risk Manager Methods

```python
# Kelly Criterion position sizing
position_size = risk_manager.calculate_kelly_position(
    win_rate=0.60,
    avg_win=100.0,
    avg_loss=80.0,
    current_capital=10000.0,
)

# Check daily loss limit (8% max)
can_trade = risk_manager.check_daily_loss_limit(
    current_pnl=-500.0,
    capital=10000.0,
)

# Check consecutive losses (5 max)
can_trade = risk_manager.check_consecutive_losses(trade_history)

# Calculate stop-loss (2x ATR)
stop_loss = risk_manager.calculate_stop_loss(
    entry_price=50000.0,
    atr=1000.0,
    direction="LONG",
)

# Calculate take-profit (3x ATR)
take_profit = risk_manager.calculate_take_profit(
    entry_price=50000.0,
    atr=1000.0,
    direction="LONG",
)

# Adjust leverage based on volatility
leverage = risk_manager.adjust_leverage(volatility=0.05)
```

## Circuit Breaker

```python
# Check circuit breaker conditions
metrics = {
    "drawdown": 0.25,         # 25% drawdown
    "volatility": 0.08,       # 8% volatility
    "error_rate": 0.15,       # 15% error rate
    "api_failure_rate": 0.30, # 30% API failures
}
should_halt, reason = circuit_breaker.check(metrics)

if should_halt:
    print(f"Trading halted: {reason}")

# Manual trigger
circuit_breaker.trigger("Manual halt due to market conditions")

# Reset after cooldown
circuit_breaker.reset()

# Check status
if circuit_breaker.is_triggered:
    print(f"Circuit breaker active: {circuit_breaker.trigger_reason}")
```

## Kill Switch

```python
# Activate emergency stop
kill_switch.activate("Emergency stop - critical issue detected")

# Check status
if kill_switch.is_active():
    print("Kill switch is active - trading halted")

# Deactivate (requires manual action)
kill_switch.deactivate()
```

## Signal Weights

| Analysis Layer | Weight | Purpose |
|---------------|--------|---------|
| Technical | 40% | Price action, momentum, volatility |
| LLM | 25% | News sentiment, qualitative analysis |
| ML | 35% | Pattern recognition, statistical validation |

## Risk Limits

| Parameter | Value | Description |
|-----------|-------|-------------|
| Kelly Fraction | 0.25 | Quarter-Kelly (conservative) |
| Max Position | 10% | Maximum position size |
| Min Position | 2% | Minimum position size |
| Max Leverage | 8x | Absolute maximum |
| Min Leverage | 2x | Minimum leverage |
| Daily Loss Limit | 8% | Max daily loss before halt |
| Consecutive Losses | 5 | Max losses before cooldown |
| Max Drawdown | 30% | Circuit breaker threshold |

## Veto Thresholds

| Check | Threshold | Action |
|-------|-----------|--------|
| Sentiment | < -0.5 | Block trade |
| Confidence | < 0.3 | Block trade |
| Signal Alignment | < 2/3 agree | Block trade |
| Daily Loss | >= 8% | Block all trades |
| Consecutive Losses | >= 5 | Block all trades |

## Decision Logic

### Signal to Action

```python
combined_score = (
    technical * 0.40 +
    llm * 0.25 +
    ml * 0.35
)

if combined_score > 0.2:
    action = "LONG"
elif combined_score < -0.2:
    action = "SHORT"
else:
    action = "HOLD"
```

### Position Sizing

```python
# 1. Calculate Kelly position
kelly_size = calculate_kelly(win_rate, avg_win, avg_loss, capital)

# 2. Adjust for confidence
adjusted_size = kelly_size * confidence

# 3. Apply limits
final_size = clamp(adjusted_size, 0.02, 0.10)  # 2% to 10%
```

### Leverage Adjustment

```python
# Base leverage from volatility
volatility = atr / price
base_leverage = adjust_leverage(volatility)

# Adjust for confidence
if confidence >= 0.8:
    leverage = min(base_leverage + 1, 8)
elif confidence <= 0.5:
    leverage = max(base_leverage - 1, 2)
else:
    leverage = base_leverage
```

### Stop Loss & Take Profit

```python
# Stop Loss: 2x ATR
stop_loss = entry - (2 * atr)  # for LONG
stop_loss = entry + (2 * atr)  # for SHORT

# Take Profit: 3x ATR (1.5:1 R/R)
take_profit = entry + (3 * atr)  # for LONG
take_profit = entry - (3 * atr)  # for SHORT
```

## Trade History

```python
from datetime import datetime, timezone

# Record a completed trade
trade = TradeHistory(
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
)

# Build history list (most recent first)
trade_history = [trade1, trade2, trade3, ...]
```

## Position Calculation

```python
# Calculate position details from decision
position_value = account_balance * decision.position_size
position_with_leverage = position_value * decision.leverage
quantity = position_with_leverage / decision.entry_price

# Risk/Reward
stop_distance = abs(decision.entry_price - decision.stop_loss)
profit_distance = abs(decision.take_profit - decision.entry_price)
risk_reward_ratio = profit_distance / stop_distance

# Risk amount
risk_amount = position_value * (stop_distance / decision.entry_price) * decision.leverage
```

## Error Handling

```python
try:
    decision = await engine.make_decision(...)

    if decision.vetoed:
        logger.warning(f"Trade vetoed: {decision.veto_reason}")
    elif decision.action != "HOLD":
        # Execute trade
        pass

except Exception as e:
    logger.error(f"Decision engine error: {e}")
    # Activate kill switch if critical
    engine.kill_switch.activate("Critical error in decision engine")
```

## Logging

All decisions are automatically logged:

```python
logger.info("decision_created", decision=decision.to_dict())
logger.debug("kelly_position_calculated", win_rate=0.6, final_size=0.05)
logger.error("daily_loss_limit_exceeded", loss_pct=0.085)
logger.critical("circuit_breaker_triggered", reason="Excessive drawdown")
```

## Complete Example

```python
from iftb.trading import create_decision_engine

async def make_trade_decision():
    # Create engine
    engine = create_decision_engine()

    # Gather inputs (from your analysis systems)
    technical_signal = await get_technical_analysis("BTCUSDT")
    llm_analysis = await get_llm_analysis("BTCUSDT", news)
    ml_prediction = await get_ml_prediction("BTCUSDT")
    market_context = await get_market_context("BTC")

    # Get account info
    balance = await exchange.get_balance()
    current_price = await exchange.get_ticker("BTCUSDT")

    # Make decision
    decision = await engine.make_decision(
        symbol="BTCUSDT",
        technical_signal=technical_signal,
        llm_analysis=llm_analysis,
        ml_prediction=ml_prediction,
        market_context=market_context,
        current_price=current_price,
        account_balance=balance,
    )

    # Execute if not vetoed
    if not decision.vetoed and decision.action != "HOLD":
        await execute_trade(decision)

    return decision
```

## See Also

- Full documentation: `/mnt/d/Develop/AQB/docs/decision_engine.md`
- Example usage: `/mnt/d/Develop/AQB/examples/decision_engine_usage.py`
- Source code: `/mnt/d/Develop/AQB/src/iftb/trading/decision_engine.py`
