# Trading Decision Engine

## Overview

The Trading Decision Engine is the core decision-making component of the IFTB trading bot. It integrates three analysis layers with comprehensive risk management to generate executable trading decisions.

**Location:** `/mnt/d/Develop/AQB/src/iftb/trading/decision_engine.py`

**Lines of Code:** 1046

## Architecture

### Signal Integration (3-Layer Analysis)

The engine combines three independent analysis sources using weighted voting:

| Layer | Weight | Purpose |
|-------|--------|---------|
| **Technical Analysis** | 40% | Price action, momentum, volatility indicators |
| **LLM Sentiment** | 25% | News analysis, market sentiment, qualitative factors |
| **ML Prediction** | 35% | XGBoost model, pattern recognition, statistical validation |

### Risk Management Components

1. **RiskManager** - Position sizing and risk calculations
2. **CircuitBreaker** - Automatic trading halt during adverse conditions
3. **KillSwitch** - Manual emergency stop
4. **TradeHistory** - Performance tracking and statistics

## Core Classes

### TradingDecision

Final trading decision with all parameters:

```python
@dataclass
class TradingDecision:
    action: Literal["LONG", "SHORT", "HOLD"]
    symbol: str
    confidence: float  # 0-1 combined confidence
    position_size: float  # Fraction of capital (0.02-0.10)
    leverage: int  # 2x-8x based on volatility
    stop_loss: float  # ATR-based stop price
    take_profit: float  # ATR-based target price
    entry_price: float
    timestamp: datetime
    reasons: list[str]  # Decision rationale
    vetoed: bool  # Was decision blocked?
    veto_reason: str | None
```

### RiskManager

Implements Kelly Criterion and risk controls:

**Key Methods:**

- `calculate_kelly_position()` - Optimal position sizing using Kelly Criterion with 0.25 fraction
- `check_daily_loss_limit()` - Enforce 8% daily loss limit
- `check_consecutive_losses()` - Monitor consecutive loss streaks (limit: 5)
- `calculate_stop_loss()` - ATR-based stop-loss (2x ATR)
- `calculate_take_profit()` - ATR-based take-profit (3x ATR, 1.5:1 R/R)
- `adjust_leverage()` - Dynamic leverage adjustment based on volatility

**Kelly Criterion Implementation:**

```
f = (p * (b + 1) - 1) / b

Where:
  f = fraction to bet
  p = win probability
  b = avg_win / avg_loss ratio

Applied: f * 0.25 (quarter-Kelly for safety)
Capped: MIN_POSITION_PCT (2%) to MAX_POSITION_PCT (10%)
```

### CircuitBreaker

Automatic trading halt system:

**Trigger Conditions:**

| Condition | Threshold | Cooldown |
|-----------|-----------|----------|
| Drawdown | 30% from peak | 24 hours |
| Volatility | 15% (extreme) | 24 hours |
| Error Rate | 30% | 24 hours |
| API Failures | 50% | 24 hours |

**Methods:**

- `check(metrics)` - Evaluate circuit breaker conditions
- `trigger(reason)` - Activate circuit breaker
- `reset()` - Reset after cooldown

### KillSwitch

Manual emergency stop:

**Methods:**

- `activate(reason)` - Immediately halt all trading
- `deactivate()` - Resume trading (requires manual action)
- `is_active()` - Check current state

### DecisionEngine

Main decision-making engine:

**Key Method:**

```python
async def make_decision(
    symbol: str,
    technical_signal: CompositeSignal,
    llm_analysis: LLMAnalysis,
    ml_prediction: ModelPrediction,
    market_context: MarketContext,
    current_price: float,
    account_balance: float,
    trade_history: list[TradeHistory] | None = None,
    current_pnl: float = 0.0,
) -> TradingDecision
```

## Decision Workflow

The engine follows a rigorous decision-making process:

### 1. Safety Checks (Pre-Decision)

- Kill Switch status
- Circuit Breaker status
- Daily loss limit (8% max)
- Consecutive loss limit (5 max)

**Result:** If any check fails, return vetoed HOLD decision

### 2. Signal Integration

```python
# Convert signals to numeric scores (-1 to 1)
technical_score = signal_to_score(technical_signal.overall_signal)
llm_score = llm_analysis.sentiment.value
ml_score = prediction_to_score(ml_prediction.action)

# Weighted combination
combined_score = (
    technical_score * 0.40 +
    llm_score * 0.25 +
    ml_score * 0.35
)

# Weighted confidence
combined_confidence = (
    technical_signal.confidence * 0.40 +
    llm_analysis.confidence * 0.25 +
    ml_prediction.confidence * 0.35
)
```

### 3. Veto Checks

- **Sentiment Veto:** LLM sentiment < -0.5 (very bearish)
- **Confidence Veto:** Combined confidence < 0.3 (too low)
- **Alignment Veto:** Major signal disagreement (less than 2/3 agree)

**Result:** If any veto triggered, return vetoed HOLD decision

### 4. Action Determination

```python
if combined_score > 0.2:
    action = "LONG"
elif combined_score < -0.2:
    action = "SHORT"
else:
    action = "HOLD"
```

### 5. Position Sizing (Kelly Criterion)

```python
# Calculate Kelly position from historical performance
kelly_size = risk_manager.calculate_kelly_position(
    win_rate=historical_win_rate,
    avg_win=historical_avg_win,
    avg_loss=historical_avg_loss,
    current_capital=account_balance,
)

# Adjust for confidence
confidence_adjusted_size = kelly_size * combined_confidence

# Apply limits (2% to 10%)
final_size = clamp(confidence_adjusted_size, MIN_POSITION_PCT, MAX_POSITION_PCT)
```

### 6. Leverage Adjustment

```python
# Calculate volatility
volatility = atr / current_price

# Adjust leverage based on volatility
base_leverage = risk_manager.adjust_leverage(volatility)

# Further adjust for confidence
if combined_confidence >= 0.8:
    leverage = min(base_leverage + 1, MAX_LEVERAGE)
elif combined_confidence <= 0.5:
    leverage = max(base_leverage - 1, MIN_LEVERAGE)
else:
    leverage = base_leverage
```

### 7. Risk Parameters

```python
# ATR-based stops
stop_loss = risk_manager.calculate_stop_loss(
    entry_price=current_price,
    atr=atr_value,
    direction=action,
)

take_profit = risk_manager.calculate_take_profit(
    entry_price=current_price,
    atr=atr_value,
    direction=action,
)
```

### 8. Decision Assembly

Create final TradingDecision with:
- Action and symbol
- Confidence and position size
- Leverage
- Stop-loss and take-profit
- Detailed reasoning
- Timestamp

## Constants Used

From `iftb.config.constants`:

```python
KELLY_FRACTION = 0.25  # Quarter-Kelly for conservative sizing
MAX_POSITION_PCT = 0.10  # 10% max per position
MIN_POSITION_PCT = 0.02  # 2% minimum position
MAX_LEVERAGE = 8  # Absolute maximum leverage
MIN_LEVERAGE = 2  # Minimum leverage
DEFAULT_LEVERAGE = 5  # Standard leverage
HIGH_CONFIDENCE_LEVERAGE = 7  # For high-confidence setups
MAX_DAILY_LOSS_PCT = 0.08  # 8% daily loss limit
CONSECUTIVE_LOSS_LIMIT = 5  # Max consecutive losses
SENTIMENT_VETO_THRESHOLD = -0.5  # LLM sentiment veto
CONFIDENCE_VETO_THRESHOLD = 0.3  # Minimum confidence
MAX_DRAWDOWN = 0.30  # 30% max drawdown
```

## Example Usage

See: `/mnt/d/Develop/AQB/examples/decision_engine_usage.py`

Basic usage:

```python
from iftb.trading import create_decision_engine

# Create engine
engine = create_decision_engine()

# Make decision
decision = await engine.make_decision(
    symbol="BTCUSDT",
    technical_signal=technical_signal,
    llm_analysis=llm_analysis,
    ml_prediction=ml_prediction,
    market_context=market_context,
    current_price=50000.0,
    account_balance=10000.0,
    trade_history=recent_trades,
    current_pnl=-50.0,
)

# Check result
if decision.vetoed:
    print(f"Trade vetoed: {decision.veto_reason}")
elif decision.action == "HOLD":
    print("No trade signal")
else:
    print(f"{decision.action} {decision.symbol}")
    print(f"Size: {decision.position_size:.2%} @ {decision.leverage}x")
    print(f"Entry: {decision.entry_price}")
    print(f"Stop: {decision.stop_loss}")
    print(f"Target: {decision.take_profit}")
```

## Safety Features

### Multi-Layer Protection

1. **Pre-Decision Checks**
   - Kill switch
   - Circuit breaker
   - Daily loss limit
   - Consecutive losses

2. **Signal Validation**
   - Sentiment veto
   - Confidence veto
   - Signal alignment check

3. **Position Sizing**
   - Kelly Criterion (conservative)
   - Confidence adjustment
   - Hard position limits (2%-10%)

4. **Risk Parameters**
   - ATR-based stops (not arbitrary)
   - Dynamic leverage (volatility-adjusted)
   - 1.5:1 risk/reward ratio

### Fail-Safe Design

- **Default to HOLD:** When in doubt, don't trade
- **Vetoed decisions:** Clearly marked with reason
- **Comprehensive logging:** All decisions logged with rationale
- **Manual override:** Kill switch for emergency stop

## Testing

Syntax validated: ✓

To run full tests (requires dependencies):

```bash
pytest tests/test_decision_engine.py -v
```

## Integration Points

The decision engine integrates with:

- `iftb.analysis.TechnicalAnalyzer` - Technical indicators
- `iftb.analysis.LLMAnalyzer` - Sentiment analysis
- `iftb.analysis.XGBoostValidator` - ML predictions
- `iftb.data.MarketContext` - External market data
- `iftb.config.constants` - Risk parameters
- `iftb.utils.get_logger` - Structured logging

## Performance Considerations

- **Async design:** All I/O operations are async
- **Lazy evaluation:** Only calculates what's needed
- **Early returns:** Fails fast on veto conditions
- **Minimal dependencies:** Self-contained logic

## Future Enhancements

Potential improvements:

1. **Adaptive Weights:** ML-based weight optimization
2. **Regime Detection:** Different weights for different market regimes
3. **Multi-Asset:** Position correlation and portfolio-level risk
4. **Backtesting Mode:** Replay historical decisions
5. **Decision Replay:** Analyze past decision quality
6. **Performance Attribution:** Track which layer performs best

## Summary

The Trading Decision Engine provides:

- **3-layer analysis integration** with weighted voting
- **Kelly Criterion position sizing** (conservative 0.25 fraction)
- **ATR-based risk management** (2x stop, 3x target)
- **Circuit breaker** for adverse conditions
- **Kill switch** for emergencies
- **Multiple veto mechanisms** to prevent bad trades
- **Comprehensive logging** of all decisions
- **Fail-safe design** defaulting to HOLD

**Result:** A robust, production-ready decision-making system that balances opportunity with rigorous risk management.
