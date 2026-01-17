# Order Execution Module

The order execution module (`iftb.trading.executor`) provides comprehensive order execution and position management with support for both paper trading and live trading modes.

## Overview

### Key Features

- **Paper Trading**: Realistic simulation with slippage, fees, and position tracking
- **Live Trading**: CCXT integration for real exchange execution
- **Order Types**: Market, limit, stop-loss, and take-profit orders
- **Position Management**: Open, close, and modify positions
- **Risk Controls**: Order validation, position size limits, leverage limits
- **Retry Logic**: Automatic retry with exponential backoff for network errors
- **Detailed Logging**: Comprehensive execution tracking

## Core Components

### Data Classes

#### Order

Represents an order with full lifecycle tracking.

```python
@dataclass
class Order:
    id: str                          # Unique order identifier
    symbol: str                       # Trading pair (e.g., "BTC/USDT")
    side: Literal["buy", "sell"]     # Order side
    type: Literal["market", "limit", "stop_loss", "take_profit"]
    amount: float                     # Order quantity in base currency
    price: Optional[float]            # Limit price (for limit orders)
    stop_price: Optional[float]       # Trigger price (for stop orders)
    status: Literal["pending", "filled", "cancelled", "failed"]
    filled_price: Optional[float]     # Actual execution price
    filled_amount: Optional[float]    # Actual filled quantity
    fee: Optional[float]              # Trading fee paid
    timestamp: datetime               # Order creation time
    exchange_order_id: Optional[str]  # Exchange-specific ID
    error_message: Optional[str]      # Error if failed
```

#### PositionState

Represents current position state with PnL tracking.

```python
@dataclass
class PositionState:
    symbol: str                       # Trading pair
    side: Literal["long", "short"]   # Position side
    entry_price: float                # Average entry price
    current_price: float              # Current market price
    amount: float                     # Position size
    leverage: int                     # Leverage multiplier
    unrealized_pnl: float             # Unrealized profit/loss
    liquidation_price: float          # Estimated liquidation price
    stop_loss: Optional[float]        # Stop-loss price
    take_profit: Optional[float]      # Take-profit price
    opened_at: datetime               # Position open time
```

#### ExecutionRequest

Request object for executing trades.

```python
@dataclass
class ExecutionRequest:
    action: Literal["long", "short", "close"]
    symbol: str
    amount: float
    entry_price: Optional[float] = None      # Uses market if None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: int = 1
    order_type: Literal["market", "limit"] = "market"
    reason: Optional[str] = None
```

## Classes

### PaperTrader

Simulated trading for backtesting and testing.

```python
from iftb.trading.executor import PaperTrader, Order
import uuid

# Initialize paper trader
trader = PaperTrader(
    initial_balance=10000.0,
    maker_fee=0.0002,  # 0.02%
    taker_fee=0.0004,  # 0.04%
)

# Place market order
order = Order(
    id=str(uuid.uuid4()),
    symbol="BTC/USDT",
    side="buy",
    type="market",
    amount=0.1,
    price=50000.0,
)

filled_order = await trader.place_order(order, current_price=50000.0)

# Check results
print(f"Filled at: ${filled_order.filled_price:.2f}")
print(f"Fee paid: ${filled_order.fee:.2f}")
print(f"Balance: ${trader.get_balance():.2f}")

# Get positions
positions = trader.get_positions()
for pos in positions:
    print(f"Position: {pos.symbol} {pos.side} @ ${pos.entry_price:.2f}")
```

#### Features

- Realistic slippage simulation (0.05%)
- Maker/taker fee calculation
- Position tracking with PnL
- Balance management
- Order history

### LiveExecutor

Real order execution via CCXT.

```python
from iftb.trading.executor import LiveExecutor
from iftb.data import ExchangeClient

# Connect to exchange
async with ExchangeClient("binance", api_key, api_secret) as client:
    executor = LiveExecutor(client, max_retries=3)

    # Place market order
    order = await executor.place_market_order(
        symbol="BTC/USDT",
        side="buy",
        amount=0.1,
    )

    # Place limit order
    limit_order = await executor.place_limit_order(
        symbol="BTC/USDT",
        side="buy",
        amount=0.1,
        price=49000.0,
    )

    # Set stop-loss
    stop_order = await executor.set_stop_loss(
        symbol="BTC/USDT",
        side="sell",
        amount=0.1,
        stop_price=48000.0,
    )

    # Get current position
    position = await executor.get_position("BTC/USDT")
    if position:
        print(f"Position PnL: ${position.unrealized_pnl:.2f}")

    # Close position
    close_order = await executor.close_position("BTC/USDT")
```

#### Features

- Automatic retry with exponential backoff
- Order validation against exchange limits
- Position fetching and tracking
- Stop-loss and take-profit management
- Order cancellation

### OrderExecutor

Unified interface with paper/live mode support.

```python
from iftb.trading.executor import OrderExecutor, ExecutionRequest

# Paper trading mode
executor = OrderExecutor(
    paper_mode=True,
    initial_balance=10000.0,
    max_position_size=0.1,  # 10% of balance max
    max_leverage=5,
)

# Create execution request
request = ExecutionRequest(
    action="long",
    symbol="BTC/USDT",
    amount=0.1,
    entry_price=50000.0,
    stop_loss=48000.0,
    take_profit=55000.0,
    leverage=2,
    order_type="market",
    reason="Bullish signal from strategy",
)

# Execute
order = await executor.execute_decision(request)

if order.status == "filled":
    print(f"Order filled at ${order.filled_price:.2f}")

    # Update stop-loss
    await executor.update_stop_loss("BTC/USDT", 49000.0)

    # Update take-profit
    await executor.update_take_profit("BTC/USDT", 56000.0)

# Get account status
status = await executor.get_account_status()
print(f"Balance: ${status['balance']:.2f}")
print(f"Total Equity: ${status['total_equity']:.2f}")
print(f"Positions: {status['positions_count']}")

# Emergency close all positions
if emergency_condition:
    closing_orders = await executor.emergency_close_all()
    print(f"Closed {len(closing_orders)} positions")
```

#### Features

- Automatic mode switching (paper/live)
- Decision validation
- Position size enforcement
- Leverage limits
- Stop-loss and take-profit management
- Emergency close functionality
- Comprehensive account status

## Integration with Decision Engine

Convert `TradingDecision` from decision engine to `ExecutionRequest`:

```python
from iftb.trading import DecisionEngine
from iftb.trading.executor import (
    OrderExecutor,
    convert_decision_to_request,
)

# Initialize components
engine = DecisionEngine(...)
executor = OrderExecutor(paper_mode=True)

# Get decision from engine
decision = await engine.make_decision(signal, context)

# Convert to execution request
if decision.action != "HOLD" and not decision.vetoed:
    request = convert_decision_to_request(
        decision,
        amount=0.1,  # Calculate based on position_size
        order_type="market"
    )

    # Execute
    order = await executor.execute_decision(request)
    print(f"Order executed: {order.id}")
```

## Complete Trading Flow Example

```python
import asyncio
from iftb.analysis import CompositeSignal
from iftb.data import fetch_latest_ticker, MarketContext
from iftb.trading import DecisionEngine
from iftb.trading.executor import (
    OrderExecutor,
    convert_decision_to_request,
)

async def trading_loop():
    # Initialize
    engine = DecisionEngine(...)
    executor = OrderExecutor(
        paper_mode=True,
        initial_balance=10000.0,
    )

    while True:
        try:
            # Get market data
            ticker = await fetch_latest_ticker("BTC/USDT")

            # Get composite signal (from analysis module)
            signal = CompositeSignal(...)  # Your signal

            # Get market context
            context = MarketContext(...)   # Your context

            # Make decision
            decision = await engine.make_decision(signal, context)

            # Log decision
            print(f"Decision: {decision.action} "
                  f"(confidence: {decision.confidence:.2%})")

            # Execute if actionable
            if decision.action != "HOLD" and not decision.vetoed:
                # Calculate position size
                account = await executor.get_account_status()
                position_value = (
                    account['balance'] * decision.position_size
                )
                amount = position_value / decision.entry_price

                # Convert and execute
                request = convert_decision_to_request(
                    decision,
                    amount=amount,
                    order_type="market"
                )

                order = await executor.execute_decision(request)

                if order.status == "filled":
                    print(f"Order filled: {order.id}")
                    print(f"  Price: ${order.filled_price:.2f}")
                    print(f"  Amount: {order.filled_amount}")
                    print(f"  Fee: ${order.fee:.2f}")
                else:
                    print(f"Order failed: {order.error_message}")

            # Get updated account status
            status = await executor.get_account_status()
            print(f"\nAccount Status:")
            print(f"  Balance: ${status['balance']:.2f}")
            print(f"  Equity: ${status['total_equity']:.2f}")
            print(f"  Open Positions: {status['positions_count']}")

            # Wait before next iteration
            await asyncio.sleep(60)  # 1 minute

        except KeyboardInterrupt:
            print("\nShutting down...")

            # Emergency close all positions
            closing_orders = await executor.emergency_close_all()
            print(f"Closed {len(closing_orders)} positions")
            break

        except Exception as e:
            print(f"Error in trading loop: {e}")
            await asyncio.sleep(5)  # Wait before retry

if __name__ == "__main__":
    asyncio.run(trading_loop())
```

## Error Handling

### Order Validation Errors

```python
try:
    request = ExecutionRequest(
        action="long",
        symbol="BTC/USDT",
        amount=0.1,
        leverage=10,  # Exceeds max_leverage
    )
    order = await executor.execute_decision(request)
except ValueError as e:
    print(f"Validation error: {e}")
```

### Network Errors

Network errors are automatically retried with exponential backoff:

```python
# LiveExecutor automatically retries on:
# - NetworkError
# - ExchangeError
# - RateLimitExceeded

# Max retries configurable:
executor = LiveExecutor(client, max_retries=5)
```

### Insufficient Balance

```python
order = await executor.execute_decision(request)

if order.status == "failed":
    if "Insufficient balance" in order.error_message:
        print("Not enough balance for trade")
        # Reduce position size or skip trade
```

## Risk Management Features

### Position Size Limits

```python
executor = OrderExecutor(
    paper_mode=True,
    max_position_size=0.1,  # 10% of balance maximum
)
```

### Leverage Limits

```python
executor = OrderExecutor(
    paper_mode=True,
    max_leverage=5,  # Maximum 5x leverage
)
```

### Validation Checks

- Order amount > 0
- Valid prices for limit/stop orders
- Stop-loss below entry for long positions
- Stop-loss above entry for short positions
- Take-profit above entry for long positions
- Take-profit below entry for short positions
- Leverage within limits
- Exchange minimum/maximum amounts

## Testing

Run the example test script:

```bash
python examples/test_executor.py
```

This will test:
- Paper trading execution
- Position management
- Order validation
- Stop-loss/take-profit updates
- Emergency close functionality

## Best Practices

1. **Always use paper trading first**: Test your strategy thoroughly before live trading
2. **Set position size limits**: Protect against oversized positions
3. **Use stop-losses**: Always set stop-losses for risk management
4. **Monitor account status**: Regularly check balance and positions
5. **Handle errors gracefully**: Implement proper error handling
6. **Log everything**: Use the built-in logging for audit trails
7. **Test edge cases**: Validate with extreme market conditions
8. **Emergency procedures**: Have a plan for emergency close scenarios

## Configuration

Settings from `iftb.config`:

```python
class TradingSettings(BaseSettings):
    paper_trading: bool = True
    default_leverage: int = 1
    max_position_size: float = 0.1
    max_leverage: int = 5
```

## Performance Considerations

- **Paper trading**: Very fast, no network latency
- **Live trading**: Subject to exchange API rate limits
- **Retry logic**: May delay order execution during network issues
- **Position updates**: Cached for performance
- **Concurrent orders**: Rate-limited to prevent API bans

## Security Notes

- API keys stored securely in `.env` file
- Never log API secrets
- Use testnet for testing live execution
- Validate all inputs before execution
- Implement maximum loss limits
- Monitor for unusual activity

## Future Enhancements

Planned features:
- Advanced order types (trailing stop, OCO)
- Multi-exchange support
- Order batching for efficiency
- Advanced slippage modeling
- Commission optimization
- Partial fills handling
- WebSocket execution updates
