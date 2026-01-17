# Order Executor Quick Reference

## Import

```python
from iftb.trading.executor import (
    Order,
    PositionState,
    ExecutionRequest,
    PaperTrader,
    LiveExecutor,
    OrderExecutor,
    convert_decision_to_request,
)
```

## Quick Start

### Paper Trading

```python
# Initialize
executor = OrderExecutor(paper_mode=True, initial_balance=10000.0)

# Create request
request = ExecutionRequest(
    action="long",
    symbol="BTC/USDT",
    amount=0.1,
    stop_loss=48000.0,
    take_profit=55000.0,
)

# Execute
order = await executor.execute_decision(request)
```

### Live Trading

```python
from iftb.data import ExchangeClient

async with ExchangeClient("binance", api_key, api_secret) as client:
    executor = OrderExecutor(paper_mode=False, exchange_client=client)
    order = await executor.execute_decision(request)
```

## Common Operations

### Place Market Order

```python
request = ExecutionRequest(
    action="long",
    symbol="BTC/USDT",
    amount=0.1,
    order_type="market",
)
order = await executor.execute_decision(request)
```

### Place Limit Order

```python
request = ExecutionRequest(
    action="long",
    symbol="BTC/USDT",
    amount=0.1,
    entry_price=49000.0,
    order_type="limit",
)
order = await executor.execute_decision(request)
```

### Close Position

```python
request = ExecutionRequest(
    action="close",
    symbol="BTC/USDT",
    amount=0.1,
)
order = await executor.execute_decision(request)
```

### Update Stop-Loss

```python
success = await executor.update_stop_loss("BTC/USDT", 49000.0)
```

### Update Take-Profit

```python
success = await executor.update_take_profit("BTC/USDT", 56000.0)
```

### Get Account Status

```python
status = await executor.get_account_status()
print(f"Balance: ${status['balance']:.2f}")
print(f"Positions: {status['positions_count']}")
```

### Emergency Close All

```python
orders = await executor.emergency_close_all()
```

## Integration with Decision Engine

```python
from iftb.trading import DecisionEngine

# Get decision
engine = DecisionEngine(...)
decision = await engine.make_decision(signal, context)

# Convert to request
if decision.action != "HOLD" and not decision.vetoed:
    request = convert_decision_to_request(decision, amount=0.1)
    order = await executor.execute_decision(request)
```

## Order Status Checking

```python
order = await executor.execute_decision(request)

if order.status == "filled":
    print(f"Filled at ${order.filled_price:.2f}")
elif order.status == "failed":
    print(f"Failed: {order.error_message}")
```

## Error Handling

```python
try:
    order = await executor.execute_decision(request)
except ValueError as e:
    print(f"Validation error: {e}")
```

## Configuration

```python
executor = OrderExecutor(
    paper_mode=True,
    initial_balance=10000.0,
    max_position_size=0.1,    # 10% max
    max_leverage=5,
)
```

## Paper Trader Direct Usage

```python
from iftb.trading.executor import PaperTrader, Order
import uuid

trader = PaperTrader(initial_balance=10000.0)

order = Order(
    id=str(uuid.uuid4()),
    symbol="BTC/USDT",
    side="buy",
    type="market",
    amount=0.1,
    price=50000.0,
)

filled = await trader.place_order(order, current_price=50000.0)
balance = trader.get_balance()
positions = trader.get_positions()
```

## Live Executor Direct Usage

```python
from iftb.trading.executor import LiveExecutor
from iftb.data import ExchangeClient

async with ExchangeClient("binance", api_key, api_secret) as client:
    executor = LiveExecutor(client)

    # Market order
    order = await executor.place_market_order("BTC/USDT", "buy", 0.1)

    # Limit order
    order = await executor.place_limit_order("BTC/USDT", "buy", 0.1, 49000.0)

    # Stop-loss
    order = await executor.set_stop_loss("BTC/USDT", "sell", 0.1, 48000.0)

    # Get position
    position = await executor.get_position("BTC/USDT")
```

## Data Structures

### ExecutionRequest

```python
ExecutionRequest(
    action="long",              # "long", "short", "close"
    symbol="BTC/USDT",
    amount=0.1,
    entry_price=50000.0,       # Optional, uses market if None
    stop_loss=48000.0,         # Optional
    take_profit=55000.0,       # Optional
    leverage=2,                # Default: 1
    order_type="market",       # "market" or "limit"
    reason="Strategy signal",  # Optional
)
```

### Order

```python
Order(
    id="uuid",
    symbol="BTC/USDT",
    side="buy",                # "buy" or "sell"
    type="market",             # "market", "limit", "stop_loss", "take_profit"
    amount=0.1,
    price=50000.0,             # Optional
    stop_price=48000.0,        # Optional
    status="filled",           # "pending", "filled", "cancelled", "failed"
    filled_price=50025.0,      # Optional
    filled_amount=0.1,         # Optional
    fee=2.0,                   # Optional
    timestamp=datetime.now(),
)
```

### PositionState

```python
PositionState(
    symbol="BTC/USDT",
    side="long",               # "long" or "short"
    entry_price=50000.0,
    current_price=51000.0,
    amount=0.1,
    leverage=2,
    unrealized_pnl=100.0,
    liquidation_price=45000.0,
    stop_loss=48000.0,         # Optional
    take_profit=55000.0,       # Optional
    opened_at=datetime.now(),
)
```

## Default Values

- **Maker Fee**: 0.02% (0.0002)
- **Taker Fee**: 0.04% (0.0004)
- **Slippage**: 0.05% (0.0005)
- **Max Retries**: 3
- **Initial Backoff**: 1 second
- **Max Position Size**: 10% of balance (0.1)
- **Max Leverage**: 5

## Testing

```bash
python examples/test_executor.py
```
