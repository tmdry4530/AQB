# IFTB Logging Quick Reference

## Setup

```python
from iftb.utils import setup_logging, get_logger, LogConfig

config = LogConfig(level="INFO", format="pretty", file_path="logs/app.log")
setup_logging(config)
logger = get_logger(__name__)
```

## Basic Logging

```python
logger.debug("debug_msg", key="value")
logger.info("info_msg", key="value")
logger.warning("warning_msg", key="value")
logger.error("error_msg", key="value", exc_info=True)
logger.critical("critical_msg", key="value")
```

## Contextual Logging

```python
from iftb.utils import add_context

with add_context(trade_id="123", symbol="BTCUSDT"):
    logger.info("processing")  # Includes context automatically
```

## Common Patterns

### Trade Logging
```python
logger.info("trade_opened", symbol="BTCUSDT", side="LONG", price=50000, qty=0.5)
logger.info("position_updated", pnl=250.50, roe_pct=5.2)
logger.warning("high_drawdown", drawdown_pct=15.5)
```

### Error Logging
```python
try:
    risky_operation()
except Exception as e:
    logger.error("operation_failed", error=str(e), exc_info=True)
```

### API Logging
```python
logger.info("api_request", endpoint="/orders", method="POST")
logger.error("api_error", status_code=429, error="Rate limited")
```

## Configuration Options

| Option | Values | Default |
|--------|--------|---------|
| `level` | DEBUG, INFO, WARNING, ERROR, CRITICAL | INFO |
| `format` | "json", "pretty" | pretty |
| `file_path` | Path string or None | None |
| `include_timestamp` | True/False | True |
| `include_caller_info` | True/False | True |
| `environment` | Any string | "dev" |
| `app_version` | Version string | "1.0.0" |

## Auto-Masked Fields

- api_key, api_secret
- password, passwd, pwd
- token, access_token, refresh_token
- secret, private_key

## Runtime Changes

```python
from iftb.utils import set_log_level, clear_context

set_log_level("DEBUG")  # Change level
clear_context()         # Clear all context
```

## Log Levels Guide

| Level | Use For |
|-------|---------|
| DEBUG | Variable values, detailed state |
| INFO | Normal operations, trades |
| WARNING | Potential issues, high latency |
| ERROR | Failed operations, API errors |
| CRITICAL | System failures, crashes |

## Best Practices

1. Use descriptive event names: `trade_opened` not `event`
2. Include relevant context: symbol, trade_id, etc.
3. Use appropriate log levels
4. Use `add_context()` for related operations
5. Don't log full sensitive objects
6. Use `exc_info=True` for exceptions

## Example: Complete Trading Flow

```python
from iftb.utils import get_logger, add_context

logger = get_logger(__name__)

def execute_trade(symbol: str, side: str):
    trade_id = generate_trade_id()

    with add_context(trade_id=trade_id, symbol=symbol, side=side):
        logger.info("trade_started")

        try:
            # Analyze
            signal = analyze_market(symbol)
            logger.info("signal_generated", signal_type=signal.type, strength=signal.strength)

            # Place order
            order = place_order(symbol, side, signal.price)
            logger.info("order_placed", order_id=order.id, price=order.price)

            # Monitor
            fill = wait_for_fill(order)
            logger.info("order_filled", fill_price=fill.price, qty=fill.quantity)

            logger.info("trade_completed", status="success")
            return fill

        except InsufficientMarginError as e:
            logger.error("trade_failed", reason="insufficient_margin", required=e.required)
            raise

        except Exception as e:
            logger.error("trade_failed", error=str(e), exc_info=True)
            raise
```
