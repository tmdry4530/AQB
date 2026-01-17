# IFTB Logging System

A comprehensive structured logging system for the Intelligent Futures Trading Bot (IFTB).

## Overview

The IFTB logging system uses `structlog` to provide structured, contextual logging with support for both human-readable (pretty) and machine-readable (JSON) output formats.

## Features

- **Structured Logging**: Log structured data as key-value pairs
- **Multiple Output Formats**: Pretty format for development, JSON for production
- **File & Console Output**: Log to both console and file simultaneously
- **Contextual Logging**: Automatically include contextual information (trade_id, symbol, etc.)
- **Sensitive Data Filtering**: Automatically mask API keys, passwords, and tokens
- **Configurable Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Caller Information**: Optional file/line/function information
- **Exception Logging**: Full stack traces for errors
- **String Truncation**: Prevent log bloat from large data

## Quick Start

### Basic Setup

```python
from iftb.utils import LogConfig, setup_logging, get_logger

# Configure logging
config = LogConfig(
    level="INFO",
    format="pretty",  # or "json" for production
    file_path="logs/iftb.log",  # Optional file output
    include_timestamp=True,
    include_caller_info=True,
)
setup_logging(config)

# Get a logger
logger = get_logger(__name__)

# Log some messages
logger.info("application_started")
logger.info("trade_opened", symbol="BTCUSDT", side="LONG", price=50000)
```

### Configuration Options

```python
@dataclass
class LogConfig:
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format: Literal["json", "pretty"] = "pretty"
    file_path: Optional[str] = None  # Path to log file
    include_timestamp: bool = True
    include_caller_info: bool = True
    console_output: bool = True
    max_string_length: int = 1000  # Truncate long strings
    environment: str = "dev"  # dev, staging, prod
    app_version: str = "1.0.0"
```

## Usage Examples

### Basic Logging

```python
logger = get_logger(__name__)

logger.debug("detailed_debug_info", state="processing")
logger.info("operation_completed", duration_ms=123)
logger.warning("high_latency_detected", latency_ms=500)
logger.error("api_error", error="Connection timeout", retry_count=3)
logger.critical("system_failure", reason="Out of memory")
```

### Structured Data

```python
# Log trade information
logger.info(
    "trade_executed",
    symbol="ETHUSDT",
    side="SHORT",
    entry_price=3000.00,
    quantity=1.5,
    leverage=10,
    stop_loss=3100.00,
    take_profit=2800.00,
)

# Log position updates
logger.info(
    "position_update",
    symbol="BTCUSDT",
    unrealized_pnl=250.50,
    realized_pnl=100.00,
    roe_pct=5.2,
)
```

### Contextual Logging

Add context that automatically appears in all log entries within a scope:

```python
from iftb.utils import add_context

logger = get_logger(__name__)

# All logs within this context will include trade_id and symbol
with add_context(trade_id="abc123", symbol="BTCUSDT"):
    logger.info("processing_trade")  # Includes trade_id and symbol
    logger.info("order_placed", order_id="xyz789")  # Also includes context

    # Nested contexts
    with add_context(strategy="momentum", timeframe="1h"):
        logger.info("signal_detected")  # Includes all context

# Context cleared outside the block
logger.info("processing_complete")  # No trade_id or symbol
```

### Error Logging with Exceptions

```python
logger = get_logger(__name__)

try:
    result = risky_operation()
except Exception as e:
    logger.error(
        "operation_failed",
        operation="risky_operation",
        error=str(e),
        exc_info=True,  # Include full stack trace
    )
```

### Sensitive Data Protection

The logger automatically masks sensitive information:

```python
logger.info(
    "api_connection",
    api_key="my_secret_key_12345",  # Will be masked
    api_secret="my_secret_123",     # Will be masked
    endpoint="https://api.exchange.com",  # Not masked
)

# Output: api_key="my***45", api_secret="my***23"
```

Automatically masked fields:
- `api_key`, `apikey`, `api_secret`, `apisecret`
- `password`, `passwd`, `pwd`
- `token`, `access_token`, `refresh_token`
- `secret`, `private_key`, `privatekey`

### Changing Log Level at Runtime

```python
from iftb.utils import set_log_level

set_log_level("DEBUG")  # Enable debug logging
# ... debug mode operations ...
set_log_level("INFO")   # Back to normal
```

## Output Formats

### Pretty Format (Development)

Human-readable colorized output for development:

```
2026-01-17T10:30:45.123456Z [info     ] trade_opened               symbol=BTCUSDT side=LONG price=50000.0
2026-01-17T10:30:46.234567Z [warning  ] high_drawdown              drawdown_pct=15.5 symbol=ETHUSDT
```

### JSON Format (Production)

Machine-readable JSON output for production logging:

```json
{
  "event": "trade_opened",
  "level": "info",
  "timestamp": "2026-01-17T10:30:45.123456Z",
  "symbol": "BTCUSDT",
  "side": "LONG",
  "price": 50000.0,
  "app": "iftb",
  "environment": "prod",
  "version": "1.0.0"
}
```

## Best Practices

### 1. Use Descriptive Event Names

```python
# Good - clear and specific
logger.info("trade_opened", ...)
logger.info("position_liquidated", ...)
logger.warning("high_drawdown_detected", ...)

# Bad - vague
logger.info("event", ...)
logger.info("thing_happened", ...)
```

### 2. Include Relevant Context

```python
# Good - includes all relevant info
logger.error(
    "order_failed",
    symbol="BTCUSDT",
    order_type="LIMIT",
    price=50000,
    quantity=0.5,
    error="Insufficient margin",
)

# Bad - missing context
logger.error("order_failed", error="Insufficient margin")
```

### 3. Use Appropriate Log Levels

- **DEBUG**: Detailed diagnostic information (variable values, state changes)
- **INFO**: General operational events (trade opened, order filled)
- **WARNING**: Potentially problematic situations (high drawdown, API delays)
- **ERROR**: Errors that need attention (failed orders, API errors)
- **CRITICAL**: System-level failures (database down, out of memory)

### 4. Use Context Managers for Related Operations

```python
with add_context(trade_id=trade.id, strategy=trade.strategy):
    logger.info("trade_analysis_started")

    # All logs here automatically include trade_id and strategy
    analyze_entry_signal()
    calculate_position_size()
    place_orders()

    logger.info("trade_analysis_completed")
```

### 5. Don't Log Sensitive Data Unnecessarily

```python
# Good - only log what's needed
logger.info("api_connected", exchange="binance", user_id="12345")

# Bad - logs full credentials
logger.info("api_connected", credentials=full_credentials_dict)
```

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI, Request
from iftb.utils import setup_logging, get_logger, add_context

app = FastAPI()
logger = get_logger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())

    with add_context(request_id=request_id, path=request.url.path):
        logger.info("request_started", method=request.method)

        response = await call_next(request)

        logger.info(
            "request_completed",
            status_code=response.status_code,
            method=request.method,
        )

    return response
```

### Trading Strategy Integration

```python
class TradingStrategy:
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(__name__)

    def analyze_and_trade(self, symbol: str):
        with add_context(strategy=self.name, symbol=symbol):
            self.logger.info("analysis_started")

            signal = self.analyze_market()
            self.logger.info("signal_generated", signal=signal.type)

            if signal.should_trade:
                trade = self.execute_trade(signal)
                self.logger.info(
                    "trade_executed",
                    trade_id=trade.id,
                    side=trade.side,
                    price=trade.entry_price,
                )
```

### Background Task Integration

```python
import asyncio
from iftb.utils import get_logger, add_context

logger = get_logger(__name__)

async def monitor_positions():
    """Background task to monitor all open positions."""
    while True:
        try:
            positions = await get_open_positions()

            for position in positions:
                with add_context(
                    position_id=position.id,
                    symbol=position.symbol,
                ):
                    if position.unrealized_pnl_pct < -10:
                        logger.warning(
                            "high_loss_detected",
                            unrealized_pnl_pct=position.unrealized_pnl_pct,
                        )

                    if position.unrealized_pnl_pct > 20:
                        logger.info(
                            "profit_target_reached",
                            unrealized_pnl_pct=position.unrealized_pnl_pct,
                        )

        except Exception as e:
            logger.error("position_monitoring_error", error=str(e), exc_info=True)

        await asyncio.sleep(10)
```

## Log Rotation

For production use, configure log rotation using `logrotate` (Linux) or Windows Task Scheduler:

### Linux logrotate Configuration

```
/var/log/iftb/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 iftb iftb
    sharedscripts
    postrotate
        systemctl reload iftb
    endscript
}
```

## Troubleshooting

### Logs Not Appearing

1. Check log level configuration
2. Verify file path permissions
3. Ensure `setup_logging()` was called
4. Check if console_output is enabled

### Performance Issues

1. Reduce log level (INFO instead of DEBUG)
2. Increase max_string_length to truncate sooner
3. Disable caller_info in production
4. Use JSON format (faster than pretty)

### Large Log Files

1. Configure log rotation
2. Reduce log level
3. Enable string truncation
4. Filter out noisy logs

## References

- [structlog Documentation](https://www.structlog.org/)
- [Python Logging HOWTO](https://docs.python.org/3/howto/logging.html)
