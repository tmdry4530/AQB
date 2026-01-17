# Storage Layer Quick Reference

## Import

```python
from iftb.data.storage import (
    DatabaseManager,
    OHLCVRepository,
    TradeRepository,
    PositionRepository,
    SystemEventRepository,
    OHLCVBar,
    Trade,
    Position,
    SystemEvent,
    TradeStatistics,
)
from iftb.config import get_settings
```

## Setup

```python
settings = get_settings()
db = DatabaseManager(settings.database.get_async_url())
await db.connect()
```

## OHLCV Operations

```python
async with db.get_session() as session:
    repo = OHLCVRepository(session)

    # Insert bars
    bars = [OHLCVBar(...)]
    count = await repo.insert_many(bars)

    # Get latest
    latest = await repo.get_latest("BTCUSDT", "1h", limit=100)

    # Get range
    bars = await repo.get_range("BTCUSDT", "1h", start_date, end_date)

    # Upsert (insert or update)
    count = await repo.upsert_many(bars)

    # Find gaps
    gaps = await repo.get_gaps("BTCUSDT", "1h")
```

## Trade Operations

```python
async with db.get_session() as session:
    repo = TradeRepository(session)

    # Insert trade
    trade = Trade(trade_id="TRD-001", ...)
    trade_id = await repo.insert(trade)

    # Update trade
    await repo.update(trade_id, exit_price=Decimal("51000"))

    # Get by ID
    trade = await repo.get_by_id(trade_id)

    # Get recent
    recent = await repo.get_recent(limit=100)

    # Get by symbol
    trades = await repo.get_by_symbol("BTCUSDT", limit=100)

    # Get statistics
    stats = await repo.get_statistics(days=30)
```

## Position Operations

```python
async with db.get_session() as session:
    repo = PositionRepository(session)

    # Create/update position
    position = Position(symbol="BTCUSDT", ...)
    await repo.update_position(position)

    # Get open positions
    open_pos = await repo.get_open_positions()

    # Get specific position
    pos = await repo.get_position("BTCUSDT")

    # Close position
    await repo.close_position(position_id)
```

## Event Logging

```python
async with db.get_session() as session:
    repo = SystemEventRepository(session)

    # Log event
    await repo.log_event(
        event_type="trade_executed",
        severity="info",
        message="Trade executed",
        symbol="BTCUSDT",
        details={"price": 50000}
    )

    # Get recent events
    events = await repo.get_recent_events(limit=100)

    # Get errors only
    errors = await repo.get_recent_events(severity="error")
```

## Data Models

### OHLCVBar
```python
OHLCVBar(
    symbol="BTCUSDT",
    exchange="binance",
    timeframe="1h",
    timestamp=datetime.utcnow(),
    open=Decimal("50000"),
    high=Decimal("51000"),
    low=Decimal("49500"),
    close=Decimal("50500"),
    volume=Decimal("100.5"),
)
```

### Trade
```python
Trade(
    trade_id="TRD-001",
    symbol="BTCUSDT",
    exchange="binance",
    side="long",  # or "short"
    action="open",  # or "close", "liquidated"
    entry_price=Decimal("50000"),
    quantity=Decimal("0.01"),
    entry_time=datetime.utcnow(),
    leverage=Decimal("5"),
    signal_score=Decimal("0.85"),
)
```

### Position
```python
Position(
    symbol="BTCUSDT",
    exchange="binance",
    side="long",
    entry_price=Decimal("50000"),
    quantity=Decimal("0.01"),
    margin=Decimal("100"),
    entry_time=datetime.utcnow(),
)
```

## Common Patterns

### Complete Transaction
```python
async with db.get_session() as session:
    # Multiple operations in one transaction
    ohlcv_repo = OHLCVRepository(session)
    trade_repo = TradeRepository(session)
    event_repo = SystemEventRepository(session)

    await ohlcv_repo.insert_many(bars)
    trade_id = await trade_repo.insert(trade)
    await event_repo.log_event("trade_opened", "info", "Trade opened")

    # All committed automatically
```

### Error Handling
```python
from sqlalchemy.exc import IntegrityError

async with db.get_session() as session:
    try:
        repo = OHLCVRepository(session)
        await repo.insert_many(bars)
    except IntegrityError:
        # Duplicate data - already exists
        pass
```

### Cleanup
```python
await db.disconnect()
```

## Performance Tips

1. **Batch operations**: Use `insert_many()` instead of loops
2. **Connection pooling**: Configure `pool_size` and `max_overflow`
3. **Indexes**: All frequently-queried columns are indexed
4. **Async**: All operations are non-blocking

## Environment Variables

```bash
DB_HOST=localhost
DB_PORT=5432
DB_DATABASE=iftb
DB_USERNAME=iftb_user
DB_PASSWORD=your_password
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
```

## Files

- **Implementation**: `src/iftb/data/storage.py`
- **Tests**: `tests/test_storage.py`
- **Examples**: `examples/storage_example.py`
- **Full Docs**: `docs/storage_layer.md`
- **Schema**: `migrations/001_initial_schema.sql`
