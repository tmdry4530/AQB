# Database Storage Layer

The storage layer provides a comprehensive async database interface using SQLAlchemy for the IFTB trading bot.

## Overview

**Module**: `iftb.data.storage`

The storage module handles all database operations including:
- OHLCV candlestick data storage and retrieval
- Trade record management
- Position tracking
- System event logging
- Trade statistics and analytics

## Architecture

### Components

1. **DatabaseManager**: Manages database connections and sessions
2. **Repository Pattern**: Specialized repositories for each data type
3. **ORM Models**: SQLAlchemy models matching the database schema
4. **Data Classes**: Type-safe domain objects for application use

### Repository Classes

- `OHLCVRepository` - OHLCV data operations
- `TradeRepository` - Trade record management
- `PositionRepository` - Position tracking
- `SystemEventRepository` - Event logging

## Installation

Ensure these dependencies are installed:

```bash
pip install sqlalchemy[asyncio] asyncpg
```

## Database Setup

1. Create PostgreSQL database:
```sql
CREATE DATABASE iftb;
```

2. Run migrations:
```bash
alembic upgrade head
```

## Usage

### Basic Connection

```python
from iftb.data.storage import DatabaseManager
from iftb.config import get_settings

settings = get_settings()
db_manager = DatabaseManager(settings.database.get_async_url())

await db_manager.connect()

# Use database
async with db_manager.get_session() as session:
    # Your database operations here
    pass

await db_manager.disconnect()
```

### OHLCV Operations

```python
from iftb.data.storage import OHLCVRepository, OHLCVBar
from datetime import datetime, timedelta
from decimal import Decimal

async with db_manager.get_session() as session:
    repo = OHLCVRepository(session)

    # Insert OHLCV data
    bars = [
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
    ]
    count = await repo.insert_many(bars)

    # Get latest bars
    latest = await repo.get_latest("BTCUSDT", "1h", limit=100)

    # Get time range
    start = datetime.utcnow() - timedelta(days=7)
    end = datetime.utcnow()
    bars = await repo.get_range("BTCUSDT", "1h", start, end)

    # Upsert (insert or update)
    count = await repo.upsert_many(bars)

    # Find data gaps
    gaps = await repo.get_gaps("BTCUSDT", "1h")
```

### Trade Management

```python
from iftb.data.storage import TradeRepository, Trade
from decimal import Decimal

async with db_manager.get_session() as session:
    repo = TradeRepository(session)

    # Insert new trade
    trade = Trade(
        trade_id="TRD-001",
        symbol="BTCUSDT",
        exchange="binance",
        side="long",
        action="open",
        entry_price=Decimal("50000"),
        quantity=Decimal("0.01"),
        entry_time=datetime.utcnow(),
        leverage=Decimal("5"),
        signal_score=Decimal("0.85"),
    )
    trade_id = await repo.insert(trade)

    # Update trade
    await repo.update(trade_id, exit_price=Decimal("51000"), realized_pnl=Decimal("50"))

    # Get trade by ID
    trade = await repo.get_by_id(trade_id)

    # Get recent trades
    recent = await repo.get_recent(limit=100)

    # Get trades by symbol
    symbol_trades = await repo.get_by_symbol("BTCUSDT", limit=100)

    # Get statistics
    stats = await repo.get_statistics(days=30)
    print(f"Win rate: {stats.win_rate}%")
    print(f"Net PnL: {stats.net_pnl}")
```

### Position Tracking

```python
from iftb.data.storage import PositionRepository, Position
from decimal import Decimal

async with db_manager.get_session() as session:
    repo = PositionRepository(session)

    # Create/update position
    position = Position(
        symbol="BTCUSDT",
        exchange="binance",
        side="long",
        entry_price=Decimal("50000"),
        quantity=Decimal("0.01"),
        margin=Decimal("100"),
        entry_time=datetime.utcnow(),
    )
    await repo.update_position(position)

    # Get open positions
    open_positions = await repo.get_open_positions()

    # Get specific position
    btc_position = await repo.get_position("BTCUSDT")

    # Close position
    await repo.close_position(btc_position.id)
```

### System Event Logging

```python
from iftb.data.storage import SystemEventRepository

async with db_manager.get_session() as session:
    repo = SystemEventRepository(session)

    # Log event
    await repo.log_event(
        event_type="trade_executed",
        severity="info",
        message="Trade executed successfully",
        symbol="BTCUSDT",
        trade_id="TRD-001",
        details={"price": 50000, "quantity": 0.01},
    )

    # Get recent events
    events = await repo.get_recent_events(limit=100)

    # Get only errors
    errors = await repo.get_recent_events(severity="error")
```

## Data Models

### OHLCVBar

Represents candlestick price data.

```python
@dataclass
class OHLCVBar:
    symbol: str
    exchange: str
    timeframe: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Optional[Decimal] = None
    trades_count: Optional[int] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
```

### Trade

Represents a trade record with entry/exit details.

```python
@dataclass
class Trade:
    trade_id: str
    symbol: str
    exchange: str
    side: str  # 'long' or 'short'
    action: str  # 'open', 'close', 'liquidated'
    entry_price: Decimal
    quantity: Decimal
    entry_time: datetime
    leverage: Decimal = Decimal("1.0")
    exit_price: Optional[Decimal] = None
    exit_time: Optional[datetime] = None
    realized_pnl: Optional[Decimal] = None
    realized_pnl_pct: Optional[Decimal] = None
    # ... additional fields
```

### Position

Represents current position state.

```python
@dataclass
class Position:
    symbol: str
    exchange: str
    side: str
    entry_price: Decimal
    quantity: Decimal
    margin: Decimal
    entry_time: datetime
    current_price: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    status: str = "open"
    # ... additional fields
```

### TradeStatistics

Aggregated trade performance metrics.

```python
@dataclass
class TradeStatistics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    gross_profit: Decimal
    gross_loss: Decimal
    net_pnl: Decimal
    total_fees: Decimal
    avg_win: Decimal
    avg_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    profit_factor: Optional[Decimal] = None
```

## Advanced Usage

### Transaction Management

The `DatabaseManager.get_session()` context manager automatically handles commits and rollbacks:

```python
async with db_manager.get_session() as session:
    repo = TradeRepository(session)
    await repo.insert(trade)
    # Automatically commits on success
    # Automatically rolls back on exception
```

### Bulk Operations

For better performance with large datasets:

```python
# Insert many OHLCV bars at once
bars = [create_bar(i) for i in range(1000)]
count = await repo.insert_many(bars)

# Upsert handles both inserts and updates
count = await repo.upsert_many(bars)
```

### Connection Pooling

Configure connection pool for optimal performance:

```python
db_manager = DatabaseManager(
    database_url=url,
    pool_size=10,        # Concurrent connections
    max_overflow=20,     # Extra connections when busy
)
```

## Performance Considerations

1. **Batch Operations**: Use `insert_many()` instead of individual inserts
2. **Indexes**: Database has indexes on frequently queried columns
3. **Connection Pooling**: Reuses connections for better performance
4. **Async Operations**: Non-blocking database operations

## Error Handling

```python
from sqlalchemy.exc import IntegrityError, OperationalError

async with db_manager.get_session() as session:
    try:
        repo = OHLCVRepository(session)
        await repo.insert_many(bars)
    except IntegrityError as e:
        # Duplicate data
        logger.error("duplicate_data", error=str(e))
    except OperationalError as e:
        # Database connection issue
        logger.error("database_error", error=str(e))
```

## Testing

Use SQLite in-memory database for tests:

```python
@pytest_asyncio.fixture
async def db_manager():
    manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
    await manager.connect()

    # Create tables
    async with manager._engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield manager

    await manager.disconnect()
```

## See Also

- [Database Schema](../migrations/001_initial_schema.sql)
- [Configuration Settings](../src/iftb/config/settings.py)
- [Example Usage](../examples/storage_example.py)
- [Tests](../tests/test_storage.py)

## API Reference

### DatabaseManager

- `__init__(database_url, pool_size=5, max_overflow=10, echo=False)`
- `async connect()` - Establish connection
- `async disconnect()` - Close connection
- `async get_session()` - Get session context manager

### OHLCVRepository

- `async insert_many(bars: list[OHLCVBar]) -> int`
- `async get_range(symbol, timeframe, start, end, exchange="binance") -> list[OHLCVBar]`
- `async get_latest(symbol, timeframe, limit=100, exchange="binance") -> list[OHLCVBar]`
- `async upsert_many(bars: list[OHLCVBar]) -> int`
- `async get_gaps(symbol, timeframe, exchange="binance") -> list[tuple[datetime, datetime]]`

### TradeRepository

- `async insert(trade: Trade) -> int`
- `async update(trade_id: int, **kwargs) -> None`
- `async get_by_id(trade_id: int) -> Trade | None`
- `async get_recent(limit=100) -> list[Trade]`
- `async get_by_symbol(symbol, limit=100, exchange="binance") -> list[Trade]`
- `async get_statistics(days=30, exchange="binance") -> TradeStatistics`

### PositionRepository

- `async get_open_positions(exchange="binance") -> list[Position]`
- `async get_position(symbol, exchange="binance") -> Position | None`
- `async update_position(position: Position) -> None`
- `async close_position(position_id: int) -> None`

### SystemEventRepository

- `async log_event(event_type, severity, message, details=None, ...) -> None`
- `async get_recent_events(severity=None, limit=100) -> list[SystemEvent]`
