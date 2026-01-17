"""
Tests for database storage layer.

Tests the DatabaseManager, repositories, and ORM models.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from src.iftb.data.storage import (
    Base,
    DatabaseManager,
    OHLCVBar,
    OHLCVRepository,
    Position,
    PositionRepository,
    SystemEventRepository,
    Trade,
    TradeRepository,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def db_engine():
    """Create async database engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Drop tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine):
    """Create async database session for testing."""
    session_maker = async_sessionmaker(
        db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_maker() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def db_manager(db_engine):
    """Create DatabaseManager instance for testing."""
    manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
    manager._engine = db_engine
    manager._session_maker = async_sessionmaker(
        db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    yield manager


# ============================================================================
# OHLCV Repository Tests
# ============================================================================


@pytest.mark.asyncio
async def test_ohlcv_insert_many(db_session):
    """Test inserting multiple OHLCV bars."""
    repo = OHLCVRepository(db_session)

    bars = [
        OHLCVBar(
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1h",
            timestamp=datetime.utcnow() - timedelta(hours=i),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49500"),
            close=Decimal("50500"),
            volume=Decimal("100.5"),
        )
        for i in range(10)
    ]

    count = await repo.insert_many(bars)
    await db_session.commit()

    assert count == 10


@pytest.mark.asyncio
async def test_ohlcv_get_latest(db_session):
    """Test getting latest OHLCV bars."""
    repo = OHLCVRepository(db_session)

    # Insert test data
    bars = [
        OHLCVBar(
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1h",
            timestamp=datetime.utcnow() - timedelta(hours=i),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49500"),
            close=Decimal("50500"),
            volume=Decimal("100.5"),
        )
        for i in range(10)
    ]
    await repo.insert_many(bars)
    await db_session.commit()

    # Get latest
    latest = await repo.get_latest("BTCUSDT", "1h", limit=5)

    assert len(latest) == 5
    assert latest[0].symbol == "BTCUSDT"


@pytest.mark.asyncio
async def test_ohlcv_get_range(db_session):
    """Test getting OHLCV bars in a time range."""
    repo = OHLCVRepository(db_session)

    now = datetime.utcnow()
    bars = [
        OHLCVBar(
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1h",
            timestamp=now - timedelta(hours=i),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49500"),
            close=Decimal("50500"),
            volume=Decimal("100.5"),
        )
        for i in range(10)
    ]
    await repo.insert_many(bars)
    await db_session.commit()

    # Get range
    start = now - timedelta(hours=5)
    end = now
    range_bars = await repo.get_range("BTCUSDT", "1h", start, end)

    assert len(range_bars) > 0
    assert all(start <= bar.timestamp <= end for bar in range_bars)


# ============================================================================
# Trade Repository Tests
# ============================================================================


@pytest.mark.asyncio
async def test_trade_insert(db_session):
    """Test inserting a trade."""
    repo = TradeRepository(db_session)

    trade = Trade(
        trade_id="test-trade-001",
        symbol="BTCUSDT",
        exchange="binance",
        side="long",
        action="open",
        entry_price=Decimal("50000"),
        quantity=Decimal("0.01"),
        entry_time=datetime.utcnow(),
    )

    trade_id = await repo.insert(trade)
    await db_session.commit()

    assert trade_id > 0


@pytest.mark.asyncio
async def test_trade_get_by_id(db_session):
    """Test getting trade by ID."""
    repo = TradeRepository(db_session)

    trade = Trade(
        trade_id="test-trade-002",
        symbol="ETHUSDT",
        exchange="binance",
        side="short",
        action="open",
        entry_price=Decimal("3000"),
        quantity=Decimal("0.1"),
        entry_time=datetime.utcnow(),
    )

    trade_id = await repo.insert(trade)
    await db_session.commit()

    # Get by ID
    retrieved = await repo.get_by_id(trade_id)

    assert retrieved is not None
    assert retrieved.trade_id == "test-trade-002"
    assert retrieved.symbol == "ETHUSDT"


@pytest.mark.asyncio
async def test_trade_get_recent(db_session):
    """Test getting recent trades."""
    repo = TradeRepository(db_session)

    # Insert multiple trades
    for i in range(5):
        trade = Trade(
            trade_id=f"test-trade-{i:03d}",
            symbol="BTCUSDT",
            exchange="binance",
            side="long",
            action="open",
            entry_price=Decimal("50000"),
            quantity=Decimal("0.01"),
            entry_time=datetime.utcnow() - timedelta(hours=i),
        )
        await repo.insert(trade)

    await db_session.commit()

    # Get recent
    recent = await repo.get_recent(limit=3)

    assert len(recent) == 3


@pytest.mark.asyncio
async def test_trade_statistics(db_session):
    """Test getting trade statistics."""
    repo = TradeRepository(db_session)

    # Insert winning and losing trades
    for i in range(10):
        pnl = Decimal("100") if i % 2 == 0 else Decimal("-50")
        trade = Trade(
            trade_id=f"test-trade-{i:03d}",
            symbol="BTCUSDT",
            exchange="binance",
            side="long",
            action="close",
            entry_price=Decimal("50000"),
            quantity=Decimal("0.01"),
            entry_time=datetime.utcnow() - timedelta(hours=i+1),
            exit_time=datetime.utcnow() - timedelta(hours=i),
            realized_pnl=pnl,
            fee=Decimal("1.0"),
        )
        await repo.insert(trade)

    await db_session.commit()

    # Get statistics
    stats = await repo.get_statistics(days=30)

    assert stats.total_trades == 10
    assert stats.winning_trades == 5
    assert stats.losing_trades == 5
    assert stats.win_rate == Decimal("50")


# ============================================================================
# Position Repository Tests
# ============================================================================


@pytest.mark.asyncio
async def test_position_update(db_session):
    """Test updating a position."""
    repo = PositionRepository(db_session)

    position = Position(
        symbol="BTCUSDT",
        exchange="binance",
        side="long",
        entry_price=Decimal("50000"),
        quantity=Decimal("0.01"),
        margin=Decimal("500"),
        entry_time=datetime.utcnow(),
    )

    await repo.update_position(position)
    await db_session.commit()

    # Get position
    retrieved = await repo.get_position("BTCUSDT")

    assert retrieved is not None
    assert retrieved.symbol == "BTCUSDT"
    assert retrieved.status == "open"


@pytest.mark.asyncio
async def test_position_get_open(db_session):
    """Test getting open positions."""
    repo = PositionRepository(db_session)

    # Insert positions
    for i in range(3):
        position = Position(
            symbol=f"BTC{i}USDT",
            exchange="binance",
            side="long",
            entry_price=Decimal("50000"),
            quantity=Decimal("0.01"),
            margin=Decimal("500"),
            entry_time=datetime.utcnow(),
        )
        await repo.update_position(position)

    await db_session.commit()

    # Get open positions
    open_positions = await repo.get_open_positions()

    assert len(open_positions) == 3


@pytest.mark.asyncio
async def test_position_close(db_session):
    """Test closing a position."""
    repo = PositionRepository(db_session)

    position = Position(
        symbol="ETHUSDT",
        exchange="binance",
        side="short",
        entry_price=Decimal("3000"),
        quantity=Decimal("0.1"),
        margin=Decimal("300"),
        entry_time=datetime.utcnow(),
    )

    await repo.update_position(position)
    await db_session.commit()

    # Get position
    retrieved = await repo.get_position("ETHUSDT")
    assert retrieved.status == "open"

    # Close position
    await repo.close_position(retrieved.id)
    await db_session.commit()

    # Verify closed
    retrieved = await repo.get_position("ETHUSDT")
    assert retrieved is None  # Should not find open position


# ============================================================================
# System Event Repository Tests
# ============================================================================


@pytest.mark.asyncio
async def test_event_log(db_session):
    """Test logging system events."""
    repo = SystemEventRepository(db_session)

    await repo.log_event(
        event_type="test_event",
        severity="info",
        message="Test event message",
        details={"key": "value"},
    )

    await db_session.commit()

    # Get recent events
    events = await repo.get_recent_events(limit=10)

    assert len(events) == 1
    assert events[0].event_type == "test_event"
    assert events[0].severity == "info"


@pytest.mark.asyncio
async def test_event_filter_by_severity(db_session):
    """Test filtering events by severity."""
    repo = SystemEventRepository(db_session)

    # Log different severity events
    await repo.log_event("event1", "info", "Info message")
    await repo.log_event("event2", "error", "Error message")
    await repo.log_event("event3", "warning", "Warning message")
    await repo.log_event("event4", "error", "Another error")

    await db_session.commit()

    # Get only errors
    errors = await repo.get_recent_events(severity="error")

    assert len(errors) == 2
    assert all(e.severity == "error" for e in errors)


# ============================================================================
# Database Manager Tests
# ============================================================================


@pytest.mark.asyncio
async def test_database_manager_session(db_manager):
    """Test DatabaseManager session context manager."""
    async with db_manager.get_session() as session:
        assert session is not None
        assert isinstance(session, AsyncSession)


@pytest.mark.asyncio
async def test_database_manager_commit(db_manager):
    """Test DatabaseManager auto-commit on success."""
    async with db_manager.get_session() as session:
        repo = OHLCVRepository(session)
        bars = [
            OHLCVBar(
                symbol="TESTUSDT",
                exchange="binance",
                timeframe="1m",
                timestamp=datetime.utcnow(),
                open=Decimal("100"),
                high=Decimal("101"),
                low=Decimal("99"),
                close=Decimal("100.5"),
                volume=Decimal("10"),
            )
        ]
        await repo.insert_many(bars)

    # Verify committed
    async with db_manager.get_session() as session:
        repo = OHLCVRepository(session)
        latest = await repo.get_latest("TESTUSDT", "1m", limit=1)
        assert len(latest) == 1
