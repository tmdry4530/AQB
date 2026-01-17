"""
Database storage layer for IFTB trading bot.

This module handles all database operations using SQLAlchemy async with comprehensive
repository patterns for OHLCV data, trades, positions, LLM logs, and system events.

Example Usage:
    ```python
    from iftb.data.storage import DatabaseManager, OHLCVRepository
    from iftb.config import get_settings

    settings = get_settings()
    db_manager = DatabaseManager(settings.database.get_async_url())

    await db_manager.connect()

    async with db_manager.get_session() as session:
        ohlcv_repo = OHLCVRepository(session)
        bars = await ohlcv_repo.get_latest("BTCUSDT", "1h", limit=100)

    await db_manager.disconnect()
    ```
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    and_,
    desc,
    func,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from iftb.utils import get_logger

logger = get_logger(__name__)

# SQLAlchemy declarative base
Base = declarative_base()


# ============================================================================
# Data Models (Domain Objects)
# ============================================================================


@dataclass
class OHLCVBar:
    """
    OHLCV candlestick bar data for database storage.

    This is the database representation with additional metadata like symbol,
    exchange, and timeframe. For exchange API data, see iftb.data.fetcher.OHLCVBar.
    """

    symbol: str
    exchange: str
    timeframe: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Decimal | None = None
    trades_count: int | None = None
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class Trade:
    """Trade record with entry/exit details."""

    trade_id: str
    symbol: str
    exchange: str
    side: str  # 'long' or 'short'
    action: str  # 'open', 'close', 'liquidated'
    entry_price: Decimal
    quantity: Decimal
    entry_time: datetime
    leverage: Decimal = Decimal("1.0")
    exit_price: Decimal | None = None
    exit_time: datetime | None = None
    realized_pnl: Decimal | None = None
    realized_pnl_pct: Decimal | None = None
    fee: Decimal = Decimal("0")
    signal_score: Decimal | None = None
    technical_score: Decimal | None = None
    llm_score: Decimal | None = None
    xgb_confidence: Decimal | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    position_size_pct: Decimal | None = None
    decision_reasons: dict | None = None
    llm_analysis: dict | None = None
    id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class Position:
    """Current position state."""

    symbol: str
    exchange: str
    side: str  # 'long' or 'short'
    entry_price: Decimal
    quantity: Decimal
    margin: Decimal
    entry_time: datetime
    leverage: Decimal = Decimal("1.0")
    current_price: Decimal | None = None
    unrealized_pnl: Decimal | None = None
    unrealized_pnl_pct: Decimal | None = None
    liquidation_price: Decimal | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    trailing_stop: Decimal | None = None
    status: str = "open"  # 'open', 'closed', 'liquidated'
    trade_id: str | None = None
    id: int | None = None
    last_updated: datetime | None = None
    created_at: datetime | None = None


@dataclass
class SystemEvent:
    """System event log entry."""

    event_type: str
    severity: str  # 'debug', 'info', 'warning', 'error', 'critical'
    message: str
    details: dict | None = None
    symbol: str | None = None
    exchange: str | None = None
    trade_id: str | None = None
    position_id: int | None = None
    stack_trace: str | None = None
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class TradeStatistics:
    """Aggregated trade statistics."""

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
    profit_factor: Decimal | None = None


# ============================================================================
# SQLAlchemy ORM Models
# ============================================================================


class OHLCVModel(Base):
    """SQLAlchemy model for OHLCV data."""

    __tablename__ = "ohlcv"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(50), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    open = Column(Numeric(20, 8), nullable=False)
    high = Column(Numeric(20, 8), nullable=False)
    low = Column(Numeric(20, 8), nullable=False)
    close = Column(Numeric(20, 8), nullable=False)
    volume = Column(Numeric(20, 8), nullable=False)
    quote_volume = Column(Numeric(20, 8))
    trades_count = Column(Integer)
    created_at = Column(DateTime(timezone=True), default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "exchange",
            "timeframe",
            "timestamp",
            name="ohlcv_unique_candle",
        ),
        Index("idx_ohlcv_symbol_timeframe", "symbol", "timeframe", "timestamp"),
        Index("idx_ohlcv_exchange_symbol", "exchange", "symbol"),
        Index("idx_ohlcv_timestamp", "timestamp"),
        Index("idx_ohlcv_lookup", "symbol", "exchange", "timeframe", "timestamp"),
    )


class TradeModel(Base):
    """SQLAlchemy model for trade records."""

    __tablename__ = "trades"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    trade_id = Column(String(100), unique=True, nullable=False)
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)
    action = Column(String(10), nullable=False)
    entry_price = Column(Numeric(20, 8), nullable=False)
    exit_price = Column(Numeric(20, 8))
    quantity = Column(Numeric(20, 8), nullable=False)
    leverage = Column(Numeric(5, 2), default=1.0)
    realized_pnl = Column(Numeric(20, 8))
    realized_pnl_pct = Column(Numeric(10, 4))
    fee = Column(Numeric(20, 8), default=0)
    signal_score = Column(Numeric(5, 4))
    technical_score = Column(Numeric(5, 4))
    llm_score = Column(Numeric(5, 4))
    xgb_confidence = Column(Numeric(5, 4))
    stop_loss = Column(Numeric(20, 8))
    take_profit = Column(Numeric(20, 8))
    position_size_pct = Column(Numeric(5, 2))
    decision_reasons = Column(JSONB)
    llm_analysis = Column(JSONB)
    entry_time = Column(DateTime(timezone=True), nullable=False, default=func.now())
    exit_time = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint("side IN ('long', 'short')", name="check_trade_side"),
        CheckConstraint("action IN ('open', 'close', 'liquidated')", name="check_trade_action"),
        Index("idx_trades_symbol", "symbol"),
        Index("idx_trades_exchange", "exchange"),
        Index("idx_trades_entry_time", "entry_time"),
        Index("idx_trades_exit_time", "exit_time"),
        Index("idx_trades_symbol_entry", "symbol", "entry_time"),
        Index("idx_trades_action", "action"),
        Index("idx_trades_pnl", "realized_pnl"),
    )


class PositionModel(Base):
    """SQLAlchemy model for current positions."""

    __tablename__ = "positions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)
    entry_price = Column(Numeric(20, 8), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    leverage = Column(Numeric(5, 2), default=1.0)
    margin = Column(Numeric(20, 8), nullable=False)
    current_price = Column(Numeric(20, 8))
    unrealized_pnl = Column(Numeric(20, 8))
    unrealized_pnl_pct = Column(Numeric(10, 4))
    liquidation_price = Column(Numeric(20, 8))
    stop_loss = Column(Numeric(20, 8))
    take_profit = Column(Numeric(20, 8))
    trailing_stop = Column(Numeric(20, 8))
    status = Column(String(20), default="open")
    trade_id = Column(String(100))
    entry_time = Column(DateTime(timezone=True), nullable=False, default=func.now())
    last_updated = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), default=func.now())

    __table_args__ = (
        CheckConstraint("side IN ('long', 'short')", name="check_position_side"),
        CheckConstraint("status IN ('open', 'closed', 'liquidated')", name="check_position_status"),
        Index("idx_positions_symbol", "symbol"),
        Index("idx_positions_status", "status"),
        Index("idx_positions_exchange", "exchange"),
        Index("idx_positions_symbol_status", "symbol", "status"),
        Index("idx_positions_trade_id", "trade_id"),
    )


class LLMAnalysisLogModel(Base):
    """SQLAlchemy model for LLM analysis logs."""

    __tablename__ = "llm_analysis_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    analysis_type = Column(String(50), nullable=False)
    symbol = Column(String(20))
    exchange = Column(String(50))
    prompt_template = Column(String(100))
    prompt_tokens = Column(Integer)
    prompt_text = Column(Text)
    response_text = Column(Text)
    response_tokens = Column(Integer)
    response_json = Column(JSONB)
    total_tokens = Column(Integer)
    estimated_cost = Column(Numeric(10, 6))
    model_name = Column(String(100))
    temperature = Column(Numeric(3, 2))
    status = Column(String(20), default="success")
    error_message = Column(Text)
    execution_time_ms = Column(Integer)
    created_at = Column(DateTime(timezone=True), default=func.now())

    __table_args__ = (
        CheckConstraint("status IN ('success', 'error', 'timeout')", name="check_llm_status"),
        Index("idx_llm_log_type", "analysis_type"),
        Index("idx_llm_log_symbol", "symbol"),
        Index("idx_llm_log_created", "created_at"),
        Index("idx_llm_log_status", "status"),
        Index("idx_llm_log_type_created", "analysis_type", "created_at"),
    )


class SystemEventModel(Base):
    """SQLAlchemy model for system events."""

    __tablename__ = "system_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    event_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(JSONB)
    symbol = Column(String(20))
    exchange = Column(String(50))
    trade_id = Column(String(100))
    position_id = Column(BigInteger)
    stack_trace = Column(Text)
    created_at = Column(DateTime(timezone=True), default=func.now())

    __table_args__ = (
        CheckConstraint(
            "severity IN ('debug', 'info', 'warning', 'error', 'critical')",
            name="check_event_severity",
        ),
        Index("idx_events_type", "event_type"),
        Index("idx_events_severity", "severity"),
        Index("idx_events_created", "created_at"),
        Index("idx_events_type_created", "event_type", "created_at"),
        Index("idx_events_symbol", "symbol"),
        Index("idx_events_trade_id", "trade_id"),
    )


class DailyPerformanceModel(Base):
    """SQLAlchemy model for daily performance metrics."""

    __tablename__ = "daily_performance"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True)
    exchange = Column(String(50), nullable=False)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Numeric(5, 2))
    gross_profit = Column(Numeric(20, 8), default=0)
    gross_loss = Column(Numeric(20, 8), default=0)
    net_pnl = Column(Numeric(20, 8), default=0)
    net_pnl_pct = Column(Numeric(10, 4))
    total_fees = Column(Numeric(20, 8), default=0)
    avg_position_size = Column(Numeric(20, 8))
    max_position_size = Column(Numeric(20, 8))
    total_volume = Column(Numeric(20, 8))
    max_drawdown = Column(Numeric(10, 4))
    max_drawdown_amount = Column(Numeric(20, 8))
    sharpe_ratio = Column(Numeric(10, 4))
    profit_factor = Column(Numeric(10, 4))
    starting_balance = Column(Numeric(20, 8))
    ending_balance = Column(Numeric(20, 8))
    peak_balance = Column(Numeric(20, 8))
    avg_trade_duration_minutes = Column(Integer)
    longest_winning_streak = Column(Integer)
    longest_losing_streak = Column(Integer)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_daily_perf_date", "date"),
        Index("idx_daily_perf_exchange", "exchange"),
        Index("idx_daily_perf_date_exchange", "date", "exchange"),
        Index("idx_daily_perf_pnl", "net_pnl"),
    )


# ============================================================================
# Database Manager
# ============================================================================


class DatabaseManager:
    """
    Async database connection manager with connection pooling.

    Handles database lifecycle operations including connection, disconnection,
    and session management.

    Example:
        ```python
        db_manager = DatabaseManager("postgresql+asyncpg://user:pass@localhost/db")
        await db_manager.connect()

        async with db_manager.get_session() as session:
            # Use session for queries
            result = await session.execute(select(OHLCVModel))

        await db_manager.disconnect()
        ```
    """

    def __init__(
        self,
        database_url: str,
        pool_size: int = 5,
        max_overflow: int = 10,
        echo: bool = False,
    ):
        """
        Initialize database manager.

        Args:
            database_url: Async database connection URL
            pool_size: Connection pool size
            max_overflow: Maximum connection overflow
            echo: Enable SQL query echo (for debugging)
        """
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.echo = echo
        self._engine: AsyncEngine | None = None
        self._session_maker: async_sessionmaker | None = None
        logger.debug(
            "database_manager_initialized",
            pool_size=pool_size,
            max_overflow=max_overflow,
        )

    async def connect(self) -> None:
        """
        Establish database connection and create engine.

        Creates the async engine and session maker for database operations.
        """
        if self._engine is not None:
            logger.warning("database_already_connected")
            return

        self._engine = create_async_engine(
            self.database_url,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            echo=self.echo,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,  # Recycle connections after 1 hour
        )

        self._session_maker = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        logger.info("database_connected", url=self.database_url.split("@")[-1])

    async def disconnect(self) -> None:
        """
        Close database connection and dispose of engine.
        """
        if self._engine is None:
            logger.warning("database_not_connected")
            return

        await self._engine.dispose()
        self._engine = None
        self._session_maker = None
        logger.info("database_disconnected")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get async database session (context manager).

        Yields:
            AsyncSession for database operations

        Raises:
            RuntimeError: If database is not connected

        Example:
            ```python
            async with db_manager.get_session() as session:
                result = await session.execute(select(OHLCVModel).limit(10))
                bars = result.scalars().all()
            ```
        """
        if self._session_maker is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        async with self._session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error("session_error", error=str(e), exc_info=True)
                raise
            finally:
                await session.close()


# ============================================================================
# Repository Base Class
# ============================================================================


class BaseRepository:
    """Base repository with common functionality."""

    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session


# ============================================================================
# OHLCV Repository
# ============================================================================


class OHLCVRepository(BaseRepository):
    """
    Repository for OHLCV data operations.

    Provides methods for inserting, querying, and managing candlestick data.
    """

    async def insert_many(self, bars: list[OHLCVBar]) -> int:
        """
        Insert multiple OHLCV bars in batch.

        Args:
            bars: List of OHLCVBar objects to insert

        Returns:
            Number of bars inserted

        Example:
            ```python
            bars = [
                OHLCVBar(
                    symbol="BTCUSDT",
                    exchange="binance",
                    timeframe="1h",
                    timestamp=datetime.now(),
                    open=Decimal("50000"),
                    high=Decimal("51000"),
                    low=Decimal("49500"),
                    close=Decimal("50500"),
                    volume=Decimal("100.5"),
                )
            ]
            count = await repo.insert_many(bars)
            ```
        """
        if not bars:
            return 0

        models = [
            OHLCVModel(
                symbol=bar.symbol,
                exchange=bar.exchange,
                timeframe=bar.timeframe,
                timestamp=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                quote_volume=bar.quote_volume,
                trades_count=bar.trades_count,
            )
            for bar in bars
        ]

        self.session.add_all(models)
        await self.session.flush()

        logger.debug(
            "ohlcv_inserted",
            count=len(bars),
            symbol=bars[0].symbol if bars else None,
            timeframe=bars[0].timeframe if bars else None,
        )
        return len(bars)

    async def get_range(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        exchange: str = "binance",
    ) -> list[OHLCVBar]:
        """
        Get OHLCV bars within a time range.

        Args:
            symbol: Trading symbol
            timeframe: Candlestick timeframe
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
            exchange: Exchange name

        Returns:
            List of OHLCVBar objects
        """
        stmt = (
            select(OHLCVModel)
            .where(
                and_(
                    OHLCVModel.symbol == symbol,
                    OHLCVModel.exchange == exchange,
                    OHLCVModel.timeframe == timeframe,
                    OHLCVModel.timestamp >= start,
                    OHLCVModel.timestamp <= end,
                )
            )
            .order_by(OHLCVModel.timestamp)
        )

        result = await self.session.execute(stmt)
        models = result.scalars().all()

        return [self._model_to_bar(model) for model in models]

    async def get_latest(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        exchange: str = "binance",
    ) -> list[OHLCVBar]:
        """
        Get latest OHLCV bars for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Candlestick timeframe
            limit: Maximum number of bars to return
            exchange: Exchange name

        Returns:
            List of OHLCVBar objects (newest first)
        """
        stmt = (
            select(OHLCVModel)
            .where(
                and_(
                    OHLCVModel.symbol == symbol,
                    OHLCVModel.exchange == exchange,
                    OHLCVModel.timeframe == timeframe,
                )
            )
            .order_by(desc(OHLCVModel.timestamp))
            .limit(limit)
        )

        result = await self.session.execute(stmt)
        models = result.scalars().all()

        return [self._model_to_bar(model) for model in models]

    async def upsert_many(self, bars: list[OHLCVBar]) -> int:
        """
        Insert or update OHLCV bars (upsert).

        If a bar with the same symbol/exchange/timeframe/timestamp exists,
        it will be updated. Otherwise, it will be inserted.

        Args:
            bars: List of OHLCVBar objects

        Returns:
            Number of bars processed
        """
        if not bars:
            return 0

        inserted = 0
        for bar in bars:
            # Check if exists
            stmt = select(OHLCVModel).where(
                and_(
                    OHLCVModel.symbol == bar.symbol,
                    OHLCVModel.exchange == bar.exchange,
                    OHLCVModel.timeframe == bar.timeframe,
                    OHLCVModel.timestamp == bar.timestamp,
                )
            )
            result = await self.session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                # Update
                existing.open = bar.open
                existing.high = bar.high
                existing.low = bar.low
                existing.close = bar.close
                existing.volume = bar.volume
                existing.quote_volume = bar.quote_volume
                existing.trades_count = bar.trades_count
            else:
                # Insert
                model = OHLCVModel(
                    symbol=bar.symbol,
                    exchange=bar.exchange,
                    timeframe=bar.timeframe,
                    timestamp=bar.timestamp,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    quote_volume=bar.quote_volume,
                    trades_count=bar.trades_count,
                )
                self.session.add(model)
            inserted += 1

        await self.session.flush()
        logger.debug("ohlcv_upserted", count=inserted)
        return inserted

    async def get_gaps(
        self,
        symbol: str,
        timeframe: str,
        exchange: str = "binance",
    ) -> list[tuple[datetime, datetime]]:
        """
        Find time gaps in OHLCV data.

        Detects missing candles by comparing expected intervals with actual data.

        Args:
            symbol: Trading symbol
            timeframe: Candlestick timeframe
            exchange: Exchange name

        Returns:
            List of (gap_start, gap_end) tuples
        """
        # Map timeframe to seconds
        timeframe_seconds = {
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "6h": 21600,
            "8h": 28800,
            "12h": 43200,
            "1d": 86400,
            "3d": 259200,
            "1w": 604800,
        }

        interval = timeframe_seconds.get(timeframe)
        if not interval:
            logger.warning("unsupported_timeframe", timeframe=timeframe)
            return []

        # Get all timestamps
        stmt = (
            select(OHLCVModel.timestamp)
            .where(
                and_(
                    OHLCVModel.symbol == symbol,
                    OHLCVModel.exchange == exchange,
                    OHLCVModel.timeframe == timeframe,
                )
            )
            .order_by(OHLCVModel.timestamp)
        )

        result = await self.session.execute(stmt)
        timestamps = [row[0] for row in result.all()]

        if len(timestamps) < 2:
            return []

        # Find gaps
        gaps = []
        for i in range(len(timestamps) - 1):
            current = timestamps[i]
            next_ts = timestamps[i + 1]
            expected_next = current + timedelta(seconds=interval)

            if next_ts > expected_next:
                gaps.append((current, next_ts))

        logger.debug(
            "ohlcv_gaps_found",
            symbol=symbol,
            timeframe=timeframe,
            gap_count=len(gaps),
        )
        return gaps

    @staticmethod
    def _model_to_bar(model: OHLCVModel) -> OHLCVBar:
        """Convert SQLAlchemy model to OHLCVBar dataclass."""
        return OHLCVBar(
            id=model.id,
            symbol=model.symbol,
            exchange=model.exchange,
            timeframe=model.timeframe,
            timestamp=model.timestamp,
            open=model.open,
            high=model.high,
            low=model.low,
            close=model.close,
            volume=model.volume,
            quote_volume=model.quote_volume,
            trades_count=model.trades_count,
            created_at=model.created_at,
        )


# ============================================================================
# Trade Repository
# ============================================================================


class TradeRepository(BaseRepository):
    """
    Repository for trade operations.

    Manages trade records including insertion, updates, and statistics.
    """

    async def insert(self, trade: Trade) -> int:
        """
        Insert a new trade record.

        Args:
            trade: Trade object to insert

        Returns:
            Trade ID of inserted record
        """
        model = TradeModel(
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            exchange=trade.exchange,
            side=trade.side,
            action=trade.action,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            quantity=trade.quantity,
            leverage=trade.leverage,
            realized_pnl=trade.realized_pnl,
            realized_pnl_pct=trade.realized_pnl_pct,
            fee=trade.fee,
            signal_score=trade.signal_score,
            technical_score=trade.technical_score,
            llm_score=trade.llm_score,
            xgb_confidence=trade.xgb_confidence,
            stop_loss=trade.stop_loss,
            take_profit=trade.take_profit,
            position_size_pct=trade.position_size_pct,
            decision_reasons=trade.decision_reasons,
            llm_analysis=trade.llm_analysis,
            entry_time=trade.entry_time,
            exit_time=trade.exit_time,
        )

        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)

        logger.info(
            "trade_inserted",
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            side=trade.side,
            action=trade.action,
        )
        return model.id

    async def update(self, trade_id: int, **kwargs) -> None:
        """
        Update trade record by ID.

        Args:
            trade_id: Database ID of trade
            **kwargs: Fields to update
        """
        stmt = update(TradeModel).where(TradeModel.id == trade_id).values(**kwargs)
        await self.session.execute(stmt)
        await self.session.flush()

        logger.debug("trade_updated", trade_id=trade_id, fields=list(kwargs.keys()))

    async def get_by_id(self, trade_id: int) -> Trade | None:
        """
        Get trade by database ID.

        Args:
            trade_id: Database ID of trade

        Returns:
            Trade object or None if not found
        """
        stmt = select(TradeModel).where(TradeModel.id == trade_id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()

        return self._model_to_trade(model) if model else None

    async def get_recent(self, limit: int = 100) -> list[Trade]:
        """
        Get recent trades.

        Args:
            limit: Maximum number of trades to return

        Returns:
            List of Trade objects (newest first)
        """
        stmt = select(TradeModel).order_by(desc(TradeModel.entry_time)).limit(limit)
        result = await self.session.execute(stmt)
        models = result.scalars().all()

        return [self._model_to_trade(model) for model in models]

    async def get_by_symbol(
        self,
        symbol: str,
        limit: int = 100,
        exchange: str = "binance",
    ) -> list[Trade]:
        """
        Get trades for a specific symbol.

        Args:
            symbol: Trading symbol
            limit: Maximum number of trades to return
            exchange: Exchange name

        Returns:
            List of Trade objects (newest first)
        """
        stmt = (
            select(TradeModel)
            .where(
                and_(
                    TradeModel.symbol == symbol,
                    TradeModel.exchange == exchange,
                )
            )
            .order_by(desc(TradeModel.entry_time))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        models = result.scalars().all()

        return [self._model_to_trade(model) for model in models]

    async def get_statistics(
        self,
        days: int = 30,
        exchange: str = "binance",
    ) -> TradeStatistics:
        """
        Get aggregated trade statistics for a time period.

        Args:
            days: Number of days to analyze
            exchange: Exchange name

        Returns:
            TradeStatistics object with aggregated metrics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Get closed trades
        stmt = select(TradeModel).where(
            and_(
                TradeModel.exchange == exchange,
                TradeModel.action == "close",
                TradeModel.exit_time >= cutoff_date,
                TradeModel.realized_pnl.isnot(None),
            )
        )
        result = await self.session.execute(stmt)
        trades = result.scalars().all()

        if not trades:
            return TradeStatistics(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=Decimal("0"),
                gross_profit=Decimal("0"),
                gross_loss=Decimal("0"),
                net_pnl=Decimal("0"),
                total_fees=Decimal("0"),
                avg_win=Decimal("0"),
                avg_loss=Decimal("0"),
                largest_win=Decimal("0"),
                largest_loss=Decimal("0"),
            )

        # Calculate statistics
        winning_trades = [t for t in trades if t.realized_pnl > 0]
        losing_trades = [t for t in trades if t.realized_pnl <= 0]

        total_trades = len(trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (
            Decimal(win_count) / Decimal(total_trades) * 100 if total_trades > 0 else Decimal("0")
        )

        gross_profit = (
            sum(t.realized_pnl for t in winning_trades) if winning_trades else Decimal("0")
        )
        gross_loss = (
            abs(sum(t.realized_pnl for t in losing_trades)) if losing_trades else Decimal("0")
        )
        net_pnl = sum(t.realized_pnl for t in trades)
        total_fees = sum(t.fee for t in trades)

        avg_win = gross_profit / len(winning_trades) if winning_trades else Decimal("0")
        avg_loss = gross_loss / len(losing_trades) if losing_trades else Decimal("0")

        largest_win = max((t.realized_pnl for t in winning_trades), default=Decimal("0"))
        largest_loss = min((t.realized_pnl for t in losing_trades), default=Decimal("0"))

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

        return TradeStatistics(
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_pnl=net_pnl,
            total_fees=total_fees,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
        )

    @staticmethod
    def _model_to_trade(model: TradeModel) -> Trade:
        """Convert SQLAlchemy model to Trade dataclass."""
        return Trade(
            id=model.id,
            trade_id=model.trade_id,
            symbol=model.symbol,
            exchange=model.exchange,
            side=model.side,
            action=model.action,
            entry_price=model.entry_price,
            exit_price=model.exit_price,
            quantity=model.quantity,
            leverage=model.leverage,
            realized_pnl=model.realized_pnl,
            realized_pnl_pct=model.realized_pnl_pct,
            fee=model.fee,
            signal_score=model.signal_score,
            technical_score=model.technical_score,
            llm_score=model.llm_score,
            xgb_confidence=model.xgb_confidence,
            stop_loss=model.stop_loss,
            take_profit=model.take_profit,
            position_size_pct=model.position_size_pct,
            decision_reasons=model.decision_reasons,
            llm_analysis=model.llm_analysis,
            entry_time=model.entry_time,
            exit_time=model.exit_time,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )


# ============================================================================
# Position Repository
# ============================================================================


class PositionRepository(BaseRepository):
    """
    Repository for position operations.

    Manages current and historical positions.
    """

    async def get_open_positions(self, exchange: str = "binance") -> list[Position]:
        """
        Get all open positions.

        Args:
            exchange: Exchange name

        Returns:
            List of Position objects
        """
        stmt = select(PositionModel).where(
            and_(
                PositionModel.exchange == exchange,
                PositionModel.status == "open",
            )
        )
        result = await self.session.execute(stmt)
        models = result.scalars().all()

        return [self._model_to_position(model) for model in models]

    async def get_position(
        self,
        symbol: str,
        exchange: str = "binance",
    ) -> Position | None:
        """
        Get open position for a symbol.

        Args:
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Position object or None if not found
        """
        stmt = select(PositionModel).where(
            and_(
                PositionModel.symbol == symbol,
                PositionModel.exchange == exchange,
                PositionModel.status == "open",
            )
        )
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()

        return self._model_to_position(model) if model else None

    async def update_position(self, position: Position) -> None:
        """
        Update or insert position.

        Args:
            position: Position object to update/insert
        """
        if position.id:
            # Update existing
            stmt = (
                update(PositionModel)
                .where(PositionModel.id == position.id)
                .values(
                    current_price=position.current_price,
                    unrealized_pnl=position.unrealized_pnl,
                    unrealized_pnl_pct=position.unrealized_pnl_pct,
                    liquidation_price=position.liquidation_price,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                    trailing_stop=position.trailing_stop,
                    status=position.status,
                )
            )
            await self.session.execute(stmt)
            logger.debug("position_updated", position_id=position.id, symbol=position.symbol)
        else:
            # Insert new
            model = PositionModel(
                symbol=position.symbol,
                exchange=position.exchange,
                side=position.side,
                entry_price=position.entry_price,
                quantity=position.quantity,
                leverage=position.leverage,
                margin=position.margin,
                current_price=position.current_price,
                unrealized_pnl=position.unrealized_pnl,
                unrealized_pnl_pct=position.unrealized_pnl_pct,
                liquidation_price=position.liquidation_price,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                trailing_stop=position.trailing_stop,
                status=position.status,
                trade_id=position.trade_id,
                entry_time=position.entry_time,
            )
            self.session.add(model)
            await self.session.flush()
            await self.session.refresh(model)
            logger.info("position_created", symbol=position.symbol, side=position.side)

    async def close_position(self, position_id: int) -> None:
        """
        Close a position by ID.

        Args:
            position_id: Database ID of position to close
        """
        stmt = update(PositionModel).where(PositionModel.id == position_id).values(status="closed")
        await self.session.execute(stmt)
        await self.session.flush()

        logger.info("position_closed", position_id=position_id)

    @staticmethod
    def _model_to_position(model: PositionModel) -> Position:
        """Convert SQLAlchemy model to Position dataclass."""
        return Position(
            id=model.id,
            symbol=model.symbol,
            exchange=model.exchange,
            side=model.side,
            entry_price=model.entry_price,
            quantity=model.quantity,
            leverage=model.leverage,
            margin=model.margin,
            current_price=model.current_price,
            unrealized_pnl=model.unrealized_pnl,
            unrealized_pnl_pct=model.unrealized_pnl_pct,
            liquidation_price=model.liquidation_price,
            stop_loss=model.stop_loss,
            take_profit=model.take_profit,
            trailing_stop=model.trailing_stop,
            status=model.status,
            trade_id=model.trade_id,
            entry_time=model.entry_time,
            last_updated=model.last_updated,
            created_at=model.created_at,
        )


# ============================================================================
# System Event Repository
# ============================================================================


class SystemEventRepository(BaseRepository):
    """
    Repository for system event logging.

    Provides structured logging to database for auditing and debugging.
    """

    async def log_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        details: dict | None = None,
        symbol: str | None = None,
        exchange: str | None = None,
        trade_id: str | None = None,
        position_id: int | None = None,
        stack_trace: str | None = None,
    ) -> None:
        """
        Log a system event.

        Args:
            event_type: Event type identifier
            severity: Log severity (debug, info, warning, error, critical)
            message: Event message
            details: Additional event details
            symbol: Related trading symbol
            exchange: Related exchange
            trade_id: Related trade ID
            position_id: Related position ID
            stack_trace: Exception stack trace
        """
        model = SystemEventModel(
            event_type=event_type,
            severity=severity,
            message=message,
            details=details,
            symbol=symbol,
            exchange=exchange,
            trade_id=trade_id,
            position_id=position_id,
            stack_trace=stack_trace,
        )

        self.session.add(model)
        await self.session.flush()

        logger.debug(
            "system_event_logged",
            event_type=event_type,
            severity=severity,
        )

    async def get_recent_events(
        self,
        severity: str | None = None,
        limit: int = 100,
    ) -> list[SystemEvent]:
        """
        Get recent system events.

        Args:
            severity: Filter by severity level (optional)
            limit: Maximum number of events to return

        Returns:
            List of SystemEvent objects (newest first)
        """
        stmt = select(SystemEventModel)

        if severity:
            stmt = stmt.where(SystemEventModel.severity == severity)

        stmt = stmt.order_by(desc(SystemEventModel.created_at)).limit(limit)

        result = await self.session.execute(stmt)
        models = result.scalars().all()

        return [self._model_to_event(model) for model in models]

    @staticmethod
    def _model_to_event(model: SystemEventModel) -> SystemEvent:
        """Convert SQLAlchemy model to SystemEvent dataclass."""
        return SystemEvent(
            id=model.id,
            event_type=model.event_type,
            severity=model.severity,
            message=model.message,
            details=model.details,
            symbol=model.symbol,
            exchange=model.exchange,
            trade_id=model.trade_id,
            position_id=model.position_id,
            stack_trace=model.stack_trace,
            created_at=model.created_at,
        )


# ============================================================================
# Module Initialization
# ============================================================================


__all__ = [
    # Models
    "Base",
    "OHLCVModel",
    "TradeModel",
    "PositionModel",
    "LLMAnalysisLogModel",
    "SystemEventModel",
    "DailyPerformanceModel",
    # Data classes
    "OHLCVBar",
    "Trade",
    "Position",
    "SystemEvent",
    "TradeStatistics",
    # Database
    "DatabaseManager",
    # Repositories
    "OHLCVRepository",
    "TradeRepository",
    "PositionRepository",
    "SystemEventRepository",
]
