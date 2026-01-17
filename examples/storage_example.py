"""
Example usage of the database storage layer.

This example demonstrates:
1. Connecting to the database
2. Inserting OHLCV data
3. Querying historical data
4. Managing trades and positions
5. Logging system events
6. Getting trade statistics
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

from src.iftb.config import get_settings
from src.iftb.data.storage import (
    DatabaseManager,
    OHLCVBar,
    OHLCVRepository,
    Position,
    PositionRepository,
    SystemEventRepository,
    Trade,
    TradeRepository,
)
from src.iftb.utils import LogConfig, get_logger, setup_logging

# Setup logging
setup_logging(LogConfig(level="INFO", format="pretty"))
logger = get_logger(__name__)


async def example_ohlcv_operations(db_manager: DatabaseManager):
    """Demonstrate OHLCV data operations."""
    logger.info("ohlcv_example_start")

    async with db_manager.get_session() as session:
        repo = OHLCVRepository(session)

        # Insert sample OHLCV data
        bars = []
        base_time = datetime.utcnow() - timedelta(hours=100)

        for i in range(100):
            bar = OHLCVBar(
                symbol="BTCUSDT",
                exchange="binance",
                timeframe="1h",
                timestamp=base_time + timedelta(hours=i),
                open=Decimal("50000") + Decimal(i * 10),
                high=Decimal("50500") + Decimal(i * 10),
                low=Decimal("49500") + Decimal(i * 10),
                close=Decimal("50200") + Decimal(i * 10),
                volume=Decimal("100.5"),
                quote_volume=Decimal("5025000"),
                trades_count=1000,
            )
            bars.append(bar)

        # Insert all bars
        count = await repo.insert_many(bars)
        logger.info("ohlcv_inserted", count=count)

        # Get latest bars
        latest = await repo.get_latest("BTCUSDT", "1h", limit=10)
        logger.info("latest_bars_fetched", count=len(latest))
        for bar in latest[:3]:
            logger.info(
                "bar_info",
                timestamp=bar.timestamp,
                open=float(bar.open),
                close=float(bar.close),
            )

        # Get bars in a time range
        start = base_time + timedelta(hours=50)
        end = base_time + timedelta(hours=60)
        range_bars = await repo.get_range("BTCUSDT", "1h", start, end)
        logger.info("range_bars_fetched", count=len(range_bars), start=start, end=end)

        # Check for gaps
        gaps = await repo.get_gaps("BTCUSDT", "1h")
        logger.info("gaps_found", gap_count=len(gaps))


async def example_trade_operations(db_manager: DatabaseManager):
    """Demonstrate trade operations."""
    logger.info("trade_example_start")

    async with db_manager.get_session() as session:
        repo = TradeRepository(session)

        # Insert a new trade (opening position)
        open_trade = Trade(
            trade_id="TRD-001",
            symbol="BTCUSDT",
            exchange="binance",
            side="long",
            action="open",
            entry_price=Decimal("50000"),
            quantity=Decimal("0.01"),
            leverage=Decimal("5"),
            stop_loss=Decimal("49000"),
            take_profit=Decimal("52000"),
            signal_score=Decimal("0.85"),
            technical_score=Decimal("0.80"),
            llm_score=Decimal("0.90"),
            entry_time=datetime.utcnow(),
            decision_reasons={
                "trend": "bullish",
                "rsi": 65.5,
                "macd": "golden_cross",
            },
        )

        trade_id = await repo.insert(open_trade)
        logger.info("trade_opened", trade_id=trade_id)

        # Simulate closing the trade
        close_trade = Trade(
            trade_id="TRD-002",
            symbol="BTCUSDT",
            exchange="binance",
            side="long",
            action="close",
            entry_price=Decimal("50000"),
            exit_price=Decimal("51500"),
            quantity=Decimal("0.01"),
            leverage=Decimal("5"),
            realized_pnl=Decimal("75"),  # (51500 - 50000) * 0.01 * 5
            realized_pnl_pct=Decimal("15"),  # 15% with 5x leverage
            fee=Decimal("2.5"),
            entry_time=datetime.utcnow() - timedelta(hours=2),
            exit_time=datetime.utcnow(),
        )

        trade_id = await repo.insert(close_trade)
        logger.info("trade_closed", trade_id=trade_id, pnl=float(close_trade.realized_pnl))

        # Get recent trades
        recent = await repo.get_recent(limit=10)
        logger.info("recent_trades_fetched", count=len(recent))

        # Get trades for specific symbol
        symbol_trades = await repo.get_by_symbol("BTCUSDT", limit=10)
        logger.info("symbol_trades_fetched", symbol="BTCUSDT", count=len(symbol_trades))

        # Get trade statistics
        stats = await repo.get_statistics(days=30)
        logger.info(
            "trade_statistics",
            total_trades=stats.total_trades,
            winning_trades=stats.winning_trades,
            win_rate=float(stats.win_rate),
            net_pnl=float(stats.net_pnl),
        )


async def example_position_operations(db_manager: DatabaseManager):
    """Demonstrate position operations."""
    logger.info("position_example_start")

    async with db_manager.get_session() as session:
        repo = PositionRepository(session)

        # Create a new position
        position = Position(
            symbol="ETHUSDT",
            exchange="binance",
            side="long",
            entry_price=Decimal("3000"),
            quantity=Decimal("0.5"),
            leverage=Decimal("3"),
            margin=Decimal("500"),
            current_price=Decimal("3100"),
            unrealized_pnl=Decimal("150"),  # (3100 - 3000) * 0.5 * 3
            unrealized_pnl_pct=Decimal("30"),
            stop_loss=Decimal("2900"),
            take_profit=Decimal("3200"),
            trade_id="TRD-003",
            entry_time=datetime.utcnow() - timedelta(hours=1),
        )

        await repo.update_position(position)
        logger.info("position_created", symbol=position.symbol)

        # Get open positions
        open_positions = await repo.get_open_positions()
        logger.info("open_positions_fetched", count=len(open_positions))

        for pos in open_positions:
            logger.info(
                "position_info",
                symbol=pos.symbol,
                side=pos.side,
                entry_price=float(pos.entry_price),
                unrealized_pnl=float(pos.unrealized_pnl) if pos.unrealized_pnl else None,
            )

        # Get specific position
        eth_position = await repo.get_position("ETHUSDT")
        if eth_position:
            logger.info(
                "position_retrieved",
                symbol=eth_position.symbol,
                status=eth_position.status,
            )


async def example_event_logging(db_manager: DatabaseManager):
    """Demonstrate system event logging."""
    logger.info("event_logging_example_start")

    async with db_manager.get_session() as session:
        repo = SystemEventRepository(session)

        # Log different types of events
        await repo.log_event(
            event_type="system_startup",
            severity="info",
            message="Trading bot started successfully",
            details={"version": "1.0.0", "environment": "production"},
        )

        await repo.log_event(
            event_type="trade_executed",
            severity="info",
            message="Trade executed successfully",
            symbol="BTCUSDT",
            exchange="binance",
            trade_id="TRD-001",
            details={"side": "long", "price": 50000, "quantity": 0.01},
        )

        await repo.log_event(
            event_type="api_error",
            severity="error",
            message="Exchange API connection failed",
            exchange="binance",
            details={"error_code": "ETIMEDOUT", "retry_count": 3},
            stack_trace="Traceback (most recent call last)...",
        )

        logger.info("events_logged", count=3)

        # Get recent events
        all_events = await repo.get_recent_events(limit=10)
        logger.info("all_events_fetched", count=len(all_events))

        # Get only error events
        errors = await repo.get_recent_events(severity="error")
        logger.info("error_events_fetched", count=len(errors))


async def main():
    """Main example function."""
    logger.info("storage_example_start")

    # Load settings
    settings = get_settings()

    # Create database manager
    db_manager = DatabaseManager(
        database_url=settings.database.get_async_url(),
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
    )

    try:
        # Connect to database
        await db_manager.connect()
        logger.info("database_connected")

        # Run examples
        await example_ohlcv_operations(db_manager)
        await example_trade_operations(db_manager)
        await example_position_operations(db_manager)
        await example_event_logging(db_manager)

        logger.info("storage_example_complete")

    except Exception as e:
        logger.error("example_error", error=str(e), exc_info=True)
        raise

    finally:
        # Disconnect from database
        await db_manager.disconnect()
        logger.info("database_disconnected")


if __name__ == "__main__":
    asyncio.run(main())
