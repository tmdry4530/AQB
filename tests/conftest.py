"""
Shared pytest fixtures for AQB test suite.

This module provides fixtures for:
- Database connections (async sessions, test database)
- Redis mock instances
- Exchange mock clients (CCXT)
- LLM mock responses (Claude)
- Sample market data (OHLCV, trades)
- Test settings overrides
"""

import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from iftb.config import Settings

# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "live: Tests requiring live API access")


# ============================================================================
# Event Loop Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Settings Fixtures
# ============================================================================

@pytest.fixture
def test_settings() -> Settings:
    """Override settings for testing environment."""
    return Settings(
        # Database
        DATABASE_URL="sqlite+aiosqlite:///:memory:",

        # Redis
        REDIS_HOST="localhost",
        REDIS_PORT=6379,
        REDIS_DB=1,  # Use different DB for tests

        # API Keys (mocked)
        ANTHROPIC_API_KEY="test-anthropic-key",

        # Exchange
        EXCHANGE_NAME="binance",
        EXCHANGE_TESTNET=True,

        # Environment
        ENVIRONMENT="test",
        DEBUG=True,

        # Rate Limits
        RATE_LIMIT_REQUESTS=100,
        RATE_LIMIT_PERIOD=60,
    )


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest_asyncio.fixture
async def async_engine(test_settings: Settings):
    """Create async database engine for testing."""
    engine = create_async_engine(
        test_settings.DATABASE_URL,
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False} if "sqlite" in test_settings.DATABASE_URL else {},
    )

    # Create tables
    # from app.db.base import Base
    # async with engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Drop tables
    # async with engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture
async def async_session_maker(async_engine):
    """Create async session maker for testing."""
    return async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@pytest_asyncio.fixture
async def db_session(async_session_maker) -> AsyncGenerator[AsyncSession, None]:
    """Create async database session for testing."""
    async with async_session_maker() as session:
        yield session
        await session.rollback()


# ============================================================================
# Redis Fixtures
# ============================================================================

@pytest.fixture
def mock_redis() -> MagicMock:
    """Create mock Redis client."""
    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.setex = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.exists = AsyncMock(return_value=False)
    redis.expire = AsyncMock(return_value=True)
    redis.ttl = AsyncMock(return_value=-1)
    redis.keys = AsyncMock(return_value=[])
    redis.flushdb = AsyncMock(return_value=True)
    redis.ping = AsyncMock(return_value=True)
    redis.close = AsyncMock()
    return redis


# ============================================================================
# Exchange Fixtures (CCXT)
# ============================================================================

@pytest.fixture
def mock_ccxt_client() -> MagicMock:
    """Create mock CCXT exchange client."""
    exchange = MagicMock()

    # Market data methods
    exchange.fetch_ohlcv = AsyncMock(return_value=[])
    exchange.fetch_ticker = AsyncMock(return_value={})
    exchange.fetch_order_book = AsyncMock(return_value={"bids": [], "asks": []})
    exchange.fetch_trades = AsyncMock(return_value=[])

    # Trading methods
    exchange.create_order = AsyncMock(return_value={"id": "test-order-123"})
    exchange.cancel_order = AsyncMock(return_value={"id": "test-order-123"})
    exchange.fetch_order = AsyncMock(return_value={})
    exchange.fetch_orders = AsyncMock(return_value=[])
    exchange.fetch_open_orders = AsyncMock(return_value=[])
    exchange.fetch_closed_orders = AsyncMock(return_value=[])

    # Account methods
    exchange.fetch_balance = AsyncMock(return_value={"free": {}, "used": {}, "total": {}})
    exchange.fetch_positions = AsyncMock(return_value=[])

    # Exchange info
    exchange.load_markets = AsyncMock(return_value={})
    exchange.markets = {}
    exchange.symbols = []
    exchange.id = "binance"
    exchange.has = {
        "fetchOHLCV": True,
        "fetchTicker": True,
        "fetchOrderBook": True,
        "createOrder": True,
        "cancelOrder": True,
        "fetchBalance": True,
    }

    return exchange


@pytest.fixture
def sample_ohlcv_data() -> list:
    """Generate sample OHLCV data for testing."""
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    data = []

    for i in range(100):
        timestamp = int((base_time + timedelta(hours=i)).timestamp() * 1000)
        open_price = 100 + i * 0.1
        high_price = open_price + 0.5
        low_price = open_price - 0.5
        close_price = open_price + 0.2
        volume = 1000 + i * 10

        data.append([timestamp, open_price, high_price, low_price, close_price, volume])

    return data


@pytest.fixture
def sample_ohlcv_dataframe(sample_ohlcv_data) -> pd.DataFrame:
    """Generate sample OHLCV DataFrame for testing."""
    df = pd.DataFrame(
        sample_ohlcv_data,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


# ============================================================================
# LLM Fixtures (Claude)
# ============================================================================

@pytest.fixture
def mock_anthropic_client() -> MagicMock:
    """Create mock Anthropic Claude client."""
    client = MagicMock()

    # Mock message response
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="This is a test response from Claude.")]
    mock_response.id = "msg_test123"
    mock_response.model = "claude-sonnet-4-5-20250929"
    mock_response.role = "assistant"
    mock_response.stop_reason = "end_turn"
    mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

    # Mock messages.create method
    client.messages.create = AsyncMock(return_value=mock_response)

    return client


@pytest.fixture
def sample_llm_analysis() -> dict:
    """Sample LLM analysis response for testing."""
    return {
        "market_sentiment": "bullish",
        "confidence": 0.75,
        "reasoning": "Price showing strong upward momentum with increasing volume.",
        "suggested_action": "buy",
        "risk_level": "medium",
        "key_factors": [
            "Strong volume increase",
            "Breaking resistance level",
            "Positive news sentiment"
        ],
        "timestamp": datetime.utcnow().isoformat(),
    }


# ============================================================================
# Sample Trade Data Fixtures
# ============================================================================

@pytest.fixture
def sample_trade_signal() -> dict:
    """Sample trade signal for testing."""
    return {
        "symbol": "BTC/USDT",
        "action": "buy",
        "price": 45000.0,
        "quantity": 0.01,
        "timestamp": datetime.utcnow(),
        "confidence": 0.8,
        "indicators": {
            "rsi": 65.5,
            "macd": 120.3,
            "ema_cross": True,
        },
        "strategy": "momentum_breakout",
    }


@pytest.fixture
def sample_executed_trade() -> dict:
    """Sample executed trade for testing."""
    return {
        "id": "trade-123",
        "symbol": "BTC/USDT",
        "side": "buy",
        "type": "limit",
        "price": 45000.0,
        "amount": 0.01,
        "cost": 450.0,
        "fee": {"cost": 0.45, "currency": "USDT"},
        "filled": 0.01,
        "remaining": 0.0,
        "status": "closed",
        "timestamp": int(datetime.utcnow().timestamp() * 1000),
    }


# ============================================================================
# Market Data Fixtures
# ============================================================================

@pytest.fixture
def sample_ticker() -> dict:
    """Sample ticker data for testing."""
    return {
        "symbol": "BTC/USDT",
        "timestamp": int(datetime.utcnow().timestamp() * 1000),
        "datetime": datetime.utcnow().isoformat(),
        "high": 46000.0,
        "low": 44000.0,
        "bid": 45000.0,
        "ask": 45100.0,
        "last": 45050.0,
        "close": 45050.0,
        "baseVolume": 1234.56,
        "quoteVolume": 55555555.0,
    }


@pytest.fixture
def sample_orderbook() -> dict:
    """Sample order book data for testing."""
    return {
        "symbol": "BTC/USDT",
        "timestamp": int(datetime.utcnow().timestamp() * 1000),
        "datetime": datetime.utcnow().isoformat(),
        "bids": [
            [45000.0, 1.5],
            [44990.0, 2.0],
            [44980.0, 1.8],
            [44970.0, 2.2],
            [44960.0, 1.3],
        ],
        "asks": [
            [45010.0, 1.2],
            [45020.0, 1.8],
            [45030.0, 2.1],
            [45040.0, 1.9],
            [45050.0, 1.6],
        ],
    }


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def freeze_time():
    """Fixture to freeze time for testing."""
    frozen_time = datetime(2024, 1, 1, 12, 0, 0)

    class FrozenTime:
        def __init__(self, dt):
            self.dt = dt

        def now(self):
            return self.dt

        def advance(self, **kwargs):
            self.dt += timedelta(**kwargs)

    return FrozenTime(frozen_time)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # Add singleton reset logic here if needed
    return
    # Cleanup after test
