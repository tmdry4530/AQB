"""
Redis caching layer for frequently accessed data.

This module provides async Redis-based caching for OHLCV data, market data,
and LLM analysis results. It uses orjson for fast JSON serialization and
implements proper async context manager patterns.

Example Usage:
    ```python
    from iftb.data.cache import CacheManager
    from iftb.config import get_settings

    settings = get_settings()

    async with CacheManager(
        host=settings.redis.host,
        port=settings.redis.port,
        password=settings.redis.password.get_secret_value() if settings.redis.password else None,
    ) as cache:
        # Cache OHLCV data
        bars = await fetch_ohlcv_data(symbol="BTCUSDT", timeframe="1h")
        await cache.ohlcv.set_bars("BTCUSDT", "1h", bars, ttl=300)

        # Retrieve cached data
        cached_bars = await cache.ohlcv.get_bars("BTCUSDT", "1h", limit=100)

        # Cache ticker data
        ticker = await fetch_ticker("BTCUSDT")
        await cache.market.set_ticker("BTCUSDT", ticker, ttl=10)

        # Cache LLM analysis
        analysis_result = {"sentiment": "bullish", "confidence": 0.85}
        await cache.llm.set_analysis("sentiment", "input_hash_123", analysis_result)
    ```
"""

import hashlib
from dataclasses import asdict
from datetime import datetime
from typing import Any, Optional

import orjson
from redis.asyncio import Redis
from redis.exceptions import ConnectionError, RedisError, TimeoutError

from iftb.data.external import FearGreedData
from iftb.data.fetcher import OHLCVBar, Ticker
from iftb.utils import get_logger

logger = get_logger(__name__)


def _serialize_datetime(obj: Any) -> Any:
    """
    Serialize datetime objects for JSON encoding.

    Args:
        obj: Object to serialize

    Returns:
        ISO format string for datetime objects, original object otherwise
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class RedisClient:
    """
    Async Redis client wrapper with connection management.

    Provides high-level async interface for Redis operations with proper
    error handling and connection lifecycle management.

    Attributes:
        host: Redis server host
        port: Redis server port
        password: Redis authentication password (optional)
        db: Redis database number
    """

    def __init__(
        self, host: str, port: int, password: Optional[str] = None, db: int = 0
    ) -> None:
        """
        Initialize Redis client.

        Args:
            host: Redis server host
            port: Redis server port
            password: Redis authentication password (optional)
            db: Redis database number (default: 0)
        """
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self._client: Optional[Redis] = None

    async def connect(self) -> None:
        """
        Establish connection to Redis server.

        Raises:
            ConnectionError: If connection fails
        """
        try:
            self._client = Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
            )
            # Test connection
            await self._client.ping()
            logger.info(
                "redis_connected",
                host=self.host,
                port=self.port,
                db=self.db,
            )
        except (ConnectionError, TimeoutError) as e:
            logger.error(
                "redis_connection_failed",
                host=self.host,
                port=self.port,
                error=str(e),
            )
            raise

    async def disconnect(self) -> None:
        """Close Redis connection gracefully."""
        if self._client:
            await self._client.aclose()
            logger.info("redis_disconnected")
            self._client = None

    async def get(self, key: str) -> Optional[str]:
        """
        Get value by key.

        Args:
            key: Redis key

        Returns:
            Value as string, or None if key doesn't exist

        Raises:
            RedisError: If Redis operation fails
        """
        if not self._client:
            raise RedisError("Redis client not connected")

        try:
            value = await self._client.get(key)
            if value:
                logger.debug("cache_hit", key=key)
            else:
                logger.debug("cache_miss", key=key)
            return value
        except RedisError as e:
            logger.error("redis_get_error", key=key, error=str(e))
            raise

    async def set(
        self, key: str, value: str, ttl: Optional[int] = None
    ) -> None:
        """
        Set key-value pair with optional TTL.

        Args:
            key: Redis key
            value: Value to store
            ttl: Time-to-live in seconds (optional)

        Raises:
            RedisError: If Redis operation fails
        """
        if not self._client:
            raise RedisError("Redis client not connected")

        try:
            if ttl:
                await self._client.setex(key, ttl, value)
            else:
                await self._client.set(key, value)
            logger.debug("cache_set", key=key, ttl=ttl)
        except RedisError as e:
            logger.error("redis_set_error", key=key, error=str(e))
            raise

    async def delete(self, key: str) -> None:
        """
        Delete key from Redis.

        Args:
            key: Redis key to delete

        Raises:
            RedisError: If Redis operation fails
        """
        if not self._client:
            raise RedisError("Redis client not connected")

        try:
            await self._client.delete(key)
            logger.debug("cache_delete", key=key)
        except RedisError as e:
            logger.error("redis_delete_error", key=key, error=str(e))
            raise

    async def exists(self, key: str) -> bool:
        """
        Check if key exists.

        Args:
            key: Redis key

        Returns:
            True if key exists, False otherwise

        Raises:
            RedisError: If Redis operation fails
        """
        if not self._client:
            raise RedisError("Redis client not connected")

        try:
            result = await self._client.exists(key)
            return bool(result)
        except RedisError as e:
            logger.error("redis_exists_error", key=key, error=str(e))
            raise

    async def get_json(self, key: str) -> Optional[dict[str, Any]]:
        """
        Get JSON value by key.

        Args:
            key: Redis key

        Returns:
            Deserialized JSON object, or None if key doesn't exist

        Raises:
            RedisError: If Redis operation fails
            orjson.JSONDecodeError: If value is not valid JSON
        """
        value = await self.get(key)
        if value is None:
            return None

        try:
            return orjson.loads(value)
        except orjson.JSONDecodeError as e:
            logger.error("json_decode_error", key=key, error=str(e))
            raise

    async def set_json(
        self, key: str, value: dict[str, Any], ttl: Optional[int] = None
    ) -> None:
        """
        Set JSON value with optional TTL.

        Args:
            key: Redis key
            value: JSON-serializable dictionary
            ttl: Time-to-live in seconds (optional)

        Raises:
            RedisError: If Redis operation fails
        """
        try:
            json_str = orjson.dumps(
                value, default=_serialize_datetime
            ).decode("utf-8")
            await self.set(key, json_str, ttl)
        except (TypeError, orjson.JSONEncodeError) as e:
            logger.error("json_encode_error", key=key, error=str(e))
            raise

    async def __aenter__(self) -> "RedisClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()


class OHLCVCache:
    """
    Cache for OHLCV (candlestick) data.

    Manages caching of time-series OHLCV bars with efficient key formatting
    and batch operations.

    Key format: `ohlcv:{symbol}:{timeframe}:{timestamp}`
    """

    def __init__(self, client: RedisClient) -> None:
        """
        Initialize OHLCV cache.

        Args:
            client: RedisClient instance
        """
        self.client = client

    def _make_key(self, symbol: str, timeframe: str, timestamp: int) -> str:
        """
        Generate Redis key for OHLCV bar.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., "1h", "4h")
            timestamp: Unix timestamp in milliseconds

        Returns:
            Redis key string
        """
        return f"ohlcv:{symbol}:{timeframe}:{timestamp}"

    def _make_pattern(self, symbol: str, timeframe: str) -> str:
        """
        Generate Redis key pattern for scanning.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Redis key pattern with wildcard
        """
        return f"ohlcv:{symbol}:{timeframe}:*"

    async def get_bars(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> Optional[list[OHLCVBar]]:
        """
        Retrieve cached OHLCV bars.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            limit: Maximum number of bars to retrieve

        Returns:
            List of OHLCV bars sorted by timestamp (newest first),
            or None if no cached data found
        """
        pattern = self._make_pattern(symbol, timeframe)

        try:
            # Scan for matching keys
            if not self.client._client:
                return None

            cursor = 0
            keys = []
            while True:
                cursor, partial_keys = await self.client._client.scan(
                    cursor, match=pattern, count=1000
                )
                keys.extend(partial_keys)
                if cursor == 0:
                    break

            if not keys:
                return None

            # Get all values
            values = await self.client._client.mget(keys)

            # Parse and sort bars
            bars = []
            for value in values:
                if value:
                    try:
                        bar_data = orjson.loads(value)
                        # Ensure we only pass fields that OHLCVBar expects
                        bar_fields = {
                            k: v for k, v in bar_data.items()
                            if k in {"timestamp", "open", "high", "low", "close", "volume"}
                        }
                        bars.append(OHLCVBar(**bar_fields))
                    except (orjson.JSONDecodeError, ValueError, TypeError) as e:
                        logger.warning("invalid_bar_data", error=str(e))
                        continue

            # Sort by timestamp descending
            bars.sort(key=lambda x: x.timestamp, reverse=True)

            # Apply limit
            result = bars[:limit]

            logger.debug(
                "ohlcv_cache_retrieved",
                symbol=symbol,
                timeframe=timeframe,
                count=len(result),
            )

            return result

        except RedisError as e:
            logger.error(
                "ohlcv_cache_get_error",
                symbol=symbol,
                timeframe=timeframe,
                error=str(e),
            )
            return None

    async def set_bars(
        self, symbol: str, timeframe: str, bars: list[OHLCVBar], ttl: int = 300
    ) -> None:
        """
        Cache OHLCV bars.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            bars: List of OHLCV bars to cache
            ttl: Time-to-live in seconds (default: 300)
        """
        try:
            for bar in bars:
                key = self._make_key(symbol, timeframe, bar.timestamp)
                bar_data = asdict(bar)
                # Add symbol and timeframe for completeness
                bar_data["symbol"] = symbol
                bar_data["timeframe"] = timeframe
                await self.client.set_json(key, bar_data, ttl)

            logger.debug(
                "ohlcv_cache_set",
                symbol=symbol,
                timeframe=timeframe,
                count=len(bars),
                ttl=ttl,
            )

        except RedisError as e:
            logger.error(
                "ohlcv_cache_set_error",
                symbol=symbol,
                timeframe=timeframe,
                error=str(e),
            )

    async def get_latest_bar(
        self, symbol: str, timeframe: str
    ) -> Optional[OHLCVBar]:
        """
        Get the most recent cached bar.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Latest OHLCV bar, or None if no cached data found
        """
        bars = await self.get_bars(symbol, timeframe, limit=1)
        return bars[0] if bars else None

    async def invalidate(self, symbol: str, timeframe: str) -> None:
        """
        Invalidate all cached bars for a symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
        """
        pattern = self._make_pattern(symbol, timeframe)

        try:
            if not self.client._client:
                return

            cursor = 0
            while True:
                cursor, keys = await self.client._client.scan(
                    cursor, match=pattern, count=1000
                )
                if keys:
                    await self.client._client.delete(*keys)
                if cursor == 0:
                    break

            logger.info(
                "ohlcv_cache_invalidated",
                symbol=symbol,
                timeframe=timeframe,
            )

        except RedisError as e:
            logger.error(
                "ohlcv_cache_invalidate_error",
                symbol=symbol,
                timeframe=timeframe,
                error=str(e),
            )


class MarketDataCache:
    """
    Cache for market data (tickers, funding rates, fear & greed index).

    Manages caching of real-time market information with short TTLs.
    """

    def __init__(self, client: RedisClient) -> None:
        """
        Initialize market data cache.

        Args:
            client: RedisClient instance
        """
        self.client = client

    async def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """
        Retrieve cached ticker data.

        Args:
            symbol: Trading symbol

        Returns:
            Ticker object, or None if not cached
        """
        key = f"ticker:{symbol}"
        data = await self.client.get_json(key)

        if data:
            try:
                return Ticker(**data)
            except ValueError as e:
                logger.warning(
                    "invalid_ticker_data", symbol=symbol, error=str(e)
                )
                return None

        return None

    async def set_ticker(
        self, symbol: str, ticker: Ticker, ttl: int = 10
    ) -> None:
        """
        Cache ticker data.

        Args:
            symbol: Trading symbol
            ticker: Ticker object
            ttl: Time-to-live in seconds (default: 10)
        """
        key = f"ticker:{symbol}"
        data = asdict(ticker)
        await self.client.set_json(key, data, ttl)

        logger.debug("ticker_cached", symbol=symbol, ttl=ttl)

    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """
        Retrieve cached funding rate.

        Args:
            symbol: Trading symbol

        Returns:
            Funding rate, or None if not cached
        """
        key = f"funding:{symbol}"
        value = await self.client.get(key)

        if value:
            try:
                return float(value)
            except ValueError as e:
                logger.warning(
                    "invalid_funding_rate", symbol=symbol, error=str(e)
                )
                return None

        return None

    async def set_funding_rate(
        self, symbol: str, rate: float, ttl: int = 60
    ) -> None:
        """
        Cache funding rate.

        Args:
            symbol: Trading symbol
            rate: Funding rate
            ttl: Time-to-live in seconds (default: 60)
        """
        key = f"funding:{symbol}"
        await self.client.set(key, str(rate), ttl)

        logger.debug("funding_rate_cached", symbol=symbol, rate=rate, ttl=ttl)

    async def get_fear_greed(self) -> Optional[FearGreedData]:
        """
        Retrieve cached Fear & Greed index data.

        Returns:
            FearGreedData object, or None if not cached
        """
        key = "fear_greed"
        data = await self.client.get_json(key)

        if data:
            try:
                # Parse timestamp if it's a string
                if "timestamp" in data and isinstance(data["timestamp"], str):
                    data["timestamp"] = datetime.fromisoformat(
                        data["timestamp"]
                    )
                return FearGreedData(**data)
            except (ValueError, TypeError) as e:
                logger.warning("invalid_fear_greed_data", error=str(e))
                return None

        return None

    async def set_fear_greed(
        self, data: FearGreedData, ttl: int = 300
    ) -> None:
        """
        Cache Fear & Greed index data.

        Args:
            data: FearGreedData object
            ttl: Time-to-live in seconds (default: 300)
        """
        key = "fear_greed"
        json_data = asdict(data)
        await self.client.set_json(key, json_data, ttl)

        logger.debug("fear_greed_cached", value=data.value, ttl=ttl)


class LLMCache:
    """
    Cache for LLM analysis results.

    Stores LLM-generated analysis to avoid redundant API calls for
    identical inputs.

    Key format: `llm:{analysis_type}:{hash_of_input}`
    """

    def __init__(self, client: RedisClient) -> None:
        """
        Initialize LLM cache.

        Args:
            client: RedisClient instance
        """
        self.client = client

    def _hash_input(self, input_data: str) -> str:
        """
        Generate hash of input data.

        Args:
            input_data: Input string to hash

        Returns:
            SHA256 hash in hexadecimal format
        """
        return hashlib.sha256(input_data.encode()).hexdigest()

    async def get_analysis(
        self, analysis_type: str, input_hash: str
    ) -> Optional[dict[str, Any]]:
        """
        Retrieve cached LLM analysis.

        Args:
            analysis_type: Type of analysis (e.g., "sentiment", "summary")
            input_hash: Hash of the input data

        Returns:
            Analysis result dictionary, or None if not cached
        """
        key = f"llm:{analysis_type}:{input_hash}"
        data = await self.client.get_json(key)

        if data:
            logger.debug(
                "llm_cache_hit", analysis_type=analysis_type, input_hash=input_hash[:16]
            )

        return data

    async def set_analysis(
        self,
        analysis_type: str,
        input_hash: str,
        result: dict[str, Any],
        ttl: int = 300,
    ) -> None:
        """
        Cache LLM analysis result.

        Args:
            analysis_type: Type of analysis
            input_hash: Hash of the input data
            result: Analysis result dictionary
            ttl: Time-to-live in seconds (default: 300)
        """
        key = f"llm:{analysis_type}:{input_hash}"
        await self.client.set_json(key, result, ttl)

        logger.debug(
            "llm_analysis_cached",
            analysis_type=analysis_type,
            input_hash=input_hash[:16],
            ttl=ttl,
        )


class CacheManager:
    """
    Unified cache access manager.

    Provides a single entry point for all caching operations with proper
    lifecycle management through async context manager.

    Attributes:
        ohlcv: OHLCV data cache
        market: Market data cache
        llm: LLM analysis cache

    Example:
        ```python
        async with CacheManager(host="localhost", port=6379) as cache:
            # Use OHLCV cache
            bars = await cache.ohlcv.get_bars("BTCUSDT", "1h")

            # Use market cache
            ticker = await cache.market.get_ticker("BTCUSDT")

            # Use LLM cache
            analysis = await cache.llm.get_analysis("sentiment", "hash123")
        ```
    """

    def __init__(
        self, host: str, port: int, password: Optional[str] = None, db: int = 0
    ) -> None:
        """
        Initialize cache manager.

        Args:
            host: Redis server host
            port: Redis server port
            password: Redis authentication password (optional)
            db: Redis database number (default: 0)
        """
        self._client = RedisClient(host, port, password, db)
        self.ohlcv = OHLCVCache(self._client)
        self.market = MarketDataCache(self._client)
        self.llm = LLMCache(self._client)

    async def connect(self) -> None:
        """Establish connection to Redis."""
        await self._client.connect()
        logger.info("cache_manager_connected")

    async def disconnect(self) -> None:
        """Close Redis connection."""
        await self._client.disconnect()
        logger.info("cache_manager_disconnected")

    async def clear_all(self) -> None:
        """
        Clear all cached data (DANGEROUS - use with caution).

        Flushes the entire Redis database.
        """
        if not self._client._client:
            raise RedisError("Redis client not connected")

        try:
            await self._client._client.flushdb()
            logger.warning("cache_cleared_all")
        except RedisError as e:
            logger.error("cache_clear_error", error=str(e))
            raise

    async def health_check(self) -> bool:
        """
        Check if Redis connection is healthy.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            if not self._client._client:
                return False
            await self._client._client.ping()
            return True
        except (ConnectionError, TimeoutError, RedisError):
            return False

    async def __aenter__(self) -> "CacheManager":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()
