# Redis Caching Layer

The Redis caching layer provides high-performance caching for frequently accessed data in the IFTB trading bot.

## Overview

The cache module (`iftb.data.cache`) implements async Redis-based caching for:
- **OHLCV data**: Candlestick bars for technical analysis
- **Market data**: Tickers, funding rates, Fear & Greed index
- **LLM analysis**: Cached LLM responses to avoid redundant API calls

## Architecture

```
CacheManager
├── RedisClient (low-level Redis operations)
├── OHLCVCache (OHLCV bar caching)
├── MarketDataCache (market data caching)
└── LLMCache (LLM analysis caching)
```

## Installation

Add to `pyproject.toml`:
```toml
dependencies = [
    "redis>=5.0.0",
    "orjson>=3.9.0",
]
```

Install dependencies:
```bash
pip install -e .
```

## Redis Setup

### Local Development

Install and start Redis:
```bash
# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis

# macOS
brew install redis
brew services start redis

# Docker
docker run -d -p 6379:6379 redis:7-alpine
```

### Environment Configuration

Add to `.env`:
```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=  # Optional
REDIS_DB=0
```

## Usage

### Basic Usage

```python
from iftb.data.cache import CacheManager
from iftb.config import get_settings

settings = get_settings()

# Using context manager (recommended)
async with CacheManager(
    host=settings.redis.host,
    port=settings.redis.port,
    password=settings.redis.password.get_secret_value() if settings.redis.password else None,
) as cache:
    # Use cache operations here
    await cache.ohlcv.set_bars("BTCUSDT", "1h", bars)
```

### Manual Connection Management

```python
cache = CacheManager(host="localhost", port=6379)
await cache.connect()

try:
    # Use cache operations
    pass
finally:
    await cache.disconnect()
```

## Cache Types

### 1. OHLCV Cache

Cache OHLCV bars with automatic timestamp-based key generation.

**Key Format**: `ohlcv:{symbol}:{timeframe}:{timestamp}`

```python
from iftb.data.fetcher import OHLCVBar

# Create bars
bars = [
    OHLCVBar(
        timestamp=1705478400000,
        open=45000.0,
        high=45500.0,
        low=44800.0,
        close=45200.0,
        volume=1234.56,
    ),
]

# Cache bars (5 minute TTL)
await cache.ohlcv.set_bars("BTCUSDT", "1h", bars, ttl=300)

# Retrieve bars
cached_bars = await cache.ohlcv.get_bars("BTCUSDT", "1h", limit=100)

# Get latest bar
latest = await cache.ohlcv.get_latest_bar("BTCUSDT", "1h")

# Invalidate cache
await cache.ohlcv.invalidate("BTCUSDT", "1h")
```

### 2. Market Data Cache

Cache real-time market data with short TTLs.

#### Ticker Data

```python
from iftb.data.fetcher import Ticker

ticker = Ticker(
    symbol="BTCUSDT",
    bid=45100.0,
    ask=45110.0,
    last=45105.0,
    volume_24h=15000.0,
    change_24h=2.5,
    high_24h=45800.0,
    low_24h=43900.0,
    timestamp=1705478400000,
)

# Cache ticker (10 second TTL)
await cache.market.set_ticker("BTCUSDT", ticker, ttl=10)

# Retrieve ticker
cached_ticker = await cache.market.get_ticker("BTCUSDT")
```

#### Funding Rates

```python
# Cache funding rate (1 minute TTL)
await cache.market.set_funding_rate("BTCUSDT", 0.0001, ttl=60)

# Retrieve funding rate
rate = await cache.market.get_funding_rate("BTCUSDT")
```

#### Fear & Greed Index

```python
from datetime import datetime
from iftb.data.external import FearGreedData

fg_data = FearGreedData(
    value=45,
    classification="Fear",
    timestamp=datetime.now(),
)

# Cache Fear & Greed (5 minute TTL)
await cache.market.set_fear_greed(fg_data, ttl=300)

# Retrieve Fear & Greed
cached_fg = await cache.market.get_fear_greed()
```

### 3. LLM Cache

Cache LLM analysis results to reduce API costs and improve response times.

**Key Format**: `llm:{analysis_type}:{hash_of_input}`

```python
# Generate input hash
input_data = "BTC market analysis for 2024-01-17"
input_hash = cache.llm._hash_input(input_data)

# Cache analysis result (5 minute TTL)
analysis_result = {
    "sentiment": "bullish",
    "confidence": 0.85,
    "key_points": [
        "Strong momentum",
        "Volume increasing",
        "Breaking resistance"
    ],
}
await cache.llm.set_analysis("sentiment", input_hash, analysis_result, ttl=300)

# Retrieve cached analysis
cached_analysis = await cache.llm.get_analysis("sentiment", input_hash)
```

## TTL Guidelines

Recommended TTL values for different data types:

| Data Type | TTL | Reason |
|-----------|-----|--------|
| Ticker | 10s | High-frequency updates |
| Funding Rate | 60s | Updates every 8h, cached for quick access |
| OHLCV Bars | 300s (5m) | Historical data changes infrequently |
| Fear & Greed | 300s (5m) | Updates once per day |
| LLM Analysis | 300s (5m) | Reusable for similar queries |

## Advanced Features

### Health Check

```python
is_healthy = await cache.health_check()
if not is_healthy:
    logger.error("Redis connection unhealthy")
```

### Clear All Cache

**Warning**: This flushes the entire Redis database.

```python
await cache.clear_all()  # Use with extreme caution
```

### Custom Redis Client

For advanced use cases, use `RedisClient` directly:

```python
from iftb.data.cache import RedisClient

async with RedisClient(host="localhost", port=6379, db=1) as client:
    # String operations
    await client.set("key", "value", ttl=60)
    value = await client.get("key")

    # JSON operations
    await client.set_json("json_key", {"data": "value"}, ttl=60)
    data = await client.get_json("json_key")

    # Key operations
    exists = await client.exists("key")
    await client.delete("key")
```

## Integration Examples

### With Exchange Data Fetcher

```python
from iftb.data.cache import CacheManager
from iftb.data.fetcher import ExchangeClient
from iftb.config import get_settings

settings = get_settings()

async def fetch_with_cache(symbol: str, timeframe: str):
    """Fetch OHLCV data with caching."""
    async with CacheManager(
        host=settings.redis.host,
        port=settings.redis.port,
    ) as cache:
        # Try cache first
        cached_bars = await cache.ohlcv.get_bars(symbol, timeframe, limit=100)
        if cached_bars:
            logger.info("cache_hit", symbol=symbol, timeframe=timeframe)
            return cached_bars

        # Cache miss - fetch from exchange
        logger.info("cache_miss", symbol=symbol, timeframe=timeframe)
        async with ExchangeClient(
            exchange="binance",
            api_key=settings.exchange.api_key.get_secret_value(),
            api_secret=settings.exchange.api_secret.get_secret_value(),
        ) as client:
            bars = await client.fetch_ohlcv(symbol, timeframe, limit=100)

            # Cache for future requests
            await cache.ohlcv.set_bars(symbol, timeframe, bars, ttl=300)

            return bars
```

### With LLM Analysis

```python
from iftb.data.cache import CacheManager
from iftb.llm import analyze_sentiment

async def get_cached_sentiment(text: str):
    """Get sentiment analysis with caching."""
    async with CacheManager(host="localhost", port=6379) as cache:
        # Generate hash
        input_hash = cache.llm._hash_input(text)

        # Check cache
        cached = await cache.llm.get_analysis("sentiment", input_hash)
        if cached:
            logger.info("llm_cache_hit")
            return cached

        # Run analysis
        logger.info("llm_cache_miss")
        result = await analyze_sentiment(text)

        # Cache result
        await cache.llm.set_analysis("sentiment", input_hash, result, ttl=300)

        return result
```

## Error Handling

The cache module handles Redis errors gracefully:

```python
from redis.exceptions import ConnectionError, RedisError

try:
    async with CacheManager(host="localhost", port=6379) as cache:
        await cache.ohlcv.set_bars(symbol, timeframe, bars)
except ConnectionError:
    logger.error("redis_connection_failed")
    # Fallback to non-cached operation
except RedisError as e:
    logger.error("redis_error", error=str(e))
    # Handle Redis-specific errors
```

## Best Practices

### 1. Use Context Managers

Always use async context managers to ensure proper connection cleanup:

```python
# Good
async with CacheManager(...) as cache:
    await cache.ohlcv.get_bars(...)

# Avoid
cache = CacheManager(...)
await cache.connect()
# ... operations ...
# Easy to forget: await cache.disconnect()
```

### 2. Choose Appropriate TTLs

- Short TTL for rapidly changing data (tickers, funding rates)
- Longer TTL for stable data (historical OHLCV, daily sentiment)
- Consider update frequency vs. staleness tolerance

### 3. Cache Invalidation

Invalidate cache when source data changes:

```python
# After receiving new OHLCV data
await cache.ohlcv.invalidate("BTCUSDT", "1h")
await cache.ohlcv.set_bars("BTCUSDT", "1h", new_bars)
```

### 4. Handle Cache Misses Gracefully

Always have a fallback when cache is unavailable:

```python
cached_data = await cache.get_data()
if not cached_data:
    # Fallback to direct fetch
    data = await fetch_from_source()
```

### 5. Use Separate Databases

Use different Redis databases for different environments:

```python
# Development: db=0
# Staging: db=1
# Production: db=2
# Testing: db=15
```

## Monitoring

### Log Messages

The cache module logs all operations:

```python
# Cache hits/misses
logger.debug("cache_hit", key=key)
logger.debug("cache_miss", key=key)

# Connection events
logger.info("redis_connected", host=host, port=port)
logger.info("redis_disconnected")

# Errors
logger.error("redis_connection_failed", error=str(e))
logger.error("redis_get_error", key=key, error=str(e))
```

### Metrics to Track

- Cache hit rate
- Average TTL
- Redis memory usage
- Connection errors
- Operation latency

## Troubleshooting

### Connection Refused

```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solution**: Ensure Redis is running and accessible:
```bash
redis-cli ping  # Should return "PONG"
```

### Memory Issues

If Redis runs out of memory, it may evict cached data prematurely.

**Solution**: Configure Redis maxmemory policy:
```redis
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
```

### Serialization Errors

```
orjson.JSONEncodeError: Type is not JSON serializable
```

**Solution**: Ensure data types are JSON-serializable. Datetime objects are automatically handled.

## Performance Considerations

### Throughput

- Redis can handle 100,000+ operations/second
- Network latency is typically the bottleneck
- Use connection pooling for high-concurrency scenarios

### Memory Usage

Approximate memory per cached item:
- OHLCV bar: ~200 bytes
- Ticker: ~300 bytes
- LLM analysis: 1-10 KB depending on size

### Optimization Tips

1. **Use shorter keys**: `ohlcv:BTC:1h:1705478400000` vs. `ohlcv:BTCUSDT:1_hour:1705478400000`
2. **Batch operations**: Use `mget` when retrieving multiple keys
3. **Compress large values**: For LLM responses >10KB, consider compression
4. **Use pipelining**: For multiple independent operations

## Testing

Run the test suite:

```bash
# Ensure Redis is running on localhost:6379
python test_cache.py
```

Expected output:
```
=== Testing RedisClient ===
✓ Basic set/get works
✓ Exists check works
✓ Delete works
✓ JSON set/get works

=== Testing OHLCVCache ===
✓ Bars cached successfully
✓ Retrieved 2 bars
✓ Latest bar retrieved correctly
✓ Cache invalidation works

...

Test Results: 5/5 passed
✓ All tests passed!
```

## API Reference

### CacheManager

Main entry point for all cache operations.

```python
CacheManager(host: str, port: int, password: str | None = None, db: int = 0)
```

**Methods**:
- `async connect() -> None`
- `async disconnect() -> None`
- `async clear_all() -> None`
- `async health_check() -> bool`

**Properties**:
- `ohlcv: OHLCVCache`
- `market: MarketDataCache`
- `llm: LLMCache`

### RedisClient

Low-level Redis client wrapper.

```python
RedisClient(host: str, port: int, password: str | None = None, db: int = 0)
```

**Methods**:
- `async get(key: str) -> str | None`
- `async set(key: str, value: str, ttl: int | None = None) -> None`
- `async delete(key: str) -> None`
- `async exists(key: str) -> bool`
- `async get_json(key: str) -> dict | None`
- `async set_json(key: str, value: dict, ttl: int | None = None) -> None`

### OHLCVCache

OHLCV bar caching.

**Methods**:
- `async get_bars(symbol: str, timeframe: str, limit: int = 100) -> list[OHLCVBar] | None`
- `async set_bars(symbol: str, timeframe: str, bars: list[OHLCVBar], ttl: int = 300) -> None`
- `async get_latest_bar(symbol: str, timeframe: str) -> OHLCVBar | None`
- `async invalidate(symbol: str, timeframe: str) -> None`

### MarketDataCache

Market data caching.

**Methods**:
- `async get_ticker(symbol: str) -> Ticker | None`
- `async set_ticker(symbol: str, ticker: Ticker, ttl: int = 10) -> None`
- `async get_funding_rate(symbol: str) -> float | None`
- `async set_funding_rate(symbol: str, rate: float, ttl: int = 60) -> None`
- `async get_fear_greed() -> FearGreedData | None`
- `async set_fear_greed(data: FearGreedData, ttl: int = 300) -> None`

### LLMCache

LLM analysis caching.

**Methods**:
- `async get_analysis(analysis_type: str, input_hash: str) -> dict | None`
- `async set_analysis(analysis_type: str, input_hash: str, result: dict, ttl: int = 300) -> None`
- `_hash_input(input_data: str) -> str` (helper method)

## License

MIT License - see LICENSE file for details.
