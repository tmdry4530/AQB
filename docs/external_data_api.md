# External Data API Documentation

## Overview

The `iftb.data.external` module provides clients for fetching market sentiment and derivatives data from external APIs:

- **Fear & Greed Index** from alternative.me
- **Funding rates, Open Interest, and Long/Short ratios** from Coinglass

## Installation

The module requires `httpx` for async HTTP requests, which is already in the project dependencies.

## Quick Start

```python
import asyncio
from iftb.data.external import ExternalDataAggregator

async def get_market_context():
    """Fetch all market context data."""
    async with ExternalDataAggregator() as aggregator:
        context = await aggregator.fetch_all(symbol="BTC")

        if context.fear_greed:
            print(f"Fear & Greed: {context.fear_greed.value}")

        if context.funding:
            print(f"Funding Rate: {context.funding.rate:.4%}")

        return context

asyncio.run(get_market_context())
```

## Classes and Data Structures

### Data Classes

#### FearGreedData

Represents Fear & Greed Index data.

```python
@dataclass
class FearGreedData:
    value: int              # 0-100 (0=Extreme Fear, 100=Extreme Greed)
    classification: str     # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    timestamp: datetime
```

#### FundingData

Represents futures funding rate data.

```python
@dataclass
class FundingData:
    symbol: str                  # e.g., "BTC"
    rate: float                  # Current rate (decimal, e.g., 0.0001 = 0.01%)
    predicted_rate: float        # Predicted next rate
    next_funding_time: datetime  # Next funding settlement time
```

#### OpenInterestData

Represents open interest data.

```python
@dataclass
class OpenInterestData:
    symbol: str            # e.g., "BTC"
    open_interest: float   # Total OI in USD
    oi_change_24h: float   # 24h change percentage
```

#### LongShortData

Represents long/short ratio data.

```python
@dataclass
class LongShortData:
    symbol: str          # e.g., "BTC"
    long_ratio: float    # Long positions ratio (0-1)
    short_ratio: float   # Short positions ratio (0-1)
    timestamp: datetime
```

#### MarketContext

Aggregated market context from all sources.

```python
@dataclass
class MarketContext:
    fear_greed: FearGreedData | None
    funding: FundingData | None
    open_interest: OpenInterestData | None
    long_short: LongShortData | None
    fetch_time: datetime
    errors: list[str]
```

### Client Classes

#### FearGreedClient

Fetches Fear & Greed Index data from alternative.me.

**Methods:**

```python
async def fetch_current() -> FearGreedData
    """Fetch current Fear & Greed Index."""

async def fetch_historical(limit: int = 30) -> list[FearGreedData]
    """Fetch historical data (max 30 points)."""
```

**Example:**

```python
async with FearGreedClient() as client:
    # Current index
    current = await client.fetch_current()
    print(f"Current: {current.value} ({current.classification})")

    # Historical data
    history = await client.fetch_historical(limit=7)
    for data in history:
        print(f"{data.timestamp.date()}: {data.value}")
```

#### CoinglassClient

Fetches derivatives data from Coinglass API.

**Configuration:**

```python
client = CoinglassClient(
    api_key="your_api_key",  # Optional, for authenticated requests
    timeout=10.0             # Request timeout in seconds
)
```

**Methods:**

```python
async def fetch_funding_rate(symbol: str = "BTC") -> FundingData
    """Fetch current funding rate."""

async def fetch_open_interest(symbol: str = "BTC") -> OpenInterestData
    """Fetch open interest data."""

async def fetch_long_short_ratio(symbol: str = "BTC") -> LongShortData
    """Fetch long/short ratio."""
```

**Example:**

```python
async with CoinglassClient(api_key="your_key") as client:
    # Funding rate
    funding = await client.fetch_funding_rate("BTC")
    print(f"Rate: {funding.rate:.4%}")

    # Open interest
    oi = await client.fetch_open_interest("BTC")
    print(f"OI: ${oi.open_interest:,.0f}")

    # Long/short ratio
    ls = await client.fetch_long_short_ratio("BTC")
    print(f"Long: {ls.long_ratio:.2%}, Short: {ls.short_ratio:.2%}")
```

#### ExternalDataAggregator

Aggregates data from all sources with caching and retry logic.

**Configuration:**

```python
aggregator = ExternalDataAggregator(
    fear_greed_client=None,  # Optional custom FearGreedClient
    coinglass_client=None,   # Optional custom CoinglassClient
    cache_ttl=300            # Cache TTL in seconds (default: 5 min)
)
```

**Methods:**

```python
async def fetch_all(
    symbol: str = "BTC",
    force_refresh: bool = False
) -> MarketContext
    """Fetch all market context data with caching."""
```

**Features:**

- **Concurrent fetching**: All APIs called in parallel for speed
- **Caching**: Reduces API calls with configurable TTL
- **Retry logic**: Exponential backoff on failures (max 3 retries)
- **Graceful degradation**: Returns partial data if some APIs fail
- **Error tracking**: All errors logged and included in response

**Example:**

```python
async with ExternalDataAggregator(cache_ttl=300) as agg:
    # First call: fetches from APIs
    context1 = await agg.fetch_all("BTC")

    # Second call: uses cache (within 5 min)
    context2 = await agg.fetch_all("BTC")

    # Force refresh: bypasses cache
    context3 = await agg.fetch_all("BTC", force_refresh=True)

    # Check for errors
    if context1.errors:
        print(f"Errors: {context1.errors}")

    # Use available data
    if context1.fear_greed:
        sentiment = context1.fear_greed.classification
        print(f"Market sentiment: {sentiment}")
```

## Error Handling

The module implements robust error handling:

### Retry Logic

- **Automatic retries**: Up to 3 attempts with exponential backoff
- **Initial backoff**: 1 second
- **Backoff multiplier**: 2x per retry

### Error Types

1. **HTTP Errors** (`httpx.HTTPError`): Network or API errors
2. **Parse Errors** (`ValueError`): Malformed API responses
3. **Unexpected Errors**: Any other exceptions

### Graceful Degradation

When a data source fails:
- Returns `None` for that data source
- Logs the error
- Includes error message in `MarketContext.errors`
- Continues fetching from other sources

**Example:**

```python
context = await aggregator.fetch_all("BTC")

# Check what data is available
if context.fear_greed is None:
    print("Fear & Greed Index unavailable")
    if any("Fear & Greed" in err for err in context.errors):
        print(f"Reason: {context.errors}")

# Use available data only
if context.funding:
    # Safe to use funding data
    analyze_funding(context.funding)
```

## Logging

All operations are logged using the IFTB logging system:

```python
from iftb.utils import get_logger

logger = get_logger(__name__)
```

**Log Events:**

- `fetching_fear_greed_current`: Fetching current F&G index
- `fear_greed_fetched`: Successfully fetched F&G data
- `fetching_funding_rate`: Fetching funding rate
- `funding_rate_fetched`: Successfully fetched funding rate
- `market_context_fetched`: All data fetched and aggregated
- `*_fetch_failed`: Retry attempt on failure
- `*_fetch_exhausted`: All retries exhausted
- `*_parse_error`: Failed to parse API response
- `*_unexpected_error`: Unexpected error occurred

## Best Practices

### 1. Use Context Managers

Always use async context managers to ensure proper cleanup:

```python
async with ExternalDataAggregator() as agg:
    context = await agg.fetch_all()
```

### 2. Handle Partial Data

Check for `None` before using data:

```python
context = await agg.fetch_all()

if context.fear_greed:
    sentiment = context.fear_greed.value
else:
    sentiment = 50  # Default neutral value
```

### 3. Configure Caching

Set appropriate cache TTL based on your use case:

```python
# High-frequency trading: short TTL
agg = ExternalDataAggregator(cache_ttl=60)

# Position trading: longer TTL
agg = ExternalDataAggregator(cache_ttl=600)
```

### 4. Monitor Errors

Log and monitor API errors:

```python
context = await agg.fetch_all()

if context.errors:
    logger.warning("external_api_errors", errors=context.errors)
    # Alert monitoring system
    send_alert(f"API errors: {len(context.errors)}")
```

### 5. Rate Limiting

Be aware of API rate limits:

- **alternative.me**: No authentication, reasonable limits
- **Coinglass**: Requires API key, has rate limits

Use caching to reduce API calls:

```python
# Good: One API call, multiple uses
context = await agg.fetch_all()
analyze_sentiment(context.fear_greed)
analyze_funding(context.funding)

# Bad: Multiple API calls
context1 = await agg.fetch_all()
analyze_sentiment(context1.fear_greed)
context2 = await agg.fetch_all()  # Unnecessary call
analyze_funding(context2.funding)
```

## Integration Example

Complete example integrating external data into trading logic:

```python
from iftb.data.external import ExternalDataAggregator
from iftb.utils import get_logger

logger = get_logger(__name__)

class MarketAnalyzer:
    """Analyzes market conditions using external data."""

    def __init__(self):
        self.aggregator = ExternalDataAggregator(cache_ttl=300)

    async def get_market_bias(self, symbol: str = "BTC") -> str:
        """Determine market bias from external data.

        Returns:
            "bullish", "bearish", or "neutral"
        """
        context = await self.aggregator.fetch_all(symbol)

        score = 0

        # Fear & Greed Index
        if context.fear_greed:
            if context.fear_greed.value < 25:
                score -= 2  # Extreme fear: contrarian bullish
            elif context.fear_greed.value > 75:
                score += 2  # Extreme greed: contrarian bearish

        # Funding Rate
        if context.funding:
            if context.funding.rate > 0.0005:
                score += 1  # High funding: bearish
            elif context.funding.rate < -0.0005:
                score -= 1  # Negative funding: bullish

        # Open Interest
        if context.open_interest:
            if context.open_interest.oi_change_24h > 10:
                score += 1  # Rising OI: trend continuation
            elif context.open_interest.oi_change_24h < -10:
                score -= 1  # Falling OI: trend reversal

        # Long/Short Ratio
        if context.long_short:
            if context.long_short.long_ratio > 0.65:
                score += 1  # Crowded long: bearish
            elif context.long_short.short_ratio > 0.65:
                score -= 1  # Crowded short: bullish

        # Determine bias
        if score >= 2:
            bias = "bearish"
        elif score <= -2:
            bias = "bullish"
        else:
            bias = "neutral"

        logger.info(
            "market_bias_calculated",
            symbol=symbol,
            bias=bias,
            score=score,
            fear_greed=context.fear_greed.value if context.fear_greed else None,
            funding_rate=context.funding.rate if context.funding else None,
        )

        return bias

    async def close(self):
        """Close aggregator and cleanup."""
        await self.aggregator.close()

# Usage
analyzer = MarketAnalyzer()
try:
    bias = await analyzer.get_market_bias("BTC")
    print(f"Market bias: {bias}")
finally:
    await analyzer.close()
```

## API Documentation

### Alternative.me (Fear & Greed Index)

- **Base URL**: `https://api.alternative.me/fng/`
- **Rate Limit**: Reasonable, no authentication required
- **Documentation**: https://alternative.me/crypto/fear-and-greed-index/

### Coinglass

- **Base URL**: `https://open-api.coinglass.com/public/v2`
- **Authentication**: API key required (via `X-API-Key` header)
- **Rate Limit**: Varies by plan
- **Documentation**: https://www.coinglass.com/api

**Note**: The Coinglass implementation is a generic template. Update endpoints and parsing based on actual API documentation.

## Testing

Run the demo script to test the implementation:

```bash
python test_external_demo.py
```

This will:
1. Fetch current Fear & Greed Index
2. Fetch historical Fear & Greed data
3. Attempt to fetch Coinglass data (may fail without API key)
4. Test the aggregator with caching

## Troubleshooting

### "No module named 'httpx'"

Install dependencies:
```bash
pip install httpx
```

### "Connection timeout"

- Check internet connection
- Increase timeout: `FearGreedClient(timeout=30)`
- Check if API is down

### Coinglass API errors

- Verify API key is correct
- Check rate limits haven't been exceeded
- Ensure endpoints match API documentation

### Cached data not updating

- Force refresh: `fetch_all(force_refresh=True)`
- Reduce cache TTL: `ExternalDataAggregator(cache_ttl=60)`

## Future Enhancements

Possible improvements:

1. **Additional data sources**: Glassnode, CryptoQuant, etc.
2. **WebSocket support**: Real-time data streaming
3. **Historical data storage**: Cache to database for analysis
4. **Rate limit handling**: Automatic backoff on 429 errors
5. **Custom indicators**: Derived metrics from multiple sources
