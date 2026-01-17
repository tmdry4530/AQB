# External Data API - Quick Reference

## Installation

Dependencies already in `pyproject.toml`:
- httpx >= 0.25.0
- structlog >= 24.1.0

## Import

```python
from iftb.data.external import (
    # Clients
    FearGreedClient,
    CoinglassClient,
    ExternalDataAggregator,
    # Data structures
    FearGreedData,
    FundingData,
    OpenInterestData,
    LongShortData,
    MarketContext,
)
```

## Quick Start

### Get All Market Data (Recommended)

```python
from iftb.data.external import ExternalDataAggregator

async with ExternalDataAggregator() as agg:
    context = await agg.fetch_all("BTC")

    # Check what's available
    if context.fear_greed:
        print(f"F&G: {context.fear_greed.value}")
    if context.funding:
        print(f"Funding: {context.funding.rate:.4%}")
```

### Fear & Greed Only

```python
from iftb.data.external import FearGreedClient

async with FearGreedClient() as client:
    # Current
    current = await client.fetch_current()
    print(f"{current.value} - {current.classification}")

    # Historical
    history = await client.fetch_historical(7)
    for h in history:
        print(f"{h.timestamp.date()}: {h.value}")
```

### Coinglass Data

```python
from iftb.data.external import CoinglassClient

async with CoinglassClient(api_key="your_key") as client:
    # Funding
    funding = await client.fetch_funding_rate("BTC")
    print(f"Rate: {funding.rate:.4%}")

    # Open Interest
    oi = await client.fetch_open_interest("BTC")
    print(f"OI: ${oi.open_interest:,.0f}")

    # Long/Short
    ls = await client.fetch_long_short_ratio("BTC")
    print(f"L/S: {ls.long_ratio:.1%}/{ls.short_ratio:.1%}")
```

## Data Structures

### FearGreedData

```python
@dataclass
class FearGreedData:
    value: int              # 0-100
    classification: str     # "Extreme Fear", "Fear", etc.
    timestamp: datetime
```

### FundingData

```python
@dataclass
class FundingData:
    symbol: str
    rate: float                  # As decimal (0.0001 = 0.01%)
    predicted_rate: float
    next_funding_time: datetime
```

### OpenInterestData

```python
@dataclass
class OpenInterestData:
    symbol: str
    open_interest: float   # USD value
    oi_change_24h: float   # Percentage
```

### LongShortData

```python
@dataclass
class LongShortData:
    symbol: str
    long_ratio: float   # 0-1
    short_ratio: float  # 0-1
    timestamp: datetime
```

### MarketContext

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

## Common Patterns

### With Error Handling

```python
async with ExternalDataAggregator() as agg:
    context = await agg.fetch_all("BTC")

    # Check errors
    if context.errors:
        logger.warning("api_errors", errors=context.errors)

    # Use available data only
    sentiment = context.fear_greed.value if context.fear_greed else 50
```

### With Custom Configuration

```python
agg = ExternalDataAggregator(
    cache_ttl=600,  # 10 minutes
)

async with agg:
    # First call: fetches from APIs
    ctx1 = await agg.fetch_all("BTC")

    # Within 10 min: uses cache
    ctx2 = await agg.fetch_all("BTC")

    # Force refresh
    ctx3 = await agg.fetch_all("BTC", force_refresh=True)
```

### Multiple Symbols

```python
async with ExternalDataAggregator() as agg:
    btc_context = await agg.fetch_all("BTC")
    eth_context = await agg.fetch_all("ETH")
```

## Features Summary

| Feature | Description |
|---------|-------------|
| **Async** | Full async/await support |
| **Context Managers** | Automatic resource cleanup |
| **Caching** | 5-minute default TTL |
| **Retry Logic** | 3 retries with exponential backoff |
| **Graceful Degradation** | Returns partial data on failure |
| **Error Tracking** | All errors logged and returned |
| **Type Safety** | Full type hints |
| **Logging** | Structured logging throughout |

## Error Handling

### Automatic Retries

```python
# Automatic:
# - 3 retries
# - Exponential backoff (1s, 2s, 4s)
# - Returns None on failure
context = await agg.fetch_all()
```

### Checking for Failures

```python
context = await agg.fetch_all()

if context.fear_greed is None:
    print("F&G unavailable")

if context.errors:
    for error in context.errors:
        print(f"Error: {error}")
```

## Configuration Options

### FearGreedClient

```python
FearGreedClient(
    timeout=10.0,  # Request timeout in seconds
)
```

### CoinglassClient

```python
CoinglassClient(
    api_key="your_key",  # Optional API key
    timeout=10.0,        # Request timeout in seconds
)
```

### ExternalDataAggregator

```python
ExternalDataAggregator(
    fear_greed_client=custom_fg_client,  # Custom client
    coinglass_client=custom_cg_client,   # Custom client
    cache_ttl=300,                       # Cache TTL in seconds
)
```

## Common Use Cases

### Trading Signal

```python
context = await agg.fetch_all("BTC")

# Extreme fear + negative funding = potential long
if (context.fear_greed and context.fear_greed.value < 25 and
    context.funding and context.funding.rate < 0):
    print("Potential long signal")
```

### Risk Assessment

```python
context = await agg.fetch_all("BTC")

risk_score = 0
if context.open_interest and context.open_interest.oi_change_24h > 20:
    risk_score += 1  # High OI increase = higher risk

if context.long_short and context.long_short.long_ratio > 0.7:
    risk_score += 1  # Crowded long = higher risk

print(f"Risk level: {risk_score}/2")
```

### Market Sentiment

```python
context = await agg.fetch_all("BTC")

if context.fear_greed:
    if context.fear_greed.value < 25:
        sentiment = "Extreme Fear (contrarian buy)"
    elif context.fear_greed.value > 75:
        sentiment = "Extreme Greed (contrarian sell)"
    else:
        sentiment = "Neutral"

    print(sentiment)
```

## Logging Events

| Event | Level | Description |
|-------|-------|-------------|
| `fetching_fear_greed_current` | DEBUG | Fetching current F&G |
| `fear_greed_fetched` | INFO | Successfully fetched |
| `*_fetch_failed` | WARNING | Retry attempt |
| `*_fetch_exhausted` | ERROR | All retries failed |
| `*_parse_error` | ERROR | Parse failed |
| `market_context_fetched` | INFO | All data aggregated |

## Files

| File | Location | Purpose |
|------|----------|---------|
| Implementation | `src/iftb/data/external.py` | Main module (666 lines) |
| Documentation | `docs/external_data_api.md` | Complete guide |
| Demo | `test_external_demo.py` | Usage examples |
| Summary | `EXTERNAL_API_IMPLEMENTATION.md` | Implementation details |
| Quick Ref | `EXTERNAL_API_QUICK_REFERENCE.md` | This file |

## Testing

```bash
# Run demo
python test_external_demo.py

# Import test
python -c "import sys; sys.path.insert(0, 'src'); from iftb.data.external import *; print('OK')"
```

## API Endpoints

### Alternative.me

- URL: `https://api.alternative.me/fng/`
- Auth: None required
- Rate Limit: Reasonable

### Coinglass

- URL: `https://open-api.coinglass.com/public/v2`
- Auth: API key (X-API-Key header)
- Rate Limit: Varies by plan
- Note: Generic implementation, customize based on docs

## Common Issues

| Issue | Solution |
|-------|----------|
| Connection timeout | Increase timeout parameter |
| Cached data stale | Use `force_refresh=True` |
| Coinglass 401 | Check API key |
| Partial data | Check `context.errors` |
| Import error | Install httpx, structlog |

## Best Practices

1. ✅ Always use `async with` context manager
2. ✅ Check for `None` before using data
3. ✅ Monitor `context.errors`
4. ✅ Use caching for frequent calls
5. ✅ Set appropriate TTL for your use case
6. ✅ Log errors for monitoring
7. ✅ Handle partial data gracefully

## Next Steps

1. Install dependencies: `pip install httpx structlog`
2. Review documentation: `docs/external_data_api.md`
3. Run demo: `python test_external_demo.py`
4. Integrate into your trading logic
5. Monitor API usage and errors

## Support

- Full documentation: `docs/external_data_api.md`
- Implementation details: `EXTERNAL_API_IMPLEMENTATION.md`
- Demo script: `test_external_demo.py`
- Source code: `src/iftb/data/external.py`
