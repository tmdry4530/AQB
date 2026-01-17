# Data Fetcher Module

The data fetcher module (`iftb.data.fetcher`) provides production-ready interfaces for fetching cryptocurrency market data from exchanges using CCXT.

## Features

- **Async CCXT Integration**: Full async/await support for high-performance data fetching
- **Robust Error Handling**: Automatic retries with exponential backoff
- **Rate Limiting**: Built-in rate limiting to respect exchange API limits
- **Data Structures**: Type-safe data models for OHLCV, tickers, and funding rates
- **Historical Downloads**: Bulk downloading with resume capability
- **Progress Tracking**: Optional callbacks for download progress
- **Context Manager Support**: Clean resource management with async context managers

## Quick Start

### Basic Usage

```python
import asyncio
from iftb.data import ExchangeClient

async def main():
    async with ExchangeClient("binance", api_key, api_secret) as client:
        # Fetch recent OHLCV data
        bars = await client.fetch_ohlcv("BTC/USDT", "1h", limit=100)

        # Fetch current ticker
        ticker = await client.fetch_ticker("BTC/USDT")
        print(f"BTC/USDT: ${ticker.last:,.2f}")

asyncio.run(main())
```

### Using Configuration

```python
from iftb.config import get_settings
from iftb.data import ExchangeClient

async def main():
    settings = get_settings()

    async with ExchangeClient(
        exchange_id="binance",
        api_key=settings.exchange.api_key.get_secret_value(),
        api_secret=settings.exchange.api_secret.get_secret_value(),
        testnet=settings.exchange.testnet,
    ) as client:
        bars = await client.fetch_ohlcv("BTC/USDT", "1h")
```

## Classes

### ExchangeClient

Main class for fetching exchange data.

**Methods:**

- `async connect()` - Initialize connection to exchange
- `async close()` - Close connection and cleanup
- `async fetch_ohlcv(symbol, timeframe, since=None, limit=500)` - Fetch OHLCV candles
- `async fetch_ohlcv_range(symbol, timeframe, start, end)` - Fetch range with pagination
- `async fetch_ticker(symbol)` - Fetch current ticker
- `async fetch_funding_rate(symbol)` - Fetch funding rate (perpetual futures)
- `async fetch_open_interest(symbol)` - Fetch open interest data

**Parameters:**

- `exchange_id`: Exchange name (e.g., "binance", "bybit")
- `api_key`: API key for authentication
- `api_secret`: API secret for authentication
- `testnet`: Whether to use testnet (default: False)

### HistoricalDataDownloader

Class for downloading bulk historical data.

**Methods:**

- `async download_historical(symbol, timeframe, start_date, end_date, output_dir, progress_callback=None, resume=True)` - Download and save to CSV

**Parameters:**

- `symbol`: Trading pair (e.g., "BTC/USDT")
- `timeframe`: Candle timeframe (e.g., "1h", "4h", "1d")
- `start_date`: Start datetime
- `end_date`: End datetime
- `output_dir`: Directory to save CSV files
- `progress_callback`: Optional callback function for progress updates
- `resume`: Whether to resume incomplete downloads (default: True)

## Data Structures

### OHLCVBar

```python
@dataclass
class OHLCVBar:
    timestamp: int      # Unix timestamp in milliseconds
    open: float         # Opening price
    high: float         # Highest price
    low: float          # Lowest price
    close: float        # Closing price
    volume: float       # Trading volume
```

### Ticker

```python
@dataclass
class Ticker:
    symbol: str         # Trading pair
    last: float         # Last traded price
    bid: float          # Best bid price
    ask: float          # Best ask price
    high_24h: float     # 24-hour high
    low_24h: float      # 24-hour low
    volume_24h: float   # 24-hour volume
    timestamp: int      # Unix timestamp in milliseconds
```

### FundingRate

```python
@dataclass
class FundingRate:
    symbol: str              # Trading pair
    rate: float              # Funding rate (positive = longs pay shorts)
    next_funding_time: int   # Next funding event timestamp
    timestamp: int           # Current timestamp
```

## Examples

### Fetch Recent OHLCV Data

```python
async with ExchangeClient("binance", api_key, api_secret) as client:
    bars = await client.fetch_ohlcv(
        symbol="BTC/USDT",
        timeframe="1h",
        limit=100
    )

    for bar in bars[-5:]:
        print(f"Close: {bar.close}, Volume: {bar.volume}")
```

### Fetch Date Range with Pagination

```python
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=30)

async with ExchangeClient("binance", api_key, api_secret) as client:
    bars = await client.fetch_ohlcv_range(
        symbol="BTC/USDT",
        timeframe="1h",
        start=start_date,
        end=end_date
    )

    print(f"Fetched {len(bars)} hourly bars")
```

### Download Historical Data

```python
from datetime import datetime, timedelta
from iftb.data import HistoricalDataDownloader

downloader = HistoricalDataDownloader("binance", api_key, api_secret)

# Download 6 years of hourly data
end_date = datetime.now()
start_date = end_date - timedelta(days=365*6)

def progress(current, total, symbol):
    print(f"Downloaded {current}/{total} bars for {symbol}")

file_path = await downloader.download_historical(
    symbol="BTC/USDT",
    timeframe="1h",
    start_date=start_date,
    end_date=end_date,
    output_dir="data/historical",
    progress_callback=progress,
    resume=True  # Resume if interrupted
)

print(f"Data saved to {file_path}")
```

### Fetch Multiple Symbols in Parallel

```python
symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]

async with ExchangeClient("binance", api_key, api_secret) as client:
    tasks = [client.fetch_ticker(symbol) for symbol in symbols]
    tickers = await asyncio.gather(*tasks)

    for ticker in tickers:
        print(f"{ticker.symbol}: ${ticker.last:,.2f}")
```

### Get Funding Rate

```python
async with ExchangeClient("binance", api_key, api_secret) as client:
    funding = await client.fetch_funding_rate("BTC/USDT")

    print(f"Funding Rate: {funding.rate * 100:.4f}%")

    if funding.rate > 0:
        print("Longs pay shorts (bullish sentiment)")
    else:
        print("Shorts pay longs (bearish sentiment)")
```

### Convenience Functions

```python
from iftb.data import fetch_latest_ohlcv, fetch_latest_ticker

# Uses default settings from config
bars = await fetch_latest_ohlcv("BTC/USDT", timeframe="1h", limit=100)
ticker = await fetch_latest_ticker("BTC/USDT")

print(f"Latest close: {bars[-1].close}")
print(f"Current price: {ticker.last}")
```

## Error Handling

The fetcher module includes robust error handling:

### Automatic Retries

```python
# Automatically retries on network errors and rate limits
async with ExchangeClient("binance", api_key, api_secret) as client:
    # Will retry up to 3 times with exponential backoff
    bars = await client.fetch_ohlcv("BTC/USDT", "1h")
```

### Manual Error Handling

```python
from ccxt.base.errors import NetworkError, RateLimitExceeded

async with ExchangeClient("binance", api_key, api_secret) as client:
    try:
        bars = await client.fetch_ohlcv("BTC/USDT", "1h")
    except RateLimitExceeded:
        # Handle rate limit
        await asyncio.sleep(60)
    except NetworkError:
        # Handle network error
        logger.error("Network connection failed")
```

## Configuration

Configuration is managed through the settings system:

```python
# .env file
EXCHANGE_API_KEY=your_api_key
EXCHANGE_API_SECRET=your_api_secret
EXCHANGE_TESTNET=true
EXCHANGE_RATE_LIMIT_PER_SECOND=10
```

```python
from iftb.config import get_settings

settings = get_settings()
print(f"Using testnet: {settings.exchange.testnet}")
print(f"Rate limit: {settings.exchange.rate_limit_per_second}/s")
```

## Supported Exchanges

The module uses CCXT, supporting 100+ exchanges including:

- Binance (binance)
- Bybit (bybit)
- OKX (okx)
- BitMEX (bitmex)
- Coinbase (coinbase)
- Kraken (kraken)

## Timeframes

Supported timeframe strings:

- `1m`, `5m`, `15m`, `30m` - Minutes
- `1h`, `4h`, `12h` - Hours
- `1d`, `3d` - Days
- `1w` - Weeks

## Rate Limiting

Built-in rate limiting features:

1. **Semaphore-based concurrency control** - Limits concurrent requests
2. **Exponential backoff** - Automatically backs off on rate limit errors
3. **Configurable limits** - Set via `EXCHANGE_RATE_LIMIT_PER_SECOND`

## Logging

All operations are logged using the structured logging system:

```python
from iftb.utils import setup_logging, LogConfig

config = LogConfig(level="INFO", format="pretty")
setup_logging(config)

# Logs will include context about all fetch operations
async with ExchangeClient("binance", api_key, api_secret) as client:
    bars = await client.fetch_ohlcv("BTC/USDT", "1h")
    # Logs: "ohlcv_fetched" with symbol, timeframe, bar count
```

## Testing

Run tests with pytest:

```bash
# Run all fetcher tests
pytest tests/data/test_fetcher.py

# Run with coverage
pytest tests/data/test_fetcher.py --cov=iftb.data.fetcher

# Run only unit tests (fast)
pytest tests/data/test_fetcher.py -m unit

# Run live tests (requires API credentials)
pytest tests/data/test_fetcher.py -m live
```

## Best Practices

1. **Always use context managers** - Ensures proper cleanup
2. **Handle rate limits gracefully** - Let automatic retries work
3. **Use resume for large downloads** - Prevents data loss on interruption
4. **Validate data quality** - Use `OHLCVValidator` after fetching
5. **Log all operations** - Helps with debugging and monitoring
6. **Use testnet for development** - Avoid production API rate limits

## Troubleshooting

### Connection Issues

```python
# Verify exchange is reachable
async with ExchangeClient("binance", api_key, api_secret) as client:
    await client.connect()
    print("Connection successful!")
```

### Authentication Errors

```python
# Check API credentials
settings = get_settings()
print(f"API Key: {settings.exchange.api_key.get_secret_value()[:10]}...")
print(f"Testnet: {settings.exchange.testnet}")
```

### Rate Limit Issues

```python
# Reduce concurrent requests
client._rate_limiter = asyncio.Semaphore(5)  # Reduce from default 10

# Or increase delay between requests
await asyncio.sleep(0.5)  # 500ms delay
```

## See Also

- [Configuration Guide](./configuration.md)
- [Data Validation](./data_validation.md)
- [Logging System](./logging.md)
- [CCXT Documentation](https://docs.ccxt.com/)
