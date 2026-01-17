# IFTB Data Module

This module handles all data collection, validation, and management for the IFTB trading bot.

## Components

### Market Data Fetching (`fetcher.py`)
- Exchange data via CCXT
- OHLCV bars, tickers, funding rates
- Historical data downloading

### External Data Sources (`external.py`)
- Fear & Greed Index
- Coinglass metrics (funding, open interest, long/short ratios)
- Market context aggregation

### Telegram News Collection (`telegram.py`)
- Real-time news from Telegram channels
- Automatic urgent news detection
- Message queue and filtering
- LLM-ready news summaries

### Data Validation (`validation.py`)
- OHLCV data quality checks
- Missing data detection
- Statistical analysis

### Data Storage (`storage.py`)
- Database models and repositories
- Trade, position, and event tracking
- Async PostgreSQL operations

## Quick Start

### Fetch Market Data

```python
from iftb.data import ExchangeClient, fetch_latest_ohlcv

client = ExchangeClient(api_key="...", api_secret="...")
ohlcv = await fetch_latest_ohlcv(client, "BTCUSDT", "1h", limit=100)
```

### Collect Telegram News

```python
from iftb.data import create_collector_from_settings

collector = await create_collector_from_settings()
async with collector:
    await collector.start()
    messages = collector.get_recent_messages(minutes=60)
```

### Get External Data

```python
from iftb.data import ExternalDataAggregator

aggregator = ExternalDataAggregator()
context = await aggregator.get_market_context("BTC")
print(f"Fear & Greed: {context.fear_greed_index}")
```

### Validate Data

```python
from iftb.data import OHLCVValidator

validator = OHLCVValidator()
report = validator.validate(df)
if report.is_valid:
    print("Data quality is good!")
```

## Documentation

- [Telegram Collector Guide](../../../docs/telegram_collector.md)
- [API Reference](../../../docs/api_reference.md)

## Examples

See `/mnt/d/Develop/AQB/examples/` for complete usage examples:
- `telegram_example.py` - Telegram news collection demo
- `data_pipeline.py` - Full data pipeline example

## Testing

```bash
# Run all data module tests
pytest tests/unit/test_telegram.py tests/unit/test_fetcher.py -v

# Run with coverage
pytest tests/unit/ --cov=iftb.data --cov-report=html
```

## Configuration

All configuration is managed through `iftb.config.settings`:

```python
from iftb.config import get_settings

settings = get_settings()

# Telegram settings
api_id = settings.telegram.api_id
channel_ids = settings.telegram.news_channel_ids

# Database settings
db_url = settings.database.get_async_url()

# Exchange settings
exchange_key = settings.exchange.api_key
```

## Environment Variables

Required `.env` configuration:

```bash
# Exchange API
EXCHANGE_API_KEY=your_key
EXCHANGE_API_SECRET=your_secret
EXCHANGE_TESTNET=true

# Telegram
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=your_hash
TELEGRAM_NEWS_CHANNEL_IDS=[-1001234567890,-1009876543210]

# Database
DB_HOST=localhost
DB_PORT=5432
DB_DATABASE=iftb
DB_USERNAME=postgres
DB_PASSWORD=your_password
```

## Architecture

```
iftb.data/
├── fetcher.py          # Exchange data fetching
├── external.py         # External API clients
├── telegram.py         # Telegram news collector
├── validation.py       # Data quality validation
├── storage.py          # Database models & repos
└── __init__.py         # Public API exports
```

## Contributing

When adding new data sources:

1. Create a new file for the source (e.g., `twitter.py`)
2. Follow the async/await pattern
3. Add comprehensive error handling
4. Include docstrings and type hints
5. Write unit tests
6. Update `__init__.py` exports
7. Add documentation

## License

MIT License - See project root for details.
