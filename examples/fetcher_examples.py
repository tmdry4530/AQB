"""
Example usage of the IFTB data fetcher module.

This script demonstrates how to use ExchangeClient and HistoricalDataDownloader
to fetch market data from cryptocurrency exchanges.

Requirements:
    - Set up .env file with exchange API credentials
    - Install dependencies: pip install -e .
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from iftb.config import get_settings
from iftb.data import (
    ExchangeClient,
    HistoricalDataDownloader,
    fetch_latest_ohlcv,
    fetch_latest_ticker,
)
from iftb.utils import LogConfig, get_logger, setup_logging


async def example_fetch_ohlcv():
    """Example: Fetch recent OHLCV data."""
    print("\n" + "=" * 60)
    print("Example 1: Fetching OHLCV Data")
    print("=" * 60)

    settings = get_settings()

    async with ExchangeClient(
        exchange_id="binance",
        api_key=settings.exchange.api_key.get_secret_value(),
        api_secret=settings.exchange.api_secret.get_secret_value(),
        testnet=settings.exchange.testnet,
    ) as client:
        # Fetch last 100 hourly candles for BTC/USDT
        bars = await client.fetch_ohlcv(
            symbol="BTC/USDT",
            timeframe="1h",
            limit=100,
        )

        print(f"\nFetched {len(bars)} OHLCV bars")
        print("\nLast 5 bars:")
        for bar in bars[-5:]:
            dt = datetime.fromtimestamp(bar.timestamp / 1000)
            print(
                f"  {dt.isoformat()} | "
                f"O: {bar.open:,.2f} | "
                f"H: {bar.high:,.2f} | "
                f"L: {bar.low:,.2f} | "
                f"C: {bar.close:,.2f} | "
                f"V: {bar.volume:,.2f}"
            )


async def example_fetch_ticker():
    """Example: Fetch current ticker data."""
    print("\n" + "=" * 60)
    print("Example 2: Fetching Ticker Data")
    print("=" * 60)

    settings = get_settings()

    async with ExchangeClient(
        exchange_id="binance",
        api_key=settings.exchange.api_key.get_secret_value(),
        api_secret=settings.exchange.api_secret.get_secret_value(),
        testnet=settings.exchange.testnet,
    ) as client:
        # Fetch current ticker for multiple symbols
        symbols = ["BTC/USDT", "ETH/USDT"]

        for symbol in symbols:
            ticker = await client.fetch_ticker(symbol)
            print(f"\n{ticker.symbol}:")
            print(f"  Last Price:     {ticker.last:,.2f}")
            print(f"  Bid:            {ticker.bid:,.2f}")
            print(f"  Ask:            {ticker.ask:,.2f}")
            print(f"  24h High:       {ticker.high_24h:,.2f}")
            print(f"  24h Low:        {ticker.low_24h:,.2f}")
            print(f"  24h Volume:     {ticker.volume_24h:,.2f}")


async def example_fetch_funding_rate():
    """Example: Fetch funding rate for perpetual futures."""
    print("\n" + "=" * 60)
    print("Example 3: Fetching Funding Rate")
    print("=" * 60)

    settings = get_settings()

    async with ExchangeClient(
        exchange_id="binance",
        api_key=settings.exchange.api_key.get_secret_value(),
        api_secret=settings.exchange.api_secret.get_secret_value(),
        testnet=settings.exchange.testnet,
    ) as client:
        funding = await client.fetch_funding_rate("BTC/USDT")

        print(f"\nFunding Rate for {funding.symbol}:")
        print(f"  Current Rate:      {funding.rate * 100:.4f}%")
        print(
            f"  Next Funding Time: {datetime.fromtimestamp(funding.next_funding_time / 1000).isoformat()}"
        )

        # Interpret funding rate
        if funding.rate > 0:
            print("  Direction:         Longs pay shorts (bullish sentiment)")
        else:
            print("  Direction:         Shorts pay longs (bearish sentiment)")


async def example_fetch_date_range():
    """Example: Fetch OHLCV data for specific date range."""
    print("\n" + "=" * 60)
    print("Example 4: Fetching Date Range")
    print("=" * 60)

    settings = get_settings()

    # Fetch last 7 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    print(f"\nFetching data from {start_date.date()} to {end_date.date()}")

    async with ExchangeClient(
        exchange_id="binance",
        api_key=settings.exchange.api_key.get_secret_value(),
        api_secret=settings.exchange.api_secret.get_secret_value(),
        testnet=settings.exchange.testnet,
    ) as client:
        bars = await client.fetch_ohlcv_range(
            symbol="BTC/USDT",
            timeframe="1h",
            start=start_date,
            end=end_date,
        )

        print(f"\nFetched {len(bars)} hourly bars")
        print(f"First bar: {datetime.fromtimestamp(bars[0].timestamp / 1000).isoformat()}")
        print(f"Last bar:  {datetime.fromtimestamp(bars[-1].timestamp / 1000).isoformat()}")

        # Calculate some statistics
        highs = [bar.high for bar in bars]
        lows = [bar.low for bar in bars]
        volumes = [bar.volume for bar in bars]

        print("\nWeek Statistics:")
        print(f"  Highest Price:  {max(highs):,.2f}")
        print(f"  Lowest Price:   {min(lows):,.2f}")
        print(f"  Avg Volume:     {sum(volumes) / len(volumes):,.2f}")


async def example_download_historical():
    """Example: Download historical data to CSV."""
    print("\n" + "=" * 60)
    print("Example 5: Downloading Historical Data")
    print("=" * 60)

    settings = get_settings()

    downloader = HistoricalDataDownloader(
        exchange_id="binance",
        api_key=settings.exchange.api_key.get_secret_value(),
        api_secret=settings.exchange.api_secret.get_secret_value(),
        testnet=settings.exchange.testnet,
    )

    # Download last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    print(f"\nDownloading BTC/USDT 1h data from {start_date.date()} to {end_date.date()}")

    # Progress callback
    def progress_callback(current: int, total: int, symbol: str):
        if total > 0:
            percent = (current / total) * 100
            print(f"\rProgress: {current}/{total} bars ({percent:.1f}%)", end="", flush=True)

    try:
        file_path = await downloader.download_historical(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=start_date,
            end_date=end_date,
            output_dir="data/historical",
            progress_callback=progress_callback,
            resume=True,
        )

        print(f"\n\nData saved to: {file_path}")
        print(f"File size: {Path(file_path).stat().st_size / 1024:.2f} KB")

    except Exception as e:
        print(f"\nError downloading data: {e}")


async def example_convenience_functions():
    """Example: Using convenience functions."""
    print("\n" + "=" * 60)
    print("Example 6: Convenience Functions")
    print("=" * 60)

    print("\nUsing convenience functions (uses default settings):")

    # Fetch OHLCV
    bars = await fetch_latest_ohlcv("BTC/USDT", timeframe="1h", limit=10)
    print(f"\nFetched {len(bars)} bars using fetch_latest_ohlcv()")
    print(f"Latest close: {bars[-1].close:,.2f}")

    # Fetch ticker
    ticker = await fetch_latest_ticker("BTC/USDT")
    print("\nFetched ticker using fetch_latest_ticker()")
    print(f"Current price: {ticker.last:,.2f}")


async def example_parallel_fetching():
    """Example: Fetch data for multiple symbols in parallel."""
    print("\n" + "=" * 60)
    print("Example 7: Parallel Fetching")
    print("=" * 60)

    settings = get_settings()

    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]

    print(f"\nFetching data for {len(symbols)} symbols in parallel...")

    async with ExchangeClient(
        exchange_id="binance",
        api_key=settings.exchange.api_key.get_secret_value(),
        api_secret=settings.exchange.api_secret.get_secret_value(),
        testnet=settings.exchange.testnet,
    ) as client:
        # Fetch all tickers concurrently
        tasks = [client.fetch_ticker(symbol) for symbol in symbols]
        tickers = await asyncio.gather(*tasks)

        print("\nResults:")
        for ticker in tickers:
            print(
                f"  {ticker.symbol:12s} | Last: {ticker.last:12,.2f} | 24h Vol: {ticker.volume_24h:12,.2f}"
            )


async def main():
    """Run all examples."""
    # Setup logging
    config = LogConfig(
        level="INFO",
        format="pretty",
        console_output=True,
    )
    setup_logging(config)

    logger = get_logger(__name__)
    logger.info("starting_fetcher_examples")

    print("=" * 60)
    print("IFTB Data Fetcher Examples")
    print("=" * 60)

    try:
        # Run examples
        await example_fetch_ohlcv()
        await example_fetch_ticker()
        await example_fetch_funding_rate()
        await example_fetch_date_range()
        await example_download_historical()
        await example_convenience_functions()
        await example_parallel_fetching()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        logger.error("example_failed", error=str(e), exc_info=True)
        print(f"\n\nError: {e}")
        print("\nMake sure you have:")
        print("  1. Set up .env file with valid API credentials")
        print("  2. Installed all dependencies")
        print("  3. Valid network connection to exchange")


if __name__ == "__main__":
    asyncio.run(main())
