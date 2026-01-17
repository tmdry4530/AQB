"""
Price Data Fetcher Module for IFTB Trading Bot.

This module handles all exchange data fetching using CCXT, providing async
interfaces for fetching OHLCV data, tickers, funding rates, and open interest.
Includes robust error handling, rate limiting, and historical data downloading.

Example Usage:
    ```python
    from iftb.data.fetcher import ExchangeClient, HistoricalDataDownloader
    from datetime import datetime, timedelta

    # Fetch recent OHLCV data
    async with ExchangeClient("binance", api_key, api_secret) as client:
        ohlcv = await client.fetch_ohlcv("BTC/USDT", "1h", limit=100)
        ticker = await client.fetch_ticker("BTC/USDT")

    # Download historical data
    downloader = HistoricalDataDownloader("binance", api_key, api_secret)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*6)  # 6 years

    file_path = await downloader.download_historical(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date=start_date,
        end_date=end_date,
        output_dir="data/historical"
    )
    ```
"""

import asyncio
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import ccxt.async_support as ccxt
from ccxt.base.errors import (
    AuthenticationError,
    BadRequest,
    ExchangeNotAvailable,
    NetworkError,
    RateLimitExceeded,
    RequestTimeout,
)

from iftb.config import get_settings
from iftb.utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class OHLCVBar:
    """
    OHLCV candlestick bar representation.

    Attributes:
        timestamp: Unix timestamp in milliseconds
        open: Opening price
        high: Highest price in period
        low: Lowest price in period
        close: Closing price
        volume: Trading volume in base currency
    """
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_ccxt(cls, data: list) -> "OHLCVBar":
        """
        Create OHLCVBar from CCXT array format.

        Args:
            data: CCXT OHLCV array [timestamp, open, high, low, close, volume]

        Returns:
            OHLCVBar instance
        """
        return cls(
            timestamp=int(data[0]),
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5]),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class Ticker:
    """
    Ticker information for a trading pair.

    Attributes:
        symbol: Trading pair symbol (e.g., "BTC/USDT")
        last: Last traded price
        bid: Best bid price
        ask: Best ask price
        high_24h: 24-hour high price
        low_24h: 24-hour low price
        volume_24h: 24-hour volume in base currency
        timestamp: Unix timestamp in milliseconds
    """
    symbol: str
    last: float
    bid: float
    ask: float
    high_24h: float
    low_24h: float
    volume_24h: float
    timestamp: int

    @classmethod
    def from_ccxt(cls, symbol: str, data: dict) -> "Ticker":
        """
        Create Ticker from CCXT dictionary format.

        Args:
            symbol: Trading pair symbol
            data: CCXT ticker dictionary

        Returns:
            Ticker instance
        """
        return cls(
            symbol=symbol,
            last=float(data.get("last", 0)),
            bid=float(data.get("bid", 0)),
            ask=float(data.get("ask", 0)),
            high_24h=float(data.get("high", 0)),
            low_24h=float(data.get("low", 0)),
            volume_24h=float(data.get("baseVolume", 0)),
            timestamp=int(data.get("timestamp", 0)),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class FundingRate:
    """
    Funding rate information for perpetual futures.

    Attributes:
        symbol: Trading pair symbol
        rate: Funding rate (positive = longs pay shorts)
        next_funding_time: Unix timestamp of next funding event in milliseconds
        timestamp: Unix timestamp in milliseconds
    """
    symbol: str
    rate: float
    next_funding_time: int
    timestamp: int

    @classmethod
    def from_ccxt(cls, symbol: str, data: dict) -> "FundingRate":
        """
        Create FundingRate from CCXT dictionary format.

        Args:
            symbol: Trading pair symbol
            data: CCXT funding rate dictionary

        Returns:
            FundingRate instance
        """
        return cls(
            symbol=symbol,
            rate=float(data.get("fundingRate", 0)),
            next_funding_time=int(data.get("fundingTimestamp", 0)),
            timestamp=int(data.get("timestamp", 0)),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return asdict(self)


# =============================================================================
# Exchange Client
# =============================================================================


class ExchangeClient:
    """
    Async CCXT exchange client wrapper with robust error handling.

    Provides high-level interface for fetching market data with automatic
    retries, rate limiting, and error handling.

    Example:
        ```python
        async with ExchangeClient("binance", api_key, api_secret) as client:
            ohlcv = await client.fetch_ohlcv("BTC/USDT", "1h")
            ticker = await client.fetch_ticker("BTC/USDT")
        ```
    """

    def __init__(
        self,
        exchange_id: str,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
    ):
        """
        Initialize exchange client.

        Args:
            exchange_id: Exchange identifier (e.g., "binance", "bybit")
            api_key: API key for authentication
            api_secret: API secret for authentication
            testnet: Whether to use testnet environment
        """
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.exchange: Optional[ccxt.Exchange] = None
        self._connected = False
        self._rate_limiter = asyncio.Semaphore(10)  # Max 10 concurrent requests

        logger.info(
            "exchange_client_initialized",
            exchange=exchange_id,
            testnet=testnet,
        )

    async def __aenter__(self) -> "ExchangeClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """
        Initialize connection to exchange.

        Creates the CCXT exchange instance and loads markets.

        Raises:
            AuthenticationError: Invalid API credentials
            ExchangeNotAvailable: Exchange is down or unreachable
        """
        if self._connected:
            logger.warning("exchange_already_connected", exchange=self.exchange_id)
            return

        try:
            # Get exchange class
            exchange_class = getattr(ccxt, self.exchange_id)

            # Configure exchange
            config = {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future",  # Use futures by default
                },
            }

            # Add testnet configuration
            if self.testnet:
                if self.exchange_id == "binance":
                    config["options"]["defaultType"] = "future"
                    config["urls"] = {
                        "api": {
                            "public": "https://testnet.binancefuture.com",
                            "private": "https://testnet.binancefuture.com",
                        }
                    }
                elif self.exchange_id == "bybit":
                    config["urls"] = {
                        "api": "https://api-testnet.bybit.com",
                    }

            self.exchange = exchange_class(config)

            # Load markets
            await self.exchange.load_markets()

            self._connected = True
            logger.info(
                "exchange_connected",
                exchange=self.exchange_id,
                testnet=self.testnet,
                markets_loaded=len(self.exchange.markets),
            )

        except AuthenticationError as e:
            logger.error(
                "exchange_authentication_failed",
                exchange=self.exchange_id,
                error=str(e),
            )
            raise
        except Exception as e:
            logger.error(
                "exchange_connection_failed",
                exchange=self.exchange_id,
                error=str(e),
            )
            raise

    async def close(self) -> None:
        """
        Close exchange connection and cleanup resources.
        """
        if self.exchange and self._connected:
            await self.exchange.close()
            self._connected = False
            logger.info("exchange_closed", exchange=self.exchange_id)

    async def _retry_request(
        self,
        func: Callable,
        *args,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        **kwargs,
    ) -> Any:
        """
        Execute request with exponential backoff retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for function
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before first retry
            **kwargs: Keyword arguments for function

        Returns:
            Result from successful function execution

        Raises:
            Exception: If all retry attempts fail
        """
        delay = initial_delay
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                async with self._rate_limiter:
                    result = await func(*args, **kwargs)

                    if attempt > 0:
                        logger.info(
                            "request_retry_succeeded",
                            function=func.__name__,
                            attempt=attempt + 1,
                        )

                    return result

            except RateLimitExceeded as e:
                last_exception = e
                wait_time = delay * (2 ** attempt)
                logger.warning(
                    "rate_limit_exceeded",
                    function=func.__name__,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    wait_time=wait_time,
                )

                if attempt < max_retries:
                    await asyncio.sleep(wait_time)
                    continue

            except (NetworkError, RequestTimeout, ExchangeNotAvailable) as e:
                last_exception = e
                wait_time = delay * (2 ** attempt)
                logger.warning(
                    "network_error",
                    function=func.__name__,
                    error=str(e),
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    wait_time=wait_time,
                )

                if attempt < max_retries:
                    await asyncio.sleep(wait_time)
                    continue

            except BadRequest as e:
                # Don't retry bad requests
                logger.error(
                    "bad_request",
                    function=func.__name__,
                    error=str(e),
                )
                raise

            except Exception as e:
                last_exception = e
                logger.error(
                    "request_failed",
                    function=func.__name__,
                    error=str(e),
                    attempt=attempt + 1,
                )

                if attempt < max_retries:
                    await asyncio.sleep(delay * (2 ** attempt))
                    continue

        # All retries exhausted
        logger.error(
            "request_failed_all_retries",
            function=func.__name__,
            max_retries=max_retries,
            error=str(last_exception),
        )
        raise last_exception

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        limit: int = 500,
    ) -> list[OHLCVBar]:
        """
        Fetch OHLCV candlestick data.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe string (e.g., "1m", "5m", "1h", "1d")
            since: Unix timestamp in milliseconds to fetch from (optional)
            limit: Maximum number of candles to fetch (default: 500)

        Returns:
            List of OHLCVBar instances

        Raises:
            ValueError: If exchange not connected
            BadRequest: Invalid symbol or timeframe
        """
        if not self._connected or not self.exchange:
            raise ValueError("Exchange not connected. Call connect() first.")

        logger.debug(
            "fetching_ohlcv",
            exchange=self.exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            limit=limit,
        )

        data = await self._retry_request(
            self.exchange.fetch_ohlcv,
            symbol,
            timeframe,
            since,
            limit,
        )

        bars = [OHLCVBar.from_ccxt(candle) for candle in data]

        logger.info(
            "ohlcv_fetched",
            exchange=self.exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            bars_count=len(bars),
        )

        return bars

    async def fetch_ohlcv_range(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[OHLCVBar]:
        """
        Fetch OHLCV data for a specific date range with pagination.

        Automatically handles pagination to fetch all data in the range.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe string (e.g., "1m", "5m", "1h", "1d")
            start: Start datetime (inclusive)
            end: End datetime (exclusive)

        Returns:
            List of OHLCVBar instances covering the entire range

        Raises:
            ValueError: If exchange not connected or invalid date range
        """
        if not self._connected or not self.exchange:
            raise ValueError("Exchange not connected. Call connect() first.")

        if start >= end:
            raise ValueError("Start date must be before end date")

        logger.info(
            "fetching_ohlcv_range",
            exchange=self.exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start=start.isoformat(),
            end=end.isoformat(),
        )

        all_bars = []
        since = int(start.timestamp() * 1000)
        end_ts = int(end.timestamp() * 1000)

        # Calculate timeframe duration in milliseconds
        timeframe_duration = self._parse_timeframe_to_ms(timeframe)

        page_count = 0
        while since < end_ts:
            bars = await self.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=1000,  # Max limit for most exchanges
            )

            if not bars:
                break

            # Filter bars within range
            filtered_bars = [
                bar for bar in bars
                if since <= bar.timestamp < end_ts
            ]

            all_bars.extend(filtered_bars)

            # Update since to last bar timestamp + timeframe duration
            since = bars[-1].timestamp + timeframe_duration
            page_count += 1

            logger.debug(
                "ohlcv_range_page_fetched",
                page=page_count,
                bars_count=len(filtered_bars),
                total_bars=len(all_bars),
            )

            # Prevent infinite loops
            if page_count > 10000:
                logger.error(
                    "ohlcv_range_pagination_limit",
                    page_count=page_count,
                )
                break

            # Rate limiting delay
            await asyncio.sleep(0.1)

        logger.info(
            "ohlcv_range_completed",
            exchange=self.exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            total_bars=len(all_bars),
            pages=page_count,
        )

        return all_bars

    async def fetch_ticker(self, symbol: str) -> Ticker:
        """
        Fetch current ticker information.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")

        Returns:
            Ticker instance with current market data

        Raises:
            ValueError: If exchange not connected
        """
        if not self._connected or not self.exchange:
            raise ValueError("Exchange not connected. Call connect() first.")

        logger.debug(
            "fetching_ticker",
            exchange=self.exchange_id,
            symbol=symbol,
        )

        data = await self._retry_request(
            self.exchange.fetch_ticker,
            symbol,
        )

        ticker = Ticker.from_ccxt(symbol, data)

        logger.info(
            "ticker_fetched",
            exchange=self.exchange_id,
            symbol=symbol,
            last=ticker.last,
        )

        return ticker

    async def fetch_funding_rate(self, symbol: str) -> FundingRate:
        """
        Fetch current funding rate for perpetual futures.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")

        Returns:
            FundingRate instance

        Raises:
            ValueError: If exchange not connected
        """
        if not self._connected or not self.exchange:
            raise ValueError("Exchange not connected. Call connect() first.")

        logger.debug(
            "fetching_funding_rate",
            exchange=self.exchange_id,
            symbol=symbol,
        )

        data = await self._retry_request(
            self.exchange.fetch_funding_rate,
            symbol,
        )

        funding_rate = FundingRate.from_ccxt(symbol, data)

        logger.info(
            "funding_rate_fetched",
            exchange=self.exchange_id,
            symbol=symbol,
            rate=funding_rate.rate,
        )

        return funding_rate

    async def fetch_open_interest(self, symbol: str) -> dict:
        """
        Fetch open interest data for futures contracts.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")

        Returns:
            Dictionary with open interest data

        Raises:
            ValueError: If exchange not connected
        """
        if not self._connected or not self.exchange:
            raise ValueError("Exchange not connected. Call connect() first.")

        logger.debug(
            "fetching_open_interest",
            exchange=self.exchange_id,
            symbol=symbol,
        )

        # Note: Not all exchanges support this endpoint
        try:
            data = await self._retry_request(
                self.exchange.fetch_open_interest,
                symbol,
            )

            logger.info(
                "open_interest_fetched",
                exchange=self.exchange_id,
                symbol=symbol,
                open_interest=data.get("openInterest", 0),
            )

            return data

        except AttributeError:
            logger.warning(
                "open_interest_not_supported",
                exchange=self.exchange_id,
            )
            return {}

    @staticmethod
    def _parse_timeframe_to_ms(timeframe: str) -> int:
        """
        Parse timeframe string to milliseconds.

        Args:
            timeframe: Timeframe string (e.g., "1m", "5m", "1h", "1d")

        Returns:
            Duration in milliseconds
        """
        units = {
            "m": 60 * 1000,
            "h": 60 * 60 * 1000,
            "d": 24 * 60 * 60 * 1000,
            "w": 7 * 24 * 60 * 60 * 1000,
        }

        if not timeframe:
            raise ValueError("Invalid timeframe")

        unit = timeframe[-1]
        if unit not in units:
            raise ValueError(f"Invalid timeframe unit: {unit}")

        try:
            value = int(timeframe[:-1])
        except ValueError:
            raise ValueError(f"Invalid timeframe value: {timeframe}")

        return value * units[unit]


# =============================================================================
# Historical Data Downloader
# =============================================================================


class HistoricalDataDownloader:
    """
    Download and manage historical OHLCV data.

    Handles bulk downloading of historical data with progress tracking,
    resume capability, and rate limiting.

    Example:
        ```python
        downloader = HistoricalDataDownloader("binance", api_key, api_secret)

        def progress_callback(current: int, total: int, symbol: str):
            print(f"Progress: {current}/{total} bars downloaded for {symbol}")

        file_path = await downloader.download_historical(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=datetime(2018, 1, 1),
            end_date=datetime.now(),
            output_dir="data/historical",
            progress_callback=progress_callback,
        )
        ```
    """

    def __init__(
        self,
        exchange_id: str,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
    ):
        """
        Initialize historical data downloader.

        Args:
            exchange_id: Exchange identifier (e.g., "binance", "bybit")
            api_key: API key for authentication
            api_secret: API secret for authentication
            testnet: Whether to use testnet environment
        """
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        logger.info(
            "historical_downloader_initialized",
            exchange=exchange_id,
        )

    async def download_historical(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        output_dir: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        resume: bool = True,
    ) -> str:
        """
        Download historical OHLCV data to CSV file.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe string (e.g., "1h", "4h", "1d")
            start_date: Start date for historical data
            end_date: End date for historical data
            output_dir: Directory to save CSV file
            progress_callback: Optional callback(current, total, symbol) for progress
            resume: Whether to resume incomplete downloads

        Returns:
            Path to the downloaded CSV file

        Raises:
            ValueError: Invalid parameters
        """
        logger.info(
            "starting_historical_download",
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        # Prepare output directory and file
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        symbol_filename = symbol.replace("/", "_")
        csv_filename = f"{symbol_filename}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        csv_path = output_path / csv_filename

        # Check for existing data if resume is enabled
        existing_bars = []
        last_timestamp = None

        if resume and csv_path.exists():
            logger.info(
                "resuming_download",
                file=str(csv_path),
            )
            existing_bars = self._load_existing_data(csv_path)
            if existing_bars:
                last_timestamp = existing_bars[-1].timestamp
                logger.info(
                    "existing_data_found",
                    bars_count=len(existing_bars),
                    last_timestamp=last_timestamp,
                )

        # Determine actual start date
        actual_start = start_date
        if last_timestamp:
            # Resume from last timestamp
            actual_start = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(hours=1)

        # Download data
        async with ExchangeClient(
            self.exchange_id,
            self.api_key,
            self.api_secret,
            self.testnet,
        ) as client:
            new_bars = await client.fetch_ohlcv_range(
                symbol=symbol,
                timeframe=timeframe,
                start=actual_start,
                end=end_date,
            )

        # Combine with existing data
        all_bars = existing_bars + new_bars

        # Remove duplicates based on timestamp
        unique_bars = {}
        for bar in all_bars:
            unique_bars[bar.timestamp] = bar

        sorted_bars = sorted(unique_bars.values(), key=lambda x: x.timestamp)

        # Save to CSV
        self._save_to_csv(csv_path, sorted_bars)

        logger.info(
            "historical_download_completed",
            symbol=symbol,
            timeframe=timeframe,
            total_bars=len(sorted_bars),
            new_bars=len(new_bars),
            file=str(csv_path),
        )

        # Call progress callback with final count
        if progress_callback:
            progress_callback(len(sorted_bars), len(sorted_bars), symbol)

        return str(csv_path)

    def _load_existing_data(self, csv_path: Path) -> list[OHLCVBar]:
        """
        Load existing OHLCV data from CSV file.

        Args:
            csv_path: Path to CSV file

        Returns:
            List of OHLCVBar instances
        """
        bars = []

        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    bar = OHLCVBar(
                        timestamp=int(row["timestamp"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                    bars.append(bar)
        except Exception as e:
            logger.warning(
                "failed_to_load_existing_data",
                file=str(csv_path),
                error=str(e),
            )

        return bars

    def _save_to_csv(self, csv_path: Path, bars: list[OHLCVBar]) -> None:
        """
        Save OHLCV bars to CSV file.

        Args:
            csv_path: Path to CSV file
            bars: List of OHLCVBar instances
        """
        with open(csv_path, "w", newline="") as f:
            if not bars:
                return

            fieldnames = ["timestamp", "open", "high", "low", "close", "volume", "datetime"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for bar in bars:
                row = bar.to_dict()
                # Add human-readable datetime
                row["datetime"] = datetime.fromtimestamp(bar.timestamp / 1000).isoformat()
                writer.writerow(row)

        logger.debug(
            "data_saved_to_csv",
            file=str(csv_path),
            bars_count=len(bars),
        )


# =============================================================================
# Convenience Functions
# =============================================================================


async def fetch_latest_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
) -> list[OHLCVBar]:
    """
    Convenience function to fetch latest OHLCV data using default settings.

    Args:
        symbol: Trading pair symbol (e.g., "BTC/USDT")
        timeframe: Timeframe string (default: "1h")
        limit: Number of candles to fetch (default: 100)

    Returns:
        List of OHLCVBar instances
    """
    settings = get_settings()

    async with ExchangeClient(
        exchange_id="binance",  # Default to Binance
        api_key=settings.exchange.api_key.get_secret_value(),
        api_secret=settings.exchange.api_secret.get_secret_value(),
        testnet=settings.exchange.testnet,
    ) as client:
        return await client.fetch_ohlcv(symbol, timeframe, limit=limit)


async def fetch_latest_ticker(symbol: str) -> Ticker:
    """
    Convenience function to fetch latest ticker using default settings.

    Args:
        symbol: Trading pair symbol (e.g., "BTC/USDT")

    Returns:
        Ticker instance
    """
    settings = get_settings()

    async with ExchangeClient(
        exchange_id="binance",  # Default to Binance
        api_key=settings.exchange.api_key.get_secret_value(),
        api_secret=settings.exchange.api_secret.get_secret_value(),
        testnet=settings.exchange.testnet,
    ) as client:
        return await client.fetch_ticker(symbol)
