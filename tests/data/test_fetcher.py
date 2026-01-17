"""
Tests for the data fetcher module.

Tests cover OHLCV fetching, ticker data, funding rates, and historical data downloading.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ccxt.base.errors import RateLimitExceeded, NetworkError

from iftb.data.fetcher import (
    ExchangeClient,
    FundingRate,
    HistoricalDataDownloader,
    OHLCVBar,
    Ticker,
)


# =============================================================================
# Test Data Structures
# =============================================================================


class TestOHLCVBar:
    """Test OHLCVBar dataclass."""

    def test_from_ccxt(self):
        """Test creating OHLCVBar from CCXT array."""
        ccxt_data = [1640000000000, 50000.0, 51000.0, 49000.0, 50500.0, 100.5]
        bar = OHLCVBar.from_ccxt(ccxt_data)

        assert bar.timestamp == 1640000000000
        assert bar.open == 50000.0
        assert bar.high == 51000.0
        assert bar.low == 49000.0
        assert bar.close == 50500.0
        assert bar.volume == 100.5

    def test_to_dict(self):
        """Test converting OHLCVBar to dictionary."""
        bar = OHLCVBar(
            timestamp=1640000000000,
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=100.5,
        )
        data = bar.to_dict()

        assert isinstance(data, dict)
        assert data["timestamp"] == 1640000000000
        assert data["close"] == 50500.0


class TestTicker:
    """Test Ticker dataclass."""

    def test_from_ccxt(self):
        """Test creating Ticker from CCXT dictionary."""
        ccxt_data = {
            "last": 50000.0,
            "bid": 49990.0,
            "ask": 50010.0,
            "high": 51000.0,
            "low": 49000.0,
            "baseVolume": 1000.0,
            "timestamp": 1640000000000,
        }
        ticker = Ticker.from_ccxt("BTC/USDT", ccxt_data)

        assert ticker.symbol == "BTC/USDT"
        assert ticker.last == 50000.0
        assert ticker.bid == 49990.0
        assert ticker.ask == 50010.0
        assert ticker.high_24h == 51000.0
        assert ticker.low_24h == 49000.0
        assert ticker.volume_24h == 1000.0


class TestFundingRate:
    """Test FundingRate dataclass."""

    def test_from_ccxt(self):
        """Test creating FundingRate from CCXT dictionary."""
        ccxt_data = {
            "fundingRate": 0.0001,
            "fundingTimestamp": 1640003600000,
            "timestamp": 1640000000000,
        }
        funding = FundingRate.from_ccxt("BTC/USDT", ccxt_data)

        assert funding.symbol == "BTC/USDT"
        assert funding.rate == 0.0001
        assert funding.next_funding_time == 1640003600000


# =============================================================================
# Test ExchangeClient
# =============================================================================


class TestExchangeClient:
    """Test ExchangeClient class."""

    @pytest.fixture
    def mock_exchange(self):
        """Create a mock CCXT exchange."""
        exchange = AsyncMock()
        exchange.markets = {"BTC/USDT": {}, "ETH/USDT": {}}
        exchange.load_markets = AsyncMock()
        exchange.close = AsyncMock()
        return exchange

    @pytest.fixture
    def client(self):
        """Create an ExchangeClient instance."""
        return ExchangeClient(
            exchange_id="binance",
            api_key="test_key",
            api_secret="test_secret",
            testnet=True,
        )

    @pytest.mark.asyncio
    async def test_connect(self, client, mock_exchange):
        """Test connecting to exchange."""
        with patch("iftb.data.fetcher.ccxt.binance", return_value=mock_exchange):
            await client.connect()

            assert client._connected
            mock_exchange.load_markets.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self, client, mock_exchange):
        """Test closing exchange connection."""
        with patch("iftb.data.fetcher.ccxt.binance", return_value=mock_exchange):
            await client.connect()
            await client.close()

            assert not client._connected
            mock_exchange.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_exchange):
        """Test using ExchangeClient as context manager."""
        with patch("iftb.data.fetcher.ccxt.binance", return_value=mock_exchange):
            async with ExchangeClient("binance", "key", "secret") as client:
                assert client._connected

            mock_exchange.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_ohlcv(self, client, mock_exchange):
        """Test fetching OHLCV data."""
        mock_ohlcv = [
            [1640000000000, 50000.0, 51000.0, 49000.0, 50500.0, 100.0],
            [1640003600000, 50500.0, 51500.0, 50000.0, 51000.0, 150.0],
        ]
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=mock_ohlcv)

        with patch("iftb.data.fetcher.ccxt.binance", return_value=mock_exchange):
            await client.connect()
            bars = await client.fetch_ohlcv("BTC/USDT", "1h", limit=2)

            assert len(bars) == 2
            assert isinstance(bars[0], OHLCVBar)
            assert bars[0].timestamp == 1640000000000
            assert bars[1].close == 51000.0

    @pytest.mark.asyncio
    async def test_fetch_ticker(self, client, mock_exchange):
        """Test fetching ticker data."""
        mock_ticker = {
            "last": 50000.0,
            "bid": 49990.0,
            "ask": 50010.0,
            "high": 51000.0,
            "low": 49000.0,
            "baseVolume": 1000.0,
            "timestamp": 1640000000000,
        }
        mock_exchange.fetch_ticker = AsyncMock(return_value=mock_ticker)

        with patch("iftb.data.fetcher.ccxt.binance", return_value=mock_exchange):
            await client.connect()
            ticker = await client.fetch_ticker("BTC/USDT")

            assert isinstance(ticker, Ticker)
            assert ticker.symbol == "BTC/USDT"
            assert ticker.last == 50000.0

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self, client, mock_exchange):
        """Test retry logic on rate limit errors."""
        # First call raises rate limit, second succeeds
        mock_exchange.fetch_ticker = AsyncMock(
            side_effect=[
                RateLimitExceeded("Rate limit exceeded"),
                {"last": 50000.0, "timestamp": 1640000000000},
            ]
        )

        with patch("iftb.data.fetcher.ccxt.binance", return_value=mock_exchange):
            await client.connect()
            # Reduce delay for testing
            ticker = await client._retry_request(
                mock_exchange.fetch_ticker,
                "BTC/USDT",
                initial_delay=0.01,
            )

            assert ticker["last"] == 50000.0
            assert mock_exchange.fetch_ticker.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, client, mock_exchange):
        """Test that retries eventually fail."""
        mock_exchange.fetch_ticker = AsyncMock(
            side_effect=NetworkError("Network error")
        )

        with patch("iftb.data.fetcher.ccxt.binance", return_value=mock_exchange):
            await client.connect()

            with pytest.raises(NetworkError):
                await client._retry_request(
                    mock_exchange.fetch_ticker,
                    "BTC/USDT",
                    max_retries=2,
                    initial_delay=0.01,
                )

    @pytest.mark.asyncio
    async def test_parse_timeframe_to_ms(self):
        """Test timeframe parsing."""
        assert ExchangeClient._parse_timeframe_to_ms("1m") == 60 * 1000
        assert ExchangeClient._parse_timeframe_to_ms("5m") == 5 * 60 * 1000
        assert ExchangeClient._parse_timeframe_to_ms("1h") == 60 * 60 * 1000
        assert ExchangeClient._parse_timeframe_to_ms("1d") == 24 * 60 * 60 * 1000

        with pytest.raises(ValueError):
            ExchangeClient._parse_timeframe_to_ms("invalid")

        with pytest.raises(ValueError):
            ExchangeClient._parse_timeframe_to_ms("1x")


# =============================================================================
# Test HistoricalDataDownloader
# =============================================================================


class TestHistoricalDataDownloader:
    """Test HistoricalDataDownloader class."""

    @pytest.fixture
    def downloader(self):
        """Create a HistoricalDataDownloader instance."""
        return HistoricalDataDownloader(
            exchange_id="binance",
            api_key="test_key",
            api_secret="test_secret",
        )

    @pytest.fixture
    def mock_bars(self):
        """Create mock OHLCV bars."""
        return [
            OHLCVBar(
                timestamp=1640000000000 + i * 3600000,
                open=50000.0 + i * 100,
                high=51000.0 + i * 100,
                low=49000.0 + i * 100,
                close=50500.0 + i * 100,
                volume=100.0,
            )
            for i in range(10)
        ]

    @pytest.mark.asyncio
    async def test_download_historical(self, downloader, mock_bars, tmp_path):
        """Test downloading historical data."""
        mock_client = AsyncMock()
        mock_client.fetch_ohlcv_range = AsyncMock(return_value=mock_bars)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("iftb.data.fetcher.ExchangeClient", return_value=mock_client):
            start_date = datetime(2021, 12, 20)
            end_date = datetime(2021, 12, 21)

            file_path = await downloader.download_historical(
                symbol="BTC/USDT",
                timeframe="1h",
                start_date=start_date,
                end_date=end_date,
                output_dir=str(tmp_path),
                resume=False,
            )

            assert Path(file_path).exists()
            assert Path(file_path).suffix == ".csv"

            # Verify CSV content
            with open(file_path, "r") as f:
                lines = f.readlines()
                assert len(lines) > 1  # Header + data rows
                assert "timestamp" in lines[0]

    def test_save_and_load_csv(self, downloader, mock_bars, tmp_path):
        """Test saving and loading CSV data."""
        csv_path = tmp_path / "test_data.csv"

        # Save
        downloader._save_to_csv(csv_path, mock_bars)
        assert csv_path.exists()

        # Load
        loaded_bars = downloader._load_existing_data(csv_path)
        assert len(loaded_bars) == len(mock_bars)
        assert loaded_bars[0].timestamp == mock_bars[0].timestamp
        assert loaded_bars[0].close == mock_bars[0].close

    @pytest.mark.asyncio
    async def test_resume_download(self, downloader, mock_bars, tmp_path):
        """Test resuming an incomplete download."""
        csv_path = tmp_path / "test_resume.csv"

        # Save first half of data
        downloader._save_to_csv(csv_path, mock_bars[:5])

        # Mock client to return second half
        mock_client = AsyncMock()
        mock_client.fetch_ohlcv_range = AsyncMock(return_value=mock_bars[5:])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("iftb.data.fetcher.ExchangeClient", return_value=mock_client):
            start_date = datetime(2021, 12, 20)
            end_date = datetime(2021, 12, 21)

            file_path = await downloader.download_historical(
                symbol="BTC/USDT",
                timeframe="1h",
                start_date=start_date,
                end_date=end_date,
                output_dir=str(tmp_path),
                resume=True,
            )

            # Verify all data is present
            loaded_bars = downloader._load_existing_data(Path(file_path))
            assert len(loaded_bars) == len(mock_bars)
