"""
External data API clients for IFTB trading bot.

This module provides clients for fetching data from external APIs:
- Fear & Greed Index from alternative.me
- Funding rates, open interest, and long/short ratios from Coinglass

Example Usage:
    ```python
    from iftb.data.external import ExternalDataAggregator

    # Initialize aggregator
    aggregator = ExternalDataAggregator()

    # Fetch all market context data
    context = await aggregator.fetch_all()

    print(f"Fear & Greed: {context.fear_greed.value}")
    print(f"Funding Rate: {context.funding.rate}")
    print(f"Open Interest: {context.open_interest.open_interest}")
    ```
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

from iftb.utils import get_logger

logger = get_logger(__name__)


@dataclass
class FearGreedData:
    """Fear & Greed Index data.

    Attributes:
        value: Index value from 0-100 (0=Extreme Fear, 100=Extreme Greed)
        classification: Human-readable classification
        timestamp: Data timestamp
    """
    value: int
    classification: str
    timestamp: datetime

    def __post_init__(self) -> None:
        """Validate data after initialization."""
        if not 0 <= self.value <= 100:
            raise ValueError(f"Fear & Greed value must be 0-100, got {self.value}")


@dataclass
class FundingData:
    """Futures funding rate data.

    Attributes:
        symbol: Trading symbol (e.g., "BTC")
        rate: Current funding rate (as decimal, e.g., 0.0001 = 0.01%)
        predicted_rate: Predicted next funding rate
        next_funding_time: Timestamp of next funding settlement
    """
    symbol: str
    rate: float
    predicted_rate: float
    next_funding_time: datetime


@dataclass
class OpenInterestData:
    """Open interest data.

    Attributes:
        symbol: Trading symbol (e.g., "BTC")
        open_interest: Total open interest in USD
        oi_change_24h: 24-hour change in open interest (percentage)
    """
    symbol: str
    open_interest: float
    oi_change_24h: float


@dataclass
class LongShortData:
    """Long/short ratio data.

    Attributes:
        symbol: Trading symbol (e.g., "BTC")
        long_ratio: Percentage of long positions (0-1)
        short_ratio: Percentage of short positions (0-1)
        timestamp: Data timestamp
    """
    symbol: str
    long_ratio: float
    short_ratio: float
    timestamp: datetime

    def __post_init__(self) -> None:
        """Validate ratios sum to approximately 1."""
        total = self.long_ratio + self.short_ratio
        if not 0.99 <= total <= 1.01:
            logger.warning(
                "long_short_ratio_invalid",
                long_ratio=self.long_ratio,
                short_ratio=self.short_ratio,
                total=total,
            )


@dataclass
class MarketContext:
    """Aggregated market context from all external sources.

    Attributes:
        fear_greed: Fear & Greed Index data (None if unavailable)
        funding: Funding rate data (None if unavailable)
        open_interest: Open interest data (None if unavailable)
        long_short: Long/short ratio data (None if unavailable)
        fetch_time: Timestamp when data was fetched
        errors: List of errors encountered during fetching
    """
    fear_greed: FearGreedData | None = None
    funding: FundingData | None = None
    open_interest: OpenInterestData | None = None
    long_short: LongShortData | None = None
    fetch_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    errors: list[str] = field(default_factory=list)


class FearGreedClient:
    """Client for fetching Fear & Greed Index from alternative.me API.

    The Fear & Greed Index is a sentiment indicator that ranges from 0 (Extreme Fear)
    to 100 (Extreme Greed), helping traders gauge market sentiment.
    """

    BASE_URL = "https://api.alternative.me/fng/"
    TIMEOUT = 10.0

    def __init__(self, timeout: float = TIMEOUT) -> None:
        """Initialize Fear & Greed client.

        Args:
            timeout: HTTP request timeout in seconds
        """
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client.

        Returns:
            Configured httpx AsyncClient
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "FearGreedClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        await self.close()

    def _parse_value(self, data: dict[str, str]) -> FearGreedData:
        """Parse Fear & Greed data from API response.

        Args:
            data: Raw data dictionary from API

        Returns:
            Parsed FearGreedData object

        Raises:
            ValueError: If data is malformed
        """
        try:
            value = int(data["value"])
            classification = data["value_classification"]
            timestamp = datetime.fromtimestamp(int(data["timestamp"]), tz=UTC)

            return FearGreedData(
                value=value,
                classification=classification,
                timestamp=timestamp,
            )
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Failed to parse Fear & Greed data: {e}") from e

    async def fetch_current(self) -> FearGreedData:
        """Fetch current Fear & Greed Index.

        Returns:
            Current Fear & Greed Index data

        Raises:
            httpx.HTTPError: If request fails
            ValueError: If response is malformed
        """
        client = await self._get_client()

        logger.debug("fetching_fear_greed_current")

        response = await client.get(self.BASE_URL)
        response.raise_for_status()

        json_data = response.json()

        if "data" not in json_data or not json_data["data"]:
            raise ValueError("No data in Fear & Greed API response")

        data = self._parse_value(json_data["data"][0])

        logger.info(
            "fear_greed_fetched",
            value=data.value,
            classification=data.classification,
        )

        return data

    async def fetch_historical(self, limit: int = 30) -> list[FearGreedData]:
        """Fetch historical Fear & Greed Index data.

        Args:
            limit: Number of historical data points to fetch (max 30)

        Returns:
            List of historical Fear & Greed data points

        Raises:
            httpx.HTTPError: If request fails
            ValueError: If response is malformed
        """
        if limit < 1 or limit > 30:
            raise ValueError(f"Limit must be between 1 and 30, got {limit}")

        client = await self._get_client()

        logger.debug("fetching_fear_greed_historical", limit=limit)

        response = await client.get(self.BASE_URL, params={"limit": limit})
        response.raise_for_status()

        json_data = response.json()

        if "data" not in json_data or not json_data["data"]:
            raise ValueError("No data in Fear & Greed API response")

        results = [self._parse_value(item) for item in json_data["data"]]

        logger.info("fear_greed_historical_fetched", count=len(results))

        return results


class CoinglassClient:
    """Client for fetching data from Coinglass API.

    Provides access to funding rates, open interest, and long/short ratios
    for cryptocurrency futures markets.

    Note: This is a placeholder implementation. Coinglass API requires authentication
    and has specific endpoints. Update BASE_URL and add API key when available.
    """

    BASE_URL = "https://open-api.coinglass.com/public/v2"
    TIMEOUT = 10.0

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = TIMEOUT,
    ) -> None:
        """Initialize Coinglass client.

        Args:
            api_key: Optional API key for authenticated requests
            timeout: HTTP request timeout in seconds
        """
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client.

        Returns:
            Configured httpx AsyncClient with optional auth headers
        """
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key

            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=headers,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "CoinglassClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        await self.close()

    async def fetch_funding_rate(self, symbol: str = "BTC") -> FundingData:
        """Fetch current funding rate for symbol.

        Args:
            symbol: Trading symbol (default: "BTC")

        Returns:
            Current funding rate data

        Raises:
            httpx.HTTPError: If request fails
            ValueError: If response is malformed
        """
        client = await self._get_client()

        logger.debug("fetching_funding_rate", symbol=symbol)

        # Coinglass API endpoint for funding rates
        url = f"{self.BASE_URL}/funding"

        response = await client.get(url, params={"symbol": symbol})
        response.raise_for_status()

        json_data = response.json()

        if "data" not in json_data:
            raise ValueError("No data in Coinglass funding rate response")

        data = json_data["data"]

        # Parse response based on Coinglass API format
        # This is a generic parser - adjust based on actual API response
        funding = FundingData(
            symbol=symbol,
            rate=float(data.get("fundingRate", 0.0)),
            predicted_rate=float(data.get("predictedRate", 0.0)),
            next_funding_time=datetime.fromtimestamp(
                int(data.get("nextFundingTime", 0)),
                tz=UTC,
            ),
        )

        logger.info(
            "funding_rate_fetched",
            symbol=symbol,
            rate=funding.rate,
        )

        return funding

    async def fetch_open_interest(self, symbol: str = "BTC") -> OpenInterestData:
        """Fetch open interest data for symbol.

        Args:
            symbol: Trading symbol (default: "BTC")

        Returns:
            Current open interest data

        Raises:
            httpx.HTTPError: If request fails
            ValueError: If response is malformed
        """
        client = await self._get_client()

        logger.debug("fetching_open_interest", symbol=symbol)

        # Coinglass API endpoint for open interest
        url = f"{self.BASE_URL}/openInterest"

        response = await client.get(url, params={"symbol": symbol})
        response.raise_for_status()

        json_data = response.json()

        if "data" not in json_data:
            raise ValueError("No data in Coinglass open interest response")

        data = json_data["data"]

        oi = OpenInterestData(
            symbol=symbol,
            open_interest=float(data.get("openInterest", 0.0)),
            oi_change_24h=float(data.get("change24h", 0.0)),
        )

        logger.info(
            "open_interest_fetched",
            symbol=symbol,
            open_interest=oi.open_interest,
            change_24h=oi.oi_change_24h,
        )

        return oi

    async def fetch_long_short_ratio(self, symbol: str = "BTC") -> LongShortData:
        """Fetch long/short ratio for symbol.

        Args:
            symbol: Trading symbol (default: "BTC")

        Returns:
            Current long/short ratio data

        Raises:
            httpx.HTTPError: If request fails
            ValueError: If response is malformed
        """
        client = await self._get_client()

        logger.debug("fetching_long_short_ratio", symbol=symbol)

        # Coinglass API endpoint for long/short ratio
        url = f"{self.BASE_URL}/longShortRatio"

        response = await client.get(url, params={"symbol": symbol})
        response.raise_for_status()

        json_data = response.json()

        if "data" not in json_data:
            raise ValueError("No data in Coinglass long/short ratio response")

        data = json_data["data"]

        ls = LongShortData(
            symbol=symbol,
            long_ratio=float(data.get("longRatio", 0.5)),
            short_ratio=float(data.get("shortRatio", 0.5)),
            timestamp=datetime.now(UTC),
        )

        logger.info(
            "long_short_ratio_fetched",
            symbol=symbol,
            long_ratio=ls.long_ratio,
            short_ratio=ls.short_ratio,
        )

        return ls


class ExternalDataAggregator:
    """Aggregates data from multiple external sources with caching.

    Fetches and caches market context data from Fear & Greed Index and Coinglass.
    Implements retry logic with exponential backoff and graceful fallback on errors.
    """

    DEFAULT_TTL = 300  # 5 minutes
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 1.0  # seconds

    def __init__(
        self,
        fear_greed_client: FearGreedClient | None = None,
        coinglass_client: CoinglassClient | None = None,
        cache_ttl: int = DEFAULT_TTL,
    ) -> None:
        """Initialize external data aggregator.

        Args:
            fear_greed_client: Optional custom Fear & Greed client
            coinglass_client: Optional custom Coinglass client
            cache_ttl: Cache time-to-live in seconds (default: 300)
        """
        self.fear_greed_client = fear_greed_client or FearGreedClient()
        self.coinglass_client = coinglass_client or CoinglassClient()
        self.cache_ttl = cache_ttl

        self._cache: MarketContext | None = None
        self._cache_time: datetime | None = None

    async def close(self) -> None:
        """Close all clients and cleanup resources."""
        await self.fear_greed_client.close()
        await self.coinglass_client.close()

    async def __aenter__(self) -> "ExternalDataAggregator":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        await self.close()

    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid.

        Returns:
            True if cache exists and is within TTL, False otherwise
        """
        if self._cache is None or self._cache_time is None:
            return False

        age = (datetime.now(UTC) - self._cache_time).total_seconds()
        return age < self.cache_ttl

    async def _fetch_with_retry(
        self,
        fetch_func: callable,
        name: str,
    ) -> tuple[object | None, str | None]:
        """Fetch data with exponential backoff retry.

        Args:
            fetch_func: Async function to call for fetching data
            name: Name of the data source for logging

        Returns:
            Tuple of (data, error_message). If successful, error is None.
            If failed after retries, data is None and error contains message.
        """
        backoff = self.INITIAL_BACKOFF

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                data = await fetch_func()
                return data, None
            except httpx.HTTPError as e:
                error_msg = f"{name} HTTP error: {e}"
                logger.warning(
                    f"{name.lower().replace(' ', '_')}_fetch_failed",
                    attempt=attempt,
                    max_retries=self.MAX_RETRIES,
                    error=str(e),
                )

                if attempt == self.MAX_RETRIES:
                    logger.error(f"{name.lower().replace(' ', '_')}_fetch_exhausted")
                    return None, error_msg

                await asyncio.sleep(backoff)
                backoff *= 2

            except ValueError as e:
                error_msg = f"{name} parse error: {e}"
                logger.error(
                    f"{name.lower().replace(' ', '_')}_parse_error",
                    error=str(e),
                )
                return None, error_msg
            except Exception as e:
                error_msg = f"{name} unexpected error: {e}"
                logger.error(
                    f"{name.lower().replace(' ', '_')}_unexpected_error",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return None, error_msg

        return None, f"{name} failed after {self.MAX_RETRIES} retries"

    async def fetch_all(self, symbol: str = "BTC", force_refresh: bool = False) -> MarketContext:
        """Fetch all external market data with caching.

        Fetches data from all sources concurrently. On failure, returns None
        for that data source and logs the error.

        Args:
            symbol: Trading symbol for Coinglass data (default: "BTC")
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            MarketContext with all available data. Unavailable data is None.
        """
        # Return cached data if valid and not forcing refresh
        if not force_refresh and self._is_cache_valid():
            logger.debug("returning_cached_market_context")
            return self._cache  # type: ignore[return-value]

        logger.info("fetching_market_context", symbol=symbol)

        errors: list[str] = []

        # Fetch all data concurrently
        fear_greed_result, funding_result, oi_result, ls_result = await asyncio.gather(
            self._fetch_with_retry(
                self.fear_greed_client.fetch_current,
                "Fear & Greed",
            ),
            self._fetch_with_retry(
                lambda: self.coinglass_client.fetch_funding_rate(symbol),
                "Funding Rate",
            ),
            self._fetch_with_retry(
                lambda: self.coinglass_client.fetch_open_interest(symbol),
                "Open Interest",
            ),
            self._fetch_with_retry(
                lambda: self.coinglass_client.fetch_long_short_ratio(symbol),
                "Long/Short Ratio",
            ),
        )

        # Unpack results and collect errors
        fear_greed, fg_error = fear_greed_result
        funding, funding_error = funding_result
        oi, oi_error = oi_result
        ls, ls_error = ls_result

        if fg_error:
            errors.append(fg_error)
        if funding_error:
            errors.append(funding_error)
        if oi_error:
            errors.append(oi_error)
        if ls_error:
            errors.append(ls_error)

        # Create market context
        context = MarketContext(
            fear_greed=fear_greed,  # type: ignore[arg-type]
            funding=funding,  # type: ignore[arg-type]
            open_interest=oi,  # type: ignore[arg-type]
            long_short=ls,  # type: ignore[arg-type]
            fetch_time=datetime.now(UTC),
            errors=errors,
        )

        # Update cache
        self._cache = context
        self._cache_time = context.fetch_time

        success_count = sum(
            1 for x in [fear_greed, funding, oi, ls] if x is not None
        )

        logger.info(
            "market_context_fetched",
            symbol=symbol,
            success_count=success_count,
            total_sources=4,
            error_count=len(errors),
        )

        return context
