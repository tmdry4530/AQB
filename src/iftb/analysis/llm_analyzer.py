"""
LLM Market Analyzer with Veto System for IFTB Trading Bot.

This module provides AI-powered market analysis using Anthropic Claude API,
with a sophisticated veto system that can block trades based on negative
sentiment or low confidence. Includes caching to avoid duplicate API calls,
rate limiting, and robust error handling with multiple fallback modes.

Example Usage:
    ```python
    from iftb.analysis.llm_analyzer import LLMAnalyzer, LLMVetoSystem
    from iftb.config import get_settings
    from iftb.data import NewsMessage, MarketContext

    settings = get_settings()
    analyzer = LLMAnalyzer(
        api_key=settings.llm.anthropic_api_key.get_secret_value(),
        model=settings.llm.model,
    )

    # Analyze market conditions
    analysis = await analyzer.analyze_market(
        symbol="BTCUSDT",
        news_messages=recent_news,
        market_context=context,
        current_price=45000.0,
    )

    # Check if trade should be vetoed
    veto_system = LLMVetoSystem()
    should_block, reason = veto_system.should_veto_trade(
        analysis=analysis,
        trade_direction="long",
    )

    if should_block:
        print(f"Trade blocked: {reason}")
    ```

Fallback Modes:
    - CONSERVATIVE: Neutral sentiment, low confidence (no veto)
    - VETO_ALL: Block all trades until API recovers
    - CACHE_ONLY: Use only cached analyses
"""

import asyncio
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
import hashlib
import json
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import Message

from iftb.config import get_settings
from iftb.config.constants import (
    CONFIDENCE_VETO_THRESHOLD,
    SENTIMENT_CAUTION_THRESHOLD,
    SENTIMENT_VETO_THRESHOLD,
)
from iftb.data import LLMCache, MarketContext, NewsMessage, RedisClient
from iftb.utils import get_logger

logger = get_logger(__name__)


class SentimentScore(Enum):
    """Market sentiment classifications with numerical values."""

    VERY_BEARISH = -1.0
    BEARISH = -0.5
    NEUTRAL = 0.0
    BULLISH = 0.5
    VERY_BULLISH = 1.0

    def __str__(self) -> str:
        """Human-readable sentiment name."""
        return self.name.replace("_", " ").title()


class FallbackMode(Enum):
    """Fallback modes when LLM API is unavailable."""

    CONSERVATIVE = "conservative"  # Neutral sentiment, low confidence
    VETO_ALL = "veto_all"  # Block all trades
    CACHE_ONLY = "cache_only"  # Use only cached analyses


@dataclass
class LLMAnalysis:
    """
    Result of LLM market analysis.

    Attributes:
        sentiment: Market sentiment classification
        confidence: Confidence level (0-1)
        summary: Brief analysis summary
        key_factors: List of key factors influencing the analysis
        should_veto: Whether this analysis suggests vetoing the trade
        veto_reason: Explanation if trade should be vetoed
        timestamp: When the analysis was performed
        model: LLM model used for analysis
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        cached: Whether this result came from cache
    """

    sentiment: SentimentScore
    confidence: float
    summary: str
    key_factors: list[str]
    should_veto: bool
    veto_reason: str | None
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool = False

    def __post_init__(self) -> None:
        """Validate confidence score."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for caching."""
        data = asdict(self)
        # Convert sentiment enum to value
        data["sentiment"] = self.sentiment.value
        # Convert timestamp to ISO format
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMAnalysis":
        """Create from cached dictionary."""
        # Parse sentiment value
        sentiment_value = data["sentiment"]
        sentiment = next(s for s in SentimentScore if s.value == sentiment_value)

        # Parse timestamp
        timestamp = datetime.fromisoformat(data["timestamp"])

        return cls(
            sentiment=sentiment,
            confidence=data["confidence"],
            summary=data["summary"],
            key_factors=data["key_factors"],
            should_veto=data["should_veto"],
            veto_reason=data.get("veto_reason"),
            timestamp=timestamp,
            model=data["model"],
            prompt_tokens=data["prompt_tokens"],
            completion_tokens=data["completion_tokens"],
            cached=True,
        )


class LLMVetoSystem:
    """
    Intelligent veto system that can block trades based on LLM analysis.

    The veto system applies multiple checks:
    1. Sentiment check: Block if sentiment is too bearish
    2. Confidence check: Block if confidence is too low
    3. Direction conflict: Block if sentiment conflicts with trade direction
    """

    def __init__(self, analyzer: "LLMAnalyzer | None" = None) -> None:
        """
        Initialize veto system with optional LLM analyzer.

        Args:
            analyzer: Optional LLM analyzer instance for direct analysis
        """
        self.analyzer = analyzer

    def should_veto_trade(self, analysis: LLMAnalysis, trade_direction: str) -> tuple[bool, str]:
        """
        Determine if a trade should be vetoed based on LLM analysis.

        Args:
            analysis: LLM market analysis result
            trade_direction: Intended trade direction ("long" or "short")

        Returns:
            Tuple of (should_veto, reason)
        """
        # Check 1: Extreme negative sentiment
        if analysis.sentiment.value <= SENTIMENT_VETO_THRESHOLD:
            return (
                True,
                f"Very bearish sentiment ({analysis.sentiment}) below veto threshold",
            )

        # Check 2: Very low confidence
        if analysis.confidence < CONFIDENCE_VETO_THRESHOLD:
            return (
                True,
                f"Low confidence ({analysis.confidence:.2f}) below veto threshold",
            )

        # Check 3: Direction conflict
        if trade_direction.lower() == "long" and analysis.sentiment.value < 0:
            return (
                True,
                f"Bearish sentiment ({analysis.sentiment}) conflicts with long direction",
            )

        if trade_direction.lower() == "short" and analysis.sentiment.value > 0:
            return (
                True,
                f"Bullish sentiment ({analysis.sentiment}) conflicts with short direction",
            )

        # Check 4: Pre-vetoed by analysis
        if analysis.should_veto:
            reason = analysis.veto_reason or "Analysis flagged trade as high-risk"
            return (True, reason)

        return (False, "")

    def calculate_position_size_multiplier(self, analysis: LLMAnalysis) -> float:
        """
        Calculate position size multiplier based on sentiment and confidence.

        Returns value between 0.5 (cautious) and 1.0 (full confidence).

        Args:
            analysis: LLM market analysis result

        Returns:
            Position size multiplier (0.5 - 1.0)
        """
        # Base multiplier from confidence
        multiplier = analysis.confidence

        # Apply sentiment penalty if below caution threshold
        if analysis.sentiment.value < SENTIMENT_CAUTION_THRESHOLD:
            sentiment_penalty = abs(analysis.sentiment.value) * 0.3
            multiplier *= 1.0 - sentiment_penalty

        # Ensure minimum multiplier of 0.5
        return max(0.5, min(1.0, multiplier))


class LLMAnalyzer:
    """
    LLM-powered market analyzer using Anthropic Claude.

    Provides intelligent market analysis by processing news, technical indicators,
    and market context. Includes rate limiting, caching, and error handling.

    Attributes:
        api_key: Anthropic API key
        model: Claude model to use
        max_tokens: Maximum tokens per request
        cache_ttl: Cache TTL in seconds
    """

    # Rate limiting: max 10 requests per minute
    MAX_REQUESTS_PER_MINUTE = 10
    RATE_LIMIT_WINDOW = 60.0  # seconds

    # Retry configuration
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 1.0  # seconds
    MAX_BACKOFF = 30.0  # seconds

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        max_tokens: int = 1000,
        cache_ttl: int = 300,
        redis_client: RedisClient | None = None,
    ) -> None:
        """
        Initialize LLM analyzer.

        Args:
            api_key: Anthropic API key
            model: Claude model to use
            max_tokens: Maximum tokens per request
            cache_ttl: Cache TTL in seconds
            redis_client: Optional Redis client for caching
        """
        from iftb.config import get_settings

        self.api_key = api_key
        self.model = model if model is not None else get_settings().llm.model
        self.max_tokens = max_tokens
        self.cache_ttl = cache_ttl

        # Initialize Anthropic client
        self._client = AsyncAnthropic(api_key=api_key)

        # Initialize cache if Redis client provided
        self._cache: LLMCache | None = None
        if redis_client:
            self._cache = LLMCache(redis_client)

        # Rate limiting
        self._request_times: list[float] = []
        self._rate_limit_lock = asyncio.Lock()

        # Error tracking
        self._consecutive_errors = 0
        self._last_error_time: datetime | None = None
        self._fallback_mode: FallbackMode | None = None

        logger.info("llm_analyzer_initialized", model=model, cache_enabled=bool(redis_client))

    async def _check_rate_limit(self) -> None:
        """
        Check and enforce rate limiting.

        Raises:
            RuntimeError: If rate limit exceeded
        """
        async with self._rate_limit_lock:
            now = asyncio.get_event_loop().time()

            # Remove old request times outside the window
            self._request_times = [
                t for t in self._request_times if now - t < self.RATE_LIMIT_WINDOW
            ]

            # Check if we're at the limit
            if len(self._request_times) >= self.MAX_REQUESTS_PER_MINUTE:
                oldest_request = self._request_times[0]
                wait_time = self.RATE_LIMIT_WINDOW - (now - oldest_request)
                logger.warning("rate_limit_exceeded", wait_time=wait_time)
                await asyncio.sleep(wait_time)

            # Record this request
            self._request_times.append(now)

    def _generate_cache_key(
        self,
        symbol: str,
        news_messages: list[NewsMessage],
        market_context: MarketContext,
        current_price: float,
    ) -> str:
        """
        Generate cache key for analysis request.

        Args:
            symbol: Trading symbol
            news_messages: Recent news messages
            market_context: Market context data
            current_price: Current price

        Returns:
            SHA256 hash of the input
        """
        # Create stable representation of inputs
        news_texts = [msg.text for msg in news_messages]
        context_data = {
            "fear_greed": market_context.fear_greed.value if market_context.fear_greed else None,
            "funding_rate": market_context.funding.rate if market_context.funding else None,
        }

        input_data = {
            "symbol": symbol,
            "news": news_texts,
            "context": context_data,
            "price": current_price,
        }

        # Generate hash
        json_str = json.dumps(input_data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _build_analysis_prompt(
        self,
        symbol: str,
        news_messages: list[NewsMessage],
        market_context: MarketContext,
        current_price: float,
    ) -> str:
        """
        Build the analysis prompt for Claude.

        Args:
            symbol: Trading symbol
            news_messages: Recent news messages
            market_context: Market context data
            current_price: Current price

        Returns:
            Formatted prompt string
        """
        # Format news section
        news_section = "Recent News:\n"
        if news_messages:
            for i, msg in enumerate(news_messages[:10], 1):  # Limit to 10 most recent
                timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M")
                urgent = "[URGENT] " if msg.is_urgent else ""
                news_section += f"{i}. {urgent}[{timestamp}] {msg.text[:200]}\n"
        else:
            news_section += "No recent news available.\n"

        # Format market context
        context_section = "Market Context:\n"
        if market_context.fear_greed:
            context_section += f"- Fear & Greed Index: {market_context.fear_greed.value} ({market_context.fear_greed.classification})\n"
        if market_context.funding:
            context_section += f"- Funding Rate: {market_context.funding.rate:.4%}\n"
        if market_context.open_interest:
            context_section += f"- Open Interest: ${market_context.open_interest.open_interest:,.0f} ({market_context.open_interest.oi_change_24h:+.2f}%)\n"
        if market_context.long_short:
            context_section += f"- Long/Short Ratio: {market_context.long_short.long_ratio:.2%} / {market_context.long_short.short_ratio:.2%}\n"

        prompt = f"""You are a cryptocurrency market analyst. Analyze the current market conditions for {symbol} and provide a sentiment assessment.

Current Price: ${current_price:,.2f}

{news_section}

{context_section}

Please analyze the market and respond in the following JSON format:
{{
    "sentiment": "<VERY_BEARISH|BEARISH|NEUTRAL|BULLISH|VERY_BULLISH>",
    "confidence": <0.0-1.0>,
    "summary": "<Brief 1-2 sentence summary>",
    "key_factors": ["<factor1>", "<factor2>", "<factor3>"],
    "should_veto": <true|false>,
    "veto_reason": "<Reason if should_veto is true, otherwise null>"
}}

Sentiment Guidelines:
- VERY_BEARISH: Strong negative indicators, high risk of downturn
- BEARISH: Negative indicators, caution advised
- NEUTRAL: Mixed signals, no clear direction
- BULLISH: Positive indicators, favorable conditions
- VERY_BULLISH: Strong positive indicators, high confidence in upside

Confidence Guidelines:
- 0.0-0.3: Low confidence, insufficient data or conflicting signals
- 0.4-0.6: Moderate confidence, some clear signals
- 0.7-0.9: High confidence, strong aligned signals
- 0.9-1.0: Very high confidence, overwhelming evidence

Veto Guidelines:
Set should_veto to true if:
- Sentiment is VERY_BEARISH or BEARISH with high confidence
- Critical negative news (exchange hack, regulatory ban, major exploit)
- Extremely one-sided market positioning suggesting reversal risk
- Low confidence (<0.3) with mixed signals

Consider all factors: news sentiment, market metrics, and timing."""

        return prompt

    def _parse_claude_response(self, response: Message) -> dict[str, Any]:
        """
        Parse Claude API response and extract JSON.

        Args:
            response: Claude Message response

        Returns:
            Parsed response dictionary

        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # Extract text content
            text_content = response.content[0].text

            # Find JSON in response (might be wrapped in markdown)
            if "```json" in text_content:
                # Extract from markdown code block
                json_start = text_content.find("```json") + 7
                json_end = text_content.find("```", json_start)
                json_str = text_content[json_start:json_end].strip()
            elif "{" in text_content:
                # Find JSON object
                json_start = text_content.find("{")
                json_end = text_content.rfind("}") + 1
                json_str = text_content[json_start:json_end]
            else:
                raise ValueError("No JSON found in response")

            # Parse JSON
            data = json.loads(json_str)

            # Validate required fields
            required_fields = ["sentiment", "confidence", "summary", "key_factors"]
            missing = [f for f in required_fields if f not in data]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")

            return data

        except (json.JSONDecodeError, IndexError, AttributeError) as e:
            logger.error("failed_to_parse_response", error=str(e), response=str(response))
            raise ValueError(f"Failed to parse Claude response: {e}")

    def _create_fallback_analysis(self, mode: FallbackMode, symbol: str) -> LLMAnalysis:
        """
        Create fallback analysis when API is unavailable.

        Args:
            mode: Fallback mode to use
            symbol: Trading symbol

        Returns:
            Fallback LLMAnalysis
        """
        if mode == FallbackMode.CONSERVATIVE:
            return LLMAnalysis(
                sentiment=SentimentScore.NEUTRAL,
                confidence=0.3,
                summary=f"Fallback analysis for {symbol} - API unavailable, using conservative neutral stance",
                key_factors=["API unavailable", "Conservative fallback mode"],
                should_veto=False,
                veto_reason=None,
                timestamp=datetime.now(UTC),
                model=self.model,
                prompt_tokens=0,
                completion_tokens=0,
            )

        if mode == FallbackMode.VETO_ALL:
            return LLMAnalysis(
                sentiment=SentimentScore.VERY_BEARISH,
                confidence=1.0,
                summary=f"Fallback veto for {symbol} - API unavailable, blocking all trades",
                key_factors=["API unavailable", "Veto all fallback mode"],
                should_veto=True,
                veto_reason="LLM API unavailable - vetoing all trades as safety measure",
                timestamp=datetime.now(UTC),
                model=self.model,
                prompt_tokens=0,
                completion_tokens=0,
            )

        # CACHE_ONLY - this shouldn't be reached, but provide safe default
        return LLMAnalysis(
            sentiment=SentimentScore.NEUTRAL,
            confidence=0.2,
            summary=f"Cache-only mode for {symbol} - no cached data available",
            key_factors=["No cache data", "Cache-only fallback mode"],
            should_veto=True,
            veto_reason="Cache-only mode with no cached data available",
            timestamp=datetime.now(UTC),
            model=self.model,
            prompt_tokens=0,
            completion_tokens=0,
        )

    async def analyze_market(
        self,
        symbol: str,
        news_messages: list[NewsMessage],
        market_context: MarketContext,
        current_price: float,
    ) -> LLMAnalysis:
        """
        Analyze market conditions using LLM.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            news_messages: Recent news messages
            market_context: Market context data
            current_price: Current market price

        Returns:
            LLM analysis result

        Raises:
            RuntimeError: If API fails and no fallback is configured
        """
        # Generate cache key
        cache_key = self._generate_cache_key(symbol, news_messages, market_context, current_price)

        # Check cache first
        if self._cache:
            try:
                cached_data = await self._cache.get_analysis("market_sentiment", cache_key)
                if cached_data:
                    logger.info("llm_cache_hit", symbol=symbol, cache_key=cache_key[:16])
                    return LLMAnalysis.from_dict(cached_data)
            except Exception as e:
                logger.warning("cache_retrieval_failed", error=str(e))

        # If in CACHE_ONLY mode and no cache hit, return fallback
        if self._fallback_mode == FallbackMode.CACHE_ONLY:
            logger.warning("cache_only_mode_no_data", symbol=symbol)
            return self._create_fallback_analysis(FallbackMode.CACHE_ONLY, symbol)

        # Try API call with retries
        for attempt in range(self.MAX_RETRIES):
            try:
                # Check rate limit
                await self._check_rate_limit()

                # Build prompt
                prompt = self._build_analysis_prompt(
                    symbol, news_messages, market_context, current_price
                )

                # Call Claude API
                logger.debug("calling_claude_api", attempt=attempt + 1, symbol=symbol)
                response = await self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )

                # Parse response
                parsed = self._parse_claude_response(response)

                # Map sentiment string to enum
                sentiment_str = parsed["sentiment"].upper()
                sentiment = SentimentScore[sentiment_str]

                # Create analysis object
                analysis = LLMAnalysis(
                    sentiment=sentiment,
                    confidence=float(parsed["confidence"]),
                    summary=parsed["summary"],
                    key_factors=parsed["key_factors"],
                    should_veto=parsed.get("should_veto", False),
                    veto_reason=parsed.get("veto_reason"),
                    timestamp=datetime.now(UTC),
                    model=self.model,
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                )

                # Cache the result
                if self._cache:
                    try:
                        await self._cache.set_analysis(
                            "market_sentiment",
                            cache_key,
                            analysis.to_dict(),
                            self.cache_ttl,
                        )
                    except Exception as e:
                        logger.warning("cache_storage_failed", error=str(e))

                # Reset error tracking on success
                self._consecutive_errors = 0
                self._fallback_mode = None

                logger.info(
                    "llm_analysis_completed",
                    symbol=symbol,
                    sentiment=str(sentiment),
                    confidence=analysis.confidence,
                    should_veto=analysis.should_veto,
                    prompt_tokens=analysis.prompt_tokens,
                    completion_tokens=analysis.completion_tokens,
                )

                return analysis

            except Exception as e:
                self._consecutive_errors += 1
                self._last_error_time = datetime.now(UTC)

                # Calculate backoff
                backoff = min(
                    self.INITIAL_BACKOFF * (2**attempt),
                    self.MAX_BACKOFF,
                )

                logger.warning(
                    "llm_api_error",
                    attempt=attempt + 1,
                    error=str(e),
                    backoff=backoff,
                    consecutive_errors=self._consecutive_errors,
                )

                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(backoff)
                else:
                    # Final retry failed
                    logger.error(
                        "llm_api_failed",
                        error=str(e),
                        consecutive_errors=self._consecutive_errors,
                    )

                    # Determine fallback mode based on error count
                    if self._consecutive_errors >= 5:
                        self._fallback_mode = FallbackMode.VETO_ALL
                        logger.warning("entering_veto_all_mode")
                    else:
                        self._fallback_mode = FallbackMode.CONSERVATIVE
                        logger.warning("entering_conservative_mode")

                    return self._create_fallback_analysis(self._fallback_mode, symbol)

        # Should never reach here
        raise RuntimeError("LLM API failed after all retries")

    async def analyze_news_urgency(self, news: NewsMessage) -> tuple[bool, str]:
        """
        Analyze if a news message requires urgent action.

        This is a lightweight check that can be used to quickly assess
        individual news items without full market analysis.

        Args:
            news: News message to analyze

        Returns:
            Tuple of (is_urgent, reason)
        """
        # If already marked urgent by keyword detection, return immediately
        if news.is_urgent:
            return (True, f"Urgent keywords detected: {', '.join(news.keywords)}")

        # For non-urgent news, we could optionally do lightweight LLM analysis
        # but for now, just return the existing classification
        return (False, "No urgent indicators detected")

    async def health_check(self) -> dict[str, Any]:
        """
        Check health status of the LLM analyzer.

        Returns:
            Health status dictionary
        """
        return {
            "status": "healthy" if self._fallback_mode is None else "degraded",
            "fallback_mode": self._fallback_mode.value if self._fallback_mode else None,
            "consecutive_errors": self._consecutive_errors,
            "last_error_time": self._last_error_time.isoformat() if self._last_error_time else None,
            "cache_enabled": self._cache is not None,
            "rate_limit_usage": len(self._request_times),
        }


async def create_analyzer_from_settings() -> LLMAnalyzer:
    """
    Create LLM analyzer from application settings.

    Returns:
        Configured LLMAnalyzer instance

    Example:
        ```python
        analyzer = await create_analyzer_from_settings()
        analysis = await analyzer.analyze_market(...)
        ```
    """
    settings = get_settings()

    # Initialize Redis client for caching
    redis_client = RedisClient(
        host=settings.redis.host,
        port=settings.redis.port,
        password=settings.redis.password.get_secret_value() if settings.redis.password else None,
        db=settings.redis.db,
    )
    await redis_client.connect()

    analyzer = LLMAnalyzer(
        api_key=settings.llm.anthropic_api_key.get_secret_value(),
        model=settings.llm.model,
        max_tokens=settings.llm.max_tokens,
        cache_ttl=settings.llm.cache_ttl_seconds,
        redis_client=redis_client,
    )

    logger.info("llm_analyzer_created_from_settings")
    return analyzer
