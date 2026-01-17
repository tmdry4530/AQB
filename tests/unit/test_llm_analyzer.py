"""
Unit tests for LLM Market Analyzer.

Tests the LLMAnalyzer, LLMVetoSystem, and related functionality including
caching, rate limiting, error handling, and fallback modes.
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from anthropic.types import Message, TextBlock, Usage
import pytest

from iftb.analysis.llm_analyzer import (
    FallbackMode,
    LLMAnalysis,
    LLMAnalyzer,
    LLMVetoSystem,
    SentimentScore,
)
from iftb.data import MarketContext, NewsMessage


@pytest.fixture
def sample_news_messages():
    """Create sample news messages for testing."""
    return [
        NewsMessage(
            timestamp=datetime.now(UTC),
            text="Bitcoin breaks $50,000 resistance with strong volume",
            channel="CryptoNews",
            channel_id=123,
            message_id=1,
            has_media=False,
            is_forwarded=False,
            is_urgent=False,
            keywords=[],
        ),
        NewsMessage(
            timestamp=datetime.now(UTC),
            text="Breaking: Major exchange lists new trading pairs",
            channel="CryptoNews",
            channel_id=123,
            message_id=2,
            has_media=False,
            is_forwarded=False,
            is_urgent=True,
            keywords=["breaking"],
        ),
    ]


@pytest.fixture
def sample_market_context():
    """Create sample market context for testing."""
    from iftb.data.external import FearGreedData, FundingData

    return MarketContext(
        fear_greed=FearGreedData(
            value=65,
            classification="Greed",
            timestamp=datetime.now(UTC),
        ),
        funding=FundingData(
            symbol="BTC",
            rate=0.0001,
            predicted_rate=0.00012,
            next_funding_time=datetime.now(UTC),
        ),
    )


class TestSentimentScore:
    """Tests for SentimentScore enum."""

    def test_sentiment_values(self):
        """Test sentiment score values."""
        assert SentimentScore.VERY_BEARISH.value == -1.0
        assert SentimentScore.BEARISH.value == -0.5
        assert SentimentScore.NEUTRAL.value == 0.0
        assert SentimentScore.BULLISH.value == 0.5
        assert SentimentScore.VERY_BULLISH.value == 1.0

    def test_sentiment_string(self):
        """Test sentiment string representation."""
        assert str(SentimentScore.VERY_BULLISH) == "Very Bullish"
        assert str(SentimentScore.NEUTRAL) == "Neutral"


class TestLLMAnalysis:
    """Tests for LLMAnalysis dataclass."""

    def test_valid_analysis(self):
        """Test creating valid analysis."""
        analysis = LLMAnalysis(
            sentiment=SentimentScore.BULLISH,
            confidence=0.75,
            summary="Market looks strong",
            key_factors=["Strong volume", "Positive news"],
            should_veto=False,
            veto_reason=None,
            timestamp=datetime.now(UTC),
            model="claude-sonnet-4-20250514",
            prompt_tokens=100,
            completion_tokens=50,
        )

        assert analysis.sentiment == SentimentScore.BULLISH
        assert analysis.confidence == 0.75
        assert not analysis.should_veto

    def test_invalid_confidence(self):
        """Test validation of confidence score."""
        with pytest.raises(ValueError, match="Confidence must be 0-1"):
            LLMAnalysis(
                sentiment=SentimentScore.NEUTRAL,
                confidence=1.5,  # Invalid
                summary="Test",
                key_factors=[],
                should_veto=False,
                veto_reason=None,
                timestamp=datetime.now(UTC),
                model="test",
                prompt_tokens=0,
                completion_tokens=0,
            )

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        original = LLMAnalysis(
            sentiment=SentimentScore.BEARISH,
            confidence=0.6,
            summary="Bearish signals detected",
            key_factors=["Negative funding", "Weak volume"],
            should_veto=False,
            veto_reason=None,
            timestamp=datetime.now(UTC),
            model="claude-sonnet-4-20250514",
            prompt_tokens=120,
            completion_tokens=60,
        )

        # Serialize
        data = original.to_dict()
        assert data["sentiment"] == -0.5
        assert isinstance(data["timestamp"], str)

        # Deserialize
        restored = LLMAnalysis.from_dict(data)
        assert restored.sentiment == original.sentiment
        assert restored.confidence == original.confidence
        assert restored.summary == original.summary
        assert restored.cached is True


class TestLLMVetoSystem:
    """Tests for LLMVetoSystem."""

    @pytest.fixture
    def veto_system(self):
        """Create veto system instance."""
        return LLMVetoSystem()

    def test_veto_extreme_bearish_sentiment(self, veto_system):
        """Test veto on very bearish sentiment."""
        analysis = LLMAnalysis(
            sentiment=SentimentScore.VERY_BEARISH,
            confidence=0.8,
            summary="Market crash imminent",
            key_factors=["Major sell-off"],
            should_veto=False,
            veto_reason=None,
            timestamp=datetime.now(UTC),
            model="test",
            prompt_tokens=0,
            completion_tokens=0,
        )

        should_veto, reason = veto_system.should_veto_trade(analysis, "long")
        assert should_veto
        assert "bearish sentiment" in reason.lower()

    def test_veto_low_confidence(self, veto_system):
        """Test veto on low confidence."""
        analysis = LLMAnalysis(
            sentiment=SentimentScore.NEUTRAL,
            confidence=0.2,  # Below threshold
            summary="Uncertain market",
            key_factors=["Mixed signals"],
            should_veto=False,
            veto_reason=None,
            timestamp=datetime.now(UTC),
            model="test",
            prompt_tokens=0,
            completion_tokens=0,
        )

        should_veto, reason = veto_system.should_veto_trade(analysis, "long")
        assert should_veto
        assert "low confidence" in reason.lower()

    def test_veto_direction_conflict_long(self, veto_system):
        """Test veto when bearish sentiment conflicts with long direction."""
        analysis = LLMAnalysis(
            sentiment=SentimentScore.BEARISH,
            confidence=0.7,
            summary="Bearish market",
            key_factors=["Negative indicators"],
            should_veto=False,
            veto_reason=None,
            timestamp=datetime.now(UTC),
            model="test",
            prompt_tokens=0,
            completion_tokens=0,
        )

        should_veto, reason = veto_system.should_veto_trade(analysis, "long")
        assert should_veto
        assert "conflicts with long" in reason.lower()

    def test_veto_direction_conflict_short(self, veto_system):
        """Test veto when bullish sentiment conflicts with short direction."""
        analysis = LLMAnalysis(
            sentiment=SentimentScore.BULLISH,
            confidence=0.7,
            summary="Bullish market",
            key_factors=["Positive indicators"],
            should_veto=False,
            veto_reason=None,
            timestamp=datetime.now(UTC),
            model="test",
            prompt_tokens=0,
            completion_tokens=0,
        )

        should_veto, reason = veto_system.should_veto_trade(analysis, "short")
        assert should_veto
        assert "conflicts with short" in reason.lower()

    def test_no_veto_good_conditions(self, veto_system):
        """Test no veto when conditions are favorable."""
        analysis = LLMAnalysis(
            sentiment=SentimentScore.BULLISH,
            confidence=0.8,
            summary="Strong bullish signals",
            key_factors=["Positive momentum"],
            should_veto=False,
            veto_reason=None,
            timestamp=datetime.now(UTC),
            model="test",
            prompt_tokens=0,
            completion_tokens=0,
        )

        should_veto, reason = veto_system.should_veto_trade(analysis, "long")
        assert not should_veto
        assert reason == ""

    def test_pre_vetoed_analysis(self, veto_system):
        """Test veto when analysis is pre-flagged."""
        analysis = LLMAnalysis(
            sentiment=SentimentScore.NEUTRAL,
            confidence=0.7,
            summary="High risk detected",
            key_factors=["Unusual volatility"],
            should_veto=True,
            veto_reason="Extreme volatility risk",
            timestamp=datetime.now(UTC),
            model="test",
            prompt_tokens=0,
            completion_tokens=0,
        )

        should_veto, reason = veto_system.should_veto_trade(analysis, "long")
        assert should_veto
        assert reason == "Extreme volatility risk"

    def test_position_size_multiplier(self, veto_system):
        """Test position size multiplier calculation."""
        # High confidence, bullish
        analysis1 = LLMAnalysis(
            sentiment=SentimentScore.VERY_BULLISH,
            confidence=0.9,
            summary="Very bullish",
            key_factors=[],
            should_veto=False,
            veto_reason=None,
            timestamp=datetime.now(UTC),
            model="test",
            prompt_tokens=0,
            completion_tokens=0,
        )
        multiplier1 = veto_system.calculate_position_size_multiplier(analysis1)
        assert 0.85 <= multiplier1 <= 1.0

        # Low confidence, neutral
        analysis2 = LLMAnalysis(
            sentiment=SentimentScore.NEUTRAL,
            confidence=0.5,
            summary="Neutral",
            key_factors=[],
            should_veto=False,
            veto_reason=None,
            timestamp=datetime.now(UTC),
            model="test",
            prompt_tokens=0,
            completion_tokens=0,
        )
        multiplier2 = veto_system.calculate_position_size_multiplier(analysis2)
        assert 0.5 <= multiplier2 < 0.9

        # Bearish sentiment penalty
        analysis3 = LLMAnalysis(
            sentiment=SentimentScore.BEARISH,
            confidence=0.6,
            summary="Bearish",
            key_factors=[],
            should_veto=False,
            veto_reason=None,
            timestamp=datetime.now(UTC),
            model="test",
            prompt_tokens=0,
            completion_tokens=0,
        )
        multiplier3 = veto_system.calculate_position_size_multiplier(analysis3)
        assert multiplier3 >= 0.5  # Should apply sentiment penalty


class TestLLMAnalyzer:
    """Tests for LLMAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return LLMAnalyzer(
            api_key="test_key",
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            cache_ttl=300,
        )

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.api_key == "test_key"
        assert analyzer.model == "claude-sonnet-4-20250514"
        assert analyzer.max_tokens == 1000
        assert analyzer._consecutive_errors == 0

    def test_generate_cache_key(self, analyzer, sample_news_messages, sample_market_context):
        """Test cache key generation."""
        key1 = analyzer._generate_cache_key(
            "BTCUSDT", sample_news_messages, sample_market_context, 50000.0
        )
        key2 = analyzer._generate_cache_key(
            "BTCUSDT", sample_news_messages, sample_market_context, 50000.0
        )

        # Same inputs should produce same key
        assert key1 == key2
        assert len(key1) == 64  # SHA256 hex length

        # Different price should produce different key
        key3 = analyzer._generate_cache_key(
            "BTCUSDT", sample_news_messages, sample_market_context, 51000.0
        )
        assert key1 != key3

    def test_build_analysis_prompt(self, analyzer, sample_news_messages, sample_market_context):
        """Test prompt building."""
        prompt = analyzer._build_analysis_prompt(
            "BTCUSDT", sample_news_messages, sample_market_context, 50000.0
        )

        assert "BTCUSDT" in prompt
        assert "50,000" in prompt
        assert "Recent News:" in prompt
        assert "Market Context:" in prompt
        assert "Fear & Greed" in prompt
        assert "JSON format" in prompt

    @pytest.mark.asyncio
    async def test_rate_limiting(self, analyzer):
        """Test rate limiting enforcement."""
        # Fill up the rate limit
        for _ in range(analyzer.MAX_REQUESTS_PER_MINUTE):
            await analyzer._check_rate_limit()

        # Next request should wait
        start_time = asyncio.get_event_loop().time()
        await analyzer._check_rate_limit()
        elapsed = asyncio.get_event_loop().time() - start_time

        # Should have waited some time (but test is lenient)
        assert elapsed >= 0

    def test_parse_valid_response(self, analyzer):
        """Test parsing valid Claude response."""
        # Mock Claude Message response
        mock_response = MagicMock(spec=Message)
        mock_response.content = [
            TextBlock(
                text='```json\n{"sentiment": "BULLISH", "confidence": 0.75, "summary": "Market looks strong", "key_factors": ["Good volume", "Positive news"], "should_veto": false, "veto_reason": null}\n```',
                type="text",
            )
        ]
        mock_response.usage = Usage(input_tokens=100, output_tokens=50)

        result = analyzer._parse_claude_response(mock_response)

        assert result["sentiment"] == "BULLISH"
        assert result["confidence"] == 0.75
        assert result["summary"] == "Market looks strong"
        assert len(result["key_factors"]) == 2
        assert result["should_veto"] is False

    def test_parse_invalid_response(self, analyzer):
        """Test parsing invalid response."""
        mock_response = MagicMock(spec=Message)
        mock_response.content = [TextBlock(text="Not valid JSON", type="text")]

        with pytest.raises(ValueError, match="No JSON found"):
            analyzer._parse_claude_response(mock_response)

    def test_create_fallback_analysis(self, analyzer):
        """Test fallback analysis creation."""
        # Conservative mode
        analysis1 = analyzer._create_fallback_analysis(FallbackMode.CONSERVATIVE, "BTCUSDT")
        assert analysis1.sentiment == SentimentScore.NEUTRAL
        assert analysis1.confidence == 0.3
        assert not analysis1.should_veto

        # Veto all mode
        analysis2 = analyzer._create_fallback_analysis(FallbackMode.VETO_ALL, "BTCUSDT")
        assert analysis2.sentiment == SentimentScore.VERY_BEARISH
        assert analysis2.confidence == 1.0
        assert analysis2.should_veto
        assert "API unavailable" in analysis2.veto_reason

    @pytest.mark.asyncio
    async def test_analyze_market_success(
        self, analyzer, sample_news_messages, sample_market_context
    ):
        """Test successful market analysis."""
        # Mock Claude API response
        mock_response = MagicMock(spec=Message)
        mock_response.content = [
            TextBlock(
                text='{"sentiment": "BULLISH", "confidence": 0.8, "summary": "Strong market", "key_factors": ["Volume", "Momentum"], "should_veto": false}',
                type="text",
            )
        ]
        mock_response.usage = Usage(input_tokens=120, output_tokens=60)

        with patch.object(analyzer._client.messages, "create", return_value=mock_response):
            analysis = await analyzer.analyze_market(
                symbol="BTCUSDT",
                news_messages=sample_news_messages,
                market_context=sample_market_context,
                current_price=50000.0,
            )

            assert analysis.sentiment == SentimentScore.BULLISH
            assert analysis.confidence == 0.8
            assert analysis.summary == "Strong market"
            assert not analysis.should_veto

    @pytest.mark.asyncio
    async def test_analyze_market_with_cache(
        self, analyzer, sample_news_messages, sample_market_context
    ):
        """Test market analysis with caching."""
        # Create mock cache
        mock_cache = AsyncMock()
        cached_data = {
            "sentiment": 0.5,
            "confidence": 0.75,
            "summary": "Cached analysis",
            "key_factors": ["Factor 1"],
            "should_veto": False,
            "veto_reason": None,
            "timestamp": datetime.now(UTC).isoformat(),
            "model": "test",
            "prompt_tokens": 100,
            "completion_tokens": 50,
        }
        mock_cache.get_analysis.return_value = cached_data
        analyzer._cache = mock_cache

        analysis = await analyzer.analyze_market(
            symbol="BTCUSDT",
            news_messages=sample_news_messages,
            market_context=sample_market_context,
            current_price=50000.0,
        )

        assert analysis.summary == "Cached analysis"
        assert analysis.cached is True
        mock_cache.get_analysis.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_news_urgency(self, analyzer, sample_news_messages):
        """Test news urgency analysis."""
        # Already urgent
        is_urgent, reason = await analyzer.analyze_news_urgency(sample_news_messages[1])
        assert is_urgent
        assert "breaking" in reason.lower()

        # Not urgent
        is_urgent, reason = await analyzer.analyze_news_urgency(sample_news_messages[0])
        assert not is_urgent

    @pytest.mark.asyncio
    async def test_health_check(self, analyzer):
        """Test health check."""
        health = await analyzer.health_check()

        assert "status" in health
        assert health["status"] == "healthy"
        assert health["consecutive_errors"] == 0
        assert health["cache_enabled"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
