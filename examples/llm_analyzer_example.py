"""
Example usage of LLM Market Analyzer.

This script demonstrates how to use the LLMAnalyzer and LLMVetoSystem
for intelligent market analysis and trade decision making.
"""

import asyncio
from datetime import UTC, datetime

from iftb.analysis import (
    LLMAnalyzer,
    LLMVetoSystem,
    SentimentScore,
    create_analyzer_from_settings,
)
from iftb.config import get_settings
from iftb.data import MarketContext, NewsMessage
from iftb.data.external import FearGreedData, FundingData
from iftb.utils import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def example_basic_analysis():
    """Example 1: Basic market analysis."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Market Analysis")
    print("=" * 60)

    settings = get_settings()

    # Create analyzer (without cache for simplicity)
    analyzer = LLMAnalyzer(
        api_key=settings.llm.anthropic_api_key.get_secret_value(),
        model=settings.llm.model,
    )

    # Sample news messages
    news_messages = [
        NewsMessage(
            timestamp=datetime.now(UTC),
            text="Bitcoin surges past $50,000 as institutional adoption accelerates",
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
            text="Major bank announces crypto trading services",
            channel="CryptoNews",
            channel_id=123,
            message_id=2,
            has_media=False,
            is_forwarded=False,
            is_urgent=True,
            keywords=["breaking"],
        ),
    ]

    # Sample market context
    market_context = MarketContext(
        fear_greed=FearGreedData(
            value=70,
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

    # Analyze market
    analysis = await analyzer.analyze_market(
        symbol="BTCUSDT",
        news_messages=news_messages,
        market_context=market_context,
        current_price=50000.0,
    )

    # Print results
    print(f"\nSentiment: {analysis.sentiment}")
    print(f"Confidence: {analysis.confidence:.2%}")
    print(f"Summary: {analysis.summary}")
    print("\nKey Factors:")
    for i, factor in enumerate(analysis.key_factors, 1):
        print(f"  {i}. {factor}")
    print(f"\nShould Veto: {analysis.should_veto}")
    if analysis.veto_reason:
        print(f"Veto Reason: {analysis.veto_reason}")
    print(f"\nToken Usage: {analysis.prompt_tokens} input, {analysis.completion_tokens} output")


async def example_veto_system():
    """Example 2: Using the veto system."""
    print("\n" + "=" * 60)
    print("Example 2: Veto System")
    print("=" * 60)

    veto_system = LLMVetoSystem()

    # Scenario 1: Bullish analysis, long trade
    analysis1 = type(
        "Analysis",
        (),
        {
            "sentiment": SentimentScore.BULLISH,
            "confidence": 0.8,
            "should_veto": False,
            "veto_reason": None,
        },
    )()

    should_veto, reason = veto_system.should_veto_trade(analysis1, "long")
    print("\nScenario 1: Bullish sentiment, Long trade")
    print(f"  Veto: {should_veto}")
    print(f"  Reason: {reason if reason else 'No veto'}")

    # Scenario 2: Bearish analysis, long trade (conflict)
    analysis2 = type(
        "Analysis",
        (),
        {
            "sentiment": SentimentScore.BEARISH,
            "confidence": 0.7,
            "should_veto": False,
            "veto_reason": None,
        },
    )()

    should_veto, reason = veto_system.should_veto_trade(analysis2, "long")
    print("\nScenario 2: Bearish sentiment, Long trade (conflict)")
    print(f"  Veto: {should_veto}")
    print(f"  Reason: {reason}")

    # Scenario 3: Low confidence
    analysis3 = type(
        "Analysis",
        (),
        {
            "sentiment": SentimentScore.NEUTRAL,
            "confidence": 0.2,
            "should_veto": False,
            "veto_reason": None,
        },
    )()

    should_veto, reason = veto_system.should_veto_trade(analysis3, "long")
    print("\nScenario 3: Low confidence")
    print(f"  Veto: {should_veto}")
    print(f"  Reason: {reason}")

    # Position size multipliers
    print("\n" + "-" * 60)
    print("Position Size Multipliers:")
    print("-" * 60)

    multiplier1 = veto_system.calculate_position_size_multiplier(analysis1)
    print(f"Bullish, High Confidence: {multiplier1:.2f}x")

    multiplier2 = veto_system.calculate_position_size_multiplier(analysis2)
    print(f"Bearish, Moderate Confidence: {multiplier2:.2f}x")

    multiplier3 = veto_system.calculate_position_size_multiplier(analysis3)
    print(f"Neutral, Low Confidence: {multiplier3:.2f}x")


async def example_with_caching():
    """Example 3: Using analyzer with caching."""
    print("\n" + "=" * 60)
    print("Example 3: Analysis with Caching")
    print("=" * 60)

    # Create analyzer with full settings (includes cache)
    analyzer = await create_analyzer_from_settings()

    # Sample data
    news_messages = [
        NewsMessage(
            timestamp=datetime.now(UTC),
            text="Bitcoin consolidates after recent rally",
            channel="CryptoNews",
            channel_id=123,
            message_id=1,
            has_media=False,
            is_forwarded=False,
            is_urgent=False,
            keywords=[],
        ),
    ]

    market_context = MarketContext(
        fear_greed=FearGreedData(
            value=60,
            classification="Greed",
            timestamp=datetime.now(UTC),
        ),
    )

    # First analysis (will call API)
    print("\nFirst analysis (API call)...")
    analysis1 = await analyzer.analyze_market(
        symbol="BTCUSDT",
        news_messages=news_messages,
        market_context=market_context,
        current_price=50000.0,
    )
    print(f"Cached: {analysis1.cached}")
    print(f"Summary: {analysis1.summary}")

    # Second analysis with same inputs (will use cache)
    print("\nSecond analysis (cache hit)...")
    analysis2 = await analyzer.analyze_market(
        symbol="BTCUSDT",
        news_messages=news_messages,
        market_context=market_context,
        current_price=50000.0,
    )
    print(f"Cached: {analysis2.cached}")
    print(f"Summary: {analysis2.summary}")


async def example_health_check():
    """Example 4: Health check monitoring."""
    print("\n" + "=" * 60)
    print("Example 4: Health Check")
    print("=" * 60)

    settings = get_settings()
    analyzer = LLMAnalyzer(
        api_key=settings.llm.anthropic_api_key.get_secret_value(),
        model=settings.llm.model,
    )

    # Get health status
    health = await analyzer.health_check()

    print(f"\nStatus: {health['status']}")
    print(f"Fallback Mode: {health['fallback_mode'] or 'None'}")
    print(f"Consecutive Errors: {health['consecutive_errors']}")
    print(f"Cache Enabled: {health['cache_enabled']}")
    print(f"Rate Limit Usage: {health['rate_limit_usage']}/{analyzer.MAX_REQUESTS_PER_MINUTE}")


async def example_error_handling():
    """Example 5: Error handling and fallback modes."""
    print("\n" + "=" * 60)
    print("Example 5: Error Handling")
    print("=" * 60)

    # Create analyzer with invalid API key to trigger fallback
    analyzer = LLMAnalyzer(
        api_key="invalid_key",
        model="claude-sonnet-4-20250514",
    )

    news_messages = []
    market_context = MarketContext()

    print("\nAttempting analysis with invalid API key...")
    try:
        analysis = await analyzer.analyze_market(
            symbol="BTCUSDT",
            news_messages=news_messages,
            market_context=market_context,
            current_price=50000.0,
        )

        # Should get fallback analysis
        print("\nFallback Mode Activated")
        print(f"Sentiment: {analysis.sentiment}")
        print(f"Confidence: {analysis.confidence:.2%}")
        print(f"Summary: {analysis.summary}")
        print(f"Should Veto: {analysis.should_veto}")

    except Exception as e:
        print(f"\nError: {e}")


async def main():
    """Run all examples."""
    try:
        # Run examples
        await example_basic_analysis()
        await example_veto_system()
        await example_with_caching()
        await example_health_check()
        await example_error_handling()

        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)

    except Exception as e:
        logger.error("example_failed", error=str(e), exc_info=True)
        print(f"\nError running examples: {e}")


if __name__ == "__main__":
    asyncio.run(main())
