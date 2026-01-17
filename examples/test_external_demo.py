"""
Demonstration script for external data API clients.

This script shows how to use the FearGreedClient, CoinglassClient,
and ExternalDataAggregator classes.

Note: This is a demo/test script and requires dependencies to be installed:
    pip install httpx structlog
"""

import asyncio


async def demo_fear_greed():
    """Demonstrate Fear & Greed Index client."""
    print("\n=== Fear & Greed Index Demo ===\n")

    # Import after dependencies note
    from iftb.data.external import FearGreedClient

    async with FearGreedClient() as client:
        # Fetch current index
        print("Fetching current Fear & Greed Index...")
        try:
            current = await client.fetch_current()
            print(f"Value: {current.value}")
            print(f"Classification: {current.classification}")
            print(f"Timestamp: {current.timestamp}")
        except Exception as e:
            print(f"Error: {e}")

        # Fetch historical data
        print("\nFetching historical data (last 7 days)...")
        try:
            historical = await client.fetch_historical(limit=7)
            print(f"Retrieved {len(historical)} data points:")
            for data in historical[:3]:
                print(f"  {data.timestamp.date()}: {data.value} ({data.classification})")
        except Exception as e:
            print(f"Error: {e}")


async def demo_coinglass():
    """Demonstrate Coinglass client."""
    print("\n=== Coinglass Demo ===\n")

    from iftb.data.external import CoinglassClient

    async with CoinglassClient() as client:
        symbol = "BTC"

        # Fetch funding rate
        print(f"Fetching funding rate for {symbol}...")
        try:
            funding = await client.fetch_funding_rate(symbol)
            print(f"Symbol: {funding.symbol}")
            print(f"Current Rate: {funding.rate * 100:.4f}%")
            print(f"Predicted Rate: {funding.predicted_rate * 100:.4f}%")
            print(f"Next Funding: {funding.next_funding_time}")
        except Exception as e:
            print(f"Error: {e}")

        # Fetch open interest
        print(f"\nFetching open interest for {symbol}...")
        try:
            oi = await client.fetch_open_interest(symbol)
            print(f"Symbol: {oi.symbol}")
            print(f"Open Interest: ${oi.open_interest:,.0f}")
            print(f"24h Change: {oi.oi_change_24h:+.2f}%")
        except Exception as e:
            print(f"Error: {e}")

        # Fetch long/short ratio
        print(f"\nFetching long/short ratio for {symbol}...")
        try:
            ls = await client.fetch_long_short_ratio(symbol)
            print(f"Symbol: {ls.symbol}")
            print(f"Long Ratio: {ls.long_ratio * 100:.2f}%")
            print(f"Short Ratio: {ls.short_ratio * 100:.2f}%")
            print(f"Timestamp: {ls.timestamp}")
        except Exception as e:
            print(f"Error: {e}")


async def demo_aggregator():
    """Demonstrate External Data Aggregator."""
    print("\n=== External Data Aggregator Demo ===\n")

    from iftb.data.external import ExternalDataAggregator

    async with ExternalDataAggregator(cache_ttl=300) as aggregator:
        # Fetch all data
        print("Fetching all market context data...")
        context = await aggregator.fetch_all(symbol="BTC")

        print(f"\nFetch Time: {context.fetch_time}")
        print(f"Errors: {len(context.errors)}")

        if context.fear_greed:
            print(f"\nFear & Greed: {context.fear_greed.value} ({context.fear_greed.classification})")
        else:
            print("\nFear & Greed: Not available")

        if context.funding:
            print(f"Funding Rate: {context.funding.rate * 100:.4f}%")
        else:
            print("Funding Rate: Not available")

        if context.open_interest:
            print(f"Open Interest: ${context.open_interest.open_interest:,.0f}")
        else:
            print("Open Interest: Not available")

        if context.long_short:
            print(f"Long/Short: {context.long_short.long_ratio:.2%} / {context.long_short.short_ratio:.2%}")
        else:
            print("Long/Short: Not available")

        # Show errors if any
        if context.errors:
            print("\nErrors encountered:")
            for error in context.errors:
                print(f"  - {error}")

        # Test caching
        print("\nFetching again (should use cache)...")
        context2 = await aggregator.fetch_all(symbol="BTC")
        print(f"Cached: {context2.fetch_time == context.fetch_time}")

        # Force refresh
        print("\nForcing refresh...")
        context3 = await aggregator.fetch_all(symbol="BTC", force_refresh=True)
        print(f"Refreshed: {context3.fetch_time != context.fetch_time}")


async def main():
    """Run all demos."""
    print("=" * 60)
    print("External Data API Clients - Demonstration")
    print("=" * 60)

    # Note about actual API availability
    print("\nNote: These demos will attempt to connect to real APIs.")
    print("Some may fail if:")
    print("  - You don't have internet connection")
    print("  - APIs are down or rate-limited")
    print("  - API authentication is required (Coinglass)")
    print()

    try:
        await demo_fear_greed()
    except Exception as e:
        print(f"\nFear & Greed demo failed: {e}")

    try:
        await demo_coinglass()
    except Exception as e:
        print(f"\nCoinglass demo failed: {e}")

    try:
        await demo_aggregator()
    except Exception as e:
        print(f"\nAggregator demo failed: {e}")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
