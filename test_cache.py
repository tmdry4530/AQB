#!/usr/bin/env python3
"""
Quick test script for Redis cache functionality.

This script tests the basic functionality of the cache module without
requiring a full test suite setup.
"""

import asyncio
from datetime import datetime

from src.iftb.data.cache import CacheManager, RedisClient
from src.iftb.data.external import FearGreedData
from src.iftb.data.fetcher import OHLCVBar, Ticker


async def test_redis_client():
    """Test RedisClient basic operations."""
    print("\n=== Testing RedisClient ===")

    try:
        async with RedisClient(host="localhost", port=6379, db=15) as client:
            # Test set/get
            await client.set("test_key", "test_value", ttl=10)
            value = await client.get("test_key")
            assert value == "test_value", f"Expected 'test_value', got {value}"
            print("✓ Basic set/get works")

            # Test exists
            exists = await client.exists("test_key")
            assert exists, "Key should exist"
            print("✓ Exists check works")

            # Test delete
            await client.delete("test_key")
            value = await client.get("test_key")
            assert value is None, "Key should be deleted"
            print("✓ Delete works")

            # Test JSON operations
            test_data = {"name": "BTC", "price": 45000.0}
            await client.set_json("test_json", test_data, ttl=10)
            retrieved = await client.get_json("test_json")
            assert retrieved == test_data, f"Expected {test_data}, got {retrieved}"
            print("✓ JSON set/get works")

            # Cleanup
            await client.delete("test_json")

    except ConnectionError:
        print("✗ Redis not running on localhost:6379")
        return False

    return True


async def test_ohlcv_cache():
    """Test OHLCVCache functionality."""
    print("\n=== Testing OHLCVCache ===")

    try:
        async with CacheManager(host="localhost", port=6379, db=15) as cache:
            # Create test bars
            bars = [
                OHLCVBar(
                    timestamp=1705478400000,
                    open=45000.0,
                    high=45500.0,
                    low=44800.0,
                    close=45200.0,
                    volume=1234.56,
                ),
                OHLCVBar(
                    timestamp=1705482000000,
                    open=45200.0,
                    high=45600.0,
                    low=45000.0,
                    close=45400.0,
                    volume=1456.78,
                ),
            ]

            # Test set_bars
            await cache.ohlcv.set_bars("BTCUSDT", "1h", bars, ttl=60)
            print("✓ Bars cached successfully")

            # Test get_bars
            retrieved_bars = await cache.ohlcv.get_bars("BTCUSDT", "1h", limit=10)
            assert retrieved_bars is not None, "Bars should be retrieved"
            assert len(retrieved_bars) == 2, f"Expected 2 bars, got {len(retrieved_bars)}"
            print(f"✓ Retrieved {len(retrieved_bars)} bars")

            # Test get_latest_bar
            latest = await cache.ohlcv.get_latest_bar("BTCUSDT", "1h")
            assert latest is not None, "Latest bar should exist"
            assert latest.timestamp == 1705482000000, "Latest bar has wrong timestamp"
            print("✓ Latest bar retrieved correctly")

            # Test invalidate
            await cache.ohlcv.invalidate("BTCUSDT", "1h")
            retrieved_bars = await cache.ohlcv.get_bars("BTCUSDT", "1h")
            assert retrieved_bars is None or len(retrieved_bars) == 0, "Bars should be invalidated"
            print("✓ Cache invalidation works")

    except ConnectionError:
        print("✗ Redis not running")
        return False

    return True


async def test_market_data_cache():
    """Test MarketDataCache functionality."""
    print("\n=== Testing MarketDataCache ===")

    try:
        async with CacheManager(host="localhost", port=6379, db=15) as cache:
            # Test ticker caching
            ticker = Ticker(
                symbol="BTCUSDT",
                bid=45100.0,
                ask=45110.0,
                last=45105.0,
                volume_24h=15000.0,
                change_24h=2.5,
                high_24h=45800.0,
                low_24h=43900.0,
                timestamp=1705478400000,
            )

            await cache.market.set_ticker("BTCUSDT", ticker, ttl=60)
            retrieved_ticker = await cache.market.get_ticker("BTCUSDT")
            assert retrieved_ticker is not None, "Ticker should be retrieved"
            assert retrieved_ticker.symbol == "BTCUSDT", "Ticker symbol mismatch"
            assert retrieved_ticker.last == 45105.0, "Ticker price mismatch"
            print("✓ Ticker caching works")

            # Test funding rate caching
            await cache.market.set_funding_rate("BTCUSDT", 0.0001, ttl=60)
            rate = await cache.market.get_funding_rate("BTCUSDT")
            assert rate is not None, "Funding rate should be retrieved"
            assert rate == 0.0001, f"Expected 0.0001, got {rate}"
            print("✓ Funding rate caching works")

            # Test fear & greed caching
            fg_data = FearGreedData(
                value=45,
                classification="Fear",
                timestamp=datetime.now(),
            )
            await cache.market.set_fear_greed(fg_data, ttl=60)
            retrieved_fg = await cache.market.get_fear_greed()
            assert retrieved_fg is not None, "Fear & Greed data should be retrieved"
            assert retrieved_fg.value == 45, "Fear & Greed value mismatch"
            print("✓ Fear & Greed caching works")

            # Cleanup
            await cache.clear_all()

    except ConnectionError:
        print("✗ Redis not running")
        return False

    return True


async def test_llm_cache():
    """Test LLMCache functionality."""
    print("\n=== Testing LLMCache ===")

    try:
        async with CacheManager(host="localhost", port=6379, db=15) as cache:
            # Test analysis caching
            analysis_result = {
                "sentiment": "bullish",
                "confidence": 0.85,
                "key_points": ["Strong momentum", "Volume increasing"],
            }

            input_hash = cache.llm._hash_input("test input data")
            await cache.llm.set_analysis("sentiment", input_hash, analysis_result, ttl=60)

            retrieved = await cache.llm.get_analysis("sentiment", input_hash)
            assert retrieved is not None, "Analysis should be retrieved"
            assert retrieved["sentiment"] == "bullish", "Sentiment mismatch"
            assert retrieved["confidence"] == 0.85, "Confidence mismatch"
            print("✓ LLM analysis caching works")

            # Test with different input
            different_hash = cache.llm._hash_input("different input")
            retrieved = await cache.llm.get_analysis("sentiment", different_hash)
            assert retrieved is None, "Should not retrieve analysis for different input"
            print("✓ Cache isolation works")

            # Cleanup
            await cache.clear_all()

    except ConnectionError:
        print("✗ Redis not running")
        return False

    return True


async def test_health_check():
    """Test health check functionality."""
    print("\n=== Testing Health Check ===")

    try:
        async with CacheManager(host="localhost", port=6379, db=15) as cache:
            is_healthy = await cache.health_check()
            assert is_healthy, "Cache should be healthy"
            print("✓ Health check passed")

    except ConnectionError:
        print("✗ Redis not running")
        return False

    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Redis Cache Module Test Suite")
    print("=" * 60)
    print("\nNote: This requires Redis running on localhost:6379")
    print("Uses database 15 to avoid conflicts with production data")

    tests = [
        test_redis_client,
        test_ohlcv_cache,
        test_market_data_cache,
        test_llm_cache,
        test_health_check,
    ]

    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"Test Results: {passed}/{total} passed")
    print("=" * 60)

    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
