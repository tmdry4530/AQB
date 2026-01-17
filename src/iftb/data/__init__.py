"""
Data fetching and management module for IFTB trading bot.

This module provides interfaces for fetching market data from exchanges,
managing historical data, handling price feeds, and collecting Telegram news.
"""

from .cache import (
    CacheManager,
    LLMCache,
    MarketDataCache,
    OHLCVCache,
    RedisClient,
)
from .external import (
    CoinglassClient,
    ExternalDataAggregator,
    FearGreedClient,
    FearGreedData,
    FundingData,
    LongShortData,
    MarketContext,
    OpenInterestData,
)
from .fetcher import (
    ExchangeClient,
    FundingRate,
    HistoricalDataDownloader,
    OHLCVBar,
    Ticker,
    fetch_latest_ohlcv,
    fetch_latest_ticker,
)
from .storage import (
    DatabaseManager,
    OHLCVRepository,
    Position,
    PositionRepository,
    SystemEvent,
    SystemEventRepository,
    Trade,
    TradeRepository,
    TradeStatistics,
)
from .storage import (
    OHLCVBar as StorageOHLCVBar,
)
from .telegram import (
    NewsMessage,
    TelegramNewsCollector,
    create_collector_from_settings,
)
from .validation import (
    DataQualityReport,
    OHLCVValidator,
    calculate_data_statistics,
)
from .websocket import (
    AggTradeMessage,
    BinanceFuturesWebSocket,
    BookTickerMessage,
    ConnectionState,
    DepthMessage,
    KlineMessage,
    MarketDataStreamer,
    MarkPriceMessage,
    MiniTickerMessage,
    RealTimeDataManager,
    StreamConfig,
    StreamType,
    TickerMessage,
    create_market_streamer,
)

__all__ = [
    # Cache
    "CacheManager",
    "RedisClient",
    "OHLCVCache",
    "MarketDataCache",
    "LLMCache",
    # Fetcher
    "ExchangeClient",
    "HistoricalDataDownloader",
    "OHLCVBar",
    "Ticker",
    "FundingRate",
    "fetch_latest_ohlcv",
    "fetch_latest_ticker",
    # Storage
    "DatabaseManager",
    "OHLCVRepository",
    "TradeRepository",
    "PositionRepository",
    "SystemEventRepository",
    "StorageOHLCVBar",
    "Trade",
    "Position",
    "SystemEvent",
    "TradeStatistics",
    # Validation
    "DataQualityReport",
    "OHLCVValidator",
    "calculate_data_statistics",
    # External data
    "FearGreedClient",
    "FearGreedData",
    "CoinglassClient",
    "FundingData",
    "OpenInterestData",
    "LongShortData",
    "ExternalDataAggregator",
    "MarketContext",
    # Telegram news
    "NewsMessage",
    "TelegramNewsCollector",
    "create_collector_from_settings",
    # WebSocket streaming
    "StreamType",
    "StreamConfig",
    "ConnectionState",
    "KlineMessage",
    "TickerMessage",
    "MiniTickerMessage",
    "AggTradeMessage",
    "BookTickerMessage",
    "MarkPriceMessage",
    "DepthMessage",
    "BinanceFuturesWebSocket",
    "MarketDataStreamer",
    "RealTimeDataManager",
    "create_market_streamer",
]
