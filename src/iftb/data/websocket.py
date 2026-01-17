"""
Real-time WebSocket receiver for Binance Futures.

Provides streaming market data via WebSocket connections with:
- Automatic reconnection with exponential backoff
- Multiple stream subscriptions (kline, ticker, depth)
- Message queuing for downstream processing
- Health monitoring and heartbeat handling
"""

from abc import ABC, abstractmethod
import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import json
import time
from typing import Any

import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
)

from iftb.config import get_settings
from iftb.utils import get_logger

logger = get_logger(__name__)


class StreamType(Enum):
    """WebSocket stream types."""

    KLINE = "kline"
    TICKER = "ticker"
    MINI_TICKER = "miniTicker"
    AGG_TRADE = "aggTrade"
    DEPTH = "depth"
    BOOK_TICKER = "bookTicker"
    MARK_PRICE = "markPrice"
    FUNDING_RATE = "fundingRate"


class ConnectionState(Enum):
    """WebSocket connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


@dataclass
class StreamConfig:
    """Configuration for a WebSocket stream subscription."""

    stream_type: StreamType
    symbol: str
    interval: str | None = None  # For kline streams
    depth_level: int | None = None  # For depth streams (5, 10, 20)
    update_speed: str | None = None  # For depth streams (100ms, 1000ms)

    def to_stream_name(self) -> str:
        """Convert config to Binance stream name format."""
        symbol_lower = self.symbol.lower()

        if self.stream_type == StreamType.KLINE:
            if not self.interval:
                raise ValueError("Interval required for kline stream")
            return f"{symbol_lower}@kline_{self.interval}"

        if self.stream_type == StreamType.TICKER:
            return f"{symbol_lower}@ticker"

        if self.stream_type == StreamType.MINI_TICKER:
            return f"{symbol_lower}@miniTicker"

        if self.stream_type == StreamType.AGG_TRADE:
            return f"{symbol_lower}@aggTrade"

        if self.stream_type == StreamType.DEPTH:
            level = self.depth_level or 20
            speed = self.update_speed or "100ms"
            return f"{symbol_lower}@depth{level}@{speed}"

        if self.stream_type == StreamType.BOOK_TICKER:
            return f"{symbol_lower}@bookTicker"

        if self.stream_type == StreamType.MARK_PRICE:
            return f"{symbol_lower}@markPrice"

        if self.stream_type == StreamType.FUNDING_RATE:
            return f"{symbol_lower}@markPrice"  # Funding rate comes with mark price

        raise ValueError(f"Unknown stream type: {self.stream_type}")


@dataclass
class KlineMessage:
    """Parsed kline/candlestick WebSocket message."""

    symbol: str
    interval: str
    open_time: int
    close_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int
    is_closed: bool
    received_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_ws_message(cls, data: dict[str, Any]) -> "KlineMessage":
        """Parse from Binance WebSocket kline message."""
        k = data["k"]
        return cls(
            symbol=data["s"],
            interval=k["i"],
            open_time=k["t"],
            close_time=k["T"],
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            quote_volume=float(k["q"]),
            trades=k["n"],
            is_closed=k["x"],
        )


@dataclass
class TickerMessage:
    """Parsed 24hr ticker WebSocket message."""

    symbol: str
    price_change: float
    price_change_percent: float
    weighted_avg_price: float
    last_price: float
    last_qty: float
    open_price: float
    high_price: float
    low_price: float
    volume: float
    quote_volume: float
    open_time: int
    close_time: int
    first_trade_id: int
    last_trade_id: int
    trade_count: int
    received_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_ws_message(cls, data: dict[str, Any]) -> "TickerMessage":
        """Parse from Binance WebSocket ticker message."""
        return cls(
            symbol=data["s"],
            price_change=float(data["p"]),
            price_change_percent=float(data["P"]),
            weighted_avg_price=float(data["w"]),
            last_price=float(data["c"]),
            last_qty=float(data["Q"]),
            open_price=float(data["o"]),
            high_price=float(data["h"]),
            low_price=float(data["l"]),
            volume=float(data["v"]),
            quote_volume=float(data["q"]),
            open_time=data["O"],
            close_time=data["C"],
            first_trade_id=data["F"],
            last_trade_id=data["L"],
            trade_count=data["n"],
        )


@dataclass
class MiniTickerMessage:
    """Parsed mini ticker WebSocket message."""

    symbol: str
    close_price: float
    open_price: float
    high_price: float
    low_price: float
    base_volume: float
    quote_volume: float
    received_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_ws_message(cls, data: dict[str, Any]) -> "MiniTickerMessage":
        """Parse from Binance WebSocket mini ticker message."""
        return cls(
            symbol=data["s"],
            close_price=float(data["c"]),
            open_price=float(data["o"]),
            high_price=float(data["h"]),
            low_price=float(data["l"]),
            base_volume=float(data["v"]),
            quote_volume=float(data["q"]),
        )


@dataclass
class AggTradeMessage:
    """Parsed aggregate trade WebSocket message."""

    symbol: str
    agg_trade_id: int
    price: float
    quantity: float
    first_trade_id: int
    last_trade_id: int
    trade_time: int
    is_buyer_maker: bool
    received_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_ws_message(cls, data: dict[str, Any]) -> "AggTradeMessage":
        """Parse from Binance WebSocket aggTrade message."""
        return cls(
            symbol=data["s"],
            agg_trade_id=data["a"],
            price=float(data["p"]),
            quantity=float(data["q"]),
            first_trade_id=data["f"],
            last_trade_id=data["l"],
            trade_time=data["T"],
            is_buyer_maker=data["m"],
        )


@dataclass
class BookTickerMessage:
    """Parsed book ticker (best bid/ask) WebSocket message."""

    symbol: str
    best_bid_price: float
    best_bid_qty: float
    best_ask_price: float
    best_ask_qty: float
    transaction_time: int
    received_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_ws_message(cls, data: dict[str, Any]) -> "BookTickerMessage":
        """Parse from Binance WebSocket bookTicker message."""
        return cls(
            symbol=data["s"],
            best_bid_price=float(data["b"]),
            best_bid_qty=float(data["B"]),
            best_ask_price=float(data["a"]),
            best_ask_qty=float(data["A"]),
            transaction_time=data.get("T", int(time.time() * 1000)),
        )


@dataclass
class MarkPriceMessage:
    """Parsed mark price and funding rate WebSocket message."""

    symbol: str
    mark_price: float
    index_price: float
    estimated_settle_price: float
    funding_rate: float
    next_funding_time: int
    received_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_ws_message(cls, data: dict[str, Any]) -> "MarkPriceMessage":
        """Parse from Binance WebSocket markPrice message."""
        return cls(
            symbol=data["s"],
            mark_price=float(data["p"]),
            index_price=float(data["i"]),
            estimated_settle_price=float(data.get("P", data["p"])),
            funding_rate=float(data["r"]),
            next_funding_time=data["T"],
        )


@dataclass
class DepthMessage:
    """Parsed order book depth WebSocket message."""

    symbol: str
    first_update_id: int
    final_update_id: int
    bids: list[tuple[float, float]]  # [(price, quantity), ...]
    asks: list[tuple[float, float]]  # [(price, quantity), ...]
    received_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_ws_message(cls, data: dict[str, Any]) -> "DepthMessage":
        """Parse from Binance WebSocket depth message."""
        return cls(
            symbol=data["s"],
            first_update_id=data["U"],
            final_update_id=data["u"],
            bids=[(float(p), float(q)) for p, q in data["b"]],
            asks=[(float(p), float(q)) for p, q in data["a"]],
        )


# Type alias for any WebSocket message
WSMessage = (
    KlineMessage
    | TickerMessage
    | MiniTickerMessage
    | AggTradeMessage
    | BookTickerMessage
    | MarkPriceMessage
    | DepthMessage
)

# Message handler callback type
MessageHandler = Callable[[WSMessage], None]
AsyncMessageHandler = Callable[[WSMessage], Any]


class BaseWebSocketClient(ABC):
    """Abstract base class for WebSocket clients."""

    def __init__(
        self,
        base_url: str,
        max_reconnect_attempts: int = 10,
        initial_reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 60.0,
        ping_interval: float = 20.0,
        ping_timeout: float = 10.0,
    ):
        self.base_url = base_url
        self.max_reconnect_attempts = max_reconnect_attempts
        self.initial_reconnect_delay = initial_reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout

        self._ws: WebSocketClientProtocol | None = None
        self._state = ConnectionState.DISCONNECTED
        self._reconnect_count = 0
        self._message_queue: asyncio.Queue[WSMessage] = asyncio.Queue()
        self._handlers: list[AsyncMessageHandler] = []
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._last_message_time: float = 0
        self._messages_received: int = 0
        self._connect_time: float | None = None

    @property
    def state(self) -> ConnectionState:
        """Current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._state == ConnectionState.CONNECTED and self._ws is not None

    @property
    def stats(self) -> dict[str, Any]:
        """Get connection statistics."""
        uptime = None
        if self._connect_time:
            uptime = time.time() - self._connect_time

        return {
            "state": self._state.value,
            "messages_received": self._messages_received,
            "reconnect_count": self._reconnect_count,
            "last_message_time": self._last_message_time,
            "uptime_seconds": uptime,
            "queue_size": self._message_queue.qsize(),
        }

    def add_handler(self, handler: AsyncMessageHandler) -> None:
        """Add a message handler callback."""
        self._handlers.append(handler)

    def remove_handler(self, handler: AsyncMessageHandler) -> None:
        """Remove a message handler callback."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    @abstractmethod
    def _build_url(self) -> str:
        """Build the WebSocket URL. Implemented by subclasses."""

    @abstractmethod
    def _parse_message(self, raw_data: str) -> WSMessage | None:
        """Parse raw WebSocket message. Implemented by subclasses."""

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        if self._state in (ConnectionState.CONNECTED, ConnectionState.CONNECTING):
            logger.warning("websocket_already_connected", state=self._state.value)
            return

        self._state = ConnectionState.CONNECTING
        url = self._build_url()

        try:
            self._ws = await websockets.connect(
                url,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout,
                close_timeout=10,
            )
            self._state = ConnectionState.CONNECTED
            self._reconnect_count = 0
            self._connect_time = time.time()

            logger.info(
                "websocket_connected",
                url=url[:100] + "..." if len(url) > 100 else url,
            )

        except Exception as e:
            self._state = ConnectionState.DISCONNECTED
            logger.error("websocket_connect_failed", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._running = False
        self._state = ConnectionState.CLOSED

        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._tasks.clear()

        # Close WebSocket
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.warning("websocket_close_error", error=str(e))
            finally:
                self._ws = None

        logger.info("websocket_disconnected", stats=self.stats)

    async def _reconnect(self) -> bool:
        """Attempt to reconnect with exponential backoff."""
        if self._reconnect_count >= self.max_reconnect_attempts:
            logger.error(
                "websocket_max_reconnect_attempts",
                attempts=self._reconnect_count,
            )
            return False

        self._state = ConnectionState.RECONNECTING
        self._reconnect_count += 1

        # Calculate delay with exponential backoff
        delay = min(
            self.initial_reconnect_delay * (2 ** (self._reconnect_count - 1)),
            self.max_reconnect_delay,
        )

        logger.info(
            "websocket_reconnecting",
            attempt=self._reconnect_count,
            delay=delay,
        )

        await asyncio.sleep(delay)

        try:
            await self.connect()
            return True
        except Exception as e:
            logger.warning(
                "websocket_reconnect_failed",
                attempt=self._reconnect_count,
                error=str(e),
            )
            return False

    async def _receive_loop(self) -> None:
        """Main loop for receiving WebSocket messages."""
        while self._running:
            try:
                if not self._ws:
                    if not await self._reconnect():
                        break
                    continue

                try:
                    raw_data = await self._ws.recv()
                    self._last_message_time = time.time()
                    self._messages_received += 1

                    message = self._parse_message(raw_data)
                    if message:
                        await self._message_queue.put(message)

                except ConnectionClosedOK:
                    logger.info("websocket_closed_ok")
                    break

                except ConnectionClosedError as e:
                    logger.warning("websocket_closed_error", code=e.code, reason=e.reason)
                    self._ws = None
                    if self._running and not await self._reconnect():
                        break

                except ConnectionClosed as e:
                    logger.warning("websocket_connection_closed", reason=str(e))
                    self._ws = None
                    if self._running and not await self._reconnect():
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("websocket_receive_error", error=str(e), exc_info=True)
                await asyncio.sleep(1)

    async def _dispatch_loop(self) -> None:
        """Loop for dispatching messages to handlers."""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0,
                )

                for handler in self._handlers:
                    try:
                        result = handler(message)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(
                            "websocket_handler_error",
                            handler=handler.__name__,
                            error=str(e),
                        )

            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("websocket_dispatch_error", error=str(e))

    async def start(self) -> None:
        """Start the WebSocket client."""
        if self._running:
            logger.warning("websocket_already_running")
            return

        self._running = True
        await self.connect()

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._receive_loop()),
            asyncio.create_task(self._dispatch_loop()),
        ]

        logger.info("websocket_client_started")

    async def stop(self) -> None:
        """Stop the WebSocket client."""
        await self.disconnect()
        logger.info("websocket_client_stopped")

    async def __aenter__(self) -> "BaseWebSocketClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    async def messages(self) -> AsyncIterator[WSMessage]:
        """Async iterator for receiving messages."""
        while self._running or not self._message_queue.empty():
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0,
                )
                yield message
            except TimeoutError:
                if not self._running:
                    break
            except asyncio.CancelledError:
                break


class BinanceFuturesWebSocket(BaseWebSocketClient):
    """
    Binance Futures WebSocket client.

    Supports multiple stream subscriptions via combined streams.
    """

    MAINNET_WS_URL = "wss://fstream.binance.com"
    TESTNET_WS_URL = "wss://stream.binancefuture.com"

    def __init__(
        self,
        streams: list[StreamConfig],
        testnet: bool = False,
        **kwargs,
    ):
        base_url = self.TESTNET_WS_URL if testnet else self.MAINNET_WS_URL
        super().__init__(base_url=base_url, **kwargs)

        self.streams = streams
        self.testnet = testnet
        self._stream_names = [s.to_stream_name() for s in streams]

    def _build_url(self) -> str:
        """Build combined streams WebSocket URL."""
        if len(self._stream_names) == 1:
            return f"{self.base_url}/ws/{self._stream_names[0]}"
        combined = "/".join(self._stream_names)
        return f"{self.base_url}/stream?streams={combined}"

    def _parse_message(self, raw_data: str) -> WSMessage | None:
        """Parse Binance WebSocket message."""
        try:
            data = json.loads(raw_data)

            # Handle combined stream format
            if "stream" in data and "data" in data:
                stream_name = data["stream"]
                payload = data["data"]
            else:
                # Single stream format
                stream_name = None
                payload = data

            # Determine message type and parse
            event_type = payload.get("e")

            if event_type == "kline":
                return KlineMessage.from_ws_message(payload)
            if event_type == "24hrTicker":
                return TickerMessage.from_ws_message(payload)
            if event_type == "24hrMiniTicker":
                return MiniTickerMessage.from_ws_message(payload)
            if event_type == "aggTrade":
                return AggTradeMessage.from_ws_message(payload)
            if event_type == "bookTicker":
                return BookTickerMessage.from_ws_message(payload)
            if event_type == "markPriceUpdate":
                return MarkPriceMessage.from_ws_message(payload)
            if event_type == "depthUpdate":
                return DepthMessage.from_ws_message(payload)
            logger.debug("websocket_unknown_event", event_type=event_type)
            return None

        except json.JSONDecodeError as e:
            logger.warning("websocket_json_decode_error", error=str(e))
            return None
        except KeyError as e:
            logger.warning("websocket_missing_field", field=str(e))
            return None
        except Exception as e:
            logger.error("websocket_parse_error", error=str(e), exc_info=True)
            return None

    async def subscribe(self, streams: list[StreamConfig]) -> None:
        """Subscribe to additional streams (requires reconnection)."""
        self.streams.extend(streams)
        self._stream_names = [s.to_stream_name() for s in self.streams]

        # Reconnect with new streams
        if self.is_connected:
            await self.disconnect()
            await self.connect()

    async def unsubscribe(self, streams: list[StreamConfig]) -> None:
        """Unsubscribe from streams (requires reconnection)."""
        stream_names_to_remove = {s.to_stream_name() for s in streams}
        self.streams = [s for s in self.streams if s.to_stream_name() not in stream_names_to_remove]
        self._stream_names = [s.to_stream_name() for s in self.streams]

        # Reconnect with remaining streams
        if self.is_connected and self.streams:
            await self.disconnect()
            await self.connect()
        elif not self.streams:
            await self.disconnect()


class MarketDataStreamer:
    """
    High-level market data streaming manager.

    Coordinates multiple WebSocket connections and provides
    a unified interface for real-time market data.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
        include_trades: bool = False,
        include_book_ticker: bool = True,
        include_mark_price: bool = True,
        testnet: bool | None = None,
    ):
        settings = get_settings()

        self.symbols = symbols or settings.trading.symbols
        self.timeframes = timeframes or settings.trading.timeframes
        self.include_trades = include_trades
        self.include_book_ticker = include_book_ticker
        self.include_mark_price = include_mark_price
        self.testnet = testnet if testnet is not None else settings.exchange.testnet

        self._clients: list[BinanceFuturesWebSocket] = []
        self._handlers: dict[StreamType, list[AsyncMessageHandler]] = {}
        self._running = False

        # Statistics
        self._klines_received: dict[str, int] = {}
        self._tickers_received: dict[str, int] = {}
        self._start_time: float | None = None

    def _build_stream_configs(self) -> list[StreamConfig]:
        """Build stream configurations based on settings."""
        configs = []

        for symbol in self.symbols:
            # Kline streams for each timeframe
            for timeframe in self.timeframes:
                configs.append(
                    StreamConfig(
                        stream_type=StreamType.KLINE,
                        symbol=symbol,
                        interval=timeframe,
                    )
                )

            # Mini ticker for price updates
            configs.append(
                StreamConfig(
                    stream_type=StreamType.MINI_TICKER,
                    symbol=symbol,
                )
            )

            # Optional streams
            if self.include_trades:
                configs.append(
                    StreamConfig(
                        stream_type=StreamType.AGG_TRADE,
                        symbol=symbol,
                    )
                )

            if self.include_book_ticker:
                configs.append(
                    StreamConfig(
                        stream_type=StreamType.BOOK_TICKER,
                        symbol=symbol,
                    )
                )

            if self.include_mark_price:
                configs.append(
                    StreamConfig(
                        stream_type=StreamType.MARK_PRICE,
                        symbol=symbol,
                    )
                )

        return configs

    def on_kline(self, handler: AsyncMessageHandler) -> None:
        """Register handler for kline messages."""
        if StreamType.KLINE not in self._handlers:
            self._handlers[StreamType.KLINE] = []
        self._handlers[StreamType.KLINE].append(handler)

    def on_ticker(self, handler: AsyncMessageHandler) -> None:
        """Register handler for ticker messages."""
        if StreamType.MINI_TICKER not in self._handlers:
            self._handlers[StreamType.MINI_TICKER] = []
        self._handlers[StreamType.MINI_TICKER].append(handler)

    def on_trade(self, handler: AsyncMessageHandler) -> None:
        """Register handler for aggregate trade messages."""
        if StreamType.AGG_TRADE not in self._handlers:
            self._handlers[StreamType.AGG_TRADE] = []
        self._handlers[StreamType.AGG_TRADE].append(handler)

    def on_book_ticker(self, handler: AsyncMessageHandler) -> None:
        """Register handler for book ticker messages."""
        if StreamType.BOOK_TICKER not in self._handlers:
            self._handlers[StreamType.BOOK_TICKER] = []
        self._handlers[StreamType.BOOK_TICKER].append(handler)

    def on_mark_price(self, handler: AsyncMessageHandler) -> None:
        """Register handler for mark price messages."""
        if StreamType.MARK_PRICE not in self._handlers:
            self._handlers[StreamType.MARK_PRICE] = []
        self._handlers[StreamType.MARK_PRICE].append(handler)

    async def _dispatch_message(self, message: WSMessage) -> None:
        """Dispatch message to appropriate handlers."""
        # Determine stream type from message type
        stream_type = None
        if isinstance(message, KlineMessage):
            stream_type = StreamType.KLINE
            key = f"{message.symbol}_{message.interval}"
            self._klines_received[key] = self._klines_received.get(key, 0) + 1
        elif isinstance(message, (TickerMessage, MiniTickerMessage)):
            stream_type = StreamType.MINI_TICKER
            self._tickers_received[message.symbol] = (
                self._tickers_received.get(message.symbol, 0) + 1
            )
        elif isinstance(message, AggTradeMessage):
            stream_type = StreamType.AGG_TRADE
        elif isinstance(message, BookTickerMessage):
            stream_type = StreamType.BOOK_TICKER
        elif isinstance(message, MarkPriceMessage):
            stream_type = StreamType.MARK_PRICE

        if stream_type and stream_type in self._handlers:
            for handler in self._handlers[stream_type]:
                try:
                    result = handler(message)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(
                        "streamer_handler_error",
                        stream_type=stream_type.value,
                        error=str(e),
                    )

    @property
    def stats(self) -> dict[str, Any]:
        """Get streaming statistics."""
        uptime = None
        if self._start_time:
            uptime = time.time() - self._start_time

        client_stats = [c.stats for c in self._clients]

        return {
            "running": self._running,
            "symbols": self.symbols,
            "timeframes": self.timeframes,
            "uptime_seconds": uptime,
            "klines_received": self._klines_received,
            "tickers_received": self._tickers_received,
            "clients": client_stats,
        }

    async def start(self) -> None:
        """Start streaming market data."""
        if self._running:
            logger.warning("streamer_already_running")
            return

        self._running = True
        self._start_time = time.time()

        # Build stream configs and create client
        configs = self._build_stream_configs()

        # Binance allows max 200 streams per connection
        # Split into multiple connections if needed
        max_streams_per_connection = 200
        for i in range(0, len(configs), max_streams_per_connection):
            batch = configs[i : i + max_streams_per_connection]
            client = BinanceFuturesWebSocket(
                streams=batch,
                testnet=self.testnet,
            )
            client.add_handler(self._dispatch_message)
            self._clients.append(client)

        # Start all clients
        await asyncio.gather(*[c.start() for c in self._clients])

        logger.info(
            "market_data_streamer_started",
            symbols=self.symbols,
            timeframes=self.timeframes,
            num_clients=len(self._clients),
            total_streams=len(configs),
        )

    async def stop(self) -> None:
        """Stop streaming market data."""
        self._running = False

        # Stop all clients
        await asyncio.gather(*[c.stop() for c in self._clients])
        self._clients.clear()

        logger.info("market_data_streamer_stopped", stats=self.stats)

    async def __aenter__(self) -> "MarketDataStreamer":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


class RealTimeDataManager:
    """
    Integrated real-time data manager.

    Combines WebSocket streaming with caching and storage
    for a complete real-time data pipeline.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
        enable_caching: bool = True,
        enable_storage: bool = False,
        testnet: bool | None = None,
    ):
        self.symbols = symbols
        self.timeframes = timeframes
        self.enable_caching = enable_caching
        self.enable_storage = enable_storage
        self.testnet = testnet

        self._streamer: MarketDataStreamer | None = None
        self._cache = None  # Will be set if caching enabled
        self._storage = None  # Will be set if storage enabled

        # Latest data cache (in-memory)
        self._latest_klines: dict[str, KlineMessage] = {}
        self._latest_tickers: dict[str, MiniTickerMessage] = {}
        self._latest_book_tickers: dict[str, BookTickerMessage] = {}
        self._latest_mark_prices: dict[str, MarkPriceMessage] = {}

        # Callbacks for completed candles
        self._candle_complete_handlers: list[AsyncMessageHandler] = []

    def on_candle_complete(self, handler: AsyncMessageHandler) -> None:
        """Register handler for completed candles."""
        self._candle_complete_handlers.append(handler)

    async def _handle_kline(self, message: KlineMessage) -> None:
        """Handle incoming kline message."""
        key = f"{message.symbol}_{message.interval}"
        self._latest_klines[key] = message

        # If candle is closed, notify handlers
        if message.is_closed:
            for handler in self._candle_complete_handlers:
                try:
                    result = handler(message)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error("candle_complete_handler_error", error=str(e))

            # Update cache if enabled
            if self.enable_caching and self._cache:
                # Cache logic would go here
                pass

            # Store to database if enabled
            if self.enable_storage and self._storage:
                # Storage logic would go here
                pass

    async def _handle_ticker(self, message: MiniTickerMessage) -> None:
        """Handle incoming ticker message."""
        self._latest_tickers[message.symbol] = message

    async def _handle_book_ticker(self, message: BookTickerMessage) -> None:
        """Handle incoming book ticker message."""
        self._latest_book_tickers[message.symbol] = message

    async def _handle_mark_price(self, message: MarkPriceMessage) -> None:
        """Handle incoming mark price message."""
        self._latest_mark_prices[message.symbol] = message

    def get_latest_price(self, symbol: str) -> float | None:
        """Get latest price for a symbol."""
        if symbol in self._latest_tickers:
            return self._latest_tickers[symbol].close_price
        return None

    def get_latest_kline(self, symbol: str, interval: str) -> KlineMessage | None:
        """Get latest kline for a symbol and interval."""
        key = f"{symbol}_{interval}"
        return self._latest_klines.get(key)

    def get_latest_book_ticker(self, symbol: str) -> BookTickerMessage | None:
        """Get latest book ticker for a symbol."""
        return self._latest_book_tickers.get(symbol)

    def get_latest_mark_price(self, symbol: str) -> MarkPriceMessage | None:
        """Get latest mark price for a symbol."""
        return self._latest_mark_prices.get(symbol)

    def get_bid_ask_spread(self, symbol: str) -> tuple[float, float, float] | None:
        """Get bid, ask, and spread for a symbol."""
        book = self._latest_book_tickers.get(symbol)
        if not book:
            return None

        spread = book.best_ask_price - book.best_bid_price
        return (book.best_bid_price, book.best_ask_price, spread)

    @property
    def stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        return {
            "symbols_tracked": list(self._latest_tickers.keys()),
            "klines_tracked": list(self._latest_klines.keys()),
            "book_tickers_tracked": list(self._latest_book_tickers.keys()),
            "mark_prices_tracked": list(self._latest_mark_prices.keys()),
            "streamer_stats": self._streamer.stats if self._streamer else None,
        }

    async def start(self) -> None:
        """Start the real-time data manager."""
        self._streamer = MarketDataStreamer(
            symbols=self.symbols,
            timeframes=self.timeframes,
            include_book_ticker=True,
            include_mark_price=True,
            testnet=self.testnet,
        )

        # Register handlers
        self._streamer.on_kline(self._handle_kline)
        self._streamer.on_ticker(self._handle_ticker)
        self._streamer.on_book_ticker(self._handle_book_ticker)
        self._streamer.on_mark_price(self._handle_mark_price)

        await self._streamer.start()

        logger.info("realtime_data_manager_started")

    async def stop(self) -> None:
        """Stop the real-time data manager."""
        if self._streamer:
            await self._streamer.stop()
            self._streamer = None

        logger.info("realtime_data_manager_stopped", stats=self.stats)

    async def __aenter__(self) -> "RealTimeDataManager":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


# Convenience function for quick setup
async def create_market_streamer(
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    testnet: bool | None = None,
) -> MarketDataStreamer:
    """Create and start a market data streamer."""
    streamer = MarketDataStreamer(
        symbols=symbols,
        timeframes=timeframes,
        testnet=testnet,
    )
    await streamer.start()
    return streamer


# Example usage and testing
async def _example_usage() -> None:
    """Example demonstrating WebSocket streaming."""
    # Setup logging
    from iftb.utils import LogConfig, setup_logging

    setup_logging(LogConfig(level="DEBUG", format="pretty"))

    # Create a simple kline handler
    async def on_kline(msg: KlineMessage) -> None:
        if msg.is_closed:
            print(
                f"[CLOSED] {msg.symbol} {msg.interval}: "
                f"O={msg.open:.2f} H={msg.high:.2f} L={msg.low:.2f} C={msg.close:.2f} "
                f"V={msg.volume:.2f}"
            )
        else:
            print(f"[LIVE] {msg.symbol} {msg.interval}: {msg.close:.2f}")

    async def on_ticker(msg: MiniTickerMessage) -> None:
        print(f"[TICKER] {msg.symbol}: {msg.close_price:.2f}")

    # Use the streamer
    async with MarketDataStreamer(
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframes=["1m"],
        testnet=True,
    ) as streamer:
        streamer.on_kline(on_kline)
        streamer.on_ticker(on_ticker)

        # Run for 30 seconds
        await asyncio.sleep(30)

        print(f"Stats: {streamer.stats}")


if __name__ == "__main__":
    asyncio.run(_example_usage())
