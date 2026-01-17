"""
Telegram news collector for IFTB trading bot.

This module collects real-time news from Telegram channels using Pyrogram,
with support for urgent message detection, message filtering, and callback hooks.

Example Usage:
    ```python
    from iftb.data.telegram import TelegramNewsCollector, NewsMessage
    from iftb.config import get_settings

    settings = get_settings()

    async def handle_urgent(message: NewsMessage):
        print(f"URGENT: {message.text}")

    collector = TelegramNewsCollector(
        api_id=settings.telegram.api_id,
        api_hash=settings.telegram.api_hash.get_secret_value(),
        channel_ids=settings.telegram.news_channel_ids,
        on_urgent_message=handle_urgent
    )

    async with collector:
        await collector.start()
        # Keep running...
        await asyncio.sleep(3600)
    ```
"""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import re
from threading import Lock

from pyrogram import Client, filters
from pyrogram.errors import FloodWait
from pyrogram.types import Message

from iftb.config import get_settings
from iftb.utils import get_logger

logger = get_logger(__name__)


@dataclass
class NewsMessage:
    """Represents a processed news message from Telegram.

    Attributes:
        timestamp: When the message was received (UTC)
        text: Message text content
        channel: Channel name/username
        channel_id: Telegram channel ID
        message_id: Telegram message ID
        has_media: Whether message contains media (photo, video, etc)
        is_forwarded: Whether message was forwarded from another source
        is_urgent: Whether message contains urgent keywords
        keywords: List of detected urgent keywords (lowercase)
    """

    timestamp: datetime
    text: str
    channel: str
    channel_id: int
    message_id: int
    has_media: bool
    is_forwarded: bool
    is_urgent: bool
    keywords: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        """String representation of the news message."""
        urgent_flag = "[URGENT]" if self.is_urgent else ""
        timestamp_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"{urgent_flag} [{timestamp_str}] {self.channel}: {text_preview}"


class TelegramNewsCollector:
    """Collects and processes news from Telegram channels.

    This collector monitors specified Telegram channels for new messages,
    detects urgent news based on keywords, and provides access to recent
    messages for analysis.

    Attributes:
        api_id: Telegram API ID
        api_hash: Telegram API hash
        channel_ids: List of channel IDs to monitor
        on_urgent_message: Optional callback for urgent messages
    """

    # Urgent keywords for detection
    URGENT_KEYWORDS_EN = {
        "breaking",
        "urgent",
        "hack",
        "exploit",
        "sec",
        "etf",
        "approved",
        "rejected",
        "lawsuit",
        "crash",
        "dump",
        "pump",
        "liquidation",
        "bankruptcy",
        "fraud",
        "alert",
        "warning",
        "critical",
    }

    URGENT_KEYWORDS_KR = {
        "속보",
        "긴급",
        "해킹",
        "승인",
        "거부",
        "폭락",
        "급등",
        "청산",
        "경고",
        "주의",
    }

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        channel_ids: list[int],
        on_urgent_message: Callable[[NewsMessage], Awaitable[None]] | None = None,
        max_queue_size: int = 200,
    ) -> None:
        """Initialize the Telegram news collector.

        Args:
            api_id: Telegram API ID from my.telegram.org
            api_hash: Telegram API hash from my.telegram.org
            channel_ids: List of channel IDs to monitor
            on_urgent_message: Optional async callback for urgent messages
            max_queue_size: Maximum number of messages to keep in memory
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.channel_ids = channel_ids
        self.on_urgent_message = on_urgent_message

        # Message queue with thread-safe access
        self._messages: deque[NewsMessage] = deque(maxlen=max_queue_size)
        self._lock = Lock()

        # Pyrogram client
        self._client: Client | None = None
        self._is_running = False
        self._reconnect_delay = 5
        self._max_reconnect_delay = 300

        logger.info(
            "telegram_collector_initialized",
            api_id=api_id,
            channel_count=len(channel_ids),
            max_queue_size=max_queue_size,
        )

    async def __aenter__(self) -> "TelegramNewsCollector":
        """Async context manager entry."""
        await self._initialize_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        """Async context manager exit."""
        await self.stop()

    async def _initialize_client(self) -> None:
        """Initialize the Pyrogram client."""
        if self._client is not None:
            logger.warning("telegram_client_already_initialized")
            return

        self._client = Client(
            name="iftb_news_collector",
            api_id=self.api_id,
            api_hash=self.api_hash,
            in_memory=True,  # Don't persist session to disk
        )

        # Register message handler for monitored channels
        @self._client.on_message(filters.chat(self.channel_ids))
        async def message_handler(_: Client, message: Message) -> None:
            """Handle incoming messages from monitored channels."""
            await self._handle_message(message)

        logger.info("telegram_client_initialized")

    async def start(self) -> None:
        """Start listening to Telegram channels.

        This method connects to Telegram and begins receiving messages.
        It includes automatic reconnection logic for handling disconnects.
        """
        if self._client is None:
            await self._initialize_client()

        if self._is_running:
            logger.warning("telegram_collector_already_running")
            return

        self._is_running = True
        retry_count = 0

        while self._is_running:
            try:
                logger.info("telegram_connecting")
                await self._client.start()  # type: ignore[union-attr]
                logger.info("telegram_connected", channels=self.channel_ids)

                # Reset retry count on successful connection
                retry_count = 0
                self._reconnect_delay = 5

                # Keep the client running
                await self._client.idle()  # type: ignore[union-attr]

            except FloodWait as e:
                # Handle Telegram rate limiting
                wait_time = e.value
                logger.warning(
                    "telegram_flood_wait",
                    wait_seconds=wait_time,
                    retry_count=retry_count,
                )
                await asyncio.sleep(wait_time)

            except Exception as e:
                retry_count += 1
                logger.error(
                    "telegram_connection_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    retry_count=retry_count,
                )

                if self._is_running:
                    # Exponential backoff for reconnection
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(
                        self._reconnect_delay * 2, self._max_reconnect_delay
                    )
                    logger.info(
                        "telegram_reconnecting",
                        next_retry_seconds=self._reconnect_delay,
                    )
                else:
                    break

    async def stop(self) -> None:
        """Stop listening and disconnect gracefully."""
        if not self._is_running:
            logger.warning("telegram_collector_not_running")
            return

        logger.info("telegram_stopping")
        self._is_running = False

        if self._client is not None:
            try:
                await self._client.stop()
                logger.info("telegram_stopped")
            except Exception as e:
                logger.error(
                    "telegram_stop_error",
                    error=str(e),
                    error_type=type(e).__name__,
                )
            finally:
                self._client = None

    async def _handle_message(self, message: Message) -> None:
        """Process incoming Telegram message.

        Args:
            message: Pyrogram Message object
        """
        try:
            news_message = self._parse_message(message)
            if news_message is None:
                return

            # Add to queue (thread-safe)
            with self._lock:
                self._messages.append(news_message)

            logger.info(
                "telegram_message_received",
                channel=news_message.channel,
                message_id=news_message.message_id,
                is_urgent=news_message.is_urgent,
                has_media=news_message.has_media,
                text_length=len(news_message.text),
            )

            # Trigger urgent callback if applicable
            if news_message.is_urgent and self.on_urgent_message is not None:
                try:
                    await self.on_urgent_message(news_message)
                except Exception as e:
                    logger.error(
                        "urgent_callback_error",
                        error=str(e),
                        error_type=type(e).__name__,
                        message_id=news_message.message_id,
                    )

        except Exception as e:
            logger.error(
                "message_handling_error",
                error=str(e),
                error_type=type(e).__name__,
                message_id=message.id if message else None,
            )

    def _parse_message(self, message: Message) -> NewsMessage | None:
        """Parse Telegram message into NewsMessage.

        Args:
            message: Pyrogram Message object

        Returns:
            Parsed NewsMessage or None if message should be skipped
        """
        # Skip messages without text
        if not message.text and not message.caption:
            return None

        # Get message text (prefer text over caption)
        text = message.text or message.caption or ""
        text = text.strip()

        if not text:
            return None

        # Get channel info
        channel_name = "Unknown"
        channel_id = 0

        if message.chat:
            channel_name = (
                message.chat.title
                or message.chat.username
                or f"Channel_{message.chat.id}"
            )
            channel_id = message.chat.id

        # Detect urgent keywords
        is_urgent, keywords = self._is_urgent(text)

        # Create NewsMessage
        news_message = NewsMessage(
            timestamp=datetime.now(UTC),
            text=text,
            channel=channel_name,
            channel_id=channel_id,
            message_id=message.id,
            has_media=bool(message.media),
            is_forwarded=bool(message.forward_date),
            is_urgent=is_urgent,
            keywords=keywords,
        )

        return news_message

    def _is_urgent(self, text: str) -> tuple[bool, list[str]]:
        """Detect urgent keywords in message text.

        Args:
            text: Message text to analyze

        Returns:
            Tuple of (is_urgent, detected_keywords)
        """
        text_lower = text.lower()
        detected_keywords: list[str] = []

        # Check English keywords
        for keyword in self.URGENT_KEYWORDS_EN:
            # Use word boundaries for more accurate matching
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, text_lower):
                detected_keywords.append(keyword)

        # Check Korean keywords (no word boundaries needed)
        for keyword in self.URGENT_KEYWORDS_KR:
            if keyword in text:
                detected_keywords.append(keyword)

        is_urgent = len(detected_keywords) > 0

        return is_urgent, detected_keywords

    def get_recent_messages(self, minutes: int = 60) -> list[NewsMessage]:
        """Get messages from the last N minutes.

        Args:
            minutes: Number of minutes to look back

        Returns:
            List of NewsMessage objects from the specified time window
        """
        cutoff_time = datetime.now(UTC) - timedelta(minutes=minutes)

        with self._lock:
            recent = [
                msg for msg in self._messages if msg.timestamp >= cutoff_time
            ]

        logger.info(
            "telegram_recent_messages_retrieved",
            minutes=minutes,
            message_count=len(recent),
        )

        return recent

    def get_news_summary(self, max_messages: int = 20) -> str:
        """Generate a summary of recent news for LLM analysis.

        Args:
            max_messages: Maximum number of messages to include

        Returns:
            Formatted string summary suitable for LLM processing
        """
        with self._lock:
            # Get most recent messages
            messages = list(self._messages)[-max_messages:]

        if not messages:
            return "No recent news messages available."

        # Build summary
        lines = [
            "=== Recent Telegram News ===",
            f"Total messages: {len(messages)}",
            "",
        ]

        urgent_count = sum(1 for msg in messages if msg.is_urgent)
        if urgent_count > 0:
            lines.append(f"URGENT messages: {urgent_count}")
            lines.append("")

        # Group by channel
        by_channel: dict[str, list[NewsMessage]] = {}
        for msg in messages:
            if msg.channel not in by_channel:
                by_channel[msg.channel] = []
            by_channel[msg.channel].append(msg)

        # Format messages by channel
        for channel, channel_messages in by_channel.items():
            lines.append(f"## {channel} ({len(channel_messages)} messages)")
            lines.append("")

            for msg in channel_messages:
                timestamp_str = msg.timestamp.strftime("%H:%M:%S")
                urgent_flag = "[URGENT] " if msg.is_urgent else ""
                keywords_str = (
                    f" (keywords: {', '.join(msg.keywords)})"
                    if msg.keywords
                    else ""
                )

                lines.append(f"- {timestamp_str} {urgent_flag}{msg.text}{keywords_str}")

            lines.append("")

        logger.info(
            "telegram_summary_generated",
            message_count=len(messages),
            urgent_count=urgent_count,
            channel_count=len(by_channel),
        )

        return "\n".join(lines)

    @property
    def is_running(self) -> bool:
        """Check if the collector is currently running."""
        return self._is_running

    @property
    def message_count(self) -> int:
        """Get the current number of messages in the queue."""
        with self._lock:
            return len(self._messages)

    def clear_messages(self) -> None:
        """Clear all messages from the queue."""
        with self._lock:
            self._messages.clear()
        logger.info("telegram_messages_cleared")


async def create_collector_from_settings(
    on_urgent_message: Callable[[NewsMessage], Awaitable[None]] | None = None,
) -> TelegramNewsCollector:
    """Create a TelegramNewsCollector from application settings.

    Args:
        on_urgent_message: Optional async callback for urgent messages

    Returns:
        Configured TelegramNewsCollector instance

    Example:
        ```python
        async def handle_urgent(msg: NewsMessage):
            print(f"Urgent: {msg.text}")

        collector = await create_collector_from_settings(
            on_urgent_message=handle_urgent
        )
        ```
    """
    settings = get_settings()

    collector = TelegramNewsCollector(
        api_id=settings.telegram.api_id,
        api_hash=settings.telegram.api_hash.get_secret_value(),
        channel_ids=settings.telegram.news_channel_ids,
        on_urgent_message=on_urgent_message,
    )

    logger.info("telegram_collector_created_from_settings")
    return collector
