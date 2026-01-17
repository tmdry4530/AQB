"""
Unit tests for Telegram news collector.

Tests the TelegramNewsCollector class including message parsing,
urgent detection, and message queue management.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iftb.data.telegram import NewsMessage, TelegramNewsCollector


class TestNewsMessage:
    """Test cases for NewsMessage dataclass."""

    def test_news_message_creation(self) -> None:
        """Test creating a NewsMessage instance."""
        msg = NewsMessage(
            timestamp=datetime.now(timezone.utc),
            text="Breaking: Bitcoin reaches new ATH",
            channel="CryptoNews",
            channel_id=123456,
            message_id=789,
            has_media=False,
            is_forwarded=False,
            is_urgent=True,
            keywords=["breaking"],
        )

        assert msg.text == "Breaking: Bitcoin reaches new ATH"
        assert msg.is_urgent
        assert "breaking" in msg.keywords

    def test_news_message_repr(self) -> None:
        """Test string representation of NewsMessage."""
        msg = NewsMessage(
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            text="Test message",
            channel="TestChannel",
            channel_id=123,
            message_id=456,
            has_media=False,
            is_forwarded=False,
            is_urgent=True,
            keywords=["urgent"],
        )

        repr_str = repr(msg)
        assert "[URGENT]" in repr_str
        assert "TestChannel" in repr_str
        assert "Test message" in repr_str


class TestTelegramNewsCollector:
    """Test cases for TelegramNewsCollector."""

    @pytest.fixture
    def collector(self) -> TelegramNewsCollector:
        """Create a collector instance for testing."""
        return TelegramNewsCollector(
            api_id=12345,
            api_hash="test_hash",
            channel_ids=[123, 456],
            max_queue_size=10,
        )

    def test_collector_initialization(self, collector: TelegramNewsCollector) -> None:
        """Test collector initialization."""
        assert collector.api_id == 12345
        assert collector.api_hash == "test_hash"
        assert collector.channel_ids == [123, 456]
        assert collector.message_count == 0
        assert not collector.is_running

    def test_is_urgent_english_keywords(self, collector: TelegramNewsCollector) -> None:
        """Test urgent detection for English keywords."""
        test_cases = [
            ("Breaking news about Bitcoin", True, ["breaking"]),
            ("URGENT: Market crash detected", True, ["urgent", "crash"]),
            ("Hack discovered in major exchange", True, ["hack"]),
            ("Normal market update", False, []),
            ("The hacker was caught", True, ["hack"]),  # Word boundary test
        ]

        for text, expected_urgent, expected_keywords in test_cases:
            is_urgent, keywords = collector._is_urgent(text)
            assert is_urgent == expected_urgent, f"Failed for: {text}"
            for kw in expected_keywords:
                assert kw in keywords, f"Keyword {kw} not found in {keywords}"

    def test_is_urgent_korean_keywords(self, collector: TelegramNewsCollector) -> None:
        """Test urgent detection for Korean keywords."""
        test_cases = [
            ("속보: 비트코인 급등", True, ["속보", "급등"]),
            ("긴급 공지사항", True, ["긴급"]),
            ("일반 뉴스입니다", False, []),
        ]

        for text, expected_urgent, expected_keywords in test_cases:
            is_urgent, keywords = collector._is_urgent(text)
            assert is_urgent == expected_urgent, f"Failed for: {text}"
            for kw in expected_keywords:
                assert kw in keywords, f"Keyword {kw} not found in {keywords}"

    def test_is_urgent_mixed_keywords(self, collector: TelegramNewsCollector) -> None:
        """Test urgent detection for mixed English and Korean keywords."""
        text = "Breaking 속보: ETF approved 승인됨"
        is_urgent, keywords = collector._is_urgent(text)

        assert is_urgent
        assert "breaking" in keywords
        assert "속보" in keywords
        assert "approved" in keywords
        assert "승인" in keywords

    def test_parse_message_with_text(self, collector: TelegramNewsCollector) -> None:
        """Test parsing a message with text."""
        mock_message = MagicMock()
        mock_message.text = "Breaking: Bitcoin hits 100k"
        mock_message.caption = None
        mock_message.id = 123
        mock_message.media = None
        mock_message.forward_date = None
        mock_message.chat = MagicMock()
        mock_message.chat.title = "CryptoNews"
        mock_message.chat.id = 456

        news_msg = collector._parse_message(mock_message)

        assert news_msg is not None
        assert news_msg.text == "Breaking: Bitcoin hits 100k"
        assert news_msg.channel == "CryptoNews"
        assert news_msg.channel_id == 456
        assert news_msg.message_id == 123
        assert news_msg.is_urgent
        assert "breaking" in news_msg.keywords

    def test_parse_message_with_caption(self, collector: TelegramNewsCollector) -> None:
        """Test parsing a message with caption (photo/video)."""
        mock_message = MagicMock()
        mock_message.text = None
        mock_message.caption = "Market analysis chart"
        mock_message.id = 789
        mock_message.media = True
        mock_message.forward_date = None
        mock_message.chat = MagicMock()
        mock_message.chat.username = "market_updates"
        mock_message.chat.id = 321

        news_msg = collector._parse_message(mock_message)

        assert news_msg is not None
        assert news_msg.text == "Market analysis chart"
        assert news_msg.has_media

    def test_parse_message_no_text(self, collector: TelegramNewsCollector) -> None:
        """Test parsing a message without text."""
        mock_message = MagicMock()
        mock_message.text = None
        mock_message.caption = None

        news_msg = collector._parse_message(mock_message)

        assert news_msg is None

    def test_parse_message_forwarded(self, collector: TelegramNewsCollector) -> None:
        """Test parsing a forwarded message."""
        mock_message = MagicMock()
        mock_message.text = "Forwarded news"
        mock_message.caption = None
        mock_message.id = 999
        mock_message.media = None
        mock_message.forward_date = datetime.now(timezone.utc)
        mock_message.chat = MagicMock()
        mock_message.chat.id = 111

        news_msg = collector._parse_message(mock_message)

        assert news_msg is not None
        assert news_msg.is_forwarded

    def test_get_recent_messages(self, collector: TelegramNewsCollector) -> None:
        """Test retrieving recent messages."""
        # Add some test messages
        for i in range(5):
            msg = NewsMessage(
                timestamp=datetime.now(timezone.utc),
                text=f"Message {i}",
                channel="TestChannel",
                channel_id=123,
                message_id=i,
                has_media=False,
                is_forwarded=False,
                is_urgent=False,
                keywords=[],
            )
            collector._messages.append(msg)

        recent = collector.get_recent_messages(minutes=60)
        assert len(recent) == 5

    def test_get_news_summary_empty(self, collector: TelegramNewsCollector) -> None:
        """Test generating summary with no messages."""
        summary = collector.get_news_summary()
        assert "No recent news messages available" in summary

    def test_get_news_summary_with_messages(
        self, collector: TelegramNewsCollector
    ) -> None:
        """Test generating summary with messages."""
        # Add test messages
        msg1 = NewsMessage(
            timestamp=datetime.now(timezone.utc),
            text="Breaking: Important news",
            channel="Channel1",
            channel_id=123,
            message_id=1,
            has_media=False,
            is_forwarded=False,
            is_urgent=True,
            keywords=["breaking"],
        )
        msg2 = NewsMessage(
            timestamp=datetime.now(timezone.utc),
            text="Regular update",
            channel="Channel2",
            channel_id=456,
            message_id=2,
            has_media=False,
            is_forwarded=False,
            is_urgent=False,
            keywords=[],
        )
        collector._messages.append(msg1)
        collector._messages.append(msg2)

        summary = collector.get_news_summary(max_messages=10)

        assert "Recent Telegram News" in summary
        assert "Channel1" in summary
        assert "Channel2" in summary
        assert "URGENT" in summary
        assert "Breaking: Important news" in summary

    def test_clear_messages(self, collector: TelegramNewsCollector) -> None:
        """Test clearing messages."""
        # Add test messages
        for i in range(3):
            msg = NewsMessage(
                timestamp=datetime.now(timezone.utc),
                text=f"Message {i}",
                channel="TestChannel",
                channel_id=123,
                message_id=i,
                has_media=False,
                is_forwarded=False,
                is_urgent=False,
                keywords=[],
            )
            collector._messages.append(msg)

        assert collector.message_count == 3

        collector.clear_messages()

        assert collector.message_count == 0

    def test_message_count_property(self, collector: TelegramNewsCollector) -> None:
        """Test message count property."""
        assert collector.message_count == 0

        msg = NewsMessage(
            timestamp=datetime.now(timezone.utc),
            text="Test",
            channel="Test",
            channel_id=123,
            message_id=1,
            has_media=False,
            is_forwarded=False,
            is_urgent=False,
            keywords=[],
        )
        collector._messages.append(msg)

        assert collector.message_count == 1

    @pytest.mark.asyncio
    async def test_handle_message_with_callback(
        self, collector: TelegramNewsCollector
    ) -> None:
        """Test handling message with urgent callback."""
        callback_called = False
        callback_message = None

        async def callback(msg: NewsMessage) -> None:
            nonlocal callback_called, callback_message
            callback_called = True
            callback_message = msg

        collector.on_urgent_message = callback

        # Create urgent mock message
        mock_message = MagicMock()
        mock_message.text = "Breaking news"
        mock_message.caption = None
        mock_message.id = 123
        mock_message.media = None
        mock_message.forward_date = None
        mock_message.chat = MagicMock()
        mock_message.chat.id = 456
        mock_message.chat.title = "Test"

        await collector._handle_message(mock_message)

        assert callback_called
        assert callback_message is not None
        assert callback_message.is_urgent

    @pytest.mark.asyncio
    async def test_handle_message_callback_error(
        self, collector: TelegramNewsCollector
    ) -> None:
        """Test handling message when callback raises error."""

        async def failing_callback(msg: NewsMessage) -> None:
            raise ValueError("Callback error")

        collector.on_urgent_message = failing_callback

        # Create urgent mock message
        mock_message = MagicMock()
        mock_message.text = "Breaking news"
        mock_message.caption = None
        mock_message.id = 123
        mock_message.media = None
        mock_message.forward_date = None
        mock_message.chat = MagicMock()
        mock_message.chat.id = 456

        # Should not raise, just log the error
        await collector._handle_message(mock_message)

        # Message should still be added to queue
        assert collector.message_count == 1


@pytest.mark.asyncio
async def test_create_collector_from_settings() -> None:
    """Test creating collector from settings."""
    with patch("iftb.data.telegram.get_settings") as mock_settings:
        mock_telegram_settings = MagicMock()
        mock_telegram_settings.api_id = 12345
        mock_telegram_settings.api_hash.get_secret_value.return_value = "test_hash"
        mock_telegram_settings.news_channel_ids = [111, 222]

        mock_settings.return_value.telegram = mock_telegram_settings

        from iftb.data.telegram import create_collector_from_settings

        collector = await create_collector_from_settings()

        assert collector.api_id == 12345
        assert collector.api_hash == "test_hash"
        assert collector.channel_ids == [111, 222]
