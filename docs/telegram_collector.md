# Telegram News Collector

The Telegram News Collector is a real-time news collection system for the IFTB trading bot. It monitors specified Telegram channels and automatically detects urgent news that may impact trading decisions.

## Features

- Real-time message collection from multiple Telegram channels
- Automatic urgent news detection (English and Korean keywords)
- Thread-safe message queue with configurable size
- Async context manager support for clean resource management
- Optional callback system for urgent message notifications
- Message filtering and search capabilities
- LLM-ready news summaries
- Automatic reconnection on network failures
- FloodWait exception handling

## Installation

The Telegram collector requires Pyrogram and TgCrypto:

```bash
pip install pyrogram tgcrypto
```

These dependencies are already included in the project's `pyproject.toml`.

## Configuration

Add the following to your `.env` file:

```bash
# Telegram API credentials (get from https://my.telegram.org)
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=your_api_hash_here
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_ALERT_CHAT_ID=your_chat_id

# News channels to monitor (comma-separated channel IDs)
TELEGRAM_NEWS_CHANNEL_IDS=[-1001234567890,-1009876543210]
```

### Finding Channel IDs

1. Add the channel to your Telegram
2. Forward any message from the channel to [@userinfobot](https://t.me/userinfobot)
3. The bot will show you the channel ID (use the negative number)

## Basic Usage

### Simple Example

```python
import asyncio
from iftb.data.telegram import create_collector_from_settings

async def main():
    # Create collector from settings
    collector = await create_collector_from_settings()

    async with collector:
        # Start collecting
        await collector.start()

        # Wait for some messages
        await asyncio.sleep(60)

        # Get recent messages
        messages = collector.get_recent_messages(minutes=10)
        print(f"Collected {len(messages)} messages")

asyncio.run(main())
```

### With Urgent Message Callback

```python
from iftb.data.telegram import NewsMessage, create_collector_from_settings

async def handle_urgent(message: NewsMessage):
    """Handle urgent news messages."""
    print(f"URGENT: {message.text}")
    print(f"Keywords: {', '.join(message.keywords)}")

    # Trigger trading analysis, send alerts, etc.
    # await analyze_news_impact(message.text)

async def main():
    collector = await create_collector_from_settings(
        on_urgent_message=handle_urgent
    )

    async with collector:
        await collector.start()

asyncio.run(main())
```

### Manual Configuration

```python
from iftb.data.telegram import TelegramNewsCollector

collector = TelegramNewsCollector(
    api_id=12345678,
    api_hash="your_api_hash",
    channel_ids=[-1001234567890, -1009876543210],
    on_urgent_message=handle_urgent,
    max_queue_size=200
)

async with collector:
    await collector.start()
```

## API Reference

### TelegramNewsCollector

Main collector class for Telegram news.

#### Constructor Parameters

- `api_id: int` - Telegram API ID from my.telegram.org
- `api_hash: str` - Telegram API hash from my.telegram.org
- `channel_ids: list[int]` - List of channel IDs to monitor
- `on_urgent_message: Optional[Callable]` - Async callback for urgent messages
- `max_queue_size: int` - Maximum messages to keep in memory (default: 200)

#### Methods

##### `async start() -> None`

Start listening to Telegram channels. Includes automatic reconnection logic.

```python
await collector.start()
```

##### `async stop() -> None`

Stop listening and disconnect gracefully.

```python
await collector.stop()
```

##### `get_recent_messages(minutes: int = 60) -> list[NewsMessage]`

Get messages from the last N minutes.

```python
messages = collector.get_recent_messages(minutes=30)
for msg in messages:
    print(f"{msg.channel}: {msg.text}")
```

##### `get_news_summary(max_messages: int = 20) -> str`

Generate a formatted summary suitable for LLM analysis.

```python
summary = collector.get_news_summary(max_messages=50)
# Feed to LLM for analysis
response = await llm_client.analyze(summary)
```

##### `clear_messages() -> None`

Clear all messages from the queue.

```python
collector.clear_messages()
```

#### Properties

##### `is_running: bool`

Check if the collector is currently running.

```python
if collector.is_running:
    print("Collector is active")
```

##### `message_count: int`

Get the current number of messages in the queue.

```python
print(f"Queue has {collector.message_count} messages")
```

### NewsMessage

Data class representing a processed Telegram message.

#### Attributes

- `timestamp: datetime` - When the message was received (UTC)
- `text: str` - Message text content
- `channel: str` - Channel name/username
- `channel_id: int` - Telegram channel ID
- `message_id: int` - Telegram message ID
- `has_media: bool` - Whether message contains media
- `is_forwarded: bool` - Whether message was forwarded
- `is_urgent: bool` - Whether message contains urgent keywords
- `keywords: list[str]` - Detected urgent keywords (lowercase)

#### Example

```python
for msg in recent_messages:
    print(f"[{msg.timestamp}] {msg.channel}")
    print(f"Text: {msg.text}")
    if msg.is_urgent:
        print(f"URGENT - Keywords: {', '.join(msg.keywords)}")
```

### Urgent Keywords

The collector automatically detects urgent news using these keywords:

#### English Keywords
- breaking, urgent, hack, exploit, sec, etf, approved, rejected
- lawsuit, crash, dump, pump, liquidation, bankruptcy, fraud
- alert, warning, critical

#### Korean Keywords
- 속보, 긴급, 해킹, 승인, 거부
- 폭락, 급등, 청산, 경고, 주의

## Advanced Usage

### Integration with Trading Bot

```python
from iftb.data.telegram import create_collector_from_settings
from iftb.decision import TradingDecisionEngine

class NewsTradingBot:
    def __init__(self):
        self.collector = None
        self.decision_engine = TradingDecisionEngine()

    async def handle_urgent_news(self, message: NewsMessage):
        """Process urgent news and make trading decisions."""
        # Generate LLM analysis
        summary = self.collector.get_news_summary(max_messages=20)

        # Analyze sentiment and impact
        analysis = await self.decision_engine.analyze_news(summary)

        # Adjust positions if needed
        if analysis.sentiment < -0.5:  # Bearish
            await self.reduce_exposure()
        elif analysis.sentiment > 0.5:  # Bullish
            await self.increase_exposure()

    async def run(self):
        self.collector = await create_collector_from_settings(
            on_urgent_message=self.handle_urgent_news
        )

        async with self.collector:
            await self.collector.start()
```

### Background Collection with Periodic Analysis

```python
import asyncio
from iftb.data.telegram import create_collector_from_settings

async def periodic_analysis(collector: TelegramNewsCollector):
    """Analyze news every 5 minutes."""
    while True:
        await asyncio.sleep(300)  # 5 minutes

        recent = collector.get_recent_messages(minutes=10)
        urgent_count = sum(1 for msg in recent if msg.is_urgent)

        if urgent_count > 3:
            print("High frequency of urgent news detected!")
            # Trigger additional analysis or alerts

async def main():
    collector = await create_collector_from_settings()

    async with collector:
        # Start both tasks
        collection = asyncio.create_task(collector.start())
        analysis = asyncio.create_task(periodic_analysis(collector))

        # Run until interrupted
        await asyncio.gather(collection, analysis)

asyncio.run(main())
```

### Custom Keyword Detection

```python
class CustomTelegramCollector(TelegramNewsCollector):
    """Extended collector with custom keywords."""

    URGENT_KEYWORDS_EN = TelegramNewsCollector.URGENT_KEYWORDS_EN | {
        "halted", "suspended", "delisted", "investigation"
    }

    URGENT_KEYWORDS_KR = TelegramNewsCollector.URGENT_KEYWORDS_KR | {
        "중단", "조사", "상장폐지"
    }

collector = CustomTelegramCollector(
    api_id=12345,
    api_hash="hash",
    channel_ids=[123, 456]
)
```

## Error Handling

The collector includes robust error handling:

### Automatic Reconnection

```python
# Collector automatically reconnects on disconnection
# with exponential backoff (5s -> 10s -> 20s ... up to 300s)
await collector.start()  # Will retry indefinitely until stopped
```

### FloodWait Handling

```python
# Telegram rate limits are automatically handled
# The collector waits the required time before retrying
```

### Callback Exceptions

```python
async def safe_callback(message: NewsMessage):
    try:
        await process_message(message)
    except Exception as e:
        logger.error(f"Callback error: {e}")
        # Error is logged but doesn't crash the collector

collector = TelegramNewsCollector(
    api_id=12345,
    api_hash="hash",
    channel_ids=[123],
    on_urgent_message=safe_callback
)
```

## Logging

The collector uses structured logging:

```python
from iftb.utils import setup_logging, LogConfig

# Setup logging
config = LogConfig(
    level="INFO",
    format="pretty",
    file_path="logs/telegram.log"
)
setup_logging(config)

# Collector will log:
# - telegram_collector_initialized
# - telegram_connecting
# - telegram_connected
# - telegram_message_received
# - telegram_flood_wait
# - telegram_connection_error
# - telegram_stopped
```

## Performance Considerations

### Memory Usage

- Default queue size: 200 messages
- Each message: ~1-2 KB (depending on text length)
- Total memory: ~200-400 KB for queue

### Rate Limits

Telegram has rate limits:
- ~30 requests per second
- FloodWait exceptions trigger automatic waiting
- No manual throttling needed

### Scaling

For multiple collectors:

```python
collectors = [
    await create_collector_from_settings(),
    # Can create multiple instances for different channel sets
]

await asyncio.gather(*[c.start() for c in collectors])
```

## Testing

Run the test suite:

```bash
pytest tests/unit/test_telegram.py -v
```

Test coverage includes:
- Message parsing
- Urgent keyword detection (English and Korean)
- Message queue management
- Callback handling
- Error scenarios

## Troubleshooting

### "FloodWait" Errors

**Cause:** Too many API requests
**Solution:** Collector automatically handles this with exponential backoff

### "Connection Timeout"

**Cause:** Network issues or Telegram downtime
**Solution:** Collector automatically reconnects

### "Invalid Channel ID"

**Cause:** Wrong channel ID format or no access
**Solution:**
1. Ensure channel ID is negative (e.g., -1001234567890)
2. Verify you have access to the channel
3. Check channel isn't private without proper authorization

### Missing Messages

**Cause:** Queue overflow (max_queue_size reached)
**Solution:** Increase max_queue_size or process messages more frequently

```python
collector = TelegramNewsCollector(
    api_id=12345,
    api_hash="hash",
    channel_ids=[123],
    max_queue_size=500  # Increased from default 200
)
```

## Best Practices

1. **Use Context Manager**: Always use `async with` for proper cleanup
2. **Handle Callbacks**: Keep callbacks fast and non-blocking
3. **Monitor Queue**: Check `message_count` to avoid overflow
4. **Log Errors**: Enable logging for production debugging
5. **Test Channels**: Test with low-volume channels first
6. **Graceful Shutdown**: Always call `stop()` before exit

## Example: Complete Trading Bot

See `/mnt/d/Develop/AQB/examples/telegram_example.py` for a complete working example.

## License

MIT License - See project root for details.
