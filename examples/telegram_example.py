"""
Example usage of the Telegram News Collector.

This example demonstrates how to:
1. Create a collector from settings
2. Handle urgent messages
3. Retrieve recent messages
4. Generate news summaries for LLM analysis
"""

import asyncio
from datetime import datetime

from iftb.data.telegram import NewsMessage, create_collector_from_settings
from iftb.utils import LogConfig, get_logger, setup_logging

# Setup logging
config = LogConfig(
    level="INFO",
    format="pretty",
    include_timestamp=True,
    include_caller_info=False,
)
setup_logging(config)

logger = get_logger(__name__)


async def handle_urgent_message(message: NewsMessage) -> None:
    """Callback for urgent messages.

    Args:
        message: The urgent news message
    """
    logger.warning(
        "urgent_news_received",
        channel=message.channel,
        keywords=message.keywords,
        text=message.text[:100],
    )
    print("\n" + "=" * 80)
    print(f"URGENT NEWS - {datetime.now().strftime('%H:%M:%S')}")
    print(f"Channel: {message.channel}")
    print(f"Keywords: {', '.join(message.keywords)}")
    print(f"Message: {message.text}")
    print("=" * 80 + "\n")


async def main() -> None:
    """Main example function."""
    logger.info("telegram_example_starting")

    # Create collector from settings
    collector = await create_collector_from_settings(on_urgent_message=handle_urgent_message)

    try:
        async with collector:
            # Start collecting in background
            collection_task = asyncio.create_task(collector.start())

            # Wait a bit for some messages
            logger.info("collecting_messages", duration_seconds=60)
            await asyncio.sleep(60)

            # Get recent messages
            recent = collector.get_recent_messages(minutes=10)
            logger.info(
                "recent_messages_retrieved",
                count=len(recent),
                urgent_count=sum(1 for msg in recent if msg.is_urgent),
            )

            # Generate summary for LLM
            summary = collector.get_news_summary(max_messages=20)
            print("\n" + summary + "\n")

            # Stop collection
            collection_task.cancel()
            try:
                await collection_task
            except asyncio.CancelledError:
                pass

    except KeyboardInterrupt:
        logger.info("telegram_example_interrupted")
    except Exception as e:
        logger.error("telegram_example_error", error=str(e), error_type=type(e).__name__)
    finally:
        logger.info("telegram_example_finished")


if __name__ == "__main__":
    asyncio.run(main())
