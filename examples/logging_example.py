"""
Example demonstrating the IFTB logging system.

This script shows various logging features including:
- Basic logging at different levels
- Contextual logging with add_context
- Sensitive data filtering
- Pretty and JSON output formats
"""

from iftb.utils import LogConfig, add_context, get_logger, setup_logging


def demo_basic_logging():
    """Demonstrate basic logging at different levels."""
    logger = get_logger(__name__)

    print("\n=== Basic Logging ===")
    logger.debug("debug_message", detail="This is a debug message")
    logger.info("info_message", detail="This is an info message")
    logger.warning("warning_message", detail="This is a warning")
    logger.error("error_message", detail="This is an error")


def demo_trade_logging():
    """Demonstrate trade-specific logging."""
    logger = get_logger(__name__)

    print("\n=== Trade Logging ===")
    logger.info(
        "trade_opened",
        symbol="BTCUSDT",
        side="LONG",
        price=50000.00,
        quantity=0.5,
        leverage=10,
    )

    logger.info(
        "position_update",
        symbol="ETHUSDT",
        unrealized_pnl=250.50,
        realized_pnl=100.00,
        entry_price=3000.00,
        current_price=3050.00,
    )

    logger.warning(
        "high_drawdown",
        symbol="BTCUSDT",
        drawdown_pct=15.5,
        max_drawdown=20.0,
        action="reducing_position",
    )


def demo_contextual_logging():
    """Demonstrate contextual logging with add_context."""
    logger = get_logger(__name__)

    print("\n=== Contextual Logging ===")

    # All logs within this context will include trade_id and symbol
    with add_context(trade_id="abc123", symbol="BTCUSDT"):
        logger.info("processing_trade")
        logger.info("order_placed", order_id="xyz789", price=50000)
        logger.info("order_filled", order_id="xyz789", filled_qty=0.5)

        # Nested context
        with add_context(stop_loss=49000, take_profit=52000):
            logger.info("risk_management_set")

    # Context is cleared outside the block
    logger.info("outside_context")


def demo_sensitive_filtering():
    """Demonstrate sensitive data filtering."""
    logger = get_logger(__name__)

    print("\n=== Sensitive Data Filtering ===")

    # These sensitive fields will be masked automatically
    logger.info(
        "api_connection",
        api_key="my_secret_api_key_12345",
        api_secret="my_secret_api_secret_67890",
        endpoint="https://api.exchange.com",
    )

    logger.info(
        "user_login",
        username="trader123",
        password="super_secret_password",
        token="bearer_token_xyz",
    )


def demo_error_logging():
    """Demonstrate error logging with exception info."""
    logger = get_logger(__name__)

    print("\n=== Error Logging ===")

    try:
        result = 1 / 0
    except ZeroDivisionError as e:
        logger.error(
            "calculation_error",
            error=str(e),
            operation="division",
            exc_info=True,
        )

    try:
        data = {"key": "value"}
        _ = data["missing_key"]
    except KeyError as e:
        logger.error(
            "data_error",
            error=str(e),
            data_keys=list(data.keys()),
            exc_info=True,
        )


def main():
    """Run all logging examples."""
    # Setup logging with pretty format for development
    print("=== PRETTY FORMAT (Development) ===")
    config = LogConfig(
        level="DEBUG",
        format="pretty",
        include_timestamp=True,
        include_caller_info=True,
        environment="dev",
        app_version="1.0.0",
    )
    setup_logging(config)

    demo_basic_logging()
    demo_trade_logging()
    demo_contextual_logging()
    demo_sensitive_filtering()
    demo_error_logging()

    # Reconfigure with JSON format for production-like output
    print("\n\n=== JSON FORMAT (Production) ===")
    config = LogConfig(
        level="INFO",
        format="json",
        file_path="logs/iftb.log",
        include_timestamp=True,
        include_caller_info=False,
        environment="prod",
        app_version="1.0.0",
    )
    setup_logging(config)

    demo_trade_logging()
    demo_contextual_logging()

    print("\n\nLogs have also been written to: logs/iftb.log")


if __name__ == "__main__":
    main()
