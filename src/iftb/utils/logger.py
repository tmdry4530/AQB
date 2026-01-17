"""
Comprehensive logging system for IFTB trading bot.

This module provides structured logging with support for both console and file output,
JSON and pretty formatting, and contextual information tracking.

Example Usage:
    ```python
    from iftb.utils.logger import setup_logging, get_logger, add_context, LogConfig

    # Setup logging
    config = LogConfig(
        level="INFO",
        format="pretty",
        file_path="logs/iftb.log",
        include_timestamp=True,
        include_caller_info=True
    )
    setup_logging(config)

    # Get logger instance
    logger = get_logger(__name__)

    # Basic logging
    logger.info("trade_opened", symbol="BTCUSDT", side="LONG", price=50000)
    logger.warning("high_drawdown", drawdown_pct=15.5)
    logger.error("api_error", error="Connection timeout", retry_count=3)

    # Context-aware logging
    with add_context(trade_id="abc123", symbol="ETHUSDT"):
        logger.info("processing_trade")
        logger.info("order_placed", order_id="xyz789")
    ```
"""

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Any, Literal

import structlog
from structlog.types import EventDict, Processor

# Global context storage for thread-local context
_context_vars: dict[str, Any] = {}


@dataclass
class LogConfig:
    """Configuration for the logging system.

    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format - "json" for production, "pretty" for development
        file_path: Optional path to log file. If None, only logs to console
        include_timestamp: Whether to include timestamps in logs
        include_caller_info: Whether to include caller file/line information
        console_output: Whether to output to console (default: True)
        max_string_length: Maximum length for string values before truncation
        environment: Environment name (dev, staging, prod)
        app_version: Application version string
    """

    level: str = "INFO"
    format: Literal["json", "pretty"] = "pretty"
    file_path: str | None = None
    include_timestamp: bool = True
    include_caller_info: bool = True
    console_output: bool = True
    max_string_length: int = 1000
    environment: str = "dev"
    app_version: str = "1.0.0"


def add_app_info(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add application-level information to log entries.

    Args:
        logger: The logger instance
        method_name: The logging method name
        event_dict: The event dictionary

    Returns:
        Updated event dictionary with app info
    """
    event_dict["app"] = "iftb"
    event_dict["environment"] = getattr(add_app_info, "environment", "unknown")
    event_dict["version"] = getattr(add_app_info, "version", "unknown")
    return event_dict


def add_trade_context(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add trade-specific context to log entries.

    Pulls context from the global context storage and adds it to the log entry.

    Args:
        logger: The logger instance
        method_name: The logging method name
        event_dict: The event dictionary

    Returns:
        Updated event dictionary with trade context
    """
    # Add any context variables that have been set
    for key, value in _context_vars.items():
        if key not in event_dict:
            event_dict[key] = value
    return event_dict


def filter_sensitive(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Filter and mask sensitive information from logs.

    Masks API keys, passwords, tokens, and other sensitive data.

    Args:
        logger: The logger instance
        method_name: The logging method name
        event_dict: The event dictionary

    Returns:
        Updated event dictionary with masked sensitive data
    """
    sensitive_keys = {
        "api_key",
        "apikey",
        "api_secret",
        "apisecret",
        "password",
        "passwd",
        "pwd",
        "token",
        "access_token",
        "refresh_token",
        "secret",
        "private_key",
        "privatekey",
    }

    def mask_value(value: Any) -> Any:
        """Mask a sensitive value."""
        if isinstance(value, str):
            if len(value) <= 4:
                return "***"
            return f"{value[:2]}***{value[-2:]}"
        return "***"

    def recursive_mask(data: Any) -> Any:
        """Recursively mask sensitive data in nested structures."""
        if isinstance(data, dict):
            return {
                key: mask_value(value) if key.lower() in sensitive_keys else recursive_mask(value)
                for key, value in data.items()
            }
        if isinstance(data, (list, tuple)):
            return type(data)(recursive_mask(item) for item in data)
        return data

    result = recursive_mask(event_dict)
    return result  # type: ignore[return-value]


def add_log_level_name(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add human-readable log level name.

    Args:
        logger: The logger instance
        method_name: The logging method name
        event_dict: The event dictionary

    Returns:
        Updated event dictionary with level name
    """
    if "level" in event_dict:
        event_dict["level_name"] = event_dict["level"]
    return event_dict


def truncate_strings(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Truncate long string values to prevent log bloat.

    Args:
        logger: The logger instance
        method_name: The logging method name
        event_dict: The event dictionary

    Returns:
        Updated event dictionary with truncated strings
    """
    max_length = getattr(truncate_strings, "max_length", 1000)

    def truncate_value(value: Any) -> Any:
        """Truncate a single value if it's a long string."""
        if isinstance(value, str) and len(value) > max_length:
            return f"{value[:max_length]}... [truncated]"
        if isinstance(value, dict):
            return {k: truncate_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(truncate_value(item) for item in value)
        return value

    return {key: truncate_value(value) for key, value in event_dict.items()}


def setup_logging(config: LogConfig) -> None:
    """Setup the logging system with the given configuration.

    This function initializes structlog with appropriate processors and handlers
    based on the provided configuration.

    Args:
        config: LogConfig instance with logging configuration

    Example:
        ```python
        config = LogConfig(
            level="INFO",
            format="json",
            file_path="logs/app.log",
            environment="production",
            app_version="2.0.0"
        )
        setup_logging(config)
        ```
    """
    # Store app info in processor functions
    add_app_info.environment = config.environment
    add_app_info.version = config.app_version
    truncate_strings.max_length = config.max_string_length

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout if config.console_output else None,
        level=getattr(logging, config.level.upper()),
    )

    # Build processor chain
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        add_log_level_name,
        add_app_info,
        add_trade_context,
        filter_sensitive,
        truncate_strings,
    ]

    # Add timestamp if requested
    if config.include_timestamp:
        processors.append(structlog.processors.TimeStamper(fmt="iso"))

    # Add caller info if requested
    if config.include_caller_info:
        processors.append(
            structlog.processors.CallsiteParameterAdder(
                [
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                ]
            )
        )

    # Add stack info on exceptions
    processors.append(structlog.processors.StackInfoRenderer())
    processors.append(structlog.processors.format_exc_info)

    # Choose renderer based on format
    if config.format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Setup file handler if path provided
    if config.file_path:
        file_path = Path(config.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(getattr(logging, config.level.upper()))

        # Use JSON format for file output regardless of console format
        file_processors: list[Processor] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            add_log_level_name,
            add_app_info,
            add_trade_context,
            filter_sensitive,
            truncate_strings,
        ]

        if config.include_timestamp:
            file_processors.append(structlog.processors.TimeStamper(fmt="iso"))

        if config.include_caller_info:
            file_processors.append(
                structlog.processors.CallsiteParameterAdder(
                    [
                        structlog.processors.CallsiteParameter.FILENAME,
                        structlog.processors.CallsiteParameter.LINENO,
                        structlog.processors.CallsiteParameter.FUNC_NAME,
                    ]
                )
            )

        file_processors.extend(
            [
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ]
        )

        # Get root logger and add file handler
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance for the given name.

    Args:
        name: Logger name, typically __name__ of the calling module

    Returns:
        Configured structlog BoundLogger instance

    Example:
        ```python
        logger = get_logger(__name__)
        logger.info("application_started")
        ```
    """
    return structlog.get_logger(name)


@contextmanager
def add_context(**kwargs: Any):
    """Context manager to add contextual information to all log entries.

    Any key-value pairs provided will be automatically added to all log entries
    made within the context.

    Args:
        **kwargs: Key-value pairs to add as context

    Yields:
        None

    Example:
        ```python
        logger = get_logger(__name__)

        with add_context(trade_id="abc123", symbol="BTCUSDT"):
            logger.info("processing_trade")  # Will include trade_id and symbol
            logger.info("order_placed", order_id="xyz789")  # Will also include context

        logger.info("outside_context")  # Will not include trade_id or symbol
        ```
    """
    # Store previous context
    previous_context = _context_vars.copy()

    # Add new context
    _context_vars.update(kwargs)

    try:
        yield
    finally:
        # Restore previous context
        _context_vars.clear()
        _context_vars.update(previous_context)


def set_log_level(level: str) -> None:
    """Change the logging level at runtime.

    Args:
        level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Example:
        ```python
        set_log_level("DEBUG")  # Enable debug logging
        ```
    """
    logging.getLogger().setLevel(getattr(logging, level.upper()))


def clear_context() -> None:
    """Clear all contextual variables.

    Useful for cleanup or when starting a new operation that should not
    inherit previous context.

    Example:
        ```python
        clear_context()  # Remove all context
        ```
    """
    _context_vars.clear()


# Pre-configured logger for this module
logger = get_logger(__name__)
