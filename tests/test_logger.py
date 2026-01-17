"""
Tests for the logging system.

This module tests the logger functionality including configuration,
context management, sensitive data filtering, and output formatting.
"""

import json
import logging
import tempfile
from pathlib import Path

import pytest
import structlog

from iftb.utils import (
    LogConfig,
    add_context,
    clear_context,
    get_logger,
    set_log_level,
    setup_logging,
)


@pytest.fixture
def temp_log_file():
    """Create a temporary log file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
        log_path = f.name
    yield log_path
    # Cleanup
    Path(log_path).unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before each test."""
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Clear structlog configuration
    structlog.reset_defaults()

    yield

    # Cleanup after test
    clear_context()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)


class TestLogConfig:
    """Tests for LogConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LogConfig()
        assert config.level == "INFO"
        assert config.format == "pretty"
        assert config.file_path is None
        assert config.include_timestamp is True
        assert config.include_caller_info is True
        assert config.console_output is True
        assert config.environment == "dev"
        assert config.app_version == "1.0.0"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LogConfig(
            level="DEBUG",
            format="json",
            file_path="/tmp/test.log",
            include_timestamp=False,
            include_caller_info=False,
            console_output=False,
            environment="prod",
            app_version="2.0.0",
        )
        assert config.level == "DEBUG"
        assert config.format == "json"
        assert config.file_path == "/tmp/test.log"
        assert config.include_timestamp is False
        assert config.include_caller_info is False
        assert config.console_output is False
        assert config.environment == "prod"
        assert config.app_version == "2.0.0"


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_pretty_logging(self):
        """Test setup with pretty format."""
        config = LogConfig(level="INFO", format="pretty")
        setup_logging(config)

        logger = get_logger(__name__)
        assert logger is not None
        assert isinstance(logger, structlog.BoundLogger)

    def test_setup_json_logging(self):
        """Test setup with JSON format."""
        config = LogConfig(level="DEBUG", format="json")
        setup_logging(config)

        logger = get_logger(__name__)
        assert logger is not None

    def test_setup_with_file_output(self, temp_log_file):
        """Test setup with file output."""
        config = LogConfig(level="INFO", format="json", file_path=temp_log_file)
        setup_logging(config)

        logger = get_logger(__name__)
        logger.info("test_message", key="value")

        # Verify file was created and contains logs
        log_path = Path(temp_log_file)
        assert log_path.exists()
        assert log_path.stat().st_size > 0

    def test_setup_with_different_levels(self):
        """Test setup with different log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = LogConfig(level=level)
            setup_logging(config)
            assert logging.getLogger().level == getattr(logging, level)

    def test_file_directory_creation(self, tmp_path):
        """Test that log file directory is created if it doesn't exist."""
        log_file = tmp_path / "logs" / "subdir" / "test.log"
        config = LogConfig(file_path=str(log_file))
        setup_logging(config)

        logger = get_logger(__name__)
        logger.info("test")

        assert log_file.parent.exists()
        assert log_file.exists()


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_bound_logger(self):
        """Test that get_logger returns a BoundLogger."""
        setup_logging(LogConfig())
        logger = get_logger(__name__)
        assert isinstance(logger, structlog.BoundLogger)

    def test_get_logger_with_different_names(self):
        """Test getting loggers with different names."""
        setup_logging(LogConfig())
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1 is not None
        assert logger2 is not None
        # Both should work independently
        logger1.info("test1")
        logger2.info("test2")


class TestContextManagement:
    """Tests for context management functionality."""

    def test_add_context_basic(self, temp_log_file):
        """Test basic context addition."""
        config = LogConfig(format="json", file_path=temp_log_file)
        setup_logging(config)
        logger = get_logger(__name__)

        with add_context(trade_id="abc123"):
            logger.info("test_message")

        # Read log file and verify context
        log_content = Path(temp_log_file).read_text()
        log_entry = json.loads(log_content.strip())
        assert log_entry["trade_id"] == "abc123"

    def test_add_context_multiple_values(self, temp_log_file):
        """Test adding multiple context values."""
        config = LogConfig(format="json", file_path=temp_log_file)
        setup_logging(config)
        logger = get_logger(__name__)

        with add_context(trade_id="abc123", symbol="BTCUSDT", side="LONG"):
            logger.info("test_message")

        log_content = Path(temp_log_file).read_text()
        log_entry = json.loads(log_content.strip())
        assert log_entry["trade_id"] == "abc123"
        assert log_entry["symbol"] == "BTCUSDT"
        assert log_entry["side"] == "LONG"

    def test_add_context_nested(self, temp_log_file):
        """Test nested context management."""
        config = LogConfig(format="json", file_path=temp_log_file)
        setup_logging(config)
        logger = get_logger(__name__)

        with add_context(outer="value1"):
            with add_context(inner="value2"):
                logger.info("nested_message")

        log_content = Path(temp_log_file).read_text()
        log_entry = json.loads(log_content.strip())
        assert log_entry["outer"] == "value1"
        assert log_entry["inner"] == "value2"

    def test_context_cleanup(self, temp_log_file):
        """Test that context is cleaned up after exiting the block."""
        config = LogConfig(format="json", file_path=temp_log_file)
        setup_logging(config)
        logger = get_logger(__name__)

        with add_context(temp_context="value"):
            pass

        # Log after context block
        logger.info("after_context")

        log_content = Path(temp_log_file).read_text()
        log_entry = json.loads(log_content.strip())
        assert "temp_context" not in log_entry

    def test_clear_context(self, temp_log_file):
        """Test clearing all context."""
        config = LogConfig(format="json", file_path=temp_log_file)
        setup_logging(config)
        logger = get_logger(__name__)

        with add_context(key1="value1", key2="value2"):
            clear_context()
            logger.info("after_clear")

        log_content = Path(temp_log_file).read_text()
        log_entry = json.loads(log_content.strip())
        assert "key1" not in log_entry
        assert "key2" not in log_entry


class TestSensitiveDataFiltering:
    """Tests for sensitive data filtering."""

    def test_filter_api_key(self, temp_log_file):
        """Test that API keys are masked."""
        config = LogConfig(format="json", file_path=temp_log_file)
        setup_logging(config)
        logger = get_logger(__name__)

        logger.info("test", api_key="my_secret_key_12345")

        log_content = Path(temp_log_file).read_text()
        log_entry = json.loads(log_content.strip())
        assert "my_secret_key_12345" not in log_content
        assert "***" in log_entry["api_key"]

    def test_filter_password(self, temp_log_file):
        """Test that passwords are masked."""
        config = LogConfig(format="json", file_path=temp_log_file)
        setup_logging(config)
        logger = get_logger(__name__)

        logger.info("test", password="super_secret_password")

        log_content = Path(temp_log_file).read_text()
        assert "super_secret_password" not in log_content

    def test_filter_multiple_sensitive_fields(self, temp_log_file):
        """Test that multiple sensitive fields are masked."""
        config = LogConfig(format="json", file_path=temp_log_file)
        setup_logging(config)
        logger = get_logger(__name__)

        logger.info(
            "test",
            api_key="key123",
            api_secret="secret456",
            password="pass789",
            token="token012",
        )

        log_content = Path(temp_log_file).read_text()
        assert "key123" not in log_content
        assert "secret456" not in log_content
        assert "pass789" not in log_content
        assert "token012" not in log_content

    def test_filter_nested_sensitive_data(self, temp_log_file):
        """Test that nested sensitive data is masked."""
        config = LogConfig(format="json", file_path=temp_log_file)
        setup_logging(config)
        logger = get_logger(__name__)

        logger.info(
            "test",
            credentials={"api_key": "secret123", "username": "user"},
        )

        log_content = Path(temp_log_file).read_text()
        assert "secret123" not in log_content


class TestLogLevels:
    """Tests for different log levels."""

    def test_all_log_levels(self, temp_log_file):
        """Test that all log levels work."""
        config = LogConfig(level="DEBUG", format="json", file_path=temp_log_file)
        setup_logging(config)
        logger = get_logger(__name__)

        logger.debug("debug_message")
        logger.info("info_message")
        logger.warning("warning_message")
        logger.error("error_message")
        logger.critical("critical_message")

        log_content = Path(temp_log_file).read_text()
        assert "debug_message" in log_content
        assert "info_message" in log_content
        assert "warning_message" in log_content
        assert "error_message" in log_content
        assert "critical_message" in log_content

    def test_log_level_filtering(self, temp_log_file):
        """Test that log level filtering works."""
        config = LogConfig(level="WARNING", format="json", file_path=temp_log_file)
        setup_logging(config)
        logger = get_logger(__name__)

        logger.debug("debug_message")
        logger.info("info_message")
        logger.warning("warning_message")
        logger.error("error_message")

        log_content = Path(temp_log_file).read_text()
        assert "debug_message" not in log_content
        assert "info_message" not in log_content
        assert "warning_message" in log_content
        assert "error_message" in log_content

    def test_set_log_level(self, temp_log_file):
        """Test changing log level at runtime."""
        config = LogConfig(level="INFO", format="json", file_path=temp_log_file)
        setup_logging(config)
        logger = get_logger(__name__)

        logger.debug("debug1")
        set_log_level("DEBUG")
        logger.debug("debug2")

        log_content = Path(temp_log_file).read_text()
        assert "debug1" not in log_content
        assert "debug2" in log_content


class TestStructuredLogging:
    """Tests for structured logging features."""

    def test_structured_fields(self, temp_log_file):
        """Test that structured fields are properly logged."""
        config = LogConfig(format="json", file_path=temp_log_file)
        setup_logging(config)
        logger = get_logger(__name__)

        logger.info(
            "trade_opened",
            symbol="BTCUSDT",
            side="LONG",
            price=50000.0,
            quantity=0.5,
        )

        log_content = Path(temp_log_file).read_text()
        log_entry = json.loads(log_content.strip())
        assert log_entry["event"] == "trade_opened"
        assert log_entry["symbol"] == "BTCUSDT"
        assert log_entry["side"] == "LONG"
        assert log_entry["price"] == 50000.0
        assert log_entry["quantity"] == 0.5

    def test_app_info_added(self, temp_log_file):
        """Test that application info is added to logs."""
        config = LogConfig(
            format="json",
            file_path=temp_log_file,
            environment="test",
            app_version="1.2.3",
        )
        setup_logging(config)
        logger = get_logger(__name__)

        logger.info("test_message")

        log_content = Path(temp_log_file).read_text()
        log_entry = json.loads(log_content.strip())
        assert log_entry["app"] == "iftb"
        assert log_entry["environment"] == "test"
        assert log_entry["version"] == "1.2.3"

    def test_timestamp_included(self, temp_log_file):
        """Test that timestamp is included when configured."""
        config = LogConfig(
            format="json", file_path=temp_log_file, include_timestamp=True
        )
        setup_logging(config)
        logger = get_logger(__name__)

        logger.info("test_message")

        log_content = Path(temp_log_file).read_text()
        log_entry = json.loads(log_content.strip())
        assert "timestamp" in log_entry

    def test_caller_info_included(self, temp_log_file):
        """Test that caller info is included when configured."""
        config = LogConfig(
            format="json", file_path=temp_log_file, include_caller_info=True
        )
        setup_logging(config)
        logger = get_logger(__name__)

        logger.info("test_message")

        log_content = Path(temp_log_file).read_text()
        log_entry = json.loads(log_content.strip())
        assert "filename" in log_entry
        assert "lineno" in log_entry
        assert "func_name" in log_entry


class TestErrorLogging:
    """Tests for error logging with exceptions."""

    def test_exception_logging(self, temp_log_file):
        """Test logging exceptions with exc_info."""
        config = LogConfig(format="json", file_path=temp_log_file)
        setup_logging(config)
        logger = get_logger(__name__)

        try:
            _ = 1 / 0
        except ZeroDivisionError:
            logger.error("calculation_error", exc_info=True)

        log_content = Path(temp_log_file).read_text()
        assert "ZeroDivisionError" in log_content
        assert "division by zero" in log_content


class TestStringTruncation:
    """Tests for string truncation to prevent log bloat."""

    def test_long_string_truncation(self, temp_log_file):
        """Test that long strings are truncated."""
        config = LogConfig(
            format="json", file_path=temp_log_file, max_string_length=100
        )
        setup_logging(config)
        logger = get_logger(__name__)

        long_string = "a" * 200
        logger.info("test", data=long_string)

        log_content = Path(temp_log_file).read_text()
        log_entry = json.loads(log_content.strip())
        assert len(log_entry["data"]) < 200
        assert "truncated" in log_entry["data"]
