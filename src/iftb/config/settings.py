"""
Configuration settings for the IFTB trading bot.

Uses pydantic-settings for environment variable management with nested models
for different configuration domains.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="iftb", alias="database", description="Database name")
    user: str = Field(default="postgres", alias="username", description="Database username")
    password: SecretStr = Field(default=SecretStr(""), description="Database password")
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Maximum connection overflow")

    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_async_url(self) -> str:
        """
        Generate async database connection URL.

        Returns:
            Async PostgreSQL connection string
        """
        password = self.password.get_secret_value()
        return (
            f"postgresql+asyncpg://{self.user}:{password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class RedisSettings(BaseSettings):
    """Redis configuration settings."""

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    password: Optional[SecretStr] = Field(default=None, description="Redis password")
    db: int = Field(default=0, description="Redis database number")

    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_url(self) -> str:
        """
        Generate Redis connection URL.

        Returns:
            Redis connection string
        """
        if self.password:
            password = self.password.get_secret_value()
            return f"redis://:{password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class ExchangeSettings(BaseSettings):
    """Exchange API configuration settings."""

    api_key: SecretStr = Field(default=SecretStr(""), description="Exchange API key")
    api_secret: SecretStr = Field(default=SecretStr(""), description="Exchange API secret")
    testnet: bool = Field(default=True, description="Use testnet environment")
    rate_limit_per_second: int = Field(
        default=10, description="API rate limit per second"
    )

    model_config = SettingsConfigDict(
        env_prefix="EXCHANGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class TelegramSettings(BaseSettings):
    """Telegram configuration settings."""

    api_id: int = Field(default=0, description="Telegram API ID")
    api_hash: SecretStr = Field(default=SecretStr(""), description="Telegram API hash")
    bot_token: SecretStr = Field(default=SecretStr(""), description="Telegram bot token")
    alert_chat_id: int = Field(default=0, description="Chat ID for alerts")
    news_channel_ids_raw: str = Field(
        default="", alias="news_channel_ids",
        description="News channel IDs to monitor (comma-separated)"
    )

    @property
    def news_channel_ids(self) -> list[int]:
        """Parse comma-separated string to list of integers."""
        if not self.news_channel_ids_raw.strip():
            return []
        return [int(x.strip()) for x in self.news_channel_ids_raw.split(",")]

    model_config = SettingsConfigDict(
        env_prefix="TELEGRAM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


class LLMSettings(BaseSettings):
    """LLM configuration settings."""

    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""), description="Anthropic API key"
    )
    model: str = Field(
        default="claude-sonnet-4-20250514", description="Claude model to use"
    )
    max_tokens: int = Field(default=1000, description="Maximum tokens per request")
    cache_ttl_seconds: int = Field(
        default=300, description="Cache TTL in seconds for LLM responses"
    )

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class TradingSettings(BaseSettings):
    """Trading configuration settings."""

    symbols_raw: str = Field(
        default="BTCUSDT", alias="symbols",
        description="Trading symbols to monitor (comma-separated)",
    )
    timeframes_raw: str = Field(
        default="1h,4h,1d", alias="timeframes",
        description="Timeframes for analysis (comma-separated)",
    )
    default_leverage: int = Field(default=1, description="Default leverage")
    paper_trading: bool = Field(
        default=True, description="Enable paper trading mode"
    )
    use_websocket: bool = Field(
        default=False, description="Use WebSocket for real-time data"
    )
    initial_balance: float = Field(
        default=10000.0, description="Initial balance for paper trading"
    )

    @property
    def symbols(self) -> list[str]:
        """Parse comma-separated string to list of strings."""
        return [x.strip() for x in self.symbols_raw.split(",") if x.strip()]

    @property
    def timeframes(self) -> list[str]:
        """Parse comma-separated string to list of strings."""
        return [x.strip() for x in self.timeframes_raw.split(",") if x.strip()]

    model_config = SettingsConfigDict(
        env_prefix="TRADING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )
    file_path: Optional[str] = Field(
        default=None, description="Optional log file path"
    )

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class Settings(BaseSettings):
    """
    Main settings class combining all configuration domains.

    Loads configuration from environment variables and .env file.
    Uses nested models for organized configuration management.
    """

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance (singleton pattern).

    Returns:
        Singleton Settings instance
    """
    return Settings()
