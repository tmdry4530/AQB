"""
Configuration module for IFTB trading bot.

Exports the main Settings class and get_settings function for application-wide
configuration management.
"""

from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
