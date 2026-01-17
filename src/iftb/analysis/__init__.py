"""
Analysis module for IFTB trading bot.

Provides technical analysis, LLM-powered market analysis, sentiment assessment,
ML validation models, and veto systems for intelligent trade decision making.
"""

from .indicators import (
    CompositeSignal,
    IndicatorResult,
    TechnicalAnalyzer,
)
from .llm_analyzer import (
    FallbackMode,
    LLMAnalysis,
    LLMAnalyzer,
    LLMVetoSystem,
    SentimentScore,
    create_analyzer_from_settings,
)
from .ml_model import (
    FeatureEngineer,
    ModelMetrics,
    ModelPrediction,
    TrainingConfig,
    XGBoostValidator,
    calculate_ensemble_confidence,
    decode_action_label,
    encode_action_label,
)

__all__ = [
    # Technical Indicators
    "TechnicalAnalyzer",
    "IndicatorResult",
    "CompositeSignal",
    # LLM Analyzer
    "LLMAnalyzer",
    "LLMVetoSystem",
    "LLMAnalysis",
    "SentimentScore",
    "FallbackMode",
    "create_analyzer_from_settings",
    # ML Model
    "XGBoostValidator",
    "FeatureEngineer",
    "ModelPrediction",
    "ModelMetrics",
    "TrainingConfig",
    "encode_action_label",
    "decode_action_label",
    "calculate_ensemble_confidence",
]
