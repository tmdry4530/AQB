"""
XGBoost Validation Model for IFTB Trading Bot.

This module implements a machine learning-based validation layer using XGBoost
for final trade decision validation. It combines technical indicators, LLM sentiment,
and market context to make probabilistic predictions with confidence intervals.

Key Features:
- Multi-class classification (LONG, SHORT, HOLD)
- Feature engineering from multiple data sources
- Model versioning and persistence
- Cross-validation and statistical significance testing
- Performance degradation detection
- Confidence interval calculation

The model serves as a final validation layer after technical and LLM analysis,
providing statistical backing for trade decisions.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from iftb.config import get_settings
from iftb.config.constants import (
    CONFIDENCE_LEVEL,
    MIN_SAMPLE_SIZE,
    TARGET_WIN_RATE,
)
from iftb.utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ModelPrediction:
    """
    Prediction output from XGBoost model.

    Contains the predicted action, confidence scores for each class,
    feature importance for interpretability, and metadata.
    """
    action: Literal["LONG", "SHORT", "HOLD"]
    confidence: float  # 0-1, max probability among classes
    probability_long: float
    probability_short: float
    probability_hold: float
    feature_importance: dict[str, float]
    model_version: str
    prediction_time: datetime

    def to_dict(self) -> dict:
        """Convert prediction to dictionary for logging/storage."""
        data = asdict(self)
        data["prediction_time"] = self.prediction_time.isoformat()
        return data


@dataclass
class ModelMetrics:
    """
    Comprehensive model performance metrics.

    Includes standard classification metrics, trading-specific metrics,
    and statistical confidence intervals.
    """
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_contribution: float  # Estimated Sharpe ratio contribution
    win_rate: float  # Percentage of profitable trades
    sample_size: int
    confidence_interval_95: tuple[float, float]  # C1: statistical validation
    confusion_matrix: list[list[int]] = field(default_factory=list)
    class_report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return asdict(self)

    def is_statistically_significant(self) -> bool:
        """
        Check if win rate is statistically different from chance (33.3%).

        Uses chi-square test for statistical significance.
        """
        if self.sample_size < MIN_SAMPLE_SIZE:
            return False

        # Chi-square test: observed vs expected (chance = 1/3)
        wins = int(self.win_rate * self.sample_size)
        losses = self.sample_size - wins
        expected_wins = self.sample_size / 3  # Chance for 3-class

        chi2_stat = ((wins - expected_wins) ** 2) / expected_wins
        chi2_critical = stats.chi2.ppf(CONFIDENCE_LEVEL, df=1)

        return chi2_stat > chi2_critical


@dataclass
class TrainingConfig:
    """XGBoost hyperparameter configuration."""

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    min_child_weight: int = 3
    gamma: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization
    scale_pos_weight: float = 1.0
    early_stopping_rounds: int = 50
    eval_metric: str = "mlogloss"  # Multi-class log loss
    objective: str = "multi:softprob"  # Multi-class probabilities
    num_class: int = 3  # LONG, SHORT, HOLD
    random_state: int = 42


# =============================================================================
# Feature Engineering
# =============================================================================


class FeatureEngineer:
    """
    Creates feature vectors for ML model from multiple data sources.

    Combines:
    - Technical indicators (14 indicators)
    - LLM sentiment and confidence
    - Market context (Fear & Greed, funding, OI)
    - Derived features (momentum, volatility, ratios)
    """

    def __init__(self):
        self.logger = get_logger(f"{__name__}.FeatureEngineer")
        self.feature_names: list[str] = []

    def create_features(
        self,
        indicators: dict[str, dict],
        llm_analysis: dict,
        market_context: dict,
        price_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Create feature DataFrame from all available data sources.

        Args:
            indicators: Dict of indicator results {name: {value, signal, etc}}
            llm_analysis: Dict with sentiment, confidence, reasoning
            market_context: Dict with fear_greed_index, funding_rate, open_interest
            price_data: Optional DataFrame with OHLCV data for derived features

        Returns:
            DataFrame with engineered features
        """
        features = {}

        # Technical Indicator Features
        features.update(self._extract_indicator_features(indicators))

        # LLM Features
        features.update(self._extract_llm_features(llm_analysis))

        # Market Context Features
        features.update(self._extract_market_features(market_context))

        # Derived Features (if price data available)
        if price_data is not None:
            features.update(self._create_derived_features(price_data))

        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Store feature names for importance tracking
        self.feature_names = list(df.columns)

        return df

    def _extract_indicator_features(self, indicators: dict[str, dict]) -> dict:
        """Extract features from technical indicators."""
        features = {}

        # Core indicator values
        indicator_map = {
            "rsi": "rsi_value",
            "macd": "macd_value",
            "macd_signal": "macd_signal_value",
            "bb_upper": "bb_upper",
            "bb_lower": "bb_lower",
            "bb_middle": "bb_middle",
            "atr": "atr_value",
            "adx": "adx_value",
            "cci": "cci_value",
            "mfi": "mfi_value",
            "stoch_k": "stoch_k",
            "stoch_d": "stoch_d",
            "williams_r": "williams_r",
            "obv": "obv_value",
        }

        for indicator_name, feature_name in indicator_map.items():
            if indicator_name in indicators:
                value = indicators[indicator_name].get("value", 0)
                features[feature_name] = float(value) if value is not None else 0.0
            else:
                features[feature_name] = 0.0

        # Signal strengths (numeric encoding)
        signal_map = {"BUY": 1.0, "SELL": -1.0, "NEUTRAL": 0.0}

        for indicator_name, indicator_data in indicators.items():
            signal = indicator_data.get("signal", "NEUTRAL")
            features[f"{indicator_name}_signal"] = signal_map.get(signal, 0.0)

        # Derived indicator features
        if "bb_upper" in features and "bb_lower" in features and "bb_middle" in features:
            bb_range = features["bb_upper"] - features["bb_lower"]
            features["bb_width"] = bb_range
            features["bb_percent"] = (
                (features["bb_middle"] - features["bb_lower"]) / bb_range
                if bb_range > 0 else 0.5
            )

        if "macd_value" in features and "macd_signal_value" in features:
            features["macd_histogram"] = (
                features["macd_value"] - features["macd_signal_value"]
            )

        return features

    def _extract_llm_features(self, llm_analysis: dict) -> dict:
        """Extract features from LLM analysis."""
        features = {
            "llm_sentiment": float(llm_analysis.get("sentiment", 0.0)),
            "llm_confidence": float(llm_analysis.get("confidence", 0.5)),
            "llm_sentiment_confidence_product": (
                float(llm_analysis.get("sentiment", 0.0)) *
                float(llm_analysis.get("confidence", 0.5))
            ),
        }

        # LLM action encoding
        action_map = {"LONG": 1.0, "SHORT": -1.0, "HOLD": 0.0}
        llm_action = llm_analysis.get("action", "HOLD")
        features["llm_action_encoded"] = action_map.get(llm_action, 0.0)

        return features

    def _extract_market_features(self, market_context: dict) -> dict:
        """Extract features from market context."""
        features = {
            "fear_greed_index": float(market_context.get("fear_greed_index", 50.0)),
            "funding_rate": float(market_context.get("funding_rate", 0.0)),
            "open_interest_change": float(
                market_context.get("open_interest_change_pct", 0.0)
            ),
        }

        # Normalized fear & greed (0-100 to -1 to 1)
        features["fear_greed_normalized"] = (features["fear_greed_index"] - 50.0) / 50.0

        # Funding rate extremes
        features["funding_rate_abs"] = abs(features["funding_rate"])

        return features

    def _create_derived_features(self, price_data: pd.DataFrame) -> dict:
        """
        Create derived features from price data.

        Includes momentum, volatility, and volume features across multiple timeframes.
        """
        features = {}

        if len(price_data) < 20:
            # Not enough data for derived features
            return features

        # Price momentum (multiple timeframes)
        features["momentum_5"] = self._calculate_momentum(price_data, 5)
        features["momentum_10"] = self._calculate_momentum(price_data, 10)
        features["momentum_20"] = self._calculate_momentum(price_data, 20)

        # Volume analysis
        if "volume" in price_data.columns:
            recent_volume = price_data["volume"].iloc[-5:].mean()
            historical_volume = price_data["volume"].iloc[-20:].mean()
            features["volume_ratio"] = (
                recent_volume / historical_volume
                if historical_volume > 0 else 1.0
            )

        # Volatility (using close prices)
        if "close" in price_data.columns:
            returns = price_data["close"].pct_change().dropna()
            features["volatility_20"] = returns.iloc[-20:].std() if len(returns) >= 20 else 0.0
            features["volatility_5"] = returns.iloc[-5:].std() if len(returns) >= 5 else 0.0

        return features

    @staticmethod
    def _calculate_momentum(price_data: pd.DataFrame, periods: int) -> float:
        """Calculate price momentum over specified periods."""
        if len(price_data) < periods + 1:
            return 0.0

        if "close" not in price_data.columns:
            return 0.0

        current_price = price_data["close"].iloc[-1]
        past_price = price_data["close"].iloc[-(periods + 1)]

        if past_price == 0:
            return 0.0

        return (current_price - past_price) / past_price


# =============================================================================
# XGBoost Model
# =============================================================================


class XGBoostValidator:
    """
    XGBoost-based trade validation model.

    Multi-class classifier that predicts LONG, SHORT, or HOLD based on
    engineered features from technical indicators, LLM analysis, and market context.

    Features:
    - Model versioning and persistence
    - Feature scaling for numerical stability
    - Cross-validation during training
    - Performance tracking and degradation detection
    - Statistical significance testing
    """

    def __init__(self, model_path: str | None = None):
        """
        Initialize XGBoost validator.

        Args:
            model_path: Path to load existing model from. If None, creates new model.
        """
        self.logger = get_logger(f"{__name__}.XGBoostValidator")
        self.settings = get_settings()

        self.config = TrainingConfig()
        self.model: XGBClassifier | None = None
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()

        self.model_version: str = "v1.0.0"
        self.trained_at: datetime | None = None
        self.feature_names: list[str] = []
        self.class_names = ["LONG", "SHORT", "HOLD"]

        # Performance tracking
        self.training_metrics: ModelMetrics | None = None
        self.validation_metrics: ModelMetrics | None = None

        # Load existing model if path provided
        if model_path:
            self.load_model(model_path)
        else:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize a new XGBoost model with configured hyperparameters."""
        self.model = XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_child_weight=self.config.min_child_weight,
            gamma=self.config.gamma,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            scale_pos_weight=self.config.scale_pos_weight,
            objective=self.config.objective,
            num_class=self.config.num_class,
            eval_metric=self.config.eval_metric,
            random_state=self.config.random_state,
            verbosity=0,
        )

        self.logger.info("Initialized new XGBoost model")

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        perform_cv: bool = True,
    ):
        """
        Train the XGBoost model with cross-validation.

        Args:
            X: Feature DataFrame
            y: Target labels (0=LONG, 1=SHORT, 2=HOLD)
            validation_split: Fraction of data for validation
            perform_cv: Whether to perform cross-validation
        """
        self.logger.info(f"Training XGBoost model on {len(X)} samples")

        # Store feature names
        self.feature_names = list(X.columns)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=self.config.random_state,
            stratify=y if len(y) > 100 else None  # Stratify if enough samples
        )

        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train model with early stopping
        eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]

        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        self.trained_at = datetime.now()

        # Calculate metrics
        self.training_metrics = self._calculate_metrics(X_train_scaled, y_train)
        self.validation_metrics = self._calculate_metrics(X_val_scaled, y_val)

        self.logger.info(
            f"Training complete. "
            f"Train accuracy: {self.training_metrics.accuracy:.3f}, "
            f"Val accuracy: {self.validation_metrics.accuracy:.3f}"
        )

        # Cross-validation
        if perform_cv and len(X) >= 100:
            self._perform_cross_validation(X, y)

    def _calculate_metrics(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Calculate comprehensive model metrics."""
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)

        # Standard classification metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y, y_pred, average="weighted", zero_division=0)

        # Trading-specific metrics
        win_rate = accuracy  # For classification, accuracy is win rate

        # Confidence interval for win rate (95%)
        n = len(y)
        z_score = stats.norm.ppf((1 + CONFIDENCE_LEVEL) / 2)
        margin = z_score * np.sqrt((win_rate * (1 - win_rate)) / n)
        ci_lower = max(0.0, win_rate - margin)
        ci_upper = min(1.0, win_rate + margin)

        # Sharpe contribution (estimated from prediction confidence)
        # Higher confidence predictions should contribute to better risk-adjusted returns
        max_probs = np.max(y_prob, axis=1)
        sharpe_contribution = np.mean(max_probs) * win_rate

        # Confusion matrix and classification report
        cm = confusion_matrix(y, y_pred).tolist()
        cr = classification_report(
            y, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )

        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            sharpe_contribution=sharpe_contribution,
            win_rate=win_rate,
            sample_size=n,
            confidence_interval_95=(ci_lower, ci_upper),
            confusion_matrix=cm,
            class_report=cr,
        )

    def _perform_cross_validation(self, X: pd.DataFrame, y: pd.Series):
        """Perform k-fold cross-validation."""
        self.logger.info("Performing 5-fold cross-validation")

        X_scaled = self.scaler.transform(X)

        cv_scores = cross_val_score(
            self.model, X_scaled, y, cv=5, scoring="accuracy"
        )

        self.logger.info(
            f"CV Scores: {cv_scores}, "
            f"Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})"
        )

    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """
        Make prediction with confidence scores.

        Args:
            features: Feature DataFrame (single row or multiple rows)

        Returns:
            ModelPrediction with action and confidence scores
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        # Ensure features match training
        features_aligned = features[self.feature_names]

        # Scale features
        X_scaled = self.scaler.transform(features_aligned)

        # Get predictions and probabilities
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)

        # Extract probabilities for first sample (if batch, take first)
        probs = y_prob[0] if len(y_prob.shape) > 1 else y_prob
        prob_long, prob_short, prob_hold = probs

        # Determine action and confidence
        action_idx = y_pred[0] if isinstance(y_pred, np.ndarray) else y_pred
        action = self.class_names[action_idx]
        confidence = float(np.max(probs))

        # Feature importance
        feature_importance = self._get_feature_importance()

        return ModelPrediction(
            action=action,
            confidence=confidence,
            probability_long=float(prob_long),
            probability_short=float(prob_short),
            probability_hold=float(prob_hold),
            feature_importance=feature_importance,
            model_version=self.model_version,
            prediction_time=datetime.now(),
        )

    def _get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores from trained model."""
        if self.model is None or not self.feature_names:
            return {}

        importance_scores = self.model.feature_importances_

        # Create dict of feature name -> importance
        importance_dict = {
            name: float(score)
            for name, score in zip(self.feature_names, importance_scores)
        }

        # Sort by importance
        return dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

    def save_model(self, path: str, version: str | None = None):
        """
        Save model, scaler, and metadata to disk.

        Args:
            path: Base path for saving (without extension)
            version: Optional version string to update
        """
        if self.model is None:
            raise RuntimeError("No model to save")

        if version:
            self.model_version = version

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save model components
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "config": asdict(self.config),
            "version": self.model_version,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "training_metrics": self.training_metrics.to_dict() if self.training_metrics else None,
            "validation_metrics": self.validation_metrics.to_dict() if self.validation_metrics else None,
        }

        joblib.dump(model_data, f"{path}.pkl")

        # Save metadata separately for easy inspection
        metadata = {
            "version": self.model_version,
            "trained_at": model_data["trained_at"],
            "training_metrics": model_data["training_metrics"],
            "validation_metrics": model_data["validation_metrics"],
            "feature_names": self.feature_names,
            "num_features": len(self.feature_names),
        }

        with open(f"{path}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Model saved to {path}.pkl (version: {self.model_version})")

    def load_model(self, path: str):
        """
        Load model, scaler, and metadata from disk.

        Args:
            path: Path to saved model (with or without .pkl extension)
        """
        if not path.endswith(".pkl"):
            path = f"{path}.pkl"

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load model data
        model_data = joblib.load(path)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.class_names = model_data["class_names"]
        self.model_version = model_data["version"]

        trained_at_str = model_data.get("trained_at")
        self.trained_at = (
            datetime.fromisoformat(trained_at_str) if trained_at_str else None
        )

        # Restore config
        config_dict = model_data.get("config", {})
        self.config = TrainingConfig(**config_dict)

        # Restore metrics
        training_metrics_dict = model_data.get("training_metrics")
        if training_metrics_dict:
            self.training_metrics = ModelMetrics(**training_metrics_dict)

        validation_metrics_dict = model_data.get("validation_metrics")
        if validation_metrics_dict:
            self.validation_metrics = ModelMetrics(**validation_metrics_dict)

        self.logger.info(
            f"Model loaded from {path} (version: {self.model_version}, "
            f"trained: {self.trained_at})"
        )

    def get_metrics(self) -> ModelMetrics | None:
        """Get validation metrics from last training."""
        return self.validation_metrics

    def should_retrain(self, recent_performance: ModelMetrics) -> bool:
        """
        Determine if model should be retrained based on performance degradation.

        Args:
            recent_performance: Metrics from recent predictions

        Returns:
            True if retraining is recommended
        """
        if self.validation_metrics is None:
            self.logger.warning("No validation metrics available for comparison")
            return True

        # Check if recent performance is significantly worse
        val_acc = self.validation_metrics.accuracy
        recent_acc = recent_performance.accuracy

        # Degradation threshold: 10% drop in accuracy
        degradation_threshold = 0.10

        if recent_acc < (val_acc - degradation_threshold):
            self.logger.warning(
                f"Performance degradation detected: "
                f"Val accuracy {val_acc:.3f} -> Recent accuracy {recent_acc:.3f}"
            )
            return True

        # Check if win rate has dropped below target
        if recent_performance.win_rate < TARGET_WIN_RATE * 0.8:  # 80% of target
            self.logger.warning(
                f"Win rate below threshold: {recent_performance.win_rate:.3f} "
                f"(target: {TARGET_WIN_RATE:.3f})"
            )
            return True

        # Check statistical significance
        if not recent_performance.is_statistically_significant():
            if recent_performance.sample_size >= MIN_SAMPLE_SIZE:
                self.logger.warning(
                    "Recent performance not statistically significant "
                    f"despite {recent_performance.sample_size} samples"
                )
                return True

        return False

    def update_hyperparameters(self, **kwargs):
        """
        Update model hyperparameters.

        Requires retraining after update.
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated {key} = {value}")
            else:
                self.logger.warning(f"Unknown hyperparameter: {key}")

        # Reinitialize model with new config
        self._initialize_model()


# =============================================================================
# Utility Functions
# =============================================================================


def encode_action_label(action: str) -> int:
    """
    Encode action string to numeric label for training.

    Args:
        action: "LONG", "SHORT", or "HOLD"

    Returns:
        Numeric label (0, 1, or 2)
    """
    action_map = {"LONG": 0, "SHORT": 1, "HOLD": 2}
    return action_map.get(action.upper(), 2)


def decode_action_label(label: int) -> str:
    """
    Decode numeric label back to action string.

    Args:
        label: 0, 1, or 2

    Returns:
        Action string
    """
    label_map = {0: "LONG", 1: "SHORT", 2: "HOLD"}
    return label_map.get(label, "HOLD")


def calculate_ensemble_confidence(
    ml_prediction: ModelPrediction,
    llm_confidence: float,
    technical_agreement: float,
) -> float:
    """
    Calculate ensemble confidence from multiple validation layers.

    Combines ML model confidence, LLM confidence, and technical indicator agreement.

    Args:
        ml_prediction: Prediction from ML model
        llm_confidence: Confidence from LLM analysis (0-1)
        technical_agreement: Technical indicator agreement score (0-1)

    Returns:
        Ensemble confidence score (0-1)
    """
    # Weighted average of confidence sources
    weights = {
        "ml": 0.4,
        "llm": 0.35,
        "technical": 0.25,
    }

    ensemble_confidence = (
        weights["ml"] * ml_prediction.confidence +
        weights["llm"] * llm_confidence +
        weights["technical"] * technical_agreement
    )

    return ensemble_confidence
