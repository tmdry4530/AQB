"""
Trading Decision Engine with Risk Management for IFTB Trading Bot.

This module implements the core decision-making engine that integrates:
1. Technical Analysis (40% weight)
2. LLM Sentiment Analysis (25% weight)
3. XGBoost ML Model (35% weight)

Key Features:
- Kelly Criterion position sizing with conservative 0.25 fraction
- Circuit breaker system for drawdown/volatility protection
- Kill switch for emergency stop
- ATR-based stop-loss and take-profit
- Dynamic leverage adjustment based on volatility
- Statistical validation before execution
- Comprehensive risk checks and veto system

The engine serves as the final decision layer, combining all analysis sources
with strict risk management to generate executable trading decisions.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

from iftb.analysis import CompositeSignal, LLMAnalysis, ModelPrediction
from iftb.config import get_settings
from iftb.config.constants import (
    CONFIDENCE_VETO_THRESHOLD,
    CONSECUTIVE_LOSS_LIMIT,
    DEFAULT_LEVERAGE,
    HIGH_CONFIDENCE_LEVERAGE,
    KELLY_FRACTION,
    MAX_DAILY_LOSS_PCT,
    MAX_DRAWDOWN,
    MAX_LEVERAGE,
    MAX_POSITION_PCT,
    MIN_LEVERAGE,
    MIN_POSITION_PCT,
    SENTIMENT_VETO_THRESHOLD,
)
from iftb.data import MarketContext
from iftb.utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TradingDecision:
    """
    Final trading decision with position sizing and risk parameters.

    Attributes:
        action: Trade direction (LONG, SHORT, or HOLD)
        symbol: Trading pair (e.g., "BTCUSDT")
        confidence: Overall confidence score (0-1)
        position_size: Position size as fraction of capital
        leverage: Leverage multiplier (2-8x)
        stop_loss: Stop-loss price
        take_profit: Take-profit price
        entry_price: Expected entry price
        timestamp: Decision timestamp
        reasons: List of reasons supporting the decision
        vetoed: Whether the decision was vetoed
        veto_reason: Reason for veto if applicable
    """
    action: Literal["LONG", "SHORT", "HOLD"]
    symbol: str
    confidence: float
    position_size: float
    leverage: int
    stop_loss: float
    take_profit: float
    entry_price: float
    timestamp: datetime
    reasons: list[str] = field(default_factory=list)
    vetoed: bool = False
    veto_reason: str | None = None

    def to_dict(self) -> dict:
        """Convert decision to dictionary for logging/storage."""
        return {
            "action": self.action,
            "symbol": self.symbol,
            "confidence": round(self.confidence, 4),
            "position_size": round(self.position_size, 4),
            "leverage": self.leverage,
            "stop_loss": round(self.stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
            "entry_price": round(self.entry_price, 2),
            "timestamp": self.timestamp.isoformat(),
            "reasons": self.reasons,
            "vetoed": self.vetoed,
            "veto_reason": self.veto_reason,
        }

    def __repr__(self) -> str:
        """Human-readable decision summary."""
        if self.vetoed:
            return (
                f"TradingDecision(VETOED: {self.veto_reason})"
            )
        if self.action == "HOLD":
            return (
                f"TradingDecision(HOLD for {self.symbol}, "
                f"confidence={self.confidence:.2f})"
            )
        return (
            f"TradingDecision({self.action} {self.symbol}, "
            f"size={self.position_size:.2%}, leverage={self.leverage}x, "
            f"confidence={self.confidence:.2f}, "
            f"entry={self.entry_price:.2f}, SL={self.stop_loss:.2f}, "
            f"TP={self.take_profit:.2f})"
        )


@dataclass
class TradeHistory:
    """Record of a completed trade."""
    symbol: str
    action: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    position_size: float
    leverage: int
    pnl: float
    pnl_pct: float
    entry_time: datetime
    exit_time: datetime
    win: bool


# =============================================================================
# Risk Manager
# =============================================================================


class RiskManager:
    """
    Risk management system implementing Kelly Criterion and position controls.

    Handles:
    - Kelly Criterion position sizing
    - Daily loss limits
    - Consecutive loss tracking
    - ATR-based stop-loss/take-profit calculation
    - Dynamic leverage adjustment
    """

    def __init__(self):
        """Initialize risk manager with default parameters."""
        self.settings = get_settings()
        logger.info("risk_manager_initialized")

    def calculate_kelly_position(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_capital: float,
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.

        Formula: f = (p * (b + 1) - 1) / b
        Where:
            f = fraction of capital to bet
            p = win probability
            b = ratio of avg_win to avg_loss

        Uses quarter-Kelly (0.25) for conservative sizing.

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount (positive value)
            current_capital: Current account capital

        Returns:
            Position size as fraction of capital (capped at MAX_POSITION_PCT)
        """
        # Validate inputs
        if win_rate <= 0 or win_rate >= 1:
            logger.warning(
                "invalid_win_rate",
                win_rate=win_rate,
                fallback=MIN_POSITION_PCT,
            )
            return MIN_POSITION_PCT

        if avg_loss <= 0:
            logger.warning(
                "invalid_avg_loss",
                avg_loss=avg_loss,
                fallback=MIN_POSITION_PCT,
            )
            return MIN_POSITION_PCT

        # Calculate Kelly fraction
        b = avg_win / avg_loss  # Odds ratio
        kelly_fraction = ((win_rate * (b + 1)) - 1) / b

        # Apply conservative quarter-Kelly
        conservative_fraction = kelly_fraction * KELLY_FRACTION

        # Clamp to position limits
        position_size = max(MIN_POSITION_PCT, min(conservative_fraction, MAX_POSITION_PCT))

        logger.debug(
            "kelly_position_calculated",
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            kelly_raw=kelly_fraction,
            kelly_conservative=conservative_fraction,
            final_size=position_size,
        )

        return position_size

    def check_daily_loss_limit(self, current_pnl: float, capital: float) -> bool:
        """
        Check if daily loss limit has been exceeded.

        Args:
            current_pnl: Today's realized + unrealized PnL
            capital: Total account capital

        Returns:
            True if trading should continue, False if limit exceeded
        """
        loss_pct = abs(current_pnl) / capital if capital > 0 else 0

        if current_pnl < 0 and loss_pct >= MAX_DAILY_LOSS_PCT:
            logger.error(
                "daily_loss_limit_exceeded",
                current_pnl=current_pnl,
                loss_pct=loss_pct,
                limit=MAX_DAILY_LOSS_PCT,
            )
            return False

        return True

    def check_consecutive_losses(self, trade_history: list[TradeHistory]) -> bool:
        """
        Check if consecutive loss limit has been exceeded.

        Args:
            trade_history: List of recent trades (most recent first)

        Returns:
            True if trading should continue, False if limit exceeded
        """
        if not trade_history:
            return True

        consecutive_losses = 0
        for trade in trade_history:
            if trade.win:
                break
            consecutive_losses += 1

        if consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
            logger.error(
                "consecutive_loss_limit_exceeded",
                consecutive_losses=consecutive_losses,
                limit=CONSECUTIVE_LOSS_LIMIT,
            )
            return False

        return True

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        direction: Literal["LONG", "SHORT"],
    ) -> float:
        """
        Calculate stop-loss price based on ATR.

        Uses 2x ATR for stop distance to avoid noise while protecting capital.

        Args:
            entry_price: Entry price for the trade
            atr: Average True Range value
            direction: Trade direction (LONG or SHORT)

        Returns:
            Stop-loss price
        """
        atr_multiplier = 2.0
        stop_distance = atr * atr_multiplier

        if direction == "LONG":
            stop_loss = entry_price - stop_distance
        else:  # SHORT
            stop_loss = entry_price + stop_distance

        logger.debug(
            "stop_loss_calculated",
            entry=entry_price,
            atr=atr,
            direction=direction,
            stop_loss=stop_loss,
        )

        return stop_loss

    def calculate_take_profit(
        self,
        entry_price: float,
        atr: float,
        direction: Literal["LONG", "SHORT"],
    ) -> float:
        """
        Calculate take-profit price based on ATR.

        Uses 3x ATR for take-profit to achieve 1.5:1 risk-reward ratio.

        Args:
            entry_price: Entry price for the trade
            atr: Average True Range value
            direction: Trade direction (LONG or SHORT)

        Returns:
            Take-profit price
        """
        atr_multiplier = 3.0
        profit_distance = atr * atr_multiplier

        if direction == "LONG":
            take_profit = entry_price + profit_distance
        else:  # SHORT
            take_profit = entry_price - profit_distance

        logger.debug(
            "take_profit_calculated",
            entry=entry_price,
            atr=atr,
            direction=direction,
            take_profit=take_profit,
        )

        return take_profit

    def adjust_leverage(self, volatility: float) -> int:
        """
        Dynamically adjust leverage based on market volatility.

        Lower volatility allows higher leverage, higher volatility requires
        lower leverage to maintain consistent risk.

        Args:
            volatility: Market volatility measure (0-1, from ATR/price ratio)

        Returns:
            Leverage multiplier (MIN_LEVERAGE to MAX_LEVERAGE)
        """
        # Volatility thresholds
        LOW_VOL = 0.02  # 2% volatility
        HIGH_VOL = 0.08  # 8% volatility

        if volatility <= LOW_VOL:
            leverage = HIGH_CONFIDENCE_LEVERAGE
        elif volatility >= HIGH_VOL:
            leverage = MIN_LEVERAGE
        else:
            # Linear interpolation between thresholds
            vol_range = HIGH_VOL - LOW_VOL
            lev_range = HIGH_CONFIDENCE_LEVERAGE - MIN_LEVERAGE
            leverage = HIGH_CONFIDENCE_LEVERAGE - int(
                ((volatility - LOW_VOL) / vol_range) * lev_range
            )

        # Ensure within bounds
        leverage = max(MIN_LEVERAGE, min(leverage, MAX_LEVERAGE))

        logger.debug(
            "leverage_adjusted",
            volatility=volatility,
            leverage=leverage,
        )

        return leverage


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitBreakerState:
    """Circuit breaker state enumeration."""
    CLOSED = "CLOSED"        # Normal operation - all requests allowed
    OPEN = "OPEN"            # Tripped - all requests blocked
    HALF_OPEN = "HALF_OPEN"  # Testing - limited requests to test recovery


class CircuitBreaker:
    """
    Circuit breaker system to halt trading during adverse conditions.

    States:
    - CLOSED: Normal operation, trading allowed
    - OPEN: Trading halted due to adverse conditions
    - HALF_OPEN: Testing recovery with limited trading (reduced position size)

    Monitors:
    - Drawdown from peak equity
    - Extreme volatility
    - System error rate
    - API failure rate

    Automatically triggers trading halt when thresholds are exceeded.
    Transitions to HALF_OPEN after cooldown to test recovery.
    """

    def __init__(self):
        """Initialize circuit breaker in CLOSED state."""
        self._state = CircuitBreakerState.CLOSED
        self.trigger_time: datetime | None = None
        self.trigger_reason: str | None = None
        self.cooldown_hours = 24
        self.half_open_hours = 4  # Time to test in HALF_OPEN before fully closing
        self.half_open_start: datetime | None = None
        self.half_open_success_count = 0
        self.half_open_failure_count = 0
        self.half_open_threshold = 3  # Successful trades needed to close
        self.half_open_max_position_pct = 0.25  # 25% of normal position in HALF_OPEN
        logger.info("circuit_breaker_initialized")

    @property
    def is_triggered(self) -> bool:
        """Check if circuit breaker is in OPEN state (fully triggered)."""
        return self._state == CircuitBreakerState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit breaker is in HALF_OPEN state."""
        return self._state == CircuitBreakerState.HALF_OPEN

    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        return self._state

    def check(self, metrics: dict) -> tuple[bool, str]:
        """
        Check if circuit breaker should be triggered or transition states.

        Args:
            metrics: Dictionary with keys:
                - drawdown: Current drawdown from peak (0-1)
                - volatility: Current market volatility (0-1)
                - error_rate: System error rate (0-1)
                - api_failure_rate: API failure rate (0-1)

        Returns:
            Tuple of (should_halt, reason)
            Note: In HALF_OPEN state, returns (False, "HALF_OPEN") to allow limited trading
        """
        # Handle OPEN state
        if self._state == CircuitBreakerState.OPEN:
            if self._is_cooldown_complete():
                self._transition_to_half_open()
                return (False, "HALF_OPEN")  # Allow limited trading for testing
            else:
                return (True, f"Circuit breaker OPEN: {self.trigger_reason}")

        # Handle HALF_OPEN state
        if self._state == CircuitBreakerState.HALF_OPEN:
            # Check if conditions improved enough to close
            if self._should_close_from_half_open():
                self.reset()
                logger.info("circuit_breaker_recovered_from_half_open")
                return (False, "")

            # Check if conditions worsened - return to OPEN
            if self._should_reopen_from_half_open(metrics):
                reason = f"Conditions worsened in HALF_OPEN state"
                self.trigger(reason)
                return (True, reason)

            # Still in HALF_OPEN - allow limited trading
            return (False, "HALF_OPEN")

        # CLOSED state - check for trigger conditions
        # Check drawdown
        drawdown = metrics.get("drawdown", 0.0)
        if drawdown >= MAX_DRAWDOWN:
            reason = f"Excessive drawdown: {drawdown:.2%} >= {MAX_DRAWDOWN:.2%}"
            self.trigger(reason)
            return (True, reason)

        # Check volatility (extreme volatility)
        volatility = metrics.get("volatility", 0.0)
        if volatility >= 0.15:  # 15% volatility
            reason = f"Extreme volatility: {volatility:.2%}"
            self.trigger(reason)
            return (True, reason)

        # Check error rate
        error_rate = metrics.get("error_rate", 0.0)
        if error_rate >= 0.3:  # 30% error rate
            reason = f"High error rate: {error_rate:.2%}"
            self.trigger(reason)
            return (True, reason)

        # Check API failure rate
        api_failure_rate = metrics.get("api_failure_rate", 0.0)
        if api_failure_rate >= 0.5:  # 50% API failure
            reason = f"High API failure rate: {api_failure_rate:.2%}"
            self.trigger(reason)
            return (True, reason)

        return (False, "")

    def trigger(self, reason: str):
        """
        Trigger circuit breaker to OPEN state.

        Args:
            reason: Reason for triggering
        """
        self._state = CircuitBreakerState.OPEN
        self.trigger_time = datetime.now(timezone.utc)
        self.trigger_reason = reason
        self.half_open_start = None
        self.half_open_success_count = 0
        self.half_open_failure_count = 0

        logger.critical(
            "circuit_breaker_triggered",
            reason=reason,
            trigger_time=self.trigger_time,
            cooldown_hours=self.cooldown_hours,
            state=self._state,
        )

    def reset(self):
        """Reset circuit breaker to CLOSED state."""
        logger.info(
            "circuit_breaker_reset",
            previous_state=self._state,
            previous_reason=self.trigger_reason,
            trigger_time=self.trigger_time,
        )

        self._state = CircuitBreakerState.CLOSED
        self.trigger_time = None
        self.trigger_reason = None
        self.half_open_start = None
        self.half_open_success_count = 0
        self.half_open_failure_count = 0

    def _transition_to_half_open(self):
        """Transition from OPEN to HALF_OPEN state for recovery testing."""
        self._state = CircuitBreakerState.HALF_OPEN
        self.half_open_start = datetime.now(timezone.utc)
        self.half_open_success_count = 0
        self.half_open_failure_count = 0

        logger.info(
            "circuit_breaker_half_open",
            previous_reason=self.trigger_reason,
            half_open_start=self.half_open_start,
            max_position_pct=self.half_open_max_position_pct,
        )

    def record_half_open_result(self, success: bool):
        """
        Record the result of a trade during HALF_OPEN state.

        Args:
            success: True if trade was successful (profitable or small loss)
        """
        if self._state != CircuitBreakerState.HALF_OPEN:
            return

        if success:
            self.half_open_success_count += 1
            logger.debug(
                "circuit_breaker_half_open_success",
                success_count=self.half_open_success_count,
                threshold=self.half_open_threshold,
            )
        else:
            self.half_open_failure_count += 1
            logger.warning(
                "circuit_breaker_half_open_failure",
                failure_count=self.half_open_failure_count,
            )

    def get_position_size_multiplier(self) -> float:
        """
        Get position size multiplier based on current state.

        Returns:
            1.0 for CLOSED (normal), reduced for HALF_OPEN, 0.0 for OPEN
        """
        if self._state == CircuitBreakerState.CLOSED:
            return 1.0
        elif self._state == CircuitBreakerState.HALF_OPEN:
            return self.half_open_max_position_pct
        else:  # OPEN
            return 0.0

    def _should_close_from_half_open(self) -> bool:
        """Check if circuit breaker should transition from HALF_OPEN to CLOSED."""
        if self._state != CircuitBreakerState.HALF_OPEN:
            return False

        # Need enough successful trades
        if self.half_open_success_count >= self.half_open_threshold:
            return True

        # Or spent enough time in HALF_OPEN without failures
        if self.half_open_start:
            elapsed = datetime.now(timezone.utc) - self.half_open_start
            if elapsed >= timedelta(hours=self.half_open_hours) and self.half_open_failure_count == 0:
                return True

        return False

    def _should_reopen_from_half_open(self, metrics: dict) -> bool:
        """Check if circuit breaker should transition from HALF_OPEN back to OPEN."""
        if self._state != CircuitBreakerState.HALF_OPEN:
            return False

        # Too many failures in HALF_OPEN
        if self.half_open_failure_count >= 2:
            return True

        # Conditions severely worsened (use stricter thresholds)
        drawdown = metrics.get("drawdown", 0.0)
        if drawdown >= MAX_DRAWDOWN * 0.8:  # 80% of threshold
            return True

        volatility = metrics.get("volatility", 0.0)
        if volatility >= 0.12:  # 12% volatility (stricter than CLOSED)
            return True

        return False

    def _is_cooldown_complete(self) -> bool:
        """Check if cooldown period has elapsed."""
        if not self.trigger_time:
            return True

        elapsed = datetime.now(timezone.utc) - self.trigger_time
        return elapsed >= timedelta(hours=self.cooldown_hours)


# =============================================================================
# Kill Switch
# =============================================================================


class KillSwitch:
    """
    Emergency kill switch for immediate trading halt.

    Can be activated manually or programmatically during critical events.
    Requires confirmation code to deactivate for additional security.

    The confirmation code is generated upon activation and must be provided
    to deactivate the kill switch, preventing accidental reactivation.
    """

    def __init__(self):
        """Initialize kill switch in inactive state."""
        self._active = False
        self.activation_time: datetime | None = None
        self.activation_reason: str | None = None
        self._confirmation_code: str | None = None
        self._deactivation_attempts = 0
        self._max_deactivation_attempts = 5
        self._lockout_until: datetime | None = None
        logger.info("kill_switch_initialized")

    def activate(self, reason: str) -> str:
        """
        Activate kill switch.

        Args:
            reason: Reason for activation

        Returns:
            Confirmation code required for deactivation
        """
        import secrets
        import hashlib

        self._active = True
        self.activation_time = datetime.now(timezone.utc)
        self.activation_reason = reason
        self._deactivation_attempts = 0
        self._lockout_until = None

        # Generate secure confirmation code (6 alphanumeric characters)
        self._confirmation_code = secrets.token_hex(3).upper()

        logger.critical(
            "kill_switch_activated",
            reason=reason,
            activation_time=self.activation_time,
            confirmation_code_hash=hashlib.sha256(
                self._confirmation_code.encode()
            ).hexdigest()[:8],  # Log partial hash for audit
        )

        return self._confirmation_code

    def deactivate(self, confirmation_code: str) -> tuple[bool, str]:
        """
        Deactivate kill switch with confirmation code verification.

        Args:
            confirmation_code: The confirmation code provided during activation

        Returns:
            Tuple of (success, message)
        """
        # Check if not active
        if not self._active:
            return (False, "Kill switch is not active")

        # Check if locked out due to too many failed attempts
        if self._lockout_until:
            if datetime.now(timezone.utc) < self._lockout_until:
                remaining = (self._lockout_until - datetime.now(timezone.utc)).seconds
                logger.warning(
                    "kill_switch_deactivation_locked_out",
                    remaining_seconds=remaining,
                )
                return (False, f"Locked out. Try again in {remaining} seconds")
            else:
                # Lockout expired
                self._lockout_until = None
                self._deactivation_attempts = 0

        # Verify confirmation code
        if confirmation_code.upper() != self._confirmation_code:
            self._deactivation_attempts += 1
            logger.warning(
                "kill_switch_deactivation_failed",
                attempts=self._deactivation_attempts,
                max_attempts=self._max_deactivation_attempts,
            )

            # Lockout after too many failed attempts
            if self._deactivation_attempts >= self._max_deactivation_attempts:
                self._lockout_until = datetime.now(timezone.utc) + timedelta(minutes=15)
                logger.error(
                    "kill_switch_deactivation_locked",
                    lockout_until=self._lockout_until,
                )
                return (False, "Too many failed attempts. Locked for 15 minutes")

            remaining_attempts = self._max_deactivation_attempts - self._deactivation_attempts
            return (False, f"Invalid confirmation code. {remaining_attempts} attempts remaining")

        # Success - deactivate
        logger.warning(
            "kill_switch_deactivated",
            previous_reason=self.activation_reason,
            activation_time=self.activation_time,
            deactivation_time=datetime.now(timezone.utc),
        )

        self._active = False
        self.activation_time = None
        self.activation_reason = None
        self._confirmation_code = None
        self._deactivation_attempts = 0
        self._lockout_until = None

        return (True, "Kill switch deactivated successfully")

    def force_deactivate(self, admin_key: str) -> tuple[bool, str]:
        """
        Force deactivate kill switch with admin key (bypass confirmation code).

        This should only be used in emergencies when the confirmation code is lost.
        Requires the KILL_SWITCH_ADMIN_KEY environment variable to be set.

        Args:
            admin_key: Admin key for force deactivation

        Returns:
            Tuple of (success, message)
        """
        import os
        import hmac

        expected_key = os.environ.get("KILL_SWITCH_ADMIN_KEY")
        if not expected_key:
            logger.error("kill_switch_admin_key_not_configured")
            return (False, "Admin key not configured")

        # Use constant-time comparison to prevent timing attacks
        if not hmac.compare_digest(admin_key, expected_key):
            logger.error("kill_switch_force_deactivation_failed_invalid_key")
            return (False, "Invalid admin key")

        logger.critical(
            "kill_switch_force_deactivated",
            previous_reason=self.activation_reason,
            activation_time=self.activation_time,
            force_deactivation_time=datetime.now(timezone.utc),
        )

        self._active = False
        self.activation_time = None
        self.activation_reason = None
        self._confirmation_code = None
        self._deactivation_attempts = 0
        self._lockout_until = None

        return (True, "Kill switch force deactivated with admin key")

    def get_status(self) -> dict:
        """
        Get current kill switch status.

        Returns:
            Dictionary with status information
        """
        return {
            "active": self._active,
            "activation_time": self.activation_time.isoformat() if self.activation_time else None,
            "activation_reason": self.activation_reason,
            "locked_out": self._lockout_until is not None and datetime.now(timezone.utc) < self._lockout_until,
            "lockout_until": self._lockout_until.isoformat() if self._lockout_until else None,
            "failed_attempts": self._deactivation_attempts,
        }

    def is_active(self) -> bool:
        """Check if kill switch is currently active."""
        return self._active


# =============================================================================
# Decision Engine
# =============================================================================


class DecisionEngine:
    """
    Core trading decision engine integrating all analysis layers.

    Combines:
    - Technical Analysis (40% weight)
    - LLM Sentiment Analysis (25% weight)
    - XGBoost ML Model (35% weight)

    With comprehensive risk management:
    - Kelly Criterion position sizing
    - ATR-based stops
    - Circuit breaker
    - Kill switch
    - Multiple veto checks
    """

    # Signal weights
    WEIGHT_TECHNICAL = 0.40
    WEIGHT_LLM = 0.25
    WEIGHT_ML = 0.35

    def __init__(
        self,
        risk_manager: RiskManager,
        circuit_breaker: CircuitBreaker,
        kill_switch: KillSwitch,
    ):
        """
        Initialize decision engine.

        Args:
            risk_manager: Risk management system
            circuit_breaker: Circuit breaker for adverse conditions
            kill_switch: Emergency kill switch
        """
        self.risk_manager = risk_manager
        self.circuit_breaker = circuit_breaker
        self.kill_switch = kill_switch
        self.settings = get_settings()

        logger.info("decision_engine_initialized")

    async def make_decision(
        self,
        symbol: str,
        technical_signal: CompositeSignal,
        llm_analysis: LLMAnalysis,
        ml_prediction: ModelPrediction,
        market_context: MarketContext,
        current_price: float,
        account_balance: float,
        trade_history: list[TradeHistory] | None = None,
        current_pnl: float = 0.0,
    ) -> TradingDecision:
        """
        Make a trading decision by integrating all analysis layers.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            technical_signal: Technical analysis composite signal
            llm_analysis: LLM sentiment analysis
            ml_prediction: XGBoost model prediction
            market_context: External market context data
            current_price: Current market price
            account_balance: Current account balance
            trade_history: Recent trade history (for risk checks)
            current_pnl: Current day's PnL

        Returns:
            TradingDecision with action, sizing, and risk parameters
        """
        timestamp = datetime.now(timezone.utc)
        trade_history = trade_history or []

        logger.info(
            "decision_making_started",
            symbol=symbol,
            price=current_price,
            balance=account_balance,
        )

        # =====================================================================
        # Safety Checks
        # =====================================================================

        # Check kill switch
        if self.kill_switch.is_active():
            return self._create_vetoed_decision(
                symbol=symbol,
                reason=f"Kill switch active: {self.kill_switch.activation_reason}",
                current_price=current_price,
                timestamp=timestamp,
            )

        # Check circuit breaker
        circuit_metrics = self._calculate_circuit_metrics(
            trade_history, account_balance, market_context
        )
        should_halt, halt_reason = self.circuit_breaker.check(circuit_metrics)
        if should_halt:
            return self._create_vetoed_decision(
                symbol=symbol,
                reason=f"Circuit breaker: {halt_reason}",
                current_price=current_price,
                timestamp=timestamp,
            )

        # Track if we're in HALF_OPEN mode (reduced position sizing)
        is_half_open = halt_reason == "HALF_OPEN"
        position_size_multiplier = self.circuit_breaker.get_position_size_multiplier()

        # Check daily loss limit
        if not self.risk_manager.check_daily_loss_limit(current_pnl, account_balance):
            return self._create_vetoed_decision(
                symbol=symbol,
                reason=f"Daily loss limit exceeded: {current_pnl:.2f}",
                current_price=current_price,
                timestamp=timestamp,
            )

        # Check consecutive losses
        if not self.risk_manager.check_consecutive_losses(trade_history):
            return self._create_vetoed_decision(
                symbol=symbol,
                reason=f"Consecutive loss limit exceeded",
                current_price=current_price,
                timestamp=timestamp,
            )

        # =====================================================================
        # Signal Integration
        # =====================================================================

        # Convert signals to numeric scores (-1 to 1)
        technical_score = self._signal_to_score(technical_signal.overall_signal)
        llm_score = llm_analysis.sentiment.value
        ml_score = self._prediction_to_score(ml_prediction.action)

        # Weighted combination
        combined_score = (
            technical_score * self.WEIGHT_TECHNICAL
            + llm_score * self.WEIGHT_LLM
            + ml_score * self.WEIGHT_ML
        )

        # Combined confidence (weighted average)
        combined_confidence = (
            technical_signal.confidence * self.WEIGHT_TECHNICAL
            + llm_analysis.confidence * self.WEIGHT_LLM
            + ml_prediction.confidence * self.WEIGHT_ML
        )

        logger.debug(
            "signals_combined",
            technical_score=technical_score,
            llm_score=llm_score,
            ml_score=ml_score,
            combined_score=combined_score,
            combined_confidence=combined_confidence,
        )

        # =====================================================================
        # Veto Checks
        # =====================================================================

        # LLM sentiment veto
        if llm_analysis.sentiment.value <= SENTIMENT_VETO_THRESHOLD:
            return self._create_vetoed_decision(
                symbol=symbol,
                reason=f"LLM sentiment too negative: {llm_analysis.sentiment.value:.2f}",
                current_price=current_price,
                timestamp=timestamp,
            )

        # Low confidence veto
        if combined_confidence < CONFIDENCE_VETO_THRESHOLD:
            return self._create_vetoed_decision(
                symbol=symbol,
                reason=f"Combined confidence too low: {combined_confidence:.2f}",
                current_price=current_price,
                timestamp=timestamp,
            )

        # Signal disagreement veto (all three disagree)
        signals = [technical_score, llm_score, ml_score]
        if not self._signals_aligned(signals):
            return self._create_vetoed_decision(
                symbol=symbol,
                reason="Major signal disagreement between analysis layers",
                current_price=current_price,
                timestamp=timestamp,
            )

        # =====================================================================
        # Action Determination
        # =====================================================================

        # Determine action based on combined score
        if combined_score > 0.2:
            action = "LONG"
        elif combined_score < -0.2:
            action = "SHORT"
        else:
            action = "HOLD"

        # If HOLD, return early
        if action == "HOLD":
            logger.info(
                "decision_hold",
                symbol=symbol,
                combined_score=combined_score,
                combined_confidence=combined_confidence,
            )
            return TradingDecision(
                action="HOLD",
                symbol=symbol,
                confidence=combined_confidence,
                position_size=0.0,
                leverage=1,
                stop_loss=current_price,
                take_profit=current_price,
                entry_price=current_price,
                timestamp=timestamp,
                reasons=["Combined signal insufficient for trade entry"],
            )

        # =====================================================================
        # Position Sizing
        # =====================================================================

        # Calculate position size using Kelly Criterion
        win_rate, avg_win, avg_loss = self._calculate_trade_statistics(trade_history)
        kelly_size = self.risk_manager.calculate_kelly_position(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_capital=account_balance,
        )

        # Adjust size based on confidence
        confidence_adjusted_size = kelly_size * combined_confidence

        # Apply circuit breaker multiplier (reduced in HALF_OPEN state)
        circuit_adjusted_size = confidence_adjusted_size * position_size_multiplier

        # Final position size (respect limits)
        position_size = max(
            MIN_POSITION_PCT,
            min(circuit_adjusted_size, MAX_POSITION_PCT)
        )

        # Log if in HALF_OPEN mode
        if is_half_open:
            logger.info(
                "half_open_position_sizing",
                original_size=confidence_adjusted_size,
                multiplier=position_size_multiplier,
                final_size=position_size,
            )

        # =====================================================================
        # Risk Parameters
        # =====================================================================

        # Get ATR from technical indicators
        atr = self._extract_atr(technical_signal)

        # Calculate volatility for leverage adjustment
        volatility = atr / current_price if current_price > 0 else 0.05

        # Adjust leverage based on volatility
        base_leverage = self.risk_manager.adjust_leverage(volatility)

        # Further adjust leverage based on confidence
        if combined_confidence >= 0.8:
            leverage = min(base_leverage + 1, MAX_LEVERAGE)
        elif combined_confidence <= 0.5:
            leverage = max(base_leverage - 1, MIN_LEVERAGE)
        else:
            leverage = base_leverage

        # Calculate stops
        stop_loss = self.risk_manager.calculate_stop_loss(
            entry_price=current_price,
            atr=atr,
            direction=action,
        )

        take_profit = self.risk_manager.calculate_take_profit(
            entry_price=current_price,
            atr=atr,
            direction=action,
        )

        # =====================================================================
        # Generate Decision Reasons
        # =====================================================================

        reasons = []
        reasons.append(
            f"Technical: {technical_signal.overall_signal} "
            f"(confidence: {technical_signal.confidence:.2f})"
        )
        reasons.append(
            f"LLM: {llm_analysis.sentiment} "
            f"(confidence: {llm_analysis.confidence:.2f})"
        )
        reasons.append(
            f"ML: {ml_prediction.action} "
            f"(confidence: {ml_prediction.confidence:.2f})"
        )
        reasons.append(f"Combined score: {combined_score:.3f}")
        reasons.append(f"Combined confidence: {combined_confidence:.2f}")
        reasons.append(f"Volatility: {volatility:.2%}")

        # =====================================================================
        # Create Final Decision
        # =====================================================================

        decision = TradingDecision(
            action=action,
            symbol=symbol,
            confidence=combined_confidence,
            position_size=position_size,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_price=current_price,
            timestamp=timestamp,
            reasons=reasons,
        )

        logger.info(
            "decision_created",
            decision=decision.to_dict(),
        )

        return decision

    def _signal_to_score(self, signal: Literal["BULLISH", "BEARISH", "NEUTRAL"]) -> float:
        """Convert signal enum to numeric score."""
        if signal == "BULLISH":
            return 1.0
        elif signal == "BEARISH":
            return -1.0
        else:
            return 0.0

    def _prediction_to_score(self, action: Literal["LONG", "SHORT", "HOLD"]) -> float:
        """Convert ML prediction to numeric score."""
        if action == "LONG":
            return 1.0
        elif action == "SHORT":
            return -1.0
        else:
            return 0.0

    def _signals_aligned(self, signals: list[float], threshold: float = 0.6) -> bool:
        """
        Check if signals are reasonably aligned.

        At least 2 out of 3 signals should agree on direction.
        """
        positive = sum(1 for s in signals if s > 0.2)
        negative = sum(1 for s in signals if s < -0.2)

        # At least 2 signals agree
        return positive >= 2 or negative >= 2

    def _extract_atr(self, technical_signal: CompositeSignal) -> float:
        """Extract ATR value from technical signal."""
        atr_indicator = technical_signal.individual_signals.get("ATR")
        if atr_indicator and isinstance(atr_indicator.value, (int, float)):
            return float(atr_indicator.value)

        # Fallback: estimate from price (2% of price)
        logger.warning("atr_not_found_using_fallback")
        return 0.02 * 50000  # Rough estimate for crypto

    def _calculate_trade_statistics(
        self,
        trade_history: list[TradeHistory]
    ) -> tuple[float, float, float]:
        """
        Calculate win rate and average win/loss from trade history.

        Returns:
            Tuple of (win_rate, avg_win, avg_loss)
        """
        if len(trade_history) < 20:
            # Insufficient data, use defaults
            logger.debug("insufficient_trade_history_using_defaults")
            return (0.55, 100.0, 80.0)

        wins = [t for t in trade_history if t.win]
        losses = [t for t in trade_history if not t.win]

        win_rate = len(wins) / len(trade_history)
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 100.0
        avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 80.0

        return (win_rate, avg_win, avg_loss)

    def _calculate_circuit_metrics(
        self,
        trade_history: list[TradeHistory],
        account_balance: float,
        market_context: MarketContext,
    ) -> dict:
        """Calculate metrics for circuit breaker check."""
        metrics = {
            "drawdown": 0.0,
            "volatility": 0.0,
            "error_rate": 0.0,
            "api_failure_rate": 0.0,
        }

        # Calculate drawdown from trade history
        if trade_history:
            peak_balance = account_balance
            for trade in trade_history:
                if trade.pnl > 0:
                    peak_balance = max(peak_balance, account_balance)

            current_drawdown = (peak_balance - account_balance) / peak_balance if peak_balance > 0 else 0.0
            metrics["drawdown"] = max(0.0, current_drawdown)

        # Estimate volatility from market context (if available)
        if market_context.fear_greed:
            # Fear & Greed index as volatility proxy
            # Extreme fear (0) or greed (100) suggests high volatility
            fg_value = market_context.fear_greed.value
            volatility_score = abs(fg_value - 50) / 50.0  # 0 at 50, 1 at 0 or 100
            metrics["volatility"] = volatility_score * 0.1  # Scale to reasonable range

        # API failure rate from market context errors
        if market_context.errors:
            # Count of errors as proxy for API issues
            metrics["api_failure_rate"] = min(len(market_context.errors) / 10.0, 1.0)

        return metrics

    def _create_vetoed_decision(
        self,
        symbol: str,
        reason: str,
        current_price: float,
        timestamp: datetime,
    ) -> TradingDecision:
        """Create a vetoed HOLD decision."""
        return TradingDecision(
            action="HOLD",
            symbol=symbol,
            confidence=0.0,
            position_size=0.0,
            leverage=1,
            stop_loss=current_price,
            take_profit=current_price,
            entry_price=current_price,
            timestamp=timestamp,
            reasons=[],
            vetoed=True,
            veto_reason=reason,
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_decision_engine() -> DecisionEngine:
    """
    Create a fully configured decision engine with all components.

    Returns:
        DecisionEngine instance ready for trading decisions
    """
    risk_manager = RiskManager()
    circuit_breaker = CircuitBreaker()
    kill_switch = KillSwitch()

    engine = DecisionEngine(
        risk_manager=risk_manager,
        circuit_breaker=circuit_breaker,
        kill_switch=kill_switch,
    )

    logger.info("decision_engine_created")
    return engine
