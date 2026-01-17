"""Technical Analysis Indicators Module.

This module provides vectorized technical analysis indicators and signal generation
for the IFTB trading system. All indicators are optimized using numpy/pandas for
high-performance computation.

Indicators:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - ATR (Average True Range)
    - ADX (Average Directional Index)
    - Stochastic Oscillator
    - CCI (Commodity Channel Index)
    - Williams %R
    - OBV (On-Balance Volume)
    - VWAP (Volume Weighted Average Price)
    - Ichimoku Cloud
    - Fibonacci Retracement
    - EMA Cross
    - Volume Profile
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

SignalType = Literal["BULLISH", "BEARISH", "NEUTRAL"]


@dataclass
class IndicatorResult:
    """Result from a technical indicator calculation.

    Attributes:
        name: Indicator name
        value: Calculated value (float for single values, dict for complex indicators)
        signal: Trading signal (BULLISH, BEARISH, or NEUTRAL)
        strength: Signal strength from 0 (weak) to 1 (strong)
        timestamp: Calculation timestamp
    """

    name: str
    value: float | dict
    signal: SignalType
    strength: float
    timestamp: datetime


@dataclass
class CompositeSignal:
    """Weighted composite signal from all indicators.

    Attributes:
        overall_signal: Aggregated signal direction
        confidence: Overall confidence score (0-1)
        bullish_indicators: Number of bullish signals
        bearish_indicators: Number of bearish signals
        neutral_indicators: Number of neutral signals
        individual_signals: Dictionary of individual indicator results
        timestamp: Signal generation timestamp
    """

    overall_signal: SignalType
    confidence: float
    bullish_indicators: int
    bearish_indicators: int
    neutral_indicators: int
    individual_signals: dict[str, IndicatorResult]
    timestamp: datetime


class TechnicalAnalyzer:
    """Technical Analysis calculator with 14 vectorized indicators.

    All calculations are optimized using numpy/pandas for performance.
    Supports both single-point and batch calculations.
    """

    def __init__(self, ohlcv_data: pd.DataFrame):
        """Initialize analyzer with OHLCV data.

        Args:
            ohlcv_data: DataFrame with columns: timestamp, open, high, low, close, volume
        """
        self.data = ohlcv_data.copy()
        self._validate_data()
        self._precompute_common()

    def _validate_data(self) -> None:
        """Validate OHLCV data format and required columns."""
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(self.data.columns):
            raise ValueError(f"Data must contain columns: {required_cols}")

        if len(self.data) < 52:  # Minimum for Ichimoku
            raise ValueError("Insufficient data: need at least 52 periods")

    def _precompute_common(self) -> None:
        """Precompute commonly used values for efficiency."""
        self.data["hl2"] = (self.data["high"] + self.data["low"]) / 2
        self.data["hlc3"] = (self.data["high"] + self.data["low"] + self.data["close"]) / 3
        self.data["ohlc4"] = (
            self.data["open"] + self.data["high"] + self.data["low"] + self.data["close"]
        ) / 4

    # ==================== RSI ====================

    def calculate_rsi(self, period: int = 14) -> IndicatorResult:
        """Calculate Relative Strength Index.

        RSI measures momentum and identifies overbought/oversold conditions.

        Args:
            period: Lookback period (default: 14)

        Returns:
            IndicatorResult with RSI value and signal
        """
        delta = self.data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        # Signal generation
        if current_rsi > 70:
            signal = "BEARISH"
            strength = min((current_rsi - 70) / 30, 1.0)
        elif current_rsi < 30:
            signal = "BULLISH"
            strength = min((30 - current_rsi) / 30, 1.0)
        else:
            signal = "NEUTRAL"
            strength = 1.0 - (abs(current_rsi - 50) / 50)

        return IndicatorResult(
            name="RSI",
            value=float(current_rsi),
            signal=signal,
            strength=float(strength),
            timestamp=datetime.now(),
        )

    # ==================== MACD ====================

    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> IndicatorResult:
        """Calculate Moving Average Convergence Divergence.

        MACD shows relationship between two moving averages and identifies trend changes.

        Args:
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)

        Returns:
            IndicatorResult with MACD values and signal
        """
        ema_fast = self.data["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = self.data["close"].ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]

        # Signal generation based on crossover and histogram
        if current_macd > current_signal and current_hist > 0:
            signal_type = "BULLISH"
            strength = min(abs(current_hist) / (abs(current_macd) + 1e-10), 1.0)
        elif current_macd < current_signal and current_hist < 0:
            signal_type = "BEARISH"
            strength = min(abs(current_hist) / (abs(current_macd) + 1e-10), 1.0)
        else:
            signal_type = "NEUTRAL"
            strength = 0.5

        # Boost strength on recent crossover
        if np.sign(current_hist) != np.sign(prev_hist):
            strength = min(strength * 1.5, 1.0)

        return IndicatorResult(
            name="MACD",
            value={
                "macd": float(current_macd),
                "signal": float(current_signal),
                "histogram": float(current_hist),
            },
            signal=signal_type,
            strength=float(strength),
            timestamp=datetime.now(),
        )

    # ==================== Bollinger Bands ====================

    def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> IndicatorResult:
        """Calculate Bollinger Bands.

        Bollinger Bands measure volatility and identify potential reversal points.

        Args:
            period: Moving average period (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)

        Returns:
            IndicatorResult with band values and signal
        """
        sma = self.data["close"].rolling(window=period).mean()
        std = self.data["close"].rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        current_close = self.data["close"].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_sma = sma.iloc[-1]

        bandwidth = current_upper - current_lower
        position = (current_close - current_lower) / bandwidth

        # Signal generation
        if position > 0.9:
            signal_type = "BEARISH"
            strength = min((position - 0.9) * 10, 1.0)
        elif position < 0.1:
            signal_type = "BULLISH"
            strength = min((0.1 - position) * 10, 1.0)
        else:
            signal_type = "NEUTRAL"
            strength = 1.0 - abs(position - 0.5) * 2

        return IndicatorResult(
            name="Bollinger_Bands",
            value={
                "upper": float(current_upper),
                "middle": float(current_sma),
                "lower": float(current_lower),
                "bandwidth": float(bandwidth),
                "position": float(position),
            },
            signal=signal_type,
            strength=float(strength),
            timestamp=datetime.now(),
        )

    # ==================== ATR ====================

    def calculate_atr(self, period: int = 14) -> IndicatorResult:
        """Calculate Average True Range.

        ATR measures market volatility. Higher values indicate higher volatility.

        Args:
            period: Lookback period (default: 14)

        Returns:
            IndicatorResult with ATR value and signal
        """
        high = self.data["high"]
        low = self.data["low"]
        close = self.data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        current_atr = atr.iloc[-1]
        avg_atr = atr.tail(50).mean()

        # ATR doesn't give directional signals, but indicates volatility
        volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0

        signal_type = "NEUTRAL"
        strength = float(np.clip(volatility_ratio, 0, 1))

        return IndicatorResult(
            name="ATR",
            value={"atr": float(current_atr), "volatility_ratio": float(volatility_ratio)},
            signal=signal_type,
            strength=strength,
            timestamp=datetime.now(),
        )

    # ==================== ADX ====================

    def calculate_adx(self, period: int = 14) -> IndicatorResult:
        """Calculate Average Directional Index.

        ADX measures trend strength (not direction). Values above 25 indicate strong trend.

        Args:
            period: Lookback period (default: 14)

        Returns:
            IndicatorResult with ADX value and signal
        """
        high = self.data["high"]
        low = self.data["low"]
        close = self.data["close"]

        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smoothed averages
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # ADX calculation
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]

        # Signal generation based on DI crossover and ADX strength
        if current_adx > 25:
            if current_plus_di > current_minus_di:
                signal_type = "BULLISH"
                strength = min(current_adx / 50, 1.0)
            else:
                signal_type = "BEARISH"
                strength = min(current_adx / 50, 1.0)
        else:
            signal_type = "NEUTRAL"
            strength = 1.0 - (current_adx / 25)

        return IndicatorResult(
            name="ADX",
            value={
                "adx": float(current_adx),
                "plus_di": float(current_plus_di),
                "minus_di": float(current_minus_di),
            },
            signal=signal_type,
            strength=float(strength),
            timestamp=datetime.now(),
        )

    # ==================== Stochastic Oscillator ====================

    def calculate_stochastic(
        self, k_period: int = 14, d_period: int = 3, smooth_k: int = 3
    ) -> IndicatorResult:
        """Calculate Stochastic Oscillator.

        Stochastic shows momentum and identifies overbought/oversold conditions.

        Args:
            k_period: %K period (default: 14)
            d_period: %D smoothing period (default: 3)
            smooth_k: %K smoothing period (default: 3)

        Returns:
            IndicatorResult with stochastic values and signal
        """
        low_min = self.data["low"].rolling(window=k_period).min()
        high_max = self.data["high"].rolling(window=k_period).max()

        k_fast = 100 * (self.data["close"] - low_min) / (high_max - low_min)
        k_slow = k_fast.rolling(window=smooth_k).mean()
        d_slow = k_slow.rolling(window=d_period).mean()

        current_k = k_slow.iloc[-1]
        current_d = d_slow.iloc[-1]

        # Signal generation
        if current_k > 80 and current_d > 80:
            signal_type = "BEARISH"
            strength = min((current_k - 80) / 20, 1.0)
        elif current_k < 20 and current_d < 20:
            signal_type = "BULLISH"
            strength = min((20 - current_k) / 20, 1.0)
        elif current_k > current_d:
            signal_type = "BULLISH"
            strength = 0.5
        elif current_k < current_d:
            signal_type = "BEARISH"
            strength = 0.5
        else:
            signal_type = "NEUTRAL"
            strength = 0.3

        return IndicatorResult(
            name="Stochastic",
            value={"k": float(current_k), "d": float(current_d)},
            signal=signal_type,
            strength=float(strength),
            timestamp=datetime.now(),
        )

    # ==================== CCI ====================

    def calculate_cci(self, period: int = 20) -> IndicatorResult:
        """Calculate Commodity Channel Index.

        CCI identifies cyclical trends and overbought/oversold conditions.

        Args:
            period: Lookback period (default: 20)

        Returns:
            IndicatorResult with CCI value and signal
        """
        tp = self.data["hlc3"]
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())

        cci = (tp - sma) / (0.015 * mad)
        current_cci = cci.iloc[-1]

        # Signal generation
        if current_cci > 100:
            signal_type = "BEARISH"
            strength = min((current_cci - 100) / 200, 1.0)
        elif current_cci < -100:
            signal_type = "BULLISH"
            strength = min((abs(current_cci) - 100) / 200, 1.0)
        else:
            signal_type = "NEUTRAL"
            strength = 1.0 - (abs(current_cci) / 100)

        return IndicatorResult(
            name="CCI",
            value=float(current_cci),
            signal=signal_type,
            strength=float(strength),
            timestamp=datetime.now(),
        )

    # ==================== Williams %R ====================

    def calculate_williams_r(self, period: int = 14) -> IndicatorResult:
        """Calculate Williams %R.

        Williams %R measures momentum and identifies overbought/oversold levels.

        Args:
            period: Lookback period (default: 14)

        Returns:
            IndicatorResult with Williams %R value and signal
        """
        high_max = self.data["high"].rolling(window=period).max()
        low_min = self.data["low"].rolling(window=period).min()

        williams_r = -100 * (high_max - self.data["close"]) / (high_max - low_min)
        current_wr = williams_r.iloc[-1]

        # Signal generation
        if current_wr > -20:
            signal_type = "BEARISH"
            strength = min((current_wr + 20) / 20, 1.0)
        elif current_wr < -80:
            signal_type = "BULLISH"
            strength = min((80 + current_wr) / 20, 1.0)
        else:
            signal_type = "NEUTRAL"
            strength = 1.0 - (abs(current_wr + 50) / 50)

        return IndicatorResult(
            name="Williams_R",
            value=float(current_wr),
            signal=signal_type,
            strength=float(strength),
            timestamp=datetime.now(),
        )

    # ==================== OBV ====================

    def calculate_obv(self) -> IndicatorResult:
        """Calculate On-Balance Volume.

        OBV uses volume flow to predict price changes.

        Returns:
            IndicatorResult with OBV value and signal
        """
        obv = (np.sign(self.data["close"].diff()) * self.data["volume"]).fillna(0).cumsum()

        current_obv = obv.iloc[-1]
        obv_sma = obv.rolling(window=20).mean()
        current_obv_sma = obv_sma.iloc[-1]

        # Signal based on OBV vs SMA
        if current_obv > current_obv_sma:
            signal_type = "BULLISH"
            strength = (
                min(abs(current_obv - current_obv_sma) / current_obv_sma, 1.0)
                if current_obv_sma != 0
                else 0.5
            )
        elif current_obv < current_obv_sma:
            signal_type = "BEARISH"
            strength = (
                min(abs(current_obv - current_obv_sma) / current_obv_sma, 1.0)
                if current_obv_sma != 0
                else 0.5
            )
        else:
            signal_type = "NEUTRAL"
            strength = 0.5

        return IndicatorResult(
            name="OBV",
            value={"obv": float(current_obv), "obv_sma": float(current_obv_sma)},
            signal=signal_type,
            strength=float(strength),
            timestamp=datetime.now(),
        )

    # ==================== VWAP ====================

    def calculate_vwap(self) -> IndicatorResult:
        """Calculate Volume Weighted Average Price.

        VWAP provides the average price weighted by volume.

        Returns:
            IndicatorResult with VWAP value and signal
        """
        typical_price = self.data["hlc3"]
        vwap = (typical_price * self.data["volume"]).cumsum() / self.data["volume"].cumsum()

        current_vwap = vwap.iloc[-1]
        current_close = self.data["close"].iloc[-1]

        # Signal based on price vs VWAP
        diff_pct = (current_close - current_vwap) / current_vwap * 100

        if diff_pct > 1:
            signal_type = "BEARISH"
            strength = min(abs(diff_pct) / 5, 1.0)
        elif diff_pct < -1:
            signal_type = "BULLISH"
            strength = min(abs(diff_pct) / 5, 1.0)
        else:
            signal_type = "NEUTRAL"
            strength = 1.0 - abs(diff_pct)

        return IndicatorResult(
            name="VWAP",
            value={
                "vwap": float(current_vwap),
                "price": float(current_close),
                "diff_pct": float(diff_pct),
            },
            signal=signal_type,
            strength=float(strength),
            timestamp=datetime.now(),
        )

    # ==================== Ichimoku Cloud ====================

    def calculate_ichimoku(
        self, tenkan: int = 9, kijun: int = 26, senkou: int = 52
    ) -> IndicatorResult:
        """Calculate Ichimoku Cloud.

        Ichimoku provides support/resistance levels and trend direction.

        Args:
            tenkan: Conversion line period (default: 9)
            kijun: Base line period (default: 26)
            senkou: Leading span B period (default: 52)

        Returns:
            IndicatorResult with Ichimoku values and signal
        """
        high = self.data["high"]
        low = self.data["low"]
        close = self.data["close"]

        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()) / 2

        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(window=kijun).max() + low.rolling(window=kijun).min()) / 2

        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)

        # Senkou Span B (Leading Span B)
        senkou_span_b = (
            (high.rolling(window=senkou).max() + low.rolling(window=senkou).min()) / 2
        ).shift(kijun)

        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-kijun)

        current_close = close.iloc[-1]
        current_tenkan = tenkan_sen.iloc[-1]
        current_kijun = kijun_sen.iloc[-1]
        current_senkou_a = senkou_span_a.iloc[-1]
        current_senkou_b = senkou_span_b.iloc[-1]

        # Signal generation
        cloud_top = max(current_senkou_a, current_senkou_b)
        cloud_bottom = min(current_senkou_a, current_senkou_b)

        bullish_signals = 0
        bearish_signals = 0

        if current_close > cloud_top:
            bullish_signals += 1
        elif current_close < cloud_bottom:
            bearish_signals += 1

        if current_tenkan > current_kijun:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if current_senkou_a > current_senkou_b:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if bullish_signals > bearish_signals:
            signal_type = "BULLISH"
            strength = bullish_signals / 3
        elif bearish_signals > bullish_signals:
            signal_type = "BEARISH"
            strength = bearish_signals / 3
        else:
            signal_type = "NEUTRAL"
            strength = 0.5

        return IndicatorResult(
            name="Ichimoku",
            value={
                "tenkan_sen": float(current_tenkan),
                "kijun_sen": float(current_kijun),
                "senkou_span_a": float(current_senkou_a),
                "senkou_span_b": float(current_senkou_b),
                "cloud_top": float(cloud_top),
                "cloud_bottom": float(cloud_bottom),
            },
            signal=signal_type,
            strength=float(strength),
            timestamp=datetime.now(),
        )

    # ==================== Fibonacci Retracement ====================

    def calculate_fibonacci(self, lookback: int = 50) -> IndicatorResult:
        """Calculate Fibonacci Retracement levels with automatic pivot detection.

        Identifies key support/resistance levels based on Fibonacci ratios.

        Args:
            lookback: Period for pivot detection (default: 50)

        Returns:
            IndicatorResult with Fibonacci levels and signal
        """
        recent_data = self.data.tail(lookback)
        swing_high = recent_data["high"].max()
        swing_low = recent_data["low"].min()

        diff = swing_high - swing_low

        # Fibonacci levels
        levels = {
            "0.0": swing_high,
            "0.236": swing_high - (diff * 0.236),
            "0.382": swing_high - (diff * 0.382),
            "0.5": swing_high - (diff * 0.5),
            "0.618": swing_high - (diff * 0.618),
            "0.786": swing_high - (diff * 0.786),
            "1.0": swing_low,
        }

        current_close = self.data["close"].iloc[-1]

        # Find closest level
        closest_level = min(levels.items(), key=lambda x: abs(x[1] - current_close))
        distance_pct = abs(current_close - closest_level[1]) / current_close * 100

        # Signal based on position relative to key levels
        if current_close > levels["0.618"]:
            signal_type = "BULLISH"
            strength = (current_close - levels["0.618"]) / (swing_high - levels["0.618"])
        elif current_close < levels["0.382"]:
            signal_type = "BEARISH"
            strength = (levels["0.382"] - current_close) / (levels["0.382"] - swing_low)
        else:
            signal_type = "NEUTRAL"
            strength = 0.5

        strength = float(np.clip(strength, 0, 1))

        return IndicatorResult(
            name="Fibonacci",
            value={
                "levels": {k: float(v) for k, v in levels.items()},
                "closest_level": closest_level[0],
                "distance_pct": float(distance_pct),
                "swing_high": float(swing_high),
                "swing_low": float(swing_low),
            },
            signal=signal_type,
            strength=strength,
            timestamp=datetime.now(),
        )

    # ==================== EMA Cross ====================

    def calculate_ema_cross(self, fast: int = 9, slow: int = 21) -> IndicatorResult:
        """Calculate EMA Crossover.

        EMA crossovers identify trend changes and entry/exit points.

        Args:
            fast: Fast EMA period (default: 9)
            slow: Slow EMA period (default: 21)

        Returns:
            IndicatorResult with EMA values and signal
        """
        ema_fast = self.data["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = self.data["close"].ewm(span=slow, adjust=False).mean()

        current_fast = ema_fast.iloc[-1]
        current_slow = ema_slow.iloc[-1]
        prev_fast = ema_fast.iloc[-2]
        prev_slow = ema_slow.iloc[-2]

        diff = current_fast - current_slow
        diff_pct = (diff / current_slow) * 100

        # Check for crossover
        crossed_up = prev_fast <= prev_slow and current_fast > current_slow
        crossed_down = prev_fast >= prev_slow and current_fast < current_slow

        # Signal generation
        if crossed_up:
            signal_type = "BULLISH"
            strength = 0.9
        elif crossed_down:
            signal_type = "BEARISH"
            strength = 0.9
        elif current_fast > current_slow:
            signal_type = "BULLISH"
            strength = min(abs(diff_pct) / 2, 0.8)
        elif current_fast < current_slow:
            signal_type = "BEARISH"
            strength = min(abs(diff_pct) / 2, 0.8)
        else:
            signal_type = "NEUTRAL"
            strength = 0.5

        return IndicatorResult(
            name="EMA_Cross",
            value={
                "ema_fast": float(current_fast),
                "ema_slow": float(current_slow),
                "diff_pct": float(diff_pct),
                "crossed_up": crossed_up,
                "crossed_down": crossed_down,
            },
            signal=signal_type,
            strength=float(strength),
            timestamp=datetime.now(),
        )

    # ==================== Volume Profile ====================

    def calculate_volume_profile(self, bins: int = 20) -> IndicatorResult:
        """Calculate Volume Profile.

        Volume Profile shows trading activity at different price levels.

        Args:
            bins: Number of price bins (default: 20)

        Returns:
            IndicatorResult with volume profile and signal
        """
        price_min = self.data["low"].min()
        price_max = self.data["high"].max()

        # Create price bins
        price_bins = np.linspace(price_min, price_max, bins + 1)

        # Assign each row to a bin
        self.data["price_bin"] = pd.cut(
            self.data["close"], bins=price_bins, labels=False, include_lowest=True
        )

        # Calculate volume per bin
        volume_profile = self.data.groupby("price_bin")["volume"].sum()

        # Find Point of Control (POC) - price level with highest volume
        poc_bin = volume_profile.idxmax()
        poc_price = (price_bins[int(poc_bin)] + price_bins[int(poc_bin) + 1]) / 2

        current_close = self.data["close"].iloc[-1]

        # Value Area (70% of volume)
        total_volume = volume_profile.sum()
        sorted_profile = volume_profile.sort_values(ascending=False)
        cumsum = 0
        value_area_bins = []

        for bin_idx, vol in sorted_profile.items():
            cumsum += vol
            value_area_bins.append(int(bin_idx))
            if cumsum >= total_volume * 0.7:
                break

        value_area_high = price_bins[max(value_area_bins) + 1]
        value_area_low = price_bins[min(value_area_bins)]

        # Signal generation
        if current_close > value_area_high:
            signal_type = "BULLISH"
            strength = min((current_close - value_area_high) / value_area_high, 1.0)
        elif current_close < value_area_low:
            signal_type = "BEARISH"
            strength = min((value_area_low - current_close) / value_area_low, 1.0)
        else:
            signal_type = "NEUTRAL"
            strength = 0.5

        return IndicatorResult(
            name="Volume_Profile",
            value={
                "poc_price": float(poc_price),
                "value_area_high": float(value_area_high),
                "value_area_low": float(value_area_low),
                "current_vs_poc": float((current_close - poc_price) / poc_price * 100),
            },
            signal=signal_type,
            strength=float(strength),
            timestamp=datetime.now(),
        )

    # ==================== Composite Analysis ====================

    def calculate_all(self) -> dict[str, IndicatorResult]:
        """Calculate all 14 technical indicators.

        Returns:
            Dictionary mapping indicator names to their results
        """
        indicators = {
            "RSI": self.calculate_rsi(),
            "MACD": self.calculate_macd(),
            "Bollinger_Bands": self.calculate_bollinger_bands(),
            "ATR": self.calculate_atr(),
            "ADX": self.calculate_adx(),
            "Stochastic": self.calculate_stochastic(),
            "CCI": self.calculate_cci(),
            "Williams_R": self.calculate_williams_r(),
            "OBV": self.calculate_obv(),
            "VWAP": self.calculate_vwap(),
            "Ichimoku": self.calculate_ichimoku(),
            "Fibonacci": self.calculate_fibonacci(),
            "EMA_Cross": self.calculate_ema_cross(),
            "Volume_Profile": self.calculate_volume_profile(),
        }

        return indicators

    def generate_composite_signal(self, weights: dict[str, float] | None = None) -> CompositeSignal:
        """Generate weighted composite signal from all indicators.

        Args:
            weights: Optional custom weights for each indicator. If None, uses equal weights.
                    Keys are indicator names, values are weights (will be normalized).

        Returns:
            CompositeSignal aggregating all individual signals
        """
        indicators = self.calculate_all()

        # Default equal weights
        if weights is None:
            weights = dict.fromkeys(indicators.keys(), 1.0)

        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        # Count signals
        bullish_count = sum(1 for ind in indicators.values() if ind.signal == "BULLISH")
        bearish_count = sum(1 for ind in indicators.values() if ind.signal == "BEARISH")
        neutral_count = sum(1 for ind in indicators.values() if ind.signal == "NEUTRAL")

        # Calculate weighted score
        score = 0.0
        for name, indicator in indicators.items():
            weight = normalized_weights.get(name, 0)
            if indicator.signal == "BULLISH":
                score += weight * indicator.strength
            elif indicator.signal == "BEARISH":
                score -= weight * indicator.strength
            # NEUTRAL contributes 0

        # Determine overall signal
        if score > 0.15:
            overall_signal = "BULLISH"
        elif score < -0.15:
            overall_signal = "BEARISH"
        else:
            overall_signal = "NEUTRAL"

        # Calculate confidence (absolute value of score)
        confidence = float(np.clip(abs(score), 0, 1))

        return CompositeSignal(
            overall_signal=overall_signal,
            confidence=confidence,
            bullish_indicators=bullish_count,
            bearish_indicators=bearish_count,
            neutral_indicators=neutral_count,
            individual_signals=indicators,
            timestamp=datetime.now(),
        )

    # ==================== Helper Methods ====================

    def detect_trend(self, period: int = 20) -> tuple[str, float]:
        """Detect current price trend.

        Args:
            period: Lookback period for trend analysis

        Returns:
            Tuple of (trend_direction, trend_strength)
        """
        close = self.data["close"]
        sma = close.rolling(window=period).mean()

        current_close = close.iloc[-1]
        current_sma = sma.iloc[-1]

        # Linear regression slope
        y = close.tail(period).values
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)

        # Normalize slope
        slope_pct = (slope / current_close) * 100

        if slope_pct > 0.1 and current_close > current_sma:
            trend = "UPTREND"
            strength = min(abs(slope_pct) / 2, 1.0)
        elif slope_pct < -0.1 and current_close < current_sma:
            trend = "DOWNTREND"
            strength = min(abs(slope_pct) / 2, 1.0)
        else:
            trend = "SIDEWAYS"
            strength = 1.0 - min(abs(slope_pct) / 0.1, 1.0)

        return trend, float(strength)

    def find_support_resistance(
        self, window: int = 20, tolerance: float = 0.02
    ) -> dict[str, list[float]]:
        """Find support and resistance levels.

        Args:
            window: Window size for pivot detection
            tolerance: Price tolerance for level clustering (as fraction)

        Returns:
            Dictionary with 'support' and 'resistance' level lists
        """
        highs = self.data["high"].rolling(window=window, center=True).max()
        lows = self.data["low"].rolling(window=window, center=True).min()

        # Find pivot highs (resistance)
        resistance_pivots = self.data[self.data["high"] == highs]["high"].values

        # Find pivot lows (support)
        support_pivots = self.data[self.data["low"] == lows]["low"].values

        # Cluster nearby levels
        def cluster_levels(levels: np.ndarray, tolerance: float) -> list[float]:
            if len(levels) == 0:
                return []

            sorted_levels = np.sort(levels)
            clusters = []
            current_cluster = [sorted_levels[0]]

            for level in sorted_levels[1:]:
                if level - current_cluster[-1] <= current_cluster[-1] * tolerance:
                    current_cluster.append(level)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [level]

            clusters.append(np.mean(current_cluster))
            return [float(c) for c in clusters]

        support_levels = cluster_levels(support_pivots, tolerance)
        resistance_levels = cluster_levels(resistance_pivots, tolerance)

        return {
            "support": sorted(support_levels)[-5:],  # Keep top 5
            "resistance": sorted(resistance_levels)[-5:],  # Keep top 5
        }
