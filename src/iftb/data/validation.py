"""
Data validation module for OHLCV data quality checks.

This module provides comprehensive validation of OHLCV (Open, High, Low, Close, Volume)
data to ensure data quality meets trading requirements. It detects missing candles,
outliers, gaps, OHLC integrity violations, and provides automated fixing capabilities.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from iftb.utils import get_logger

logger = get_logger(__name__)


@dataclass
class DataQualityReport:
    """
    Report containing data quality metrics and validation results.

    Attributes:
        total_rows: Total number of rows in the dataset
        valid_rows: Number of rows that passed all validation checks
        missing_candles: Count of missing candles in the time series
        outliers_detected: Number of outliers detected using statistical methods
        gaps_detected: Number of significant price gaps detected
        duplicate_timestamps: Number of duplicate timestamp entries
        quality_score: Overall quality score from 0-100 (100 = perfect)
        issues: List of human-readable issue descriptions
        is_acceptable: Whether the data quality meets minimum standards
    """
    total_rows: int
    valid_rows: int
    missing_candles: int
    outliers_detected: int
    gaps_detected: int
    duplicate_timestamps: int
    quality_score: float
    issues: list[str] = field(default_factory=list)
    is_acceptable: bool = False

    def __str__(self) -> str:
        """Generate a human-readable summary of the quality report."""
        return (
            f"Data Quality Report:\n"
            f"  Total Rows: {self.total_rows}\n"
            f"  Valid Rows: {self.valid_rows}\n"
            f"  Quality Score: {self.quality_score:.2f}%\n"
            f"  Missing Candles: {self.missing_candles}\n"
            f"  Outliers: {self.outliers_detected}\n"
            f"  Gaps: {self.gaps_detected}\n"
            f"  Duplicates: {self.duplicate_timestamps}\n"
            f"  Status: {'ACCEPTABLE' if self.is_acceptable else 'NEEDS ATTENTION'}\n"
            f"  Issues: {len(self.issues)}"
        )


class OHLCVValidator:
    """
    Validator for OHLCV data quality checks and automated fixes.

    This class performs comprehensive validation of OHLCV data including:
    - Missing candle detection
    - OHLC integrity checks (High >= Low, etc.)
    - Statistical outlier detection
    - Price gap detection
    - Duplicate timestamp detection

    It also provides automated fixing capabilities for common issues.

    Attributes:
        outlier_zscore_threshold: Z-score threshold for outlier detection (default: 4.0)
        max_single_candle_change: Maximum allowed price change per candle (default: 0.20)
        volume_outlier_multiplier: Volume outlier detection multiplier (default: 10.0)
        min_quality_score: Minimum acceptable quality score (default: 95.0)
    """

    def __init__(
        self,
        outlier_zscore_threshold: float = 4.0,
        max_single_candle_change: float = 0.20,
        volume_outlier_multiplier: float = 10.0,
        min_quality_score: float = 95.0
    ):
        """
        Initialize the OHLCV validator with configuration parameters.

        Args:
            outlier_zscore_threshold: Z-score threshold for statistical outlier detection
            max_single_candle_change: Maximum allowed fractional price change per candle
            volume_outlier_multiplier: Volume values this many times above median are outliers
            min_quality_score: Minimum quality score (0-100) for acceptable data
        """
        self.outlier_zscore_threshold = outlier_zscore_threshold
        self.max_single_candle_change = max_single_candle_change
        self.volume_outlier_multiplier = volume_outlier_multiplier
        self.min_quality_score = min_quality_score

        logger.info(
            f"OHLCVValidator initialized with zscore_threshold={outlier_zscore_threshold}, "
            f"max_change={max_single_candle_change}, min_quality={min_quality_score}"
        )

    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Perform comprehensive validation of OHLCV data.

        Args:
            df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
               timestamp should be datetime64[ns] or convertible to datetime

        Returns:
            DataQualityReport containing all validation metrics and issues

        Raises:
            ValueError: If required columns are missing or data is empty
        """
        logger.info(f"Starting validation of {len(df)} rows")

        # Validate input
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if df.empty:
            raise ValueError("DataFrame is empty")

        # Make a copy to avoid modifying original
        df = df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        issues = []
        total_rows = len(df)

        # Check for duplicate timestamps
        duplicate_timestamps = df['timestamp'].duplicated().sum()
        if duplicate_timestamps > 0:
            issues.append(f"Found {duplicate_timestamps} duplicate timestamps")
            logger.warning(f"Detected {duplicate_timestamps} duplicate timestamps")

        # Detect expected interval (most common time difference)
        time_diffs = df['timestamp'].diff().dropna()
        if len(time_diffs) > 0:
            mode_interval = time_diffs.mode()
            expected_interval = mode_interval[0] if not mode_interval.empty else time_diffs.median()
        else:
            expected_interval = timedelta(minutes=1)  # Default fallback

        # Detect missing candles
        missing_candles = self._detect_missing_candles(df, expected_interval)
        if missing_candles > 0:
            issues.append(f"Found {missing_candles} missing candles")
            logger.warning(f"Detected {missing_candles} missing candles")

        # Check OHLC integrity
        ohlc_violations = self._check_ohlc_integrity(df)
        if ohlc_violations > 0:
            issues.append(f"Found {ohlc_violations} OHLC integrity violations")
            logger.warning(f"Detected {ohlc_violations} OHLC integrity violations")

        # Detect outliers
        outliers_detected = self._detect_outliers(df)
        if outliers_detected > 0:
            issues.append(f"Found {outliers_detected} outliers")
            logger.warning(f"Detected {outliers_detected} outliers")

        # Detect gaps
        gaps_detected = self._detect_gaps(df)
        if gaps_detected > 0:
            issues.append(f"Found {gaps_detected} significant price gaps")
            logger.warning(f"Detected {gaps_detected} price gaps")

        # Calculate valid rows (rows without any issues)
        valid_rows = total_rows - ohlc_violations - outliers_detected

        # Calculate quality score
        # Base score from valid rows ratio
        base_score = (valid_rows / total_rows) * 100 if total_rows > 0 else 0

        # Penalize for various issues (normalized by dataset size)
        missing_penalty = (missing_candles / max(total_rows, 1)) * 10
        gap_penalty = (gaps_detected / max(total_rows, 1)) * 5
        duplicate_penalty = (duplicate_timestamps / max(total_rows, 1)) * 10

        quality_score = max(0, base_score - missing_penalty - gap_penalty - duplicate_penalty)

        is_acceptable = quality_score >= self.min_quality_score and len(issues) == 0

        report = DataQualityReport(
            total_rows=total_rows,
            valid_rows=valid_rows,
            missing_candles=missing_candles,
            outliers_detected=outliers_detected,
            gaps_detected=gaps_detected,
            duplicate_timestamps=duplicate_timestamps,
            quality_score=quality_score,
            issues=issues,
            is_acceptable=is_acceptable
        )

        logger.info(
            f"Validation complete: Quality Score = {quality_score:.2f}%, "
            f"Acceptable = {is_acceptable}"
        )

        return report

    def _detect_missing_candles(self, df: pd.DataFrame, expected_interval: timedelta) -> int:
        """
        Detect missing candles in the time series based on expected interval.

        Args:
            df: DataFrame with timestamp column
            expected_interval: Expected time difference between consecutive candles

        Returns:
            Number of missing candles detected
        """
        if len(df) < 2:
            return 0

        time_diffs = df['timestamp'].diff().dropna()

        # Allow 10% tolerance for interval matching
        tolerance = expected_interval * 0.1
        lower_bound = expected_interval - tolerance
        upper_bound = expected_interval + tolerance

        # Count gaps larger than expected interval
        missing_count = 0
        for diff in time_diffs:
            if diff > upper_bound:
                # Calculate how many candles are missing
                missing = int((diff - expected_interval) / expected_interval)
                missing_count += missing

        return missing_count

    def _check_ohlc_integrity(self, df: pd.DataFrame) -> int:
        """
        Check OHLC integrity rules: High >= Low, High >= Open, High >= Close, etc.

        Args:
            df: DataFrame with open, high, low, close columns

        Returns:
            Number of integrity violations detected
        """
        violations = 0

        # High must be >= Low
        violations += (df['high'] < df['low']).sum()

        # High must be >= Open
        violations += (df['high'] < df['open']).sum()

        # High must be >= Close
        violations += (df['high'] < df['close']).sum()

        # Low must be <= Open
        violations += (df['low'] > df['open']).sum()

        # Low must be <= Close
        violations += (df['low'] > df['close']).sum()

        # Volume must be non-negative
        violations += (df['volume'] < 0).sum()

        # Check for NaN values
        violations += df[['open', 'high', 'low', 'close', 'volume']].isna().sum().sum()

        return int(violations)

    def _detect_outliers(self, df: pd.DataFrame) -> int:
        """
        Detect outliers using Z-score method and sudden change detection.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            Number of outliers detected
        """
        if len(df) < 3:
            return 0

        outliers = 0

        # Z-score based outlier detection for prices
        for col in ['open', 'high', 'low', 'close']:
            if df[col].std() > 0:  # Avoid division by zero
                z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
                outliers += (z_scores > self.outlier_zscore_threshold).sum()

        # Sudden change detection (price jumps)
        df['price_change'] = df['close'].pct_change().abs()
        sudden_changes = (df['price_change'] > self.max_single_candle_change).sum()
        outliers += sudden_changes

        # Volume outliers (extreme volume spikes)
        median_volume = df['volume'].median()
        if median_volume > 0:
            volume_outliers = (df['volume'] > median_volume * self.volume_outlier_multiplier).sum()
            outliers += volume_outliers

        return int(outliers)

    def _detect_gaps(self, df: pd.DataFrame, gap_threshold: float = 0.02) -> int:
        """
        Detect significant price gaps between consecutive candles.

        A gap is defined as a significant difference between the close of one candle
        and the open of the next candle.

        Args:
            df: DataFrame with open and close columns
            gap_threshold: Threshold for gap detection as fraction (default: 0.02 = 2%)

        Returns:
            Number of significant gaps detected
        """
        if len(df) < 2:
            return 0

        # Calculate gap between close[i] and open[i+1]
        df['next_open'] = df['open'].shift(-1)
        df['gap'] = ((df['next_open'] - df['close']) / df['close']).abs()

        gaps = (df['gap'] > gap_threshold).sum()

        return int(gaps)

    def fix_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically fix detected issues in the OHLCV data.

        Fixes applied:
        - Remove duplicate timestamps (keep first occurrence)
        - Interpolate missing candles (linear interpolation)
        - Fix OHLC integrity violations (adjust values to maintain rules)
        - Clip extreme outliers to ±3 standard deviations

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Fixed DataFrame with same structure
        """
        logger.info(f"Starting automatic fixes on {len(df)} rows")

        # Make a copy to avoid modifying original
        df = df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # 1. Remove duplicates (keep first)
        original_len = len(df)
        df = df.drop_duplicates(subset='timestamp', keep='first')
        removed = original_len - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate timestamps")

        # 2. Fix OHLC integrity violations
        # Ensure High is the maximum
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)

        # Ensure Low is the minimum
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

        # Ensure non-negative volume
        df['volume'] = df['volume'].clip(lower=0)

        # Fill NaN values with forward fill, then backward fill
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        df[ohlcv_cols] = df[ohlcv_cols].ffill().bfill()

        logger.info("Fixed OHLC integrity violations")

        # 3. Clip extreme outliers (±3 standard deviations)
        for col in ['open', 'high', 'low', 'close']:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                original_values = df[col].copy()
                df[col] = df[col].clip(lower_bound, upper_bound)
                clipped = (original_values != df[col]).sum()
                if clipped > 0:
                    logger.info(f"Clipped {clipped} outliers in {col} column")

        # 4. Interpolate missing candles
        # Detect expected interval
        time_diffs = df['timestamp'].diff().dropna()
        if len(time_diffs) > 0:
            mode_interval = time_diffs.mode()
            expected_interval = mode_interval[0] if not mode_interval.empty else time_diffs.median()

            # Create complete time range
            start_time = df['timestamp'].min()
            end_time = df['timestamp'].max()
            complete_range = pd.date_range(start=start_time, end=end_time, freq=expected_interval)

            # Reindex to complete range
            df = df.set_index('timestamp')
            df = df.reindex(complete_range)

            # Interpolate missing values
            original_na = df.isna().sum().sum()
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            df[ohlcv_cols] = df[ohlcv_cols].interpolate(method='linear')
            df = df.ffill().bfill()  # Fill any remaining NaN at edges
            filled = original_na - df.isna().sum().sum()
            if filled > 0:
                logger.info(f"Interpolated {int(filled / 5)} missing candles")

            # Reset index
            df = df.reset_index()
            df.rename(columns={'index': 'timestamp'}, inplace=True)

        logger.info(f"Automatic fixes complete. Final row count: {len(df)}")

        return df


def calculate_data_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate comprehensive statistics for OHLCV data.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Dictionary containing statistical metrics:
        - row_count: Total number of rows
        - time_range: Tuple of (start_time, end_time)
        - price_stats: Statistics for each price column (mean, std, min, max, median)
        - volume_stats: Volume statistics
        - completeness: Percentage of non-null values

    Raises:
        ValueError: If required columns are missing
    """
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        return {
            'row_count': 0,
            'time_range': (None, None),
            'price_stats': {},
            'volume_stats': {},
            'completeness': 0.0
        }

    # Ensure timestamp is datetime
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])

    # Calculate statistics for price columns
    price_stats = {}
    for col in ['open', 'high', 'low', 'close']:
        price_stats[col] = {
            'mean': float(df_copy[col].mean()),
            'std': float(df_copy[col].std()),
            'min': float(df_copy[col].min()),
            'max': float(df_copy[col].max()),
            'median': float(df_copy[col].median()),
            'q25': float(df_copy[col].quantile(0.25)),
            'q75': float(df_copy[col].quantile(0.75))
        }

    # Calculate volume statistics
    volume_stats = {
        'mean': float(df_copy['volume'].mean()),
        'std': float(df_copy['volume'].std()),
        'min': float(df_copy['volume'].min()),
        'max': float(df_copy['volume'].max()),
        'median': float(df_copy['volume'].median()),
        'total': float(df_copy['volume'].sum()),
        'q25': float(df_copy['volume'].quantile(0.25)),
        'q75': float(df_copy['volume'].quantile(0.75))
    }

    # Calculate completeness (percentage of non-null values)
    total_cells = len(df_copy) * len(required_cols)
    non_null_cells = df_copy[required_cols].notna().sum().sum()
    completeness = (non_null_cells / total_cells * 100) if total_cells > 0 else 0

    # Time range
    time_range = (
        df_copy['timestamp'].min().isoformat() if pd.notna(df_copy['timestamp'].min()) else None,
        df_copy['timestamp'].max().isoformat() if pd.notna(df_copy['timestamp'].max()) else None
    )

    # Calculate time interval statistics
    time_diffs = df_copy['timestamp'].diff().dropna()
    interval_stats = {}
    if len(time_diffs) > 0:
        interval_stats = {
            'mean_seconds': time_diffs.mean().total_seconds(),
            'median_seconds': time_diffs.median().total_seconds(),
            'min_seconds': time_diffs.min().total_seconds(),
            'max_seconds': time_diffs.max().total_seconds()
        }

    return {
        'row_count': len(df_copy),
        'time_range': time_range,
        'price_stats': price_stats,
        'volume_stats': volume_stats,
        'completeness': float(completeness),
        'interval_stats': interval_stats
    }
