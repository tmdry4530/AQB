# IFTB (AI 통합 선물 퀀트 봇) 작업 계획서

**작성일**: 2026-01-17
**버전**: v1.1 (리뷰 반영)
**상태**: Draft
**접근 방식**: 3-Layer 병렬 구현 + 보수적 테스트 + 전체 DevOps

---

## 1. 프로젝트 개요

### 1.1 목표
- LLM + 기술분석 + XGBoost 3-Layer 시스템 구축
- 바이낸스 BTC/ETH 선물 자동 매매
- **리스크 조정 수익률 극대화** (Sharpe Ratio ≥ 1.5 목표)

### 1.2 성능 목표 (통계적 정의) [C1 수정]

| 지표 | 목표 | 통계적 검증 기준 |
|------|------|------------------|
| 승률 | ≥60% | n≥500 거래, 95% CI [57%, 63%] |
| 유의성 | p<0.05 | χ² 검정 vs 무작위 (50%) |
| Sharpe Ratio | ≥1.5 | 연환산, 252 거래일 기준 |
| Sortino Ratio | ≥2.0 | 하방 변동성만 고려 |
| Profit Factor | ≥1.8 | 총이익/총손실 |
| 최대 MDD | ≤30% | 롤링 12개월 기준 |

**검증 방법:**
```python
# 승률 신뢰구간 계산
from scipy import stats
import numpy as np

def validate_win_rate(wins: int, total: int, target: float = 0.60):
    """
    승률 통계적 검증
    - n ≥ 500 거래 필수
    - 95% 신뢰구간 하한 ≥ target 확인
    - χ² 검정으로 무작위 대비 유의성 검증
    """
    if total < 500:
        raise ValueError(f"최소 500 거래 필요. 현재: {total}")

    # 점 추정
    win_rate = wins / total

    # 95% 신뢰구간 (Wilson score interval)
    ci_low, ci_high = stats.proportion_confint(wins, total, alpha=0.05, method='wilson')

    # χ² 검정 vs 무작위 (50%)
    expected = [total * 0.5, total * 0.5]
    observed = [wins, total - wins]
    chi2, p_value = stats.chisquare(observed, expected)

    return {
        'win_rate': win_rate,
        'ci_95': (ci_low, ci_high),
        'chi2_stat': chi2,
        'p_value': p_value,
        'statistically_significant': p_value < 0.05 and ci_low >= target
    }
```

### 1.3 핵심 결정 사항

| 항목 | 결정 | 근거 |
|------|------|------|
| Python 환경 | uv | 빠른 의존성 해결, 모던 툴체인 |
| 웹 프레임워크 | FastAPI | 비동기 지원, 타입 힌트 |
| 데이터베이스 | PostgreSQL + Redis | 영속성 + 실시간 캐싱 |
| 테스트 | pytest + hypothesis | 속성 기반 테스트 포함 |
| CI/CD | GitHub Actions | PR 검증, 자동 배포 |
| 인프라 | AWS EC2 + Docker | 컨테이너 기반 배포 |
| 모니터링 | Prometheus + Grafana | 메트릭 수집 및 시각화 |

---

## 2. 리스크 관리 명세 (강화)

### 2.1 포지션 사이징 상수 [C2 수정]

```python
# risk/constants.py

# Kelly Criterion 제한
KELLY_FRACTION = 0.25          # Quarter-Kelly (풀 Kelly의 25%)
MAX_POSITION_PCT = 0.10        # 단일 포지션 최대 10%
MIN_POSITION_PCT = 0.02        # 최소 포지션 2%

# 레버리지 제한
MAX_LEVERAGE = 8               # 절대 최대 레버리지
DEFAULT_LEVERAGE = 5           # 기본 레버리지
MIN_LEVERAGE = 2               # 최소 레버리지
HIGH_CONFIDENCE_LEVERAGE = 7   # 고확신 시 레버리지

# 자본 보호
MAX_DAILY_LOSS_PCT = 0.08      # 일일 최대 손실 8%
MAX_WEEKLY_LOSS_PCT = 0.15     # 주간 최대 손실 15%
MARGIN_CALL_THRESHOLD = 0.20   # 마진률 20% 이하 시 경고
EMERGENCY_EXIT_MARGIN = 0.10   # 마진률 10% 이하 시 강제 청산
```

### 2.2 회로 차단기 (Circuit Breaker) [C3 수정]

```python
# risk/circuit_breaker.py

from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

class BreakerState(Enum):
    CLOSED = "closed"      # 정상 운영
    OPEN = "open"          # 거래 중단
    HALF_OPEN = "half_open"  # 제한적 운영

@dataclass
class CircuitBreakerConfig:
    """회로 차단기 설정"""

    # 일일 손실 한도
    daily_loss_limit_pct: float = 0.08  # 8%

    # 연속 손실 한도
    consecutive_loss_limit: int = 5

    # 시간당 거래 한도
    hourly_trade_limit: int = 3

    # 쿨다운 시간
    cooldown_after_daily_limit: timedelta = timedelta(hours=24)
    cooldown_after_consecutive_loss: timedelta = timedelta(hours=12)

    # 복구 조건
    half_open_trade_limit: int = 1  # 반개방 시 1회만 허용
    recovery_win_required: int = 2   # 정상 복귀에 2연승 필요


class CircuitBreaker:
    """회로 차단기 구현"""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = BreakerState.CLOSED
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.hourly_trades = 0
        self.last_trade_time = None
        self.breaker_triggered_at = None
        self.recovery_wins = 0

    def can_trade(self) -> tuple[bool, str]:
        """거래 가능 여부 확인"""

        # 강제 중단 상태
        if self.state == BreakerState.OPEN:
            remaining = self._get_remaining_cooldown()
            if remaining > timedelta(0):
                return False, f"회로 차단 중. 남은 시간: {remaining}"
            else:
                self._transition_to_half_open()

        # 반개방 상태
        if self.state == BreakerState.HALF_OPEN:
            if self.hourly_trades >= self.config.half_open_trade_limit:
                return False, "반개방 상태: 거래 한도 도달"

        # 일일 손실 한도
        if self.daily_pnl <= -self.config.daily_loss_limit_pct:
            self._trigger_breaker("daily_loss")
            return False, f"일일 손실 한도 도달: {self.daily_pnl:.2%}"

        # 연속 손실 한도
        if self.consecutive_losses >= self.config.consecutive_loss_limit:
            self._trigger_breaker("consecutive_loss")
            return False, f"연속 {self.consecutive_losses}회 손실"

        # 시간당 거래 한도
        if self.hourly_trades >= self.config.hourly_trade_limit:
            return False, "시간당 거래 한도 도달"

        return True, "OK"

    def record_trade(self, pnl_pct: float):
        """거래 결과 기록"""
        self.daily_pnl += pnl_pct
        self.hourly_trades += 1
        self.last_trade_time = datetime.now()

        if pnl_pct < 0:
            self.consecutive_losses += 1
            self.recovery_wins = 0
        else:
            self.consecutive_losses = 0
            self.recovery_wins += 1

            # 반개방 → 정상 복귀
            if (self.state == BreakerState.HALF_OPEN and
                self.recovery_wins >= self.config.recovery_win_required):
                self.state = BreakerState.CLOSED

    def emergency_stop(self, reason: str):
        """수동 긴급 정지"""
        self.state = BreakerState.OPEN
        self.breaker_triggered_at = datetime.now()
        # 무기한 정지 (수동 해제 필요)
        return f"긴급 정지 활성화: {reason}"

    def manual_reset(self, operator: str):
        """수동 리셋 (감사 로깅)"""
        self.state = BreakerState.CLOSED
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.hourly_trades = 0
        # 감사 로그
        return f"회로 차단기 리셋 by {operator} at {datetime.now()}"
```

### 2.3 킬 스위치 [C3 수정]

```python
# risk/kill_switch.py

class KillSwitch:
    """긴급 정지 스위치"""

    def __init__(self, telegram_bot, exchange_client):
        self.telegram = telegram_bot
        self.exchange = exchange_client
        self.is_active = False
        self.activation_reason = None
        self.activation_time = None

    async def activate(self, reason: str, close_positions: bool = True):
        """
        킬 스위치 활성화
        1. 모든 신규 주문 차단
        2. 미체결 주문 취소
        3. (선택) 모든 포지션 청산
        4. 텔레그램 알림
        """
        self.is_active = True
        self.activation_reason = reason
        self.activation_time = datetime.now()

        # 미체결 주문 전체 취소
        await self.exchange.cancel_all_orders()

        # 포지션 청산 (선택)
        if close_positions:
            positions = await self.exchange.get_positions()
            for pos in positions:
                await self.exchange.close_position(pos.symbol, "market")

        # 긴급 알림
        await self.telegram.send_alert(
            level="CRITICAL",
            message=f"🚨 킬 스위치 활성화\n"
                   f"사유: {reason}\n"
                   f"시간: {self.activation_time}\n"
                   f"포지션 청산: {close_positions}"
        )

        return True

    async def deactivate(self, operator: str, confirmation_code: str):
        """킬 스위치 비활성화 (2단계 인증)"""
        if confirmation_code != self._generate_confirmation_code():
            raise ValueError("확인 코드 불일치")

        self.is_active = False

        await self.telegram.send_alert(
            level="INFO",
            message=f"킬 스위치 비활성화 by {operator}"
        )
```

### 2.4 LLM 거부권 로직 [C4 수정]

```python
# llm/veto.py

from dataclasses import dataclass
from enum import Enum

class VetoDecision(Enum):
    APPROVE = "approve"           # 승인
    SOFT_VETO = "soft_veto"       # 포지션 축소
    HARD_VETO = "hard_veto"       # 완전 거부

@dataclass
class LLMVetoConfig:
    """LLM 거부권 설정"""

    # 감정 점수 임계값
    sentiment_veto_threshold: float = -0.5    # 이하면 hard veto
    sentiment_caution_threshold: float = -0.2  # 이하면 soft veto

    # 확신도 임계값
    confidence_veto_threshold: float = 0.3    # 이하면 hard veto
    confidence_caution_threshold: float = 0.5  # 이하면 soft veto

    # 뉴스 정렬 가중치
    news_conflict_penalty: float = 0.5  # 뉴스 충돌 시 포지션 50% 감소

    # API 장애 시 동작
    fallback_mode: str = "conservative"  # conservative | technical_only | halt


class LLMVetoSystem:
    """LLM 거부권 시스템"""

    def __init__(self, config: LLMVetoConfig):
        self.config = config
        self.api_failures = 0
        self.max_consecutive_failures = 3

    def evaluate_signal(
        self,
        signal_direction: str,  # "LONG" | "SHORT"
        llm_analysis: dict | None,  # LLM 응답 (None if API 실패)
    ) -> tuple[VetoDecision, float, str]:
        """
        신호 평가 및 거부권 결정

        Returns:
            (결정, 포지션 비율 조정, 사유)
        """

        # API 장애 시 폴백
        if llm_analysis is None:
            return self._handle_api_failure(signal_direction)

        # API 복구
        self.api_failures = 0

        sentiment = llm_analysis.get("sentiment_score", 0)
        confidence = llm_analysis.get("confidence", 0.5)
        trade_env = llm_analysis.get("trade_environment", {})
        news_alignment = llm_analysis.get("news_alignment", "neutral")

        # 1. 방향성 거부 (Hard Veto)
        if signal_direction == "LONG" and not trade_env.get("long_allowed", True):
            return VetoDecision.HARD_VETO, 0, trade_env.get("reason", "롱 금지")

        if signal_direction == "SHORT" and not trade_env.get("short_allowed", True):
            return VetoDecision.HARD_VETO, 0, trade_env.get("reason", "숏 금지")

        # 2. 감정 점수 기반 거부
        if sentiment <= self.config.sentiment_veto_threshold:
            return VetoDecision.HARD_VETO, 0, f"부정적 감정: {sentiment:.2f}"

        if sentiment <= self.config.sentiment_caution_threshold:
            return VetoDecision.SOFT_VETO, 0.5, f"주의 감정: {sentiment:.2f}"

        # 3. 확신도 기반 거부
        if confidence <= self.config.confidence_veto_threshold:
            return VetoDecision.HARD_VETO, 0, f"낮은 확신도: {confidence:.2f}"

        if confidence <= self.config.confidence_caution_threshold:
            return VetoDecision.SOFT_VETO, 0.7, f"중간 확신도: {confidence:.2f}"

        # 4. 뉴스 충돌 감지
        position_multiplier = 1.0
        if news_alignment == "conflicting":
            position_multiplier = 1 - self.config.news_conflict_penalty
            return VetoDecision.SOFT_VETO, position_multiplier, "뉴스-신호 충돌"

        # 5. 승인
        return VetoDecision.APPROVE, 1.0, "승인"

    def _handle_api_failure(self, signal_direction: str):
        """API 장애 시 폴백 처리"""
        self.api_failures += 1

        if self.api_failures >= self.max_consecutive_failures:
            mode = self.config.fallback_mode

            if mode == "halt":
                return VetoDecision.HARD_VETO, 0, "LLM API 연속 실패: 거래 중단"

            elif mode == "conservative":
                # 보수적: 50% 포지션만
                return VetoDecision.SOFT_VETO, 0.5, "LLM API 실패: 보수적 모드"

            elif mode == "technical_only":
                # 기술 신호만으로 진행
                return VetoDecision.APPROVE, 0.7, "LLM API 실패: 기술 신호만 사용"

        # 단일 실패: 보수적 진행
        return VetoDecision.SOFT_VETO, 0.7, "LLM API 일시 실패"
```

---

## 3. 데이터 품질 관리 [C5 수정]

### 3.1 데이터 검증 파이프라인

```python
# data/validation.py

from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Tuple

@dataclass
class DataQualityReport:
    """데이터 품질 리포트"""
    total_rows: int
    valid_rows: int
    missing_candles: int
    outliers_detected: int
    gaps_detected: int
    duplicate_timestamps: int
    quality_score: float  # 0-100
    issues: List[str]
    is_acceptable: bool


class OHLCVValidator:
    """OHLCV 데이터 검증기"""

    def __init__(self):
        # 이상치 기준 (Z-score)
        self.outlier_zscore_threshold = 4.0

        # 가격 변동 한계 (단일 캔들)
        self.max_single_candle_change = 0.20  # 20%

        # 볼륨 이상치 배수
        self.volume_outlier_multiplier = 10.0

        # 최소 품질 점수
        self.min_quality_score = 95.0

    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        """전체 데이터 검증"""
        issues = []

        # 1. 필수 컬럼 확인
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = set(required) - set(df.columns)
        if missing_cols:
            issues.append(f"누락 컬럼: {missing_cols}")

        # 2. 타임스탬프 검증
        dup_timestamps = df['timestamp'].duplicated().sum()
        if dup_timestamps > 0:
            issues.append(f"중복 타임스탬프: {dup_timestamps}개")

        # 3. 누락 캔들 감지
        missing_candles = self._detect_missing_candles(df)
        if missing_candles > 0:
            issues.append(f"누락 캔들: {missing_candles}개")

        # 4. OHLC 무결성
        ohlc_violations = self._check_ohlc_integrity(df)
        if ohlc_violations > 0:
            issues.append(f"OHLC 무결성 위반: {ohlc_violations}개")

        # 5. 이상치 감지
        outliers = self._detect_outliers(df)
        if outliers > 0:
            issues.append(f"이상치: {outliers}개")

        # 6. 갭 감지
        gaps = self._detect_gaps(df)
        if gaps > 0:
            issues.append(f"가격 갭: {gaps}개")

        # 품질 점수 계산
        total = len(df)
        problems = dup_timestamps + missing_candles + ohlc_violations + outliers
        quality_score = max(0, (1 - problems / total) * 100)

        return DataQualityReport(
            total_rows=total,
            valid_rows=total - problems,
            missing_candles=missing_candles,
            outliers_detected=outliers,
            gaps_detected=gaps,
            duplicate_timestamps=dup_timestamps,
            quality_score=quality_score,
            issues=issues,
            is_acceptable=quality_score >= self.min_quality_score
        )

    def _detect_missing_candles(self, df: pd.DataFrame) -> int:
        """누락 캔들 감지"""
        df = df.sort_values('timestamp')
        expected_interval = pd.Timedelta(hours=1)  # 1시간봉

        time_diffs = df['timestamp'].diff()
        missing = (time_diffs > expected_interval * 1.5).sum()

        return int(missing)

    def _check_ohlc_integrity(self, df: pd.DataFrame) -> int:
        """OHLC 무결성 검사: high >= low, high >= open/close, low <= open/close"""
        violations = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ).sum()

        return int(violations)

    def _detect_outliers(self, df: pd.DataFrame) -> int:
        """가격 이상치 감지 (Z-score + 급변동)"""
        # Z-score 기반
        returns = df['close'].pct_change()
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        zscore_outliers = (z_scores > self.outlier_zscore_threshold).sum()

        # 급변동 기반
        sudden_change = (np.abs(returns) > self.max_single_candle_change).sum()

        return int(zscore_outliers + sudden_change)

    def _detect_gaps(self, df: pd.DataFrame) -> int:
        """가격 갭 감지 (이전 캔들과 겹치지 않음)"""
        gaps = 0
        for i in range(1, len(df)):
            prev_low, prev_high = df.iloc[i-1]['low'], df.iloc[i-1]['high']
            curr_low, curr_high = df.iloc[i]['low'], df.iloc[i]['high']

            # 갭 업 또는 갭 다운
            if curr_low > prev_high or curr_high < prev_low:
                gap_size = min(abs(curr_low - prev_high), abs(curr_high - prev_low))
                gap_pct = gap_size / prev_high
                if gap_pct > 0.02:  # 2% 이상 갭만 카운트
                    gaps += 1

        return gaps

    def fix_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """감지된 문제 자동 수정"""
        df = df.copy()

        # 1. 중복 제거
        df = df.drop_duplicates(subset=['timestamp'], keep='first')

        # 2. 정렬
        df = df.sort_values('timestamp').reset_index(drop=True)

        # 3. 누락 캔들 보간 (선형)
        df = df.set_index('timestamp')
        df = df.resample('1h').asfreq()
        df = df.interpolate(method='linear')
        df = df.reset_index()

        # 4. OHLC 무결성 수정
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

        # 5. 극단적 이상치 클리핑 (±3σ)
        returns = df['close'].pct_change()
        mean, std = returns.mean(), returns.std()
        df.loc[returns > mean + 3*std, 'close'] = df['close'].shift(1) * (1 + mean + 3*std)
        df.loc[returns < mean - 3*std, 'close'] = df['close'].shift(1) * (1 + mean - 3*std)

        return df
```

---

## 4. 데이터베이스 스키마 [C6 수정]

### 4.1 핵심 테이블 정의

```sql
-- migrations/001_initial_schema.sql

-- OHLCV 가격 데이터
CREATE TABLE ohlcv (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,           -- 'BTCUSDT', 'ETHUSDT'
    exchange VARCHAR(20) NOT NULL,         -- 'binance', 'bybit'
    timeframe VARCHAR(10) NOT NULL,        -- '1h', '4h', '1d'
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(30, 8) NOT NULL,
    quote_volume DECIMAL(30, 8),           -- USDT 거래량
    trades_count INTEGER,                  -- 거래 횟수
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- 인덱스용 유니크 제약
    CONSTRAINT ohlcv_unique UNIQUE (symbol, exchange, timeframe, timestamp)
);

-- 인덱스
CREATE INDEX idx_ohlcv_symbol_time ON ohlcv (symbol, timestamp DESC);
CREATE INDEX idx_ohlcv_timeframe ON ohlcv (timeframe, timestamp DESC);


-- 거래 기록
CREATE TABLE trades (
    id BIGSERIAL PRIMARY KEY,
    trade_id VARCHAR(50) UNIQUE NOT NULL,  -- 거래소 주문 ID
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,             -- 'LONG', 'SHORT'
    action VARCHAR(10) NOT NULL,           -- 'OPEN', 'CLOSE', 'PARTIAL_CLOSE'

    -- 가격 정보
    entry_price DECIMAL(20, 8),
    exit_price DECIMAL(20, 8),
    quantity DECIMAL(20, 8) NOT NULL,
    leverage INTEGER NOT NULL,

    -- 손익
    realized_pnl DECIMAL(20, 8),
    realized_pnl_pct DECIMAL(10, 6),
    fee DECIMAL(20, 8),

    -- 신호 정보
    signal_score DECIMAL(5, 4),            -- 최종 신호 점수
    technical_score DECIMAL(5, 4),         -- 기술적 점수
    llm_score DECIMAL(5, 4),               -- LLM 점수
    xgb_confidence DECIMAL(5, 4),          -- XGBoost 확신도

    -- 리스크 정보
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    position_size_pct DECIMAL(5, 4),       -- 자본 대비 %

    -- 메타데이터
    decision_reasons JSONB,                -- 의사결정 상세
    llm_analysis JSONB,                    -- LLM 응답 원본

    -- 타임스탬프
    signal_time TIMESTAMPTZ,               -- 신호 발생 시간
    entry_time TIMESTAMPTZ,                -- 진입 시간
    exit_time TIMESTAMPTZ,                 -- 청산 시간
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 인덱스
CREATE INDEX idx_trades_symbol ON trades (symbol, entry_time DESC);
CREATE INDEX idx_trades_pnl ON trades (realized_pnl_pct DESC);
CREATE INDEX idx_trades_side ON trades (side, action);


-- 포지션 현황 (실시간)
CREATE TABLE positions (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,             -- 'LONG', 'SHORT'

    -- 포지션 상세
    entry_price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    leverage INTEGER NOT NULL,
    margin DECIMAL(20, 8) NOT NULL,

    -- 현재 상태
    current_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),
    unrealized_pnl_pct DECIMAL(10, 6),
    liquidation_price DECIMAL(20, 8),
    margin_ratio DECIMAL(5, 4),

    -- 손절/익절
    stop_loss DECIMAL(20, 8),
    take_profit_1 DECIMAL(20, 8),          -- 50% 익절
    take_profit_2 DECIMAL(20, 8),          -- 100% 익절
    trailing_stop DECIMAL(20, 8),

    -- 상태
    status VARCHAR(20) DEFAULT 'OPEN',     -- 'OPEN', 'CLOSED', 'LIQUIDATED'

    -- 타임스탬프
    opened_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ,
    last_updated TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT positions_unique UNIQUE (symbol, exchange, status)
        WHERE status = 'OPEN'
);


-- LLM 분석 로그 (감사용)
CREATE TABLE llm_analysis_log (
    id BIGSERIAL PRIMARY KEY,
    analysis_type VARCHAR(20) NOT NULL,    -- 'REGULAR', 'SIGNAL_VALIDATION', 'URGENT'

    -- 입력
    prompt_template VARCHAR(50),
    market_data JSONB,
    news_context TEXT,

    -- 출력
    raw_response TEXT,
    parsed_response JSONB,
    parse_success BOOLEAN,

    -- 메타
    model_used VARCHAR(50),
    tokens_input INTEGER,
    tokens_output INTEGER,
    latency_ms INTEGER,
    cost_usd DECIMAL(10, 6),

    created_at TIMESTAMPTZ DEFAULT NOW()
);


-- 시스템 이벤트 로그 (감사용)
CREATE TABLE system_events (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,       -- 'CIRCUIT_BREAKER', 'KILL_SWITCH', 'API_ERROR', etc.
    severity VARCHAR(20) NOT NULL,         -- 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

    message TEXT NOT NULL,
    details JSONB,

    -- 관련 엔티티
    related_trade_id BIGINT REFERENCES trades(id),
    related_position_id BIGINT REFERENCES positions(id),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_events_type ON system_events (event_type, created_at DESC);
CREATE INDEX idx_events_severity ON system_events (severity, created_at DESC);


-- 일일 성과 요약
CREATE TABLE daily_performance (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,

    -- 거래 통계
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,

    -- 손익
    gross_profit DECIMAL(20, 8) DEFAULT 0,
    gross_loss DECIMAL(20, 8) DEFAULT 0,
    net_pnl DECIMAL(20, 8) DEFAULT 0,
    net_pnl_pct DECIMAL(10, 6) DEFAULT 0,

    -- 리스크 메트릭
    max_drawdown DECIMAL(10, 6),
    sharpe_ratio DECIMAL(10, 6),

    -- 신호 통계
    signals_generated INTEGER DEFAULT 0,
    signals_approved INTEGER DEFAULT 0,
    signals_vetoed INTEGER DEFAULT 0,

    -- 자본
    starting_balance DECIMAL(20, 8),
    ending_balance DECIMAL(20, 8),

    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 5. 백테스트 슬리피지 모델 [누락 요소]

```python
# backtest/slippage.py

from dataclasses import dataclass
from enum import Enum
import numpy as np

class SlippageModel(Enum):
    FIXED = "fixed"              # 고정 슬리피지
    VOLUME_BASED = "volume"      # 거래량 기반
    VOLATILITY_BASED = "volatility"  # 변동성 기반
    REALISTIC = "realistic"      # 복합 모델

@dataclass
class SlippageConfig:
    """슬리피지 설정"""

    model: SlippageModel = SlippageModel.REALISTIC

    # 고정 슬리피지
    fixed_slippage_pct: float = 0.0005  # 0.05%

    # 거래량 기반 파라미터
    volume_impact_factor: float = 0.1   # 거래량 영향 계수

    # 변동성 기반 파라미터
    volatility_multiplier: float = 0.5  # ATR 대비 슬리피지

    # 현실적 모델 파라미터
    base_slippage: float = 0.0002       # 기본 0.02%
    market_impact_coefficient: float = 0.05

    # 거래 수수료
    maker_fee: float = 0.0002           # 0.02%
    taker_fee: float = 0.0004           # 0.04%


class SlippageSimulator:
    """백테스트용 슬리피지 시뮬레이터"""

    def __init__(self, config: SlippageConfig):
        self.config = config

    def calculate_slippage(
        self,
        order_type: str,          # 'market', 'limit'
        side: str,                # 'buy', 'sell'
        price: float,
        quantity: float,
        volume_24h: float,
        atr: float,
        spread: float = 0.0001    # 기본 스프레드 0.01%
    ) -> tuple[float, float]:
        """
        슬리피지 계산

        Returns:
            (실제 체결가, 슬리피지 비용)
        """

        if self.config.model == SlippageModel.FIXED:
            slippage_pct = self.config.fixed_slippage_pct

        elif self.config.model == SlippageModel.VOLUME_BASED:
            # 주문량 / 24시간 거래량 비율
            volume_ratio = (quantity * price) / volume_24h
            slippage_pct = self.config.volume_impact_factor * np.sqrt(volume_ratio)

        elif self.config.model == SlippageModel.VOLATILITY_BASED:
            # ATR 기반
            atr_pct = atr / price
            slippage_pct = atr_pct * self.config.volatility_multiplier

        elif self.config.model == SlippageModel.REALISTIC:
            # 복합 모델: 기본 + 거래량 영향 + 변동성
            order_value = quantity * price
            volume_ratio = order_value / volume_24h
            atr_pct = atr / price

            slippage_pct = (
                self.config.base_slippage +
                self.config.market_impact_coefficient * np.sqrt(volume_ratio) +
                atr_pct * 0.1 +
                spread / 2
            )

        # 방향에 따른 가격 조정
        if side == 'buy':
            execution_price = price * (1 + slippage_pct)
        else:
            execution_price = price * (1 - slippage_pct)

        # 수수료
        fee_rate = self.config.taker_fee if order_type == 'market' else self.config.maker_fee
        fee_cost = quantity * execution_price * fee_rate

        slippage_cost = abs(execution_price - price) * quantity
        total_cost = slippage_cost + fee_cost

        return execution_price, total_cost
```

---

## 6. 프로젝트 구조 (업데이트)

```
iftb/
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── cd.yml
│       └── model-retrain.yml
├── migrations/                    # [신규] DB 마이그레이션
│   ├── 001_initial_schema.sql
│   └── ...
├── src/
│   └── iftb/
│       ├── __init__.py
│       ├── main.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py
│       │   └── constants.py       # [신규] 리스크 상수
│       ├── data/
│       │   ├── __init__.py
│       │   ├── fetcher.py
│       │   ├── telegram.py
│       │   ├── storage.py
│       │   └── validation.py      # [신규] 데이터 검증
│       ├── analysis/
│       │   └── ...
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── analyzer.py
│       │   ├── prompts.py
│       │   ├── parser.py
│       │   └── veto.py            # [신규] 거부권 로직
│       ├── ml/
│       │   ├── __init__.py
│       │   ├── model.py
│       │   ├── training.py
│       │   ├── features.py
│       │   ├── calibration.py
│       │   └── versioning.py      # [신규] 모델 버전 관리
│       ├── decision/
│       │   └── ...
│       ├── execution/
│       │   └── ...
│       ├── risk/
│       │   ├── __init__.py
│       │   ├── manager.py
│       │   ├── sizing.py
│       │   ├── filters.py
│       │   ├── circuit_breaker.py # [신규]
│       │   └── kill_switch.py     # [신규]
│       ├── backtest/              # [신규]
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   ├── slippage.py
│       │   └── statistics.py
│       ├── monitoring/
│       │   └── ...
│       └── utils/
│           └── ...
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_circuit_breaker.py  # [신규]
│   │   ├── test_kill_switch.py      # [신규]
│   │   ├── test_llm_veto.py         # [신규]
│   │   ├── test_data_validation.py  # [신규]
│   │   ├── test_slippage.py         # [신규]
│   │   └── ...
│   └── ...
├── scripts/
│   └── ...
├── data/
│   └── ...
├── configs/
│   ├── dev.yaml
│   ├── staging.yaml
│   └── prod.yaml
└── docs/
    ├── architecture.md
    ├── api.md
    ├── runbook.md
    ├── disaster-recovery.md       # [신규]
    └── rollback-procedures.md     # [신규]
```

---

## 7. 개발 페이즈 (업데이트)

### Phase 0: 프로젝트 초기화 (1주)

| ID | 작업 | 예상 시간 | 비고 |
|----|------|----------|------|
| 0.1 | 프로젝트 구조 생성 | 2h | |
| 0.2 | uv + pyproject.toml 설정 | 1h | |
| 0.3 | 핵심 의존성 설치 | 1h | |
| 0.4 | Docker 환경 구성 | 3h | |
| 0.5 | GitHub Actions CI 파이프라인 | 3h | |
| 0.6 | pre-commit hooks 설정 | 1h | |
| 0.7 | Pydantic Settings 구성 | 2h | |
| 0.8 | 로깅 시스템 구축 | 2h | |
| 0.9 | PostgreSQL + Redis Docker 설정 | 2h | |
| 0.10 | **DB 스키마 마이그레이션** | 2h | [C6] |
| 0.11 | **리스크 상수 정의** | 1h | [C2] |
| 0.12 | 기본 테스트 구조 설정 | 1h | |

### Phase 1: 데이터 레이어 (2주)

**Stream 1A: 가격 데이터** (기존 + 검증 추가)

| ID | 작업 | 예상 시간 | 비고 |
|----|------|----------|------|
| 1A.1 | CCXT 바이낸스 연동 | 3h | |
| 1A.2 | 6년치 1H OHLCV 다운로드 스크립트 | 4h | |
| 1A.3 | **데이터 검증 파이프라인** | 4h | [C5] |
| 1A.4 | **이상치 감지 & 수정** | 3h | [C5] |
| 1A.5 | 실시간 WebSocket 수신기 | 4h | |
| 1A.6 | PostgreSQL 저장 레이어 | 3h | |
| 1A.7 | Redis 실시간 캐싱 | 2h | |
| 1A.8 | 단위 테스트 | 4h | |

### Phase 2: 분석 레이어 (2주)

**Stream 2B: LLM 분석** (거부권 강화)

| ID | 작업 | 예상 시간 | 비고 |
|----|------|----------|------|
| 2B.1 | Claude API 연동 | 2h | |
| 2B.2 | 정기 분석 프롬프트 | 3h | |
| 2B.3 | 신호 검증 프롬프트 | 3h | |
| 2B.4 | JSON 응답 파싱 | 3h | |
| 2B.5 | **거부권 시스템 구현** | 4h | [C4] |
| 2B.6 | **API 장애 폴백 로직** | 3h | [C4] |
| 2B.7 | 응답 캐싱 (5분 TTL) | 2h | |
| 2B.8 | Rate Limiting | 2h | |
| 2B.9 | 단위 테스트 (Mock) | 4h | |

**Stream 2C: XGBoost 모델** (버전 관리 추가)

| ID | 작업 | 예상 시간 | 비고 |
|----|------|----------|------|
| 2C.1 | 레이블 생성 (대칭) | 3h | |
| 2C.2 | Feature 엔지니어링 | 4h | |
| 2C.3 | Train/Val/Test 분할 | 2h | |
| 2C.4 | 클래스 균형 처리 | 2h | |
| 2C.5 | XGBoost 모델 구현 | 4h | |
| 2C.6 | 시계열 교차검증 | 3h | |
| 2C.7 | Probability Calibration | 3h | |
| 2C.8 | Optuna 하이퍼파라미터 튜닝 | 4h | |
| 2C.9 | **모델 버전 관리 시스템** | 3h | [W2] |
| 2C.10 | 모델 저장/로드 | 2h | |
| 2C.11 | **통계적 성능 검증** | 3h | [C1] |
| 2C.12 | 단위 테스트 | 4h | |

### Phase 3: 의사결정 & 실행 (2주)

**Stream 3B: 리스크 관리** (회로 차단기/킬 스위치)

| ID | 작업 | 예상 시간 | 비고 |
|----|------|----------|------|
| 3B.1 | Kelly Criterion 포지션 사이징 | 3h | |
| 3B.2 | **Kelly 비율 제한 적용** | 2h | [C2] |
| 3B.3 | 동적 레버리지 조정 | 3h | |
| 3B.4 | ATR 기반 손절/익절 | 3h | |
| 3B.5 | **회로 차단기 구현** | 4h | [C3] |
| 3B.6 | **킬 스위치 구현** | 3h | [C3] |
| 3B.7 | 리스크 필터 구현 | 4h | |
| 3B.8 | MDD 모니터링 | 2h | |
| 3B.9 | 연속 손실 관리 | 2h | |
| 3B.10 | 단위 테스트 | 5h | |

### Phase 4: 통합 & 백테스트 (2주)

**Stream 4B: 백테스트 시스템** (슬리피지 모델 추가)

| ID | 작업 | 예상 시간 | 비고 |
|----|------|----------|------|
| 4B.1 | Backtrader 통합 | 4h | |
| 4B.2 | 커스텀 전략 어댑터 | 4h | |
| 4B.3 | **슬리피지 시뮬레이터** | 4h | [누락] |
| 4B.4 | 성과 분석기 | 3h | |
| 4B.5 | **통계적 검증 모듈** | 4h | [C1] |
| 4B.6 | 시각화 (matplotlib) | 3h | |
| 4B.7 | 2020-2024 백테스트 실행 | 4h | |
| 4B.8 | 워크포워드 테스트 (2024-2025) | 4h | |
| 4B.9 | 백테스트 리포트 생성 | 3h | |

### Phase 5: 모니터링 & 배포 (2주)

(기존과 동일 + 롤백/DR 문서 추가)

| ID | 작업 | 예상 시간 | 비고 |
|----|------|----------|------|
| 5B.6 | 배포 Runbook 작성 | 3h | |
| 5B.7 | **롤백 절차 문서화** | 2h | [누락] |
| 5B.8 | **재해 복구 계획** | 3h | [누락] |

---

## 8. 경고 사항 대응

| 경고 | 대응 |
|------|------|
| W1: 월 40% 비현실적 | Sharpe Ratio ≥ 1.5로 목표 재정의 |
| W2: 모델 버전 관리 없음 | Phase 2C에 버전 관리 시스템 추가 |
| W3: 텔레그램 신뢰성 | 다중 채널 + 신뢰도 점수화 적용 |
| W4: 부하 테스트 없음 | Phase 5에 부하 테스트 추가 |
| W5: 외부 API 폴백 | 각 외부 API에 폴백 로직 구현 |
| W6: 바이낸스 특수 상황 | 펀딩비/ADL 핸들러 추가 |

---

## 9. 성공 기준 체크리스트 (업데이트)

### 개발 완료

- [ ] 모든 모듈 단위 테스트 통과 (85%+)
- [ ] 통합 테스트 통과
- [ ] CI/CD 파이프라인 작동
- [ ] **백테스트 승률 60%+ (n≥500, 95% CI, p<0.05)** [C1]
- [ ] **회로 차단기 테스트 통과** [C3]
- [ ] **킬 스위치 테스트 통과** [C3]
- [ ] **데이터 품질 점수 95%+** [C5]
- [ ] 워크포워드 테스트 통과

### 운영 준비

- [ ] AWS 배포 완료
- [ ] 모니터링 대시보드 작동
- [ ] Telegram 알림 작동
- [ ] **롤백 절차 문서화 완료** [누락]
- [ ] **재해 복구 계획 수립** [누락]
- [ ] Paper Trading 2주 완료
- [ ] MDD 30% 이내 유지

---

## 10. 다음 단계

**계획 승인 시 즉시 실행:**

1. GitHub 레포지토리 생성
2. 프로젝트 구조 초기화 (Phase 0)
3. DB 스키마 적용
4. Phase 1 병렬 스트림 시작

---

**계획 버전**: v1.1
**변경 사항**: C1-C6 치명적 문제 해결, 경고 사항 대응 추가
**예상 총 기간**: 10-12주 (Paper Trading 포함)
**병렬화 수준**: 최대 3개 스트림 동시 진행
