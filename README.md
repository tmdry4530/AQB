# IFTB - Intelligent Futures Trading Bot

AI-integrated cryptocurrency futures trading bot with 3-layer decision system.

## Architecture

```
Technical Analysis (40%)
         ↓
   LLM Analysis (25%)  →  Decision Engine  →  Order Executor
         ↓
   ML Validation (35%)
```

## Features

- **Technical Analysis**: 14 indicators (RSI, MACD, Bollinger Bands, ATR, ADX, etc.)
- **LLM Integration**: Claude API for market sentiment analysis with veto system
- **ML Validation**: XGBoost classifier for LONG/SHORT/HOLD decisions
- **Risk Management**: Kelly Criterion sizing, Circuit Breaker, Kill Switch
- **Order Execution**: Paper trading and live trading via CCXT (Binance Futures)
- **Real-time Data**: WebSocket streaming for market data
- **External Data**: Fear & Greed Index, Coinglass metrics

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/iftb.git
cd iftb

# Install with uv
uv pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

## Usage

```bash
# Run in paper trading mode
python -m iftb.main

# Or use the CLI entry point
iftb
```

## Configuration

`.env.example`을 `.env`로 복사한 후 아래 가이드를 참고하여 API 키를 발급받으세요.

### 1. Binance API 키 발급

#### 테스트넷 (권장 - 모의거래)

1. [Binance Futures Testnet](https://testnet.binancefuture.com/) 접속
2. GitHub 계정으로 로그인
3. 우측 상단 프로필 → **API Management** 클릭
4. **Create API** 버튼 클릭
5. API Key와 Secret Key 복사

```env
EXCHANGE_API_KEY=your_testnet_api_key
EXCHANGE_API_SECRET=your_testnet_api_secret
EXCHANGE_TESTNET=true
```

#### 실거래 계정

1. [Binance](https://www.binance.com/) 로그인
2. 프로필 → **API Management** 이동
3. **Create API** → **System generated** 선택
4. 2FA 인증 완료
5. API 제한 설정:
   - ✅ Enable Reading
   - ✅ Enable Futures
   - ❌ Enable Withdrawals (보안상 비활성화 권장)
6. IP 제한 설정 (권장)

```env
EXCHANGE_API_KEY=your_live_api_key
EXCHANGE_API_SECRET=your_live_api_secret
EXCHANGE_TESTNET=false
```

> ⚠️ **주의**: API Secret은 생성 시 한 번만 표시됩니다. 안전하게 보관하세요.

### 2. Anthropic (Claude) API 키 발급

1. [Anthropic Console](https://console.anthropic.com/) 접속
2. 계정 생성 또는 로그인
3. **API Keys** 메뉴 이동
4. **Create Key** 클릭
5. 키 이름 입력 후 생성

```env
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxx
LLM_MODEL=claude-sonnet-4-20250514
```

> 💡 무료 크레딧으로 시작 가능. 사용량에 따라 과금됩니다.

### 3. Telegram 설정 (채널 데이터 수집)

참여 중인 뉴스/시그널 채널의 메시지를 수집하여 시장 분석에 활용합니다.

#### API ID/Hash 발급 (필수)

1. [my.telegram.org](https://my.telegram.org/apps) 접속
2. 본인 전화번호로 로그인
3. **API development tools** 선택
4. 앱 정보 입력:
   - App title: `IFTB` (자유롭게)
   - Short name: `iftb`
   - Platform: `Desktop`
5. API ID와 API Hash 복사

```env
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=your_api_hash_here
```

#### 채널 ID 확인

모니터링할 채널의 ID를 확인하는 방법:

1. [Telegram Web](https://web.telegram.org/) 접속
2. 채널 입장 후 URL 확인: `https://web.telegram.org/k/#-1001234567890`
3. URL의 숫자 부분이 채널 ID (`-1001234567890`)

```env
# 모니터링할 채널 ID (쉼표로 구분)
TELEGRAM_NEWS_CHANNEL_IDS=-1001111111111,-1002222222222
```

> 💡 **팁**: 크립토 뉴스, 시그널 채널 등에 참여한 상태여야 메시지를 수집할 수 있습니다.

#### Bot Token (알림 전송용, 선택)

트레이딩 알림을 받고 싶다면 봇을 생성하세요:

1. [@BotFather](https://t.me/BotFather)에서 `/newbot` 실행
2. 생성된 봇과 대화 시작 후 Chat ID 확인

```env
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_ALERT_CHAT_ID=123456789
```

### 4. 데이터베이스 설정

#### PostgreSQL

```bash
# Docker로 빠르게 시작
docker run -d \
  --name iftb-postgres \
  -e POSTGRES_USER=iftb \
  -e POSTGRES_PASSWORD=your_secure_password \
  -e POSTGRES_DB=iftb \
  -p 5432:5432 \
  postgres:15
```

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=iftb
DB_USER=iftb
DB_PASSWORD=your_secure_password
```

#### Redis

```bash
# Docker로 빠르게 시작
docker run -d \
  --name iftb-redis \
  -p 6379:6379 \
  redis:7
```

```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

### 5. 거래 설정

```env
# 거래 심볼 (쉼표로 구분)
TRADING_SYMBOLS=BTCUSDT,ETHUSDT

# 타임프레임
TRADING_TIMEFRAMES=1h,4h

# 레버리지 (1-125, 테스트넷에서 먼저 테스트)
TRADING_DEFAULT_LEVERAGE=5

# 모의거래 모드 (true 권장)
TRADING_PAPER_MODE=true
```

### 전체 환경변수 목록

| 변수명 | 필수 | 설명 |
|--------|------|------|
| `EXCHANGE_API_KEY` | ✅ | Binance API 키 |
| `EXCHANGE_API_SECRET` | ✅ | Binance API 시크릿 |
| `EXCHANGE_TESTNET` | ✅ | 테스트넷 사용 여부 |
| `ANTHROPIC_API_KEY` | ✅ | Claude API 키 |
| `DB_HOST` | ✅ | PostgreSQL 호스트 |
| `DB_PASSWORD` | ✅ | PostgreSQL 비밀번호 |
| `REDIS_HOST` | - | Redis 호스트 |
| `TELEGRAM_API_ID` | - | 채널 데이터 수집용 |
| `TELEGRAM_API_HASH` | - | 채널 데이터 수집용 |
| `TELEGRAM_NEWS_CHANNEL_IDS` | - | 모니터링할 채널 ID |
| `TELEGRAM_BOT_TOKEN` | - | 알림 전송용 (선택) |
| `TRADING_PAPER_MODE` | ✅ | 모의거래 모드 |

## Project Structure

```
src/iftb/
├── analysis/        # Technical, LLM, and ML analysis
├── config/          # Settings and constants
├── data/            # Data fetching, caching, storage
├── trading/         # Decision engine and order execution
├── utils/           # Logging and utilities
└── main.py          # Entry point
```

## License

MIT
