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

### 3. Telegram 설정 (선택)

#### Bot Token 발급

1. Telegram에서 [@BotFather](https://t.me/BotFather) 검색
2. `/newbot` 명령어 입력
3. 봇 이름과 username 설정
4. 발급된 토큰 복사

```env
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
```

#### Chat ID 확인

1. 봇에게 아무 메시지 전송
2. 브라우저에서 접속: `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. `chat.id` 값 확인

```env
TELEGRAM_ALERT_CHAT_ID=123456789
```

#### API ID/Hash (고급 기능용)

1. [my.telegram.org](https://my.telegram.org/apps) 접속
2. 전화번호로 로그인
3. **API development tools** 선택
4. 앱 정보 입력 후 생성

```env
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=your_api_hash_here
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
| `TELEGRAM_BOT_TOKEN` | - | 텔레그램 알림용 |
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
