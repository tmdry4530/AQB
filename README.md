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

See `.env.example` for all configuration options:

- Exchange API credentials (Binance)
- Database (PostgreSQL)
- Cache (Redis)
- LLM (Anthropic Claude)
- Trading parameters

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
