# Database Schema Reference

Complete reference for all tables, views, and functions in the IFTB Trading Bot database.

## Tables

### Core Trading Tables (001_initial_schema.sql)

#### `ohlcv`
OHLCV candlestick price data from exchanges
- `id` BIGSERIAL PRIMARY KEY
- `symbol` VARCHAR(20) - Trading pair (e.g., BTC/USDT)
- `exchange` VARCHAR(50) - Exchange name
- `timeframe` VARCHAR(10) - Candle timeframe (1m, 5m, 1h, etc.)
- `timestamp` TIMESTAMPTZ - Candle open time
- `open`, `high`, `low`, `close` DECIMAL(20,8) - Price data
- `volume` DECIMAL(20,8) - Base volume
- `quote_volume` DECIMAL(20,8) - Quote volume
- `trades_count` INTEGER - Number of trades
- `created_at` TIMESTAMPTZ

**Indexes**: symbol+timeframe+timestamp, exchange+symbol, timestamp
**Unique**: (symbol, exchange, timeframe, timestamp)

#### `trades`
Historical trade records with entry/exit details
- `id` BIGSERIAL PRIMARY KEY
- `trade_id` VARCHAR(100) UNIQUE - Unique trade identifier
- `symbol` VARCHAR(20)
- `exchange` VARCHAR(50)
- `side` VARCHAR(10) - 'long' or 'short'
- `action` VARCHAR(10) - 'open', 'close', 'liquidated'
- `entry_price`, `exit_price` DECIMAL(20,8)
- `quantity` DECIMAL(20,8)
- `leverage` DECIMAL(5,2)
- `realized_pnl`, `realized_pnl_pct` DECIMAL
- `fee` DECIMAL(20,8)
- `signal_score`, `technical_score`, `llm_score`, `xgb_confidence` DECIMAL(5,4)
- `stop_loss`, `take_profit` DECIMAL(20,8)
- `position_size_pct` DECIMAL(5,2)
- `decision_reasons` JSONB - Why trade was taken
- `llm_analysis` JSONB - LLM analysis data
- `entry_time`, `exit_time` TIMESTAMPTZ
- `created_at`, `updated_at` TIMESTAMPTZ

**Indexes**: symbol, exchange, entry_time, exit_time, action, pnl, GIN(decision_reasons)

#### `positions`
Current open and recently closed positions
- `id` BIGSERIAL PRIMARY KEY
- `symbol` VARCHAR(20)
- `exchange` VARCHAR(50)
- `side` VARCHAR(10) - 'long' or 'short'
- `entry_price` DECIMAL(20,8)
- `quantity` DECIMAL(20,8)
- `leverage` DECIMAL(5,2)
- `margin` DECIMAL(20,8)
- `current_price` DECIMAL(20,8)
- `unrealized_pnl`, `unrealized_pnl_pct` DECIMAL
- `liquidation_price` DECIMAL(20,8)
- `stop_loss`, `take_profit`, `trailing_stop` DECIMAL(20,8)
- `status` VARCHAR(20) - 'open', 'closed', 'liquidated'
- `trade_id` VARCHAR(100) - Reference to trades table
- `entry_time`, `last_updated` TIMESTAMPTZ

**Indexes**: symbol, status, exchange, symbol+status, trade_id
**Unique**: (symbol, exchange, side, status) WHERE status='open'

#### `llm_analysis_log`
Audit log for all LLM API calls
- `id` BIGSERIAL PRIMARY KEY
- `analysis_type` VARCHAR(50) - Type of analysis
- `symbol`, `exchange` VARCHAR
- `prompt_template` VARCHAR(100)
- `prompt_tokens`, `response_tokens`, `total_tokens` INTEGER
- `prompt_text`, `response_text` TEXT
- `response_json` JSONB
- `estimated_cost` DECIMAL(10,6)
- `model_name` VARCHAR(100)
- `temperature` DECIMAL(3,2)
- `status` VARCHAR(20) - 'success', 'error', 'timeout'
- `error_message` TEXT
- `execution_time_ms` INTEGER
- `created_at` TIMESTAMPTZ

**Indexes**: type, symbol, created_at, status, type+created_at, GIN(response_json)

#### `system_events`
System-wide event and audit log
- `id` BIGSERIAL PRIMARY KEY
- `event_type` VARCHAR(50)
- `severity` VARCHAR(20) - 'debug', 'info', 'warning', 'error', 'critical'
- `message` TEXT
- `details` JSONB
- `symbol`, `exchange`, `trade_id` VARCHAR
- `position_id` BIGINT
- `stack_trace` TEXT
- `created_at` TIMESTAMPTZ

**Indexes**: type, severity, created_at, type+created_at, symbol, trade_id, GIN(details)

#### `daily_performance`
Daily aggregated trading performance metrics
- `id` BIGSERIAL PRIMARY KEY
- `date` DATE UNIQUE
- `exchange` VARCHAR(50)
- `total_trades`, `winning_trades`, `losing_trades` INTEGER
- `win_rate` DECIMAL(5,2)
- `gross_profit`, `gross_loss`, `net_pnl` DECIMAL(20,8)
- `net_pnl_pct` DECIMAL(10,4)
- `total_fees` DECIMAL(20,8)
- `avg_position_size`, `max_position_size`, `total_volume` DECIMAL(20,8)
- `max_drawdown`, `max_drawdown_amount` DECIMAL
- `sharpe_ratio`, `profit_factor` DECIMAL(10,4)
- `starting_balance`, `ending_balance`, `peak_balance` DECIMAL(20,8)
- `avg_trade_duration_minutes` INTEGER
- `longest_winning_streak`, `longest_losing_streak` INTEGER
- `created_at`, `updated_at` TIMESTAMPTZ

**Indexes**: date, exchange, date+exchange, pnl

### Analysis Tables (002_analysis_tables.sql)

#### `technical_signals`
Aggregated technical analysis from multiple indicators
- `id` BIGSERIAL PRIMARY KEY
- `symbol` VARCHAR(20)
- `timestamp` TIMESTAMPTZ
- `overall_signal` VARCHAR(10) - 'bullish', 'bearish', 'neutral'
- `confidence` DECIMAL(5,4) - 0.0 to 1.0
- `bullish_count`, `bearish_count`, `neutral_count` INTEGER
- `indicators_json` JSONB - Full indicator breakdown
- `rsi_value`, `rsi_signal` - RSI indicator
- `macd_histogram`, `macd_signal` - MACD indicator
- `bb_position`, `bb_signal` - Bollinger Bands
- `timeframe` VARCHAR(10)
- `created_at` TIMESTAMPTZ

**Indexes**: symbol, timestamp, symbol+timestamp, overall_signal, confidence, GIN(indicators_json)
**Partial**: High confidence (>=0.7)

#### `llm_analyses`
LLM-based market analysis and sentiment
- `id` BIGSERIAL PRIMARY KEY
- `symbol` VARCHAR(20)
- `timestamp` TIMESTAMPTZ
- `sentiment` VARCHAR(10) - 'bullish', 'bearish', 'neutral', 'uncertain'
- `confidence` DECIMAL(5,4)
- `summary` TEXT
- `key_factors_json` JSONB - Key factors influencing analysis
- `model` VARCHAR(50) - LLM model used
- `cached` BOOLEAN
- `prompt_tokens`, `completion_tokens`, `total_tokens` INTEGER
- `estimated_cost` DECIMAL(10,6)
- `response_time_ms` INTEGER
- `temperature` DECIMAL(3,2)
- `prompt_text`, `response_text` TEXT
- `status` VARCHAR(20) - 'success', 'error', 'timeout', 'cached'
- `error_message` TEXT
- `created_at` TIMESTAMPTZ

**Indexes**: symbol, timestamp, sentiment, confidence, model, status, cached, GIN(key_factors_json)
**Partial**: High confidence (>=0.7) and successful

#### `ml_predictions`
Machine learning model predictions (XGBoost)
- `id` BIGSERIAL PRIMARY KEY
- `symbol` VARCHAR(20)
- `timestamp` TIMESTAMPTZ
- `action` VARCHAR(10) - 'long', 'short', 'hold', 'close'
- `confidence` DECIMAL(5,4)
- `prob_long`, `prob_short`, `prob_hold` DECIMAL(5,4)
- `feature_importance_json` JSONB - Feature importance for explainability
- `model_version` VARCHAR(50)
- `model_type` VARCHAR(50) - Default 'xgboost'
- `prediction_time_ms` INTEGER
- `features_used` INTEGER
- `historical_accuracy` DECIMAL(5,4)
- `created_at` TIMESTAMPTZ

**Indexes**: symbol, timestamp, action, confidence, model_version, GIN(feature_importance_json)
**Partial**: High confidence (>=0.7)

#### `trading_decisions`
Final trading decisions combining all signals
- `id` BIGSERIAL PRIMARY KEY
- `symbol` VARCHAR(20)
- `timestamp` TIMESTAMPTZ
- `action` VARCHAR(10) - 'long', 'short', 'close', 'hold', 'vetoed'
- `confidence` DECIMAL(5,4)
- `position_size`, `position_size_pct` DECIMAL
- `leverage` INTEGER
- `stop_loss`, `take_profit`, `entry_price` DECIMAL(20,8)
- `reasons_json` JSONB - Decision reasoning
- `technical_signal_id` → technical_signals.id
- `llm_analysis_id` → llm_analyses.id
- `ml_prediction_id` → ml_predictions.id
- `vetoed` BOOLEAN
- `veto_reason` TEXT
- `veto_source` VARCHAR(50)
- `executed` BOOLEAN
- `order_id` → orders.id
- `execution_price` DECIMAL(20,8)
- `executed_at` TIMESTAMPTZ
- `decision_latency_ms` INTEGER
- `created_at` TIMESTAMPTZ

**Indexes**: symbol, timestamp, action, confidence, vetoed, executed, GIN(reasons_json), foreign keys
**Partial**: Executed decisions, Vetoed decisions

### Risk Management Tables (003_risk_tables.sql)

#### `circuit_breaker_events`
Automatic trading halts triggered by risk thresholds
- `id` BIGSERIAL PRIMARY KEY
- `triggered_at` TIMESTAMPTZ
- `reason` VARCHAR(100)
- `trigger_type` VARCHAR(50) - 'max_daily_loss', 'max_drawdown', 'rapid_loss', 'consecutive_losses', 'position_limit', 'exchange_error', 'network_issue', 'manual'
- `threshold_value`, `actual_value` DECIMAL(20,8)
- `symbol` VARCHAR(20)
- `daily_pnl` DECIMAL(20,8)
- `drawdown_pct` DECIMAL(10,4)
- `consecutive_losses` INTEGER
- `open_positions` INTEGER
- `account_balance` DECIMAL(20,8)
- `context_json` JSONB - Full context at trigger time
- `resolved_at` TIMESTAMPTZ
- `cooldown_hours` DECIMAL(5,2)
- `cooldown_expires_at` TIMESTAMPTZ (auto-calculated)
- `status` VARCHAR(20) - 'active', 'resolved', 'overridden', 'expired'
- `resolved_by` VARCHAR(50)
- `resolution_reason` TEXT
- `severity` VARCHAR(20) - 'info', 'warning', 'critical', 'emergency'
- `created_at`, `updated_at` TIMESTAMPTZ

**Indexes**: triggered_at, status, trigger_type, severity, symbol, cooldown_expires_at, GIN(context_json)
**Partial**: Active circuit breakers

#### `kill_switch_events`
Emergency system shutdowns and manual interventions
- `id` BIGSERIAL PRIMARY KEY
- `activated_at` TIMESTAMPTZ
- `reason` TEXT
- `trigger_source` VARCHAR(50) - 'manual', 'api', 'circuit_breaker_cascade', 'system_health', 'exchange_emergency', 'security_alert', 'critical_error'
- `activated_by` VARCHAR(100)
- `positions_closed`, `orders_cancelled` INTEGER
- `emergency_exit_used` BOOLEAN
- `balance_before`, `equity_before` DECIMAL(20,8)
- `open_positions_before`, `pending_orders_before` INTEGER
- `balance_after`, `equity_after`, `emergency_pnl` DECIMAL(20,8)
- `action_log_json` JSONB - Detailed action log
- `system_state_json` JSONB
- `deactivated_at` TIMESTAMPTZ
- `deactivated_by` VARCHAR(100)
- `deactivation_reason` TEXT
- `status` VARCHAR(20) - 'active', 'deactivated', 'partial_recovery'
- `severity` VARCHAR(20) - 'warning', 'critical', 'emergency'
- `recovery_started_at`, `recovery_completed_at` TIMESTAMPTZ
- `recovery_notes` TEXT
- `created_at`, `updated_at` TIMESTAMPTZ

**Indexes**: activated_at, status, trigger_source, severity, activated_by, GIN(action_log_json)
**Partial**: Active kill switches

#### `daily_pnl`
Daily profit/loss tracking with comprehensive metrics
- `id` BIGSERIAL PRIMARY KEY
- `date` DATE UNIQUE
- `realized_pnl`, `unrealized_pnl`, `total_pnl` DECIMAL(20,8)
- `balance_start`, `balance_end` DECIMAL(20,8)
- `equity_start`, `equity_end` DECIMAL(20,8)
- `peak_balance`, `lowest_balance`, `peak_equity`, `lowest_equity` DECIMAL(20,8)
- `max_drawdown`, `max_drawdown_pct` DECIMAL
- `current_drawdown`, `current_drawdown_pct` DECIMAL
- `total_trades`, `winning_trades`, `losing_trades` INTEGER
- `win_rate` DECIMAL(5,2)
- `gross_profit`, `gross_loss` DECIMAL(20,8)
- `average_win`, `average_loss` DECIMAL(20,8)
- `profit_factor` DECIMAL(10,4)
- `long_trades`, `short_trades` INTEGER
- `avg_position_size`, `max_position_size`, `total_volume` DECIMAL(20,8)
- `total_fees`, `fee_pct_of_volume` DECIMAL
- `circuit_breakers_triggered` INTEGER
- `kill_switch_activated` BOOLEAN
- `sharpe_ratio`, `sortino_ratio`, `calmar_ratio` DECIMAL(10,4)
- `avg_trade_duration_minutes`, `shortest_trade_minutes`, `longest_trade_minutes` INTEGER
- `current_win_streak`, `current_loss_streak` INTEGER
- `longest_win_streak`, `longest_loss_streak` INTEGER
- `best_trade_pnl`, `worst_trade_pnl` DECIMAL(20,8)
- `exchange` VARCHAR(50)
- `calculation_completed_at` TIMESTAMPTZ
- `created_at`, `updated_at` TIMESTAMPTZ

**Indexes**: date, total_pnl, win_rate
**Partial**: Days with circuit breakers, Days with kill switch

### Market Data Tables (004_market_data_tables.sql)

#### `ohlcv_data`
Enhanced OHLCV candlestick data for technical analysis
- `id` BIGSERIAL PRIMARY KEY
- `symbol` VARCHAR(20)
- `timeframe` VARCHAR(10) - '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w'
- `timestamp` TIMESTAMPTZ
- `open`, `high`, `low`, `close` DECIMAL(20,8)
- `volume` DECIMAL(20,8)
- `quote_volume`, `taker_buy_volume`, `taker_buy_quote_volume` DECIMAL(20,8)
- `trades_count` INTEGER
- `is_complete`, `has_gaps` BOOLEAN - Data quality flags
- `exchange` VARCHAR(50)
- `data_source` VARCHAR(50)
- `created_at` TIMESTAMPTZ

**Indexes**: symbol, timestamp, symbol+timeframe, symbol+timeframe+timestamp, exchange, volume
**Unique**: (symbol, timeframe, timestamp, exchange)
**Partial**: Incomplete data, Recent data (30 days)
**Constraint**: OHLCV values must be consistent (high >= all, low <= all)

#### `fear_greed_index`
Crypto Fear & Greed Index for market sentiment
- `id` BIGSERIAL PRIMARY KEY
- `timestamp` TIMESTAMPTZ UNIQUE
- `value` INTEGER (0-100)
- `classification` VARCHAR(20) - 'extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed' (auto-calculated)
- `value_yesterday`, `value_last_week`, `value_last_month` INTEGER
- `volatility_score`, `market_momentum_score` INTEGER (0-100)
- `social_media_score`, `surveys_score` INTEGER (0-100)
- `dominance_score`, `trends_score` INTEGER (0-100)
- `data_source` VARCHAR(50) - Default 'alternative.me'
- `created_at` TIMESTAMPTZ

**Indexes**: timestamp, value, classification, created_at
**Partial**: Extreme fear/greed conditions

#### `funding_rates`
Perpetual futures funding rates
- `id` BIGSERIAL PRIMARY KEY
- `symbol` VARCHAR(20)
- `timestamp` TIMESTAMPTZ
- `rate` DECIMAL(10,8) - Funding rate as decimal
- `rate_bps` DECIMAL(10,4) - Rate in basis points (auto-calculated)
- `annualized_rate` DECIMAL(10,6) - Annualized rate (auto-calculated)
- `mark_price`, `index_price` DECIMAL(20,8)
- `premium` DECIMAL(10,8) - Mark/index premium (auto-calculated)
- `predicted_rate` DECIMAL(10,8)
- `funding_interval_hours` INTEGER - Default 8
- `exchange` VARCHAR(50)
- `is_estimated` BOOLEAN
- `created_at` TIMESTAMPTZ

**Indexes**: symbol, timestamp, symbol+timestamp, exchange, rate, created_at
**Unique**: (symbol, timestamp, exchange)
**Partial**: High positive funding (>0.0005), High negative funding (<-0.0005)

## Views

### v_active_positions (001_initial_schema.sql)
All currently open positions with signal scores from trades table
- All position columns + signal_score, technical_score, llm_score, minutes_held

### v_recent_trades (001_initial_schema.sql)
Last 100 closed trades with key metrics and duration

### v_decision_analysis (002_analysis_tables.sql)
Complete trading decision breakdown with all component analyses
- Joins trading_decisions with technical_signals, llm_analyses, ml_predictions

### v_analysis_performance (002_analysis_tables.sql)
Track accuracy of different analysis methods against actual trade outcomes
- Correlates decisions with trade results to measure prediction accuracy

### v_risk_summary (003_risk_tables.sql)
Real-time overview of risk management status
- Active circuit breakers/kill switches, today's P&L, recent events, 7-day performance

### v_market_data_summary (004_market_data_tables.sql)
Latest market data snapshot for each symbol
- Latest OHLCV, Fear & Greed Index, Funding Rate per symbol

### v_funding_rate_analysis (004_market_data_tables.sql)
Funding rate trend analysis for trading cost optimization
- Current, average, extreme funding rates over different periods

### v_volatility_analysis (004_market_data_tables.sql)
Price volatility metrics for risk management
- Volatility, ATR proxy, volume analysis, price changes

### v_market_correlation (004_market_data_tables.sql)
Correlation analysis between market metrics
- Daily correlation between Fear & Greed, funding rates, and price changes

## Functions

### get_latest_price(p_symbol, p_exchange, p_timeframe) → DECIMAL(20,8)
Get the most recent closing price from ohlcv table

### calculate_position_pnl(p_position_id, p_current_price) → TABLE
Calculate unrealized P&L and percentage for a position

### is_trading_allowed() → TABLE(allowed BOOLEAN, reason TEXT, cooldown_expires TIMESTAMP)
Check if trading is currently allowed based on circuit breakers and kill switch

### calculate_daily_pnl(p_date) → VOID
Calculate and update daily P&L metrics for a given date

### get_latest_ohlcv_price(p_symbol, p_timeframe) → DECIMAL(20,8)
Get the most recent close price from ohlcv_data table

### get_latest_fear_greed() → TABLE(value INTEGER, classification VARCHAR, timestamp TIMESTAMP)
Get the most recent Fear & Greed Index reading

### get_latest_funding_rate(p_symbol, p_exchange) → TABLE(rate, rate_bps, annualized_rate, timestamp)
Get the most recent funding rate for a symbol

## Triggers

### update_updated_at_column()
Automatically updates `updated_at` timestamp on row modifications
- Applied to: trades, positions, daily_performance, circuit_breaker_events, kill_switch_events, daily_pnl

### set_circuit_breaker_cooldown_expiry()
Auto-calculates `cooldown_expires_at` based on `triggered_at` + `cooldown_hours`
- Applied to: circuit_breaker_events

### set_fear_greed_classification()
Auto-sets `classification` based on `value` (0-25: extreme_fear, 26-45: fear, etc.)
- Applied to: fear_greed_index

### calculate_funding_rate_metrics()
Auto-calculates `rate_bps`, `annualized_rate`, and `premium`
- Applied to: funding_rates

## Data Types Reference

- **BIGSERIAL**: Auto-incrementing 64-bit integer
- **VARCHAR(N)**: Variable-length string with max length
- **TEXT**: Unlimited length string
- **DECIMAL(20,8)**: 20 total digits, 8 after decimal (for prices)
- **DECIMAL(5,4)**: 5 total digits, 4 after decimal (for 0-1 scores)
- **DECIMAL(10,4)**: 10 total digits, 4 after decimal (for percentages)
- **INTEGER**: 32-bit integer
- **BOOLEAN**: True/False
- **TIMESTAMPTZ**: Timestamp with timezone
- **DATE**: Date without time
- **JSONB**: Binary JSON with indexing support

## Index Types

- **B-tree** (default): Standard index for equality and range queries
- **GIN**: Generalized Inverted Index for JSONB and full-text search
- **Unique**: Ensures no duplicate values
- **Partial**: Index only rows matching WHERE clause

## Naming Conventions

- **Tables**: snake_case, plural nouns (trades, positions)
- **Columns**: snake_case (entry_price, win_rate)
- **Indexes**: idx_{table}_{columns} (idx_trades_symbol)
- **Views**: v_{name} (v_active_positions)
- **Functions**: {verb}_{noun}_{suffix} (get_latest_price)
- **Triggers**: {action}_{table}_{purpose} (update_trades_updated_at)

## Size Estimates

For 1 year of continuous trading (estimated):

| Table | Row Size | Est. Rows/Year | Total Size |
|-------|----------|----------------|------------|
| ohlcv_data | ~150 bytes | ~5M (1m candles) | ~750 MB |
| trades | ~500 bytes | ~10K | ~5 MB |
| positions | ~300 bytes | ~10K | ~3 MB |
| technical_signals | ~400 bytes | ~100K | ~40 MB |
| llm_analyses | ~2KB | ~50K | ~100 MB |
| ml_predictions | ~300 bytes | ~100K | ~30 MB |
| trading_decisions | ~500 bytes | ~100K | ~50 MB |
| funding_rates | ~150 bytes | ~30K (8h intervals) | ~5 MB |
| fear_greed_index | ~100 bytes | ~365 | <1 MB |

**Total estimated size**: ~1-2 GB per year with indexes
