-- Migration: 001_initial_schema.sql
-- Description: Initial database schema for AQB trading bot
-- Created: 2026-01-17

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- TABLE: ohlcv - Price Data
-- ============================================================================
CREATE TABLE ohlcv (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    quote_volume DECIMAL(20, 8),
    trades_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Ensure no duplicate candles
    CONSTRAINT ohlcv_unique_candle UNIQUE (symbol, exchange, timeframe, timestamp)
);

-- Indexes for common query patterns
CREATE INDEX idx_ohlcv_symbol_timeframe ON ohlcv(symbol, timeframe, timestamp DESC);
CREATE INDEX idx_ohlcv_exchange_symbol ON ohlcv(exchange, symbol);
CREATE INDEX idx_ohlcv_timestamp ON ohlcv(timestamp DESC);
CREATE INDEX idx_ohlcv_lookup ON ohlcv(symbol, exchange, timeframe, timestamp);

COMMENT ON TABLE ohlcv IS 'OHLCV candlestick price data from exchanges';

-- ============================================================================
-- TABLE: trades - Trade Records
-- ============================================================================
CREATE TABLE trades (
    id BIGSERIAL PRIMARY KEY,
    trade_id VARCHAR(100) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('long', 'short')),
    action VARCHAR(10) NOT NULL CHECK (action IN ('open', 'close', 'liquidated')),

    -- Price and quantity
    entry_price DECIMAL(20, 8) NOT NULL,
    exit_price DECIMAL(20, 8),
    quantity DECIMAL(20, 8) NOT NULL,
    leverage DECIMAL(5, 2) DEFAULT 1.0,

    -- P&L
    realized_pnl DECIMAL(20, 8),
    realized_pnl_pct DECIMAL(10, 4),
    fee DECIMAL(20, 8) DEFAULT 0,

    -- Signal scores
    signal_score DECIMAL(5, 4),
    technical_score DECIMAL(5, 4),
    llm_score DECIMAL(5, 4),
    xgb_confidence DECIMAL(5, 4),

    -- Risk management
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    position_size_pct DECIMAL(5, 2),

    -- Analysis data
    decision_reasons JSONB,
    llm_analysis JSONB,

    -- Timestamps
    entry_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    exit_time TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for trade queries
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_exchange ON trades(exchange);
CREATE INDEX idx_trades_entry_time ON trades(entry_time DESC);
CREATE INDEX idx_trades_exit_time ON trades(exit_time DESC) WHERE exit_time IS NOT NULL;
CREATE INDEX idx_trades_symbol_entry ON trades(symbol, entry_time DESC);
CREATE INDEX idx_trades_action ON trades(action);
CREATE INDEX idx_trades_pnl ON trades(realized_pnl) WHERE realized_pnl IS NOT NULL;
CREATE INDEX idx_trades_decision_reasons ON trades USING GIN(decision_reasons);

COMMENT ON TABLE trades IS 'Historical trade records with entry/exit details';

-- ============================================================================
-- TABLE: positions - Current Positions
-- ============================================================================
CREATE TABLE positions (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('long', 'short')),

    -- Entry details
    entry_price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    leverage DECIMAL(5, 2) DEFAULT 1.0,
    margin DECIMAL(20, 8) NOT NULL,

    -- Current state
    current_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),
    unrealized_pnl_pct DECIMAL(10, 4),
    liquidation_price DECIMAL(20, 8),

    -- Risk management
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    trailing_stop DECIMAL(20, 8),

    -- Status
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'closed', 'liquidated')),

    -- Metadata
    trade_id VARCHAR(100),
    entry_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Only one open position per symbol/exchange/side
    CONSTRAINT positions_unique_open UNIQUE (symbol, exchange, side, status)
        WHERE status = 'open'
);

-- Indexes for position queries
CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_positions_exchange ON positions(exchange);
CREATE INDEX idx_positions_symbol_status ON positions(symbol, status);
CREATE INDEX idx_positions_trade_id ON positions(trade_id);

COMMENT ON TABLE positions IS 'Current open and recently closed positions';

-- ============================================================================
-- TABLE: llm_analysis_log - LLM Audit Log
-- ============================================================================
CREATE TABLE llm_analysis_log (
    id BIGSERIAL PRIMARY KEY,
    analysis_type VARCHAR(50) NOT NULL,
    symbol VARCHAR(20),
    exchange VARCHAR(50),

    -- Prompt information
    prompt_template VARCHAR(100),
    prompt_tokens INTEGER,
    prompt_text TEXT,

    -- Response data
    response_text TEXT,
    response_tokens INTEGER,
    response_json JSONB,

    -- Token and cost tracking
    total_tokens INTEGER,
    estimated_cost DECIMAL(10, 6),

    -- Model information
    model_name VARCHAR(100),
    temperature DECIMAL(3, 2),

    -- Status and timing
    status VARCHAR(20) DEFAULT 'success' CHECK (status IN ('success', 'error', 'timeout')),
    error_message TEXT,
    execution_time_ms INTEGER,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for analysis log queries
CREATE INDEX idx_llm_log_type ON llm_analysis_log(analysis_type);
CREATE INDEX idx_llm_log_symbol ON llm_analysis_log(symbol);
CREATE INDEX idx_llm_log_created ON llm_analysis_log(created_at DESC);
CREATE INDEX idx_llm_log_status ON llm_analysis_log(status);
CREATE INDEX idx_llm_log_type_created ON llm_analysis_log(analysis_type, created_at DESC);
CREATE INDEX idx_llm_log_response ON llm_analysis_log USING GIN(response_json);

COMMENT ON TABLE llm_analysis_log IS 'Audit log for all LLM API calls and responses';

-- ============================================================================
-- TABLE: system_events - System Audit Log
-- ============================================================================
CREATE TABLE system_events (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('debug', 'info', 'warning', 'error', 'critical')),
    message TEXT NOT NULL,
    details JSONB,

    -- Related entity references
    symbol VARCHAR(20),
    exchange VARCHAR(50),
    trade_id VARCHAR(100),
    position_id BIGINT,

    -- Stack trace for errors
    stack_trace TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for event log queries
CREATE INDEX idx_events_type ON system_events(event_type);
CREATE INDEX idx_events_severity ON system_events(severity);
CREATE INDEX idx_events_created ON system_events(created_at DESC);
CREATE INDEX idx_events_type_created ON system_events(event_type, created_at DESC);
CREATE INDEX idx_events_symbol ON system_events(symbol) WHERE symbol IS NOT NULL;
CREATE INDEX idx_events_trade_id ON system_events(trade_id) WHERE trade_id IS NOT NULL;
CREATE INDEX idx_events_details ON system_events USING GIN(details);

COMMENT ON TABLE system_events IS 'System-wide event and audit log';

-- ============================================================================
-- TABLE: daily_performance - Daily Summary
-- ============================================================================
CREATE TABLE daily_performance (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    exchange VARCHAR(50) NOT NULL,

    -- Trade statistics
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 2),

    -- P&L metrics
    gross_profit DECIMAL(20, 8) DEFAULT 0,
    gross_loss DECIMAL(20, 8) DEFAULT 0,
    net_pnl DECIMAL(20, 8) DEFAULT 0,
    net_pnl_pct DECIMAL(10, 4),
    total_fees DECIMAL(20, 8) DEFAULT 0,

    -- Position metrics
    avg_position_size DECIMAL(20, 8),
    max_position_size DECIMAL(20, 8),
    total_volume DECIMAL(20, 8),

    -- Risk metrics
    max_drawdown DECIMAL(10, 4),
    max_drawdown_amount DECIMAL(20, 8),
    sharpe_ratio DECIMAL(10, 4),
    profit_factor DECIMAL(10, 4),

    -- Balance tracking
    starting_balance DECIMAL(20, 8),
    ending_balance DECIMAL(20, 8),
    peak_balance DECIMAL(20, 8),

    -- Additional metrics
    avg_trade_duration_minutes INTEGER,
    longest_winning_streak INTEGER,
    longest_losing_streak INTEGER,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance queries
CREATE INDEX idx_daily_perf_date ON daily_performance(date DESC);
CREATE INDEX idx_daily_perf_exchange ON daily_performance(exchange);
CREATE INDEX idx_daily_perf_date_exchange ON daily_performance(date DESC, exchange);
CREATE INDEX idx_daily_perf_pnl ON daily_performance(net_pnl DESC);

COMMENT ON TABLE daily_performance IS 'Daily aggregated trading performance metrics';

-- ============================================================================
-- TRIGGERS: Auto-update timestamps
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to relevant tables
CREATE TRIGGER update_trades_updated_at
    BEFORE UPDATE ON trades
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_daily_performance_updated_at
    BEFORE UPDATE ON daily_performance
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger to update last_updated on positions
CREATE TRIGGER update_positions_last_updated
    BEFORE UPDATE ON positions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- VIEWS: Useful queries
-- ============================================================================

-- View: Active positions with current P&L
CREATE VIEW v_active_positions AS
SELECT
    p.*,
    t.signal_score,
    t.technical_score,
    t.llm_score,
    EXTRACT(EPOCH FROM (NOW() - p.entry_time))/60 AS minutes_held
FROM positions p
LEFT JOIN trades t ON p.trade_id = t.trade_id
WHERE p.status = 'open';

COMMENT ON VIEW v_active_positions IS 'All currently open positions with signal scores';

-- View: Recent trade performance
CREATE VIEW v_recent_trades AS
SELECT
    symbol,
    exchange,
    side,
    entry_price,
    exit_price,
    quantity,
    realized_pnl,
    realized_pnl_pct,
    signal_score,
    technical_score,
    llm_score,
    entry_time,
    exit_time,
    EXTRACT(EPOCH FROM (exit_time - entry_time))/60 AS duration_minutes
FROM trades
WHERE action = 'close'
ORDER BY exit_time DESC
LIMIT 100;

COMMENT ON VIEW v_recent_trades IS 'Last 100 closed trades with key metrics';

-- ============================================================================
-- FUNCTIONS: Utility functions
-- ============================================================================

-- Function: Get latest price for a symbol
CREATE OR REPLACE FUNCTION get_latest_price(
    p_symbol VARCHAR,
    p_exchange VARCHAR,
    p_timeframe VARCHAR DEFAULT '1m'
)
RETURNS DECIMAL(20, 8) AS $$
DECLARE
    latest_price DECIMAL(20, 8);
BEGIN
    SELECT close INTO latest_price
    FROM ohlcv
    WHERE symbol = p_symbol
        AND exchange = p_exchange
        AND timeframe = p_timeframe
    ORDER BY timestamp DESC
    LIMIT 1;

    RETURN latest_price;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_latest_price IS 'Get the most recent closing price for a symbol';

-- Function: Calculate position P&L
CREATE OR REPLACE FUNCTION calculate_position_pnl(
    p_position_id BIGINT,
    p_current_price DECIMAL(20, 8)
)
RETURNS TABLE(
    unrealized_pnl DECIMAL(20, 8),
    unrealized_pnl_pct DECIMAL(10, 4)
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        CASE
            WHEN side = 'long' THEN
                (p_current_price - entry_price) * quantity * leverage
            WHEN side = 'short' THEN
                (entry_price - p_current_price) * quantity * leverage
        END AS unrealized_pnl,
        CASE
            WHEN side = 'long' THEN
                ((p_current_price - entry_price) / entry_price) * 100 * leverage
            WHEN side = 'short' THEN
                ((entry_price - p_current_price) / entry_price) * 100 * leverage
        END AS unrealized_pnl_pct
    FROM positions
    WHERE id = p_position_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION calculate_position_pnl IS 'Calculate unrealized P&L for a position';

-- ============================================================================
-- GRANTS: Set up permissions (adjust as needed)
-- ============================================================================

-- Grant usage on sequences
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO PUBLIC;

-- Grant select, insert, update on tables
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO PUBLIC;

-- ============================================================================
-- INITIAL DATA: None required for initial schema
-- ============================================================================

-- Migration complete
