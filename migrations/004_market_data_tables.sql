-- Migration: 004_market_data_tables.sql
-- Description: Market data tables for OHLCV, Fear & Greed Index, and funding rates
-- Created: 2026-01-18

-- =============================================================================
-- OHLCV DATA TABLE
-- =============================================================================
-- Stores candlestick price data for technical analysis
-- Note: This is separate from the ohlcv table in 001_initial_schema.sql
-- This version is optimized for the IFTB Trading Bot's specific needs
CREATE TABLE IF NOT EXISTS ohlcv_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL CHECK (timeframe IN ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w')),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,

    -- OHLCV values
    open DECIMAL(20, 8) NOT NULL CHECK (open > 0),
    high DECIMAL(20, 8) NOT NULL CHECK (high > 0),
    low DECIMAL(20, 8) NOT NULL CHECK (low > 0),
    close DECIMAL(20, 8) NOT NULL CHECK (close > 0),
    volume DECIMAL(20, 8) NOT NULL CHECK (volume >= 0),

    -- Additional volume metrics
    quote_volume DECIMAL(20, 8),
    taker_buy_volume DECIMAL(20, 8),
    taker_buy_quote_volume DECIMAL(20, 8),

    -- Trade count
    trades_count INTEGER,

    -- Data quality flags
    is_complete BOOLEAN DEFAULT TRUE,
    has_gaps BOOLEAN DEFAULT FALSE,

    -- Source information
    exchange VARCHAR(50) NOT NULL DEFAULT 'binance',
    data_source VARCHAR(50),

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Unique constraint: one candle per symbol/timeframe/timestamp
CREATE UNIQUE INDEX idx_ohlcv_data_unique ON ohlcv_data(symbol, timeframe, timestamp, exchange);

-- Indexes for ohlcv_data table
CREATE INDEX idx_ohlcv_data_symbol ON ohlcv_data(symbol);
CREATE INDEX idx_ohlcv_data_timestamp ON ohlcv_data(timestamp DESC);
CREATE INDEX idx_ohlcv_data_symbol_timeframe ON ohlcv_data(symbol, timeframe);
CREATE INDEX idx_ohlcv_data_symbol_timeframe_timestamp ON ohlcv_data(symbol, timeframe, timestamp DESC);
CREATE INDEX idx_ohlcv_data_exchange ON ohlcv_data(exchange);
CREATE INDEX idx_ohlcv_data_volume ON ohlcv_data(volume DESC);

-- Partial index for incomplete data
CREATE INDEX idx_ohlcv_data_incomplete ON ohlcv_data(symbol, timeframe, timestamp)
    WHERE is_complete = FALSE OR has_gaps = TRUE;

-- Partial index for recent data (last 30 days)
CREATE INDEX idx_ohlcv_data_recent ON ohlcv_data(symbol, timeframe, timestamp DESC)
    WHERE timestamp >= NOW() - INTERVAL '30 days';

-- Check constraint: OHLCV values must be consistent
ALTER TABLE ohlcv_data ADD CONSTRAINT check_ohlcv_values
    CHECK (high >= open AND high >= close AND high >= low AND low <= open AND low <= close);

COMMENT ON TABLE ohlcv_data IS 'OHLCV candlestick price data for technical analysis';

-- =============================================================================
-- FEAR & GREED INDEX TABLE
-- =============================================================================
-- Stores the Crypto Fear & Greed Index for market sentiment analysis
CREATE TABLE IF NOT EXISTS fear_greed_index (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Index value (0-100)
    value INTEGER NOT NULL CHECK (value >= 0 AND value <= 100),

    -- Classification
    classification VARCHAR(20) NOT NULL CHECK (classification IN (
        'extreme_fear',
        'fear',
        'neutral',
        'greed',
        'extreme_greed'
    )),

    -- Value change
    value_yesterday INTEGER,
    value_last_week INTEGER,
    value_last_month INTEGER,

    -- Component scores (if available)
    volatility_score INTEGER CHECK (volatility_score >= 0 AND volatility_score <= 100),
    market_momentum_score INTEGER CHECK (market_momentum_score >= 0 AND market_momentum_score <= 100),
    social_media_score INTEGER CHECK (social_media_score >= 0 AND social_media_score <= 100),
    surveys_score INTEGER CHECK (surveys_score >= 0 AND surveys_score <= 100),
    dominance_score INTEGER CHECK (dominance_score >= 0 AND dominance_score <= 100),
    trends_score INTEGER CHECK (trends_score >= 0 AND trends_score <= 100),

    -- Data source
    data_source VARCHAR(50) DEFAULT 'alternative.me',

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Unique constraint: one reading per timestamp
CREATE UNIQUE INDEX idx_fear_greed_timestamp_unique ON fear_greed_index(timestamp);

-- Indexes for fear_greed_index table
CREATE INDEX idx_fear_greed_timestamp ON fear_greed_index(timestamp DESC);
CREATE INDEX idx_fear_greed_value ON fear_greed_index(value);
CREATE INDEX idx_fear_greed_classification ON fear_greed_index(classification);
CREATE INDEX idx_fear_greed_created_at ON fear_greed_index(created_at DESC);

-- Partial indexes for extreme conditions
CREATE INDEX idx_fear_greed_extreme_fear ON fear_greed_index(timestamp DESC)
    WHERE classification IN ('extreme_fear', 'fear');

CREATE INDEX idx_fear_greed_extreme_greed ON fear_greed_index(timestamp DESC)
    WHERE classification IN ('extreme_greed', 'greed');

-- Trigger to automatically set classification based on value
CREATE OR REPLACE FUNCTION set_fear_greed_classification()
RETURNS TRIGGER AS $$
BEGIN
    NEW.classification = CASE
        WHEN NEW.value <= 25 THEN 'extreme_fear'
        WHEN NEW.value <= 45 THEN 'fear'
        WHEN NEW.value <= 55 THEN 'neutral'
        WHEN NEW.value <= 75 THEN 'greed'
        ELSE 'extreme_greed'
    END;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_fear_greed_classification
    BEFORE INSERT OR UPDATE ON fear_greed_index
    FOR EACH ROW
    EXECUTE FUNCTION set_fear_greed_classification();

COMMENT ON TABLE fear_greed_index IS 'Crypto Fear & Greed Index for market sentiment analysis';

-- =============================================================================
-- FUNDING RATES TABLE
-- =============================================================================
-- Stores perpetual futures funding rates
CREATE TABLE IF NOT EXISTS funding_rates (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Funding rate (typically as a percentage, e.g., 0.0001 = 0.01%)
    rate DECIMAL(10, 8) NOT NULL,

    -- Rate in basis points for easier reading
    rate_bps DECIMAL(10, 4),

    -- Annualized rate (rate * funding_periods_per_year)
    annualized_rate DECIMAL(10, 6),

    -- Mark price at time of funding
    mark_price DECIMAL(20, 8),
    index_price DECIMAL(20, 8),

    -- Premium/discount
    premium DECIMAL(10, 8),

    -- Estimated next funding rate (if available)
    predicted_rate DECIMAL(10, 8),

    -- Funding interval (typically 8 hours)
    funding_interval_hours INTEGER DEFAULT 8,

    -- Exchange information
    exchange VARCHAR(50) NOT NULL DEFAULT 'binance',

    -- Data quality
    is_estimated BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Unique constraint: one funding rate per symbol/timestamp/exchange
CREATE UNIQUE INDEX idx_funding_rates_unique ON funding_rates(symbol, timestamp, exchange);

-- Indexes for funding_rates table
CREATE INDEX idx_funding_rates_symbol ON funding_rates(symbol);
CREATE INDEX idx_funding_rates_timestamp ON funding_rates(timestamp DESC);
CREATE INDEX idx_funding_rates_symbol_timestamp ON funding_rates(symbol, timestamp DESC);
CREATE INDEX idx_funding_rates_exchange ON funding_rates(exchange);
CREATE INDEX idx_funding_rates_rate ON funding_rates(rate);
CREATE INDEX idx_funding_rates_created_at ON funding_rates(created_at DESC);

-- Partial indexes for extreme funding rates
CREATE INDEX idx_funding_rates_high_positive ON funding_rates(symbol, timestamp DESC)
    WHERE rate > 0.0005; -- High positive funding (expensive to be long)

CREATE INDEX idx_funding_rates_high_negative ON funding_rates(symbol, timestamp DESC)
    WHERE rate < -0.0005; -- High negative funding (expensive to be short)

-- Trigger to calculate derived values
CREATE OR REPLACE FUNCTION calculate_funding_rate_metrics()
RETURNS TRIGGER AS $$
BEGIN
    -- Convert rate to basis points (1 bps = 0.0001 = 0.01%)
    NEW.rate_bps = NEW.rate * 10000;

    -- Annualize the rate (assuming 8-hour funding periods = 3 times per day = 1095 times per year)
    IF NEW.funding_interval_hours IS NOT NULL THEN
        NEW.annualized_rate = NEW.rate * (24.0 / NEW.funding_interval_hours) * 365;
    END IF;

    -- Calculate premium if both prices available
    IF NEW.mark_price IS NOT NULL AND NEW.index_price IS NOT NULL AND NEW.index_price > 0 THEN
        NEW.premium = (NEW.mark_price - NEW.index_price) / NEW.index_price;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_funding_rate_metrics
    BEFORE INSERT OR UPDATE ON funding_rates
    FOR EACH ROW
    EXECUTE FUNCTION calculate_funding_rate_metrics();

COMMENT ON TABLE funding_rates IS 'Perpetual futures funding rates for trading cost analysis';

-- =============================================================================
-- MARKET DATA SUMMARY VIEW
-- =============================================================================
-- Real-time view of latest market data for each symbol
CREATE VIEW v_market_data_summary AS
SELECT
    o.symbol,
    o.timeframe,

    -- Latest OHLCV
    o.timestamp AS last_candle_time,
    o.close AS last_price,
    o.volume AS last_volume,
    ((o.close - o.open) / o.open * 100) AS last_candle_change_pct,

    -- Latest Fear & Greed
    fg.value AS fear_greed_value,
    fg.classification AS fear_greed_classification,
    fg.timestamp AS fear_greed_time,

    -- Latest Funding Rate
    fr.rate AS funding_rate,
    fr.rate_bps AS funding_rate_bps,
    fr.annualized_rate AS funding_rate_annual,
    fr.timestamp AS funding_rate_time

FROM (
    -- Latest OHLCV per symbol/timeframe
    SELECT DISTINCT ON (symbol, timeframe)
        symbol, timeframe, timestamp, open, high, low, close, volume
    FROM ohlcv_data
    ORDER BY symbol, timeframe, timestamp DESC
) o

LEFT JOIN LATERAL (
    -- Latest Fear & Greed Index
    SELECT value, classification, timestamp
    FROM fear_greed_index
    ORDER BY timestamp DESC
    LIMIT 1
) fg ON TRUE

LEFT JOIN LATERAL (
    -- Latest Funding Rate per symbol
    SELECT rate, rate_bps, annualized_rate, timestamp
    FROM funding_rates
    WHERE funding_rates.symbol = o.symbol
    ORDER BY timestamp DESC
    LIMIT 1
) fr ON TRUE

ORDER BY o.symbol, o.timeframe;

COMMENT ON VIEW v_market_data_summary IS 'Latest market data snapshot for each symbol';

-- =============================================================================
-- FUNDING RATE ANALYSIS VIEW
-- =============================================================================
-- View to analyze funding rate trends
CREATE VIEW v_funding_rate_analysis AS
SELECT
    symbol,
    exchange,

    -- Current funding rate
    (SELECT rate FROM funding_rates fr1
     WHERE fr1.symbol = fr.symbol AND fr1.exchange = fr.exchange
     ORDER BY timestamp DESC LIMIT 1) AS current_rate,

    -- Average rates over different periods
    AVG(rate) FILTER (WHERE timestamp >= NOW() - INTERVAL '24 hours') AS avg_rate_24h,
    AVG(rate) FILTER (WHERE timestamp >= NOW() - INTERVAL '7 days') AS avg_rate_7d,
    AVG(rate) FILTER (WHERE timestamp >= NOW() - INTERVAL '30 days') AS avg_rate_30d,

    -- Extreme rates in last 24 hours
    MAX(rate) FILTER (WHERE timestamp >= NOW() - INTERVAL '24 hours') AS max_rate_24h,
    MIN(rate) FILTER (WHERE timestamp >= NOW() - INTERVAL '24 hours') AS min_rate_24h,

    -- Count of extreme funding periods
    COUNT(*) FILTER (WHERE rate > 0.0005 AND timestamp >= NOW() - INTERVAL '7 days') AS high_positive_periods_7d,
    COUNT(*) FILTER (WHERE rate < -0.0005 AND timestamp >= NOW() - INTERVAL '7 days') AS high_negative_periods_7d,

    -- Latest timestamp
    MAX(timestamp) AS last_update

FROM funding_rates fr
GROUP BY symbol, exchange
ORDER BY symbol, exchange;

COMMENT ON VIEW v_funding_rate_analysis IS 'Funding rate trend analysis for trading cost optimization';

-- =============================================================================
-- VOLATILITY ANALYSIS VIEW
-- =============================================================================
-- View to analyze price volatility from OHLCV data
CREATE VIEW v_volatility_analysis AS
SELECT
    symbol,
    timeframe,

    -- Current volatility (from latest candles)
    STDDEV((high - low) / low * 100) FILTER (WHERE timestamp >= NOW() - INTERVAL '24 hours') AS volatility_24h,
    STDDEV((high - low) / low * 100) FILTER (WHERE timestamp >= NOW() - INTERVAL '7 days') AS volatility_7d,

    -- Average true range (ATR proxy)
    AVG((high - low) / low * 100) FILTER (WHERE timestamp >= NOW() - INTERVAL '24 hours') AS avg_range_24h,
    AVG((high - low) / low * 100) FILTER (WHERE timestamp >= NOW() - INTERVAL '7 days') AS avg_range_7d,

    -- Volume analysis
    AVG(volume) FILTER (WHERE timestamp >= NOW() - INTERVAL '24 hours') AS avg_volume_24h,
    MAX(volume) FILTER (WHERE timestamp >= NOW() - INTERVAL '24 hours') AS max_volume_24h,

    -- Price change
    (
        (SELECT close FROM ohlcv_data o2
         WHERE o2.symbol = o.symbol AND o2.timeframe = o.timeframe
         ORDER BY timestamp DESC LIMIT 1)
        -
        (SELECT close FROM ohlcv_data o3
         WHERE o3.symbol = o.symbol AND o3.timeframe = o.timeframe
         AND o3.timestamp >= NOW() - INTERVAL '24 hours'
         ORDER BY timestamp ASC LIMIT 1)
    ) AS price_change_24h,

    -- Latest data
    MAX(timestamp) AS last_update

FROM ohlcv_data o
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY symbol, timeframe
ORDER BY symbol, timeframe;

COMMENT ON VIEW v_volatility_analysis IS 'Price volatility metrics for risk management';

-- =============================================================================
-- MARKET CORRELATION VIEW
-- =============================================================================
-- View to analyze correlation between Fear & Greed and funding rates
CREATE VIEW v_market_correlation AS
SELECT
    fr.symbol,
    DATE_TRUNC('day', fr.timestamp) AS date,

    -- Average funding rate for the day
    AVG(fr.rate) AS avg_funding_rate,

    -- Fear & Greed value for the day (using first reading)
    (SELECT value FROM fear_greed_index
     WHERE DATE_TRUNC('day', timestamp) = DATE_TRUNC('day', fr.timestamp)
     ORDER BY timestamp ASC LIMIT 1) AS fear_greed_value,

    -- Price change for the day
    (
        (SELECT close FROM ohlcv_data
         WHERE symbol = fr.symbol AND timeframe = '1d'
         AND DATE_TRUNC('day', timestamp) = DATE_TRUNC('day', fr.timestamp)
         ORDER BY timestamp DESC LIMIT 1)
        -
        (SELECT close FROM ohlcv_data
         WHERE symbol = fr.symbol AND timeframe = '1d'
         AND DATE_TRUNC('day', timestamp) = DATE_TRUNC('day', fr.timestamp) - INTERVAL '1 day'
         ORDER BY timestamp DESC LIMIT 1)
    ) AS price_change

FROM funding_rates fr
WHERE fr.timestamp >= NOW() - INTERVAL '30 days'
GROUP BY fr.symbol, DATE_TRUNC('day', fr.timestamp)
ORDER BY date DESC, fr.symbol;

COMMENT ON VIEW v_market_correlation IS 'Correlation analysis between market metrics';

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to get latest price for a symbol
CREATE OR REPLACE FUNCTION get_latest_ohlcv_price(
    p_symbol VARCHAR,
    p_timeframe VARCHAR DEFAULT '1m'
)
RETURNS DECIMAL(20, 8) AS $$
DECLARE
    v_price DECIMAL(20, 8);
BEGIN
    SELECT close INTO v_price
    FROM ohlcv_data
    WHERE symbol = p_symbol AND timeframe = p_timeframe
    ORDER BY timestamp DESC
    LIMIT 1;

    RETURN v_price;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_latest_ohlcv_price IS 'Get the most recent close price for a symbol';

-- Function to get latest Fear & Greed Index
CREATE OR REPLACE FUNCTION get_latest_fear_greed()
RETURNS TABLE(
    value INTEGER,
    classification VARCHAR(20),
    timestamp TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        fg.value,
        fg.classification,
        fg.timestamp
    FROM fear_greed_index fg
    ORDER BY fg.timestamp DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_latest_fear_greed IS 'Get the most recent Fear & Greed Index reading';

-- Function to get latest funding rate for a symbol
CREATE OR REPLACE FUNCTION get_latest_funding_rate(
    p_symbol VARCHAR,
    p_exchange VARCHAR DEFAULT 'binance'
)
RETURNS TABLE(
    rate DECIMAL(10, 8),
    rate_bps DECIMAL(10, 4),
    annualized_rate DECIMAL(10, 6),
    timestamp TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        fr.rate,
        fr.rate_bps,
        fr.annualized_rate,
        fr.timestamp
    FROM funding_rates fr
    WHERE fr.symbol = p_symbol AND fr.exchange = p_exchange
    ORDER BY fr.timestamp DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_latest_funding_rate IS 'Get the most recent funding rate for a symbol';

-- Migration complete
