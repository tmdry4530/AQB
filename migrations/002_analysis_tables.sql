-- Migration: 002_analysis_tables.sql
-- Description: Analysis history tables for technical signals, LLM analysis, ML predictions, and trading decisions
-- Created: 2026-01-18

-- =============================================================================
-- TECHNICAL SIGNALS TABLE
-- =============================================================================
-- Stores aggregated technical analysis signals from multiple indicators
CREATE TABLE IF NOT EXISTS technical_signals (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Overall signal aggregation
    overall_signal VARCHAR(10) NOT NULL CHECK (overall_signal IN ('bullish', 'bearish', 'neutral')),
    confidence DECIMAL(5, 4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),

    -- Signal counts
    bullish_count INTEGER DEFAULT 0,
    bearish_count INTEGER DEFAULT 0,
    neutral_count INTEGER DEFAULT 0,

    -- Detailed indicator signals stored as JSON
    -- Example: {"RSI": {"signal": "bullish", "value": 65}, "MACD": {"signal": "bearish", "histogram": -0.5}}
    indicators_json JSONB NOT NULL,

    -- Individual key indicators (for quick filtering)
    rsi_value DECIMAL(10, 4),
    rsi_signal VARCHAR(10),
    macd_histogram DECIMAL(20, 8),
    macd_signal VARCHAR(10),
    bb_position DECIMAL(10, 4),
    bb_signal VARCHAR(10),

    -- Metadata
    timeframe VARCHAR(10) NOT NULL DEFAULT '1h',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for technical_signals table
CREATE INDEX idx_technical_signals_symbol ON technical_signals(symbol);
CREATE INDEX idx_technical_signals_timestamp ON technical_signals(timestamp DESC);
CREATE INDEX idx_technical_signals_symbol_timestamp ON technical_signals(symbol, timestamp DESC);
CREATE INDEX idx_technical_signals_overall_signal ON technical_signals(overall_signal);
CREATE INDEX idx_technical_signals_confidence ON technical_signals(confidence DESC);
CREATE INDEX idx_technical_signals_indicators ON technical_signals USING GIN(indicators_json);

-- Partial index for high-confidence signals
CREATE INDEX idx_technical_signals_high_confidence ON technical_signals(symbol, timestamp DESC)
    WHERE confidence >= 0.7;

COMMENT ON TABLE technical_signals IS 'Aggregated technical analysis signals from multiple indicators';

-- =============================================================================
-- LLM ANALYSES TABLE
-- =============================================================================
-- Stores LLM-based market analysis and sentiment
CREATE TABLE IF NOT EXISTS llm_analyses (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Analysis results
    sentiment VARCHAR(10) NOT NULL CHECK (sentiment IN ('bullish', 'bearish', 'neutral', 'uncertain')),
    confidence DECIMAL(5, 4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    summary TEXT NOT NULL,

    -- Detailed factors as JSON array
    -- Example: [{"factor": "strong momentum", "impact": "positive", "weight": 0.8}]
    key_factors_json JSONB,

    -- Model information
    model VARCHAR(50) NOT NULL,
    cached BOOLEAN DEFAULT FALSE,

    -- Token usage tracking
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,

    -- Cost tracking
    estimated_cost DECIMAL(10, 6),

    -- Response metadata
    response_time_ms INTEGER,
    temperature DECIMAL(3, 2),

    -- Raw data (optional, for debugging)
    prompt_text TEXT,
    response_text TEXT,

    -- Status
    status VARCHAR(20) DEFAULT 'success' CHECK (status IN ('success', 'error', 'timeout', 'cached')),
    error_message TEXT,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for llm_analyses table
CREATE INDEX idx_llm_analyses_symbol ON llm_analyses(symbol);
CREATE INDEX idx_llm_analyses_timestamp ON llm_analyses(timestamp DESC);
CREATE INDEX idx_llm_analyses_symbol_timestamp ON llm_analyses(symbol, timestamp DESC);
CREATE INDEX idx_llm_analyses_sentiment ON llm_analyses(sentiment);
CREATE INDEX idx_llm_analyses_confidence ON llm_analyses(confidence DESC);
CREATE INDEX idx_llm_analyses_model ON llm_analyses(model);
CREATE INDEX idx_llm_analyses_status ON llm_analyses(status);
CREATE INDEX idx_llm_analyses_cached ON llm_analyses(cached);
CREATE INDEX idx_llm_analyses_key_factors ON llm_analyses USING GIN(key_factors_json);

-- Partial index for successful high-confidence analyses
CREATE INDEX idx_llm_analyses_high_confidence ON llm_analyses(symbol, timestamp DESC)
    WHERE confidence >= 0.7 AND status = 'success';

COMMENT ON TABLE llm_analyses IS 'LLM-based market analysis and sentiment with token tracking';

-- =============================================================================
-- ML PREDICTIONS TABLE
-- =============================================================================
-- Stores machine learning model predictions (e.g., XGBoost)
CREATE TABLE IF NOT EXISTS ml_predictions (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Prediction results
    action VARCHAR(10) NOT NULL CHECK (action IN ('long', 'short', 'hold', 'close')),
    confidence DECIMAL(5, 4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),

    -- Probability distribution
    prob_long DECIMAL(5, 4) CHECK (prob_long >= 0 AND prob_long <= 1),
    prob_short DECIMAL(5, 4) CHECK (prob_short >= 0 AND prob_short <= 1),
    prob_hold DECIMAL(5, 4) CHECK (prob_hold >= 0 AND prob_hold <= 1),

    -- Feature importance for explainability
    -- Example: {"rsi": 0.25, "macd": 0.20, "volume": 0.15}
    feature_importance_json JSONB,

    -- Model information
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) DEFAULT 'xgboost',

    -- Performance metadata
    prediction_time_ms INTEGER,
    features_used INTEGER,

    -- Backtesting correlation (if available)
    historical_accuracy DECIMAL(5, 4),

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for ml_predictions table
CREATE INDEX idx_ml_predictions_symbol ON ml_predictions(symbol);
CREATE INDEX idx_ml_predictions_timestamp ON ml_predictions(timestamp DESC);
CREATE INDEX idx_ml_predictions_symbol_timestamp ON ml_predictions(symbol, timestamp DESC);
CREATE INDEX idx_ml_predictions_action ON ml_predictions(action);
CREATE INDEX idx_ml_predictions_confidence ON ml_predictions(confidence DESC);
CREATE INDEX idx_ml_predictions_model_version ON ml_predictions(model_version);
CREATE INDEX idx_ml_predictions_feature_importance ON ml_predictions USING GIN(feature_importance_json);

-- Partial index for high-confidence predictions
CREATE INDEX idx_ml_predictions_high_confidence ON ml_predictions(symbol, timestamp DESC)
    WHERE confidence >= 0.7;

COMMENT ON TABLE ml_predictions IS 'Machine learning model predictions with confidence scores';

-- =============================================================================
-- TRADING DECISIONS TABLE
-- =============================================================================
-- Stores final trading decisions after combining all signals
CREATE TABLE IF NOT EXISTS trading_decisions (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Decision outcome
    action VARCHAR(10) NOT NULL CHECK (action IN ('long', 'short', 'close', 'hold', 'vetoed')),
    confidence DECIMAL(5, 4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),

    -- Position sizing
    position_size DECIMAL(20, 8),
    position_size_pct DECIMAL(5, 2),
    leverage INTEGER CHECK (leverage > 0 AND leverage <= 125),

    -- Risk parameters
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    entry_price DECIMAL(20, 8),

    -- Decision reasoning
    -- Example: [{"source": "technical", "signal": "bullish", "weight": 0.4}, {"source": "llm", "signal": "bullish", "weight": 0.3}]
    reasons_json JSONB NOT NULL,

    -- Component signals (references to other tables)
    technical_signal_id BIGINT REFERENCES technical_signals(id) ON DELETE SET NULL,
    llm_analysis_id BIGINT REFERENCES llm_analyses(id) ON DELETE SET NULL,
    ml_prediction_id BIGINT REFERENCES ml_predictions(id) ON DELETE SET NULL,

    -- Veto information
    vetoed BOOLEAN DEFAULT FALSE,
    veto_reason TEXT,
    veto_source VARCHAR(50),

    -- Execution tracking
    executed BOOLEAN DEFAULT FALSE,
    order_id BIGINT REFERENCES orders(id) ON DELETE SET NULL,
    execution_price DECIMAL(20, 8),
    executed_at TIMESTAMP WITH TIME ZONE,

    -- Metadata
    decision_latency_ms INTEGER,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for trading_decisions table
CREATE INDEX idx_trading_decisions_symbol ON trading_decisions(symbol);
CREATE INDEX idx_trading_decisions_timestamp ON trading_decisions(timestamp DESC);
CREATE INDEX idx_trading_decisions_symbol_timestamp ON trading_decisions(symbol, timestamp DESC);
CREATE INDEX idx_trading_decisions_action ON trading_decisions(action);
CREATE INDEX idx_trading_decisions_confidence ON trading_decisions(confidence DESC);
CREATE INDEX idx_trading_decisions_vetoed ON trading_decisions(vetoed);
CREATE INDEX idx_trading_decisions_executed ON trading_decisions(executed);
CREATE INDEX idx_trading_decisions_reasons ON trading_decisions USING GIN(reasons_json);
CREATE INDEX idx_trading_decisions_technical_signal_id ON trading_decisions(technical_signal_id);
CREATE INDEX idx_trading_decisions_llm_analysis_id ON trading_decisions(llm_analysis_id);
CREATE INDEX idx_trading_decisions_ml_prediction_id ON trading_decisions(ml_prediction_id);
CREATE INDEX idx_trading_decisions_order_id ON trading_decisions(order_id);

-- Partial indexes for specific decision types
CREATE INDEX idx_trading_decisions_executed_decisions ON trading_decisions(symbol, timestamp DESC)
    WHERE executed = TRUE;

CREATE INDEX idx_trading_decisions_vetoed_decisions ON trading_decisions(symbol, timestamp DESC)
    WHERE vetoed = TRUE;

COMMENT ON TABLE trading_decisions IS 'Final trading decisions combining all analysis signals';

-- =============================================================================
-- ANALYSIS CORRELATION VIEW
-- =============================================================================
-- View to see all analysis components for a decision
CREATE VIEW v_decision_analysis AS
SELECT
    td.id AS decision_id,
    td.symbol,
    td.timestamp,
    td.action,
    td.confidence AS decision_confidence,
    td.executed,
    td.vetoed,

    -- Technical analysis
    ts.overall_signal AS technical_signal,
    ts.confidence AS technical_confidence,
    ts.bullish_count,
    ts.bearish_count,

    -- LLM analysis
    la.sentiment AS llm_sentiment,
    la.confidence AS llm_confidence,
    la.summary AS llm_summary,
    la.cached AS llm_cached,

    -- ML prediction
    mp.action AS ml_action,
    mp.confidence AS ml_confidence,
    mp.prob_long,
    mp.prob_short,

    td.reasons_json
FROM trading_decisions td
LEFT JOIN technical_signals ts ON td.technical_signal_id = ts.id
LEFT JOIN llm_analyses la ON td.llm_analysis_id = la.id
LEFT JOIN ml_predictions mp ON td.ml_prediction_id = mp.id
ORDER BY td.timestamp DESC;

COMMENT ON VIEW v_decision_analysis IS 'Complete view of trading decisions with all component analyses';

-- =============================================================================
-- PERFORMANCE TRACKING VIEW
-- =============================================================================
-- View to track analysis accuracy over time
CREATE VIEW v_analysis_performance AS
SELECT
    td.symbol,
    td.action AS decision_action,
    td.executed,
    td.entry_price AS decision_price,
    t.realized_pnl,
    t.realized_pnl_pct,

    -- Was the decision correct?
    CASE
        WHEN t.realized_pnl > 0 THEN TRUE
        WHEN t.realized_pnl < 0 THEN FALSE
        ELSE NULL
    END AS was_profitable,

    ts.overall_signal AS technical_signal,
    la.sentiment AS llm_sentiment,
    mp.action AS ml_action,

    td.timestamp AS decision_time,
    t.entry_time,
    t.exit_time
FROM trading_decisions td
LEFT JOIN orders o ON td.order_id = o.id
LEFT JOIN trades t ON o.trade_id = t.id
LEFT JOIN technical_signals ts ON td.technical_signal_id = ts.id
LEFT JOIN llm_analyses la ON td.llm_analysis_id = la.id
LEFT JOIN ml_predictions mp ON td.ml_prediction_id = mp.id
WHERE td.executed = TRUE AND t.realized_pnl IS NOT NULL;

COMMENT ON VIEW v_analysis_performance IS 'Track accuracy of different analysis methods against actual trade outcomes';

-- Migration complete
