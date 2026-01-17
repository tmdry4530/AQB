-- Migration: 003_risk_tables.sql
-- Description: Risk management tables for circuit breakers, kill switch, and daily P&L tracking
-- Created: 2026-01-18

-- =============================================================================
-- CIRCUIT BREAKER EVENTS TABLE
-- =============================================================================
-- Tracks automatic trading halts due to risk thresholds
CREATE TABLE IF NOT EXISTS circuit_breaker_events (
    id BIGSERIAL PRIMARY KEY,
    triggered_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Trigger reason and details
    reason VARCHAR(100) NOT NULL,
    trigger_type VARCHAR(50) NOT NULL CHECK (trigger_type IN (
        'max_daily_loss',
        'max_drawdown',
        'rapid_loss',
        'consecutive_losses',
        'position_limit',
        'exchange_error',
        'network_issue',
        'manual'
    )),

    -- Threshold details
    threshold_value DECIMAL(20, 8),
    actual_value DECIMAL(20, 8),

    -- Context at trigger time
    symbol VARCHAR(20),
    daily_pnl DECIMAL(20, 8),
    drawdown_pct DECIMAL(10, 4),
    consecutive_losses INTEGER,
    open_positions INTEGER,
    account_balance DECIMAL(20, 8),

    -- Detailed context as JSON
    -- Example: {"last_trades": [...], "risk_metrics": {...}, "market_conditions": {...}}
    context_json JSONB,

    -- Resolution information
    resolved_at TIMESTAMP WITH TIME ZONE,
    cooldown_hours DECIMAL(5, 2) DEFAULT 1.0,
    cooldown_expires_at TIMESTAMP WITH TIME ZONE,

    -- Status
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'resolved', 'overridden', 'expired')),

    -- Who resolved it
    resolved_by VARCHAR(50),
    resolution_reason TEXT,

    -- Severity level
    severity VARCHAR(20) NOT NULL DEFAULT 'warning' CHECK (severity IN ('info', 'warning', 'critical', 'emergency')),

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for circuit_breaker_events table
CREATE INDEX idx_circuit_breaker_triggered_at ON circuit_breaker_events(triggered_at DESC);
CREATE INDEX idx_circuit_breaker_status ON circuit_breaker_events(status);
CREATE INDEX idx_circuit_breaker_trigger_type ON circuit_breaker_events(trigger_type);
CREATE INDEX idx_circuit_breaker_severity ON circuit_breaker_events(severity);
CREATE INDEX idx_circuit_breaker_symbol ON circuit_breaker_events(symbol) WHERE symbol IS NOT NULL;
CREATE INDEX idx_circuit_breaker_cooldown_expires ON circuit_breaker_events(cooldown_expires_at)
    WHERE status = 'active';
CREATE INDEX idx_circuit_breaker_context ON circuit_breaker_events USING GIN(context_json);

-- Partial index for active circuit breakers
CREATE INDEX idx_circuit_breaker_active ON circuit_breaker_events(triggered_at DESC)
    WHERE status = 'active';

-- Trigger to auto-calculate cooldown expiration
CREATE OR REPLACE FUNCTION set_circuit_breaker_cooldown_expiry()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.cooldown_hours IS NOT NULL AND NEW.triggered_at IS NOT NULL THEN
        NEW.cooldown_expires_at = NEW.triggered_at + (NEW.cooldown_hours || ' hours')::INTERVAL;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_circuit_breaker_cooldown
    BEFORE INSERT OR UPDATE ON circuit_breaker_events
    FOR EACH ROW
    EXECUTE FUNCTION set_circuit_breaker_cooldown_expiry();

COMMENT ON TABLE circuit_breaker_events IS 'Automatic trading halts triggered by risk thresholds';

-- =============================================================================
-- KILL SWITCH EVENTS TABLE
-- =============================================================================
-- Tracks emergency system shutdowns and manual interventions
CREATE TABLE IF NOT EXISTS kill_switch_events (
    id BIGSERIAL PRIMARY KEY,
    activated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Activation details
    reason TEXT NOT NULL,
    trigger_source VARCHAR(50) NOT NULL CHECK (trigger_source IN (
        'manual',
        'api',
        'circuit_breaker_cascade',
        'system_health',
        'exchange_emergency',
        'security_alert',
        'critical_error'
    )),

    -- User/system who activated
    activated_by VARCHAR(100) NOT NULL,

    -- Actions taken on activation
    positions_closed INTEGER DEFAULT 0,
    orders_cancelled INTEGER DEFAULT 0,
    emergency_exit_used BOOLEAN DEFAULT FALSE,

    -- State before kill switch
    balance_before DECIMAL(20, 8),
    equity_before DECIMAL(20, 8),
    open_positions_before INTEGER,
    pending_orders_before INTEGER,

    -- State after kill switch
    balance_after DECIMAL(20, 8),
    equity_after DECIMAL(20, 8),
    emergency_pnl DECIMAL(20, 8),

    -- Detailed action log
    -- Example: [{"time": "...", "action": "cancel_order", "order_id": "...", "status": "success"}]
    action_log_json JSONB,

    -- System state snapshot
    system_state_json JSONB,

    -- Deactivation
    deactivated_at TIMESTAMP WITH TIME ZONE,
    deactivated_by VARCHAR(100),
    deactivation_reason TEXT,

    -- Status
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'deactivated', 'partial_recovery')),

    -- Severity
    severity VARCHAR(20) NOT NULL DEFAULT 'emergency' CHECK (severity IN ('warning', 'critical', 'emergency')),

    -- Recovery status
    recovery_started_at TIMESTAMP WITH TIME ZONE,
    recovery_completed_at TIMESTAMP WITH TIME ZONE,
    recovery_notes TEXT,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for kill_switch_events table
CREATE INDEX idx_kill_switch_activated_at ON kill_switch_events(activated_at DESC);
CREATE INDEX idx_kill_switch_status ON kill_switch_events(status);
CREATE INDEX idx_kill_switch_trigger_source ON kill_switch_events(trigger_source);
CREATE INDEX idx_kill_switch_severity ON kill_switch_events(severity);
CREATE INDEX idx_kill_switch_activated_by ON kill_switch_events(activated_by);
CREATE INDEX idx_kill_switch_action_log ON kill_switch_events USING GIN(action_log_json);

-- Partial index for active kill switches
CREATE INDEX idx_kill_switch_active ON kill_switch_events(activated_at DESC)
    WHERE status = 'active';

COMMENT ON TABLE kill_switch_events IS 'Emergency system shutdowns and manual interventions';

-- =============================================================================
-- DAILY PNL TABLE
-- =============================================================================
-- Daily profit and loss tracking with balance snapshots
CREATE TABLE IF NOT EXISTS daily_pnl (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,

    -- P&L metrics
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    total_pnl DECIMAL(20, 8) DEFAULT 0,

    -- Balance tracking
    balance_start DECIMAL(20, 8) NOT NULL,
    balance_end DECIMAL(20, 8) NOT NULL,
    equity_start DECIMAL(20, 8),
    equity_end DECIMAL(20, 8),

    -- Intraday metrics
    peak_balance DECIMAL(20, 8),
    lowest_balance DECIMAL(20, 8),
    peak_equity DECIMAL(20, 8),
    lowest_equity DECIMAL(20, 8),

    -- Drawdown tracking
    max_drawdown DECIMAL(20, 8),
    max_drawdown_pct DECIMAL(10, 4),
    current_drawdown DECIMAL(20, 8),
    current_drawdown_pct DECIMAL(10, 4),

    -- Trading activity
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 2),

    -- Trade P&L breakdown
    gross_profit DECIMAL(20, 8) DEFAULT 0,
    gross_loss DECIMAL(20, 8) DEFAULT 0,
    average_win DECIMAL(20, 8),
    average_loss DECIMAL(20, 8),
    profit_factor DECIMAL(10, 4),

    -- Position metrics
    long_trades INTEGER DEFAULT 0,
    short_trades INTEGER DEFAULT 0,
    avg_position_size DECIMAL(20, 8),
    max_position_size DECIMAL(20, 8),
    total_volume DECIMAL(20, 8),

    -- Fee tracking
    total_fees DECIMAL(20, 8) DEFAULT 0,
    fee_pct_of_volume DECIMAL(10, 6),

    -- Risk events
    circuit_breakers_triggered INTEGER DEFAULT 0,
    kill_switch_activated BOOLEAN DEFAULT FALSE,

    -- Performance ratios
    sharpe_ratio DECIMAL(10, 4),
    sortino_ratio DECIMAL(10, 4),
    calmar_ratio DECIMAL(10, 4),

    -- Trade duration
    avg_trade_duration_minutes INTEGER,
    shortest_trade_minutes INTEGER,
    longest_trade_minutes INTEGER,

    -- Streak tracking
    current_win_streak INTEGER DEFAULT 0,
    current_loss_streak INTEGER DEFAULT 0,
    longest_win_streak INTEGER DEFAULT 0,
    longest_loss_streak INTEGER DEFAULT 0,

    -- Additional metrics
    best_trade_pnl DECIMAL(20, 8),
    worst_trade_pnl DECIMAL(20, 8),

    -- Metadata
    exchange VARCHAR(50),
    calculation_completed_at TIMESTAMP WITH TIME ZONE,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for daily_pnl table
CREATE INDEX idx_daily_pnl_date ON daily_pnl(date DESC);
CREATE INDEX idx_daily_pnl_total_pnl ON daily_pnl(total_pnl DESC);
CREATE INDEX idx_daily_pnl_win_rate ON daily_pnl(win_rate DESC);
CREATE INDEX idx_daily_pnl_circuit_breakers ON daily_pnl(date DESC)
    WHERE circuit_breakers_triggered > 0;
CREATE INDEX idx_daily_pnl_kill_switch ON daily_pnl(date DESC)
    WHERE kill_switch_activated = TRUE;

COMMENT ON TABLE daily_pnl IS 'Daily profit/loss tracking with comprehensive performance metrics';

-- =============================================================================
-- UPDATE TIMESTAMP TRIGGERS
-- =============================================================================
-- Reuse the existing update_updated_at_column function

CREATE TRIGGER update_circuit_breaker_updated_at
    BEFORE UPDATE ON circuit_breaker_events
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_kill_switch_updated_at
    BEFORE UPDATE ON kill_switch_events
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_daily_pnl_updated_at
    BEFORE UPDATE ON daily_pnl
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- RISK SUMMARY VIEW
-- =============================================================================
-- Real-time view of risk status
CREATE VIEW v_risk_summary AS
SELECT
    -- Active circuit breakers
    (SELECT COUNT(*) FROM circuit_breaker_events WHERE status = 'active') AS active_circuit_breakers,
    (SELECT COUNT(*) FROM kill_switch_events WHERE status = 'active') AS active_kill_switches,

    -- Today's P&L
    (SELECT total_pnl FROM daily_pnl WHERE date = CURRENT_DATE) AS today_pnl,
    (SELECT balance_end FROM daily_pnl WHERE date = CURRENT_DATE) AS current_balance,

    -- Recent circuit breaker
    (SELECT triggered_at FROM circuit_breaker_events ORDER BY triggered_at DESC LIMIT 1) AS last_circuit_breaker,
    (SELECT reason FROM circuit_breaker_events ORDER BY triggered_at DESC LIMIT 1) AS last_circuit_breaker_reason,

    -- Recent kill switch
    (SELECT activated_at FROM kill_switch_events ORDER BY activated_at DESC LIMIT 1) AS last_kill_switch,
    (SELECT reason FROM kill_switch_events ORDER BY activated_at DESC LIMIT 1) AS last_kill_switch_reason,

    -- 7-day performance
    (SELECT AVG(win_rate) FROM daily_pnl WHERE date >= CURRENT_DATE - INTERVAL '7 days') AS avg_win_rate_7d,
    (SELECT SUM(total_pnl) FROM daily_pnl WHERE date >= CURRENT_DATE - INTERVAL '7 days') AS total_pnl_7d;

COMMENT ON VIEW v_risk_summary IS 'Real-time overview of risk management status';

-- =============================================================================
-- CIRCUIT BREAKER CHECK FUNCTION
-- =============================================================================
-- Function to check if trading is currently allowed
CREATE OR REPLACE FUNCTION is_trading_allowed()
RETURNS TABLE(
    allowed BOOLEAN,
    reason TEXT,
    cooldown_expires TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        CASE
            WHEN EXISTS (SELECT 1 FROM kill_switch_events WHERE status = 'active') THEN FALSE
            WHEN EXISTS (SELECT 1 FROM circuit_breaker_events WHERE status = 'active' AND cooldown_expires_at > NOW()) THEN FALSE
            ELSE TRUE
        END AS allowed,
        CASE
            WHEN EXISTS (SELECT 1 FROM kill_switch_events WHERE status = 'active') THEN 'Kill switch active'
            WHEN EXISTS (SELECT 1 FROM circuit_breaker_events WHERE status = 'active' AND cooldown_expires_at > NOW()) THEN
                (SELECT reason FROM circuit_breaker_events WHERE status = 'active' ORDER BY triggered_at DESC LIMIT 1)
            ELSE 'Trading allowed'
        END AS reason,
        (SELECT MIN(cooldown_expires_at) FROM circuit_breaker_events WHERE status = 'active') AS cooldown_expires;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION is_trading_allowed IS 'Check if trading is currently allowed based on circuit breakers and kill switch';

-- =============================================================================
-- DAILY PNL CALCULATION FUNCTION
-- =============================================================================
-- Function to calculate and update daily P&L metrics
CREATE OR REPLACE FUNCTION calculate_daily_pnl(p_date DATE)
RETURNS VOID AS $$
DECLARE
    v_record RECORD;
BEGIN
    -- Aggregate trade data for the day
    SELECT
        COALESCE(SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END), 0) AS gross_profit,
        COALESCE(SUM(CASE WHEN realized_pnl < 0 THEN realized_pnl ELSE 0 END), 0) AS gross_loss,
        COALESCE(SUM(realized_pnl), 0) AS realized_pnl,
        COUNT(*) AS total_trades,
        COUNT(*) FILTER (WHERE realized_pnl > 0) AS winning_trades,
        COUNT(*) FILTER (WHERE realized_pnl < 0) AS losing_trades,
        COUNT(*) FILTER (WHERE side = 'long') AS long_trades,
        COUNT(*) FILTER (WHERE side = 'short') AS short_trades,
        COALESCE(SUM(fee), 0) AS total_fees,
        MAX(realized_pnl) AS best_trade,
        MIN(realized_pnl) AS worst_trade
    INTO v_record
    FROM trades
    WHERE DATE(exit_time) = p_date AND action = 'close';

    -- Insert or update daily_pnl record
    INSERT INTO daily_pnl (
        date,
        realized_pnl,
        total_trades,
        winning_trades,
        losing_trades,
        win_rate,
        gross_profit,
        gross_loss,
        long_trades,
        short_trades,
        total_fees,
        best_trade_pnl,
        worst_trade_pnl,
        balance_start,
        balance_end
    )
    VALUES (
        p_date,
        v_record.realized_pnl,
        v_record.total_trades,
        v_record.winning_trades,
        v_record.losing_trades,
        CASE WHEN v_record.total_trades > 0 THEN (v_record.winning_trades::DECIMAL / v_record.total_trades * 100) ELSE 0 END,
        v_record.gross_profit,
        v_record.gross_loss,
        v_record.long_trades,
        v_record.short_trades,
        v_record.total_fees,
        v_record.best_trade,
        v_record.worst_trade,
        0, -- balance_start (should be updated from account_snapshots)
        0  -- balance_end (should be updated from account_snapshots)
    )
    ON CONFLICT (date) DO UPDATE SET
        realized_pnl = EXCLUDED.realized_pnl,
        total_trades = EXCLUDED.total_trades,
        winning_trades = EXCLUDED.winning_trades,
        losing_trades = EXCLUDED.losing_trades,
        win_rate = EXCLUDED.win_rate,
        gross_profit = EXCLUDED.gross_profit,
        gross_loss = EXCLUDED.gross_loss,
        long_trades = EXCLUDED.long_trades,
        short_trades = EXCLUDED.short_trades,
        total_fees = EXCLUDED.total_fees,
        best_trade_pnl = EXCLUDED.best_trade_pnl,
        worst_trade_pnl = EXCLUDED.worst_trade_pnl,
        calculation_completed_at = NOW();
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION calculate_daily_pnl IS 'Calculate and update daily P&L metrics for a given date';

-- Migration complete
