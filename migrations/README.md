# Database Migrations for IFTB Trading Bot

This directory contains PostgreSQL migration scripts for the IFTB (If This, Then Bot) Trading Bot database schema.

## Migration Files

### 001_initial_schema.sql
Core trading tables including:
- **ohlcv** - OHLCV candlestick price data with exchange information
- **trades** - Historical trade records with P&L and signal scores
- **positions** - Current and historical positions with risk management
- **llm_analysis_log** - Audit log for all LLM API calls
- **system_events** - System-wide event and audit log
- **daily_performance** - Daily aggregated performance metrics

Includes views, triggers, and utility functions for trade management.

### 002_analysis_tables.sql
Analysis history and decision tracking:
- **technical_signals** - Aggregated technical analysis from multiple indicators (RSI, MACD, BB, etc.)
- **llm_analyses** - LLM-based market analysis with sentiment, confidence, and token tracking
- **ml_predictions** - Machine learning predictions (XGBoost) with probability distributions
- **trading_decisions** - Final trading decisions combining all signals with veto tracking

Includes views for:
- `v_decision_analysis` - Complete decision breakdown with all component analyses
- `v_analysis_performance` - Track accuracy of analysis methods vs actual outcomes

### 003_risk_tables.sql
Risk management and safeguards:
- **circuit_breaker_events** - Automatic trading halts due to risk thresholds
- **kill_switch_events** - Emergency system shutdowns and manual interventions
- **daily_pnl** - Comprehensive daily P&L with performance metrics and risk ratios

Includes:
- `v_risk_summary` - Real-time risk status overview
- `is_trading_allowed()` - Function to check if trading is permitted
- `calculate_daily_pnl()` - Function to compute daily metrics

### 004_market_data_tables.sql
Market data for analysis:
- **ohlcv_data** - Enhanced OHLCV data with data quality flags
- **fear_greed_index** - Crypto Fear & Greed Index with component scores
- **funding_rates** - Perpetual futures funding rates with annualized calculations

Includes views for:
- `v_market_data_summary` - Latest market data snapshot per symbol
- `v_funding_rate_analysis` - Funding rate trends and extremes
- `v_volatility_analysis` - Price volatility metrics
- `v_market_correlation` - Correlation between Fear & Greed and funding rates

## Running Migrations

### Prerequisites
- PostgreSQL 12+ (for JSONB support and modern features)
- Database user with CREATE TABLE, CREATE INDEX, and CREATE FUNCTION privileges

### Method 1: Using psql

```bash
# Run all migrations in order
psql -U your_username -d your_database -f migrations/001_initial_schema.sql
psql -U your_username -d your_database -f migrations/002_analysis_tables.sql
psql -U your_username -d your_database -f migrations/003_risk_tables.sql
psql -U your_username -d your_database -f migrations/004_market_data_tables.sql
```

### Method 2: Using a migration script

```bash
#!/bin/bash
DATABASE_URL="postgresql://username:password@localhost:5432/dbname"

for migration in migrations/*.sql; do
    echo "Running $migration..."
    psql "$DATABASE_URL" -f "$migration"
done
```

### Method 3: Using Python (asyncpg)

```python
import asyncpg
import asyncio
from pathlib import Path

async def run_migrations():
    conn = await asyncpg.connect('postgresql://user:pass@localhost/dbname')

    migration_files = sorted(Path('migrations').glob('*.sql'))

    for migration in migration_files:
        print(f"Running {migration.name}...")
        sql = migration.read_text()
        await conn.execute(sql)

    await conn.close()

asyncio.run(run_migrations())
```

## Database Schema Overview

### Core Data Flow
```
Market Data → Technical Analysis → LLM Analysis → ML Predictions
                    ↓                   ↓               ↓
              Trading Decision → Order Execution → Position Management
                    ↓
              Risk Management (Circuit Breakers, Kill Switch)
                    ↓
              Daily P&L Tracking
```

### Key Relationships
- `trading_decisions` references `technical_signals`, `llm_analyses`, and `ml_predictions`
- `trading_decisions` links to `orders` for execution tracking
- `orders` reference `positions` and `trades`
- `daily_pnl` aggregates from `trades` and `account_snapshots`

## Important Features

### Automatic Triggers
- **Updated_at timestamps** - Automatically maintained on all core tables
- **Cooldown expiration** - Circuit breaker cooldowns calculated automatically
- **Fear & Greed classification** - Auto-set based on value (0-100 scale)
- **Funding rate metrics** - Basis points and annualized rates calculated automatically

### Data Integrity
- **Unique constraints** - Prevent duplicate candles, funding rates, daily records
- **Check constraints** - Ensure valid price relationships, percentage ranges, enum values
- **Foreign keys** - Maintain referential integrity between related tables
- **Partial indexes** - Optimize queries for common patterns (active, recent, high-confidence)

### Performance Optimizations
- **JSONB indexes (GIN)** - Fast queries on JSON columns (indicators, factors, reasons)
- **Composite indexes** - Optimized for common query patterns (symbol + timestamp)
- **Partial indexes** - Index only relevant subsets (active status, recent dates)
- **Time-based partitioning ready** - Structure supports future partitioning by date

## Views and Functions

### Key Views
- **v_active_positions** - Currently open positions with signal scores
- **v_recent_trades** - Last 100 closed trades with performance metrics
- **v_decision_analysis** - Complete trading decision breakdown
- **v_risk_summary** - Real-time risk management status
- **v_market_data_summary** - Latest market data for all symbols

### Key Functions
- **get_latest_price()** - Retrieve current price for a symbol
- **calculate_position_pnl()** - Compute unrealized P&L for a position
- **is_trading_allowed()** - Check if trading is permitted (circuit breakers/kill switch)
- **calculate_daily_pnl()** - Aggregate daily performance metrics
- **get_latest_fear_greed()** - Current market sentiment
- **get_latest_funding_rate()** - Current funding cost for a symbol

## Column Conventions

- **Timestamps** - All use `TIMESTAMP WITH TIME ZONE` for proper timezone handling
- **Decimals** - `DECIMAL(20, 8)` for prices/amounts, `DECIMAL(5, 4)` for scores/confidence
- **Percents** - Stored as decimals (0.0 to 1.0 for 0% to 100%)
- **JSON** - Use `JSONB` for structured data (better performance than JSON)
- **Enums** - Enforced via `CHECK` constraints for type safety

## Maintenance

### Recommended Indexes to Monitor
```sql
-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan ASC;

-- Find missing indexes (tables with many sequential scans)
SELECT schemaname, tablename, seq_scan, seq_tup_read
FROM pg_stat_user_tables
WHERE schemaname = 'public' AND seq_scan > 1000
ORDER BY seq_scan DESC;
```

### Vacuum and Analyze
```sql
-- Regularly update statistics
ANALYZE technical_signals;
ANALYZE llm_analyses;
ANALYZE ml_predictions;
ANALYZE trading_decisions;

-- Vacuum to reclaim space
VACUUM ANALYZE ohlcv_data;
```

### Data Retention
Consider implementing data retention policies:
```sql
-- Archive old OHLCV data (keep 90 days)
DELETE FROM ohlcv_data WHERE timestamp < NOW() - INTERVAL '90 days';

-- Archive old LLM analyses (keep 30 days)
DELETE FROM llm_analyses WHERE timestamp < NOW() - INTERVAL '30 days';

-- Keep trade history longer (365 days)
DELETE FROM trades WHERE exit_time < NOW() - INTERVAL '365 days';
```

## Security

### Recommended Grants
```sql
-- Read-only user for analytics
GRANT CONNECT ON DATABASE trading_bot TO analyst;
GRANT USAGE ON SCHEMA public TO analyst;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analyst;

-- Trading bot user (read/write)
GRANT CONNECT ON DATABASE trading_bot TO trading_bot;
GRANT USAGE ON SCHEMA public TO trading_bot;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO trading_bot;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO trading_bot;
```

### Sensitive Data
Consider encrypting:
- LLM API responses (may contain sensitive analysis)
- Account balance information
- Trade execution details

## Troubleshooting

### Common Issues

**Issue: Foreign key constraint violations**
```sql
-- Check for orphaned records
SELECT * FROM trading_decisions
WHERE technical_signal_id NOT IN (SELECT id FROM technical_signals);
```

**Issue: Slow queries on JSONB columns**
```sql
-- Create additional GIN indexes if needed
CREATE INDEX idx_custom_json_path ON table_name USING GIN ((json_column -> 'specific_key'));
```

**Issue: Deadlocks on high-frequency updates**
```sql
-- Check for lock contention
SELECT * FROM pg_locks WHERE NOT granted;
```

## Future Enhancements

Potential additions to consider:
- Time-series partitioning for OHLCV and trades tables
- Materialized views for expensive aggregations
- Streaming replication for high availability
- TimescaleDB extension for better time-series performance
- PgBouncer for connection pooling
- Automated backup scripts

## Contributing

When adding new migrations:
1. Use sequential numbering (005_, 006_, etc.)
2. Include descriptive comments
3. Add appropriate indexes
4. Document in this README
5. Test with sample data before deployment

## License

Part of the IFTB Trading Bot project.
