# Quick Start Guide

Get your IFTB Trading Bot database up and running in 5 minutes.

## Prerequisites

- PostgreSQL 12+ installed and running
- Database created (or permission to create one)
- One of the following:
  - `psql` command-line tool (comes with PostgreSQL)
  - Python 3.7+ with `psycopg2-binary` or `asyncpg`

## Option 1: Quick Setup with psql (Recommended)

### Step 1: Create Database
```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE aqb_trading;

# Create user (optional)
CREATE USER aqb_bot WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE aqb_trading TO aqb_bot;

# Exit
\q
```

### Step 2: Run Migrations
```bash
cd /mnt/d/Develop/AQB/migrations

# Make script executable
chmod +x run_migrations.sh

# Run migrations
./run_migrations.sh postgresql://aqb_bot:your_secure_password@localhost:5432/aqb_trading
```

Done! Your database is ready.

## Option 2: Quick Setup with Python

### Step 1: Install Dependencies
```bash
# Using psycopg2 (sync)
pip install psycopg2-binary

# OR using asyncpg (async)
pip install asyncpg
```

### Step 2: Set Environment Variable
```bash
export DATABASE_URL="postgresql://aqb_bot:your_secure_password@localhost:5432/aqb_trading"
```

### Step 3: Run Migrations
```bash
cd /mnt/d/Develop/AQB/migrations

# Make script executable
chmod +x run_migrations.py

# Run with psycopg2 (default)
./run_migrations.py

# OR run with asyncpg
./run_migrations.py --async
```

Done! Your database is ready.

## Option 3: Manual Setup

If you prefer to run migrations manually:

```bash
cd /mnt/d/Develop/AQB/migrations

psql postgresql://aqb_bot:password@localhost:5432/aqb_trading -f 001_initial_schema.sql
psql postgresql://aqb_bot:password@localhost:5432/aqb_trading -f 002_analysis_tables.sql
psql postgresql://aqb_bot:password@localhost:5432/aqb_trading -f 003_risk_tables.sql
psql postgresql://aqb_bot:password@localhost:5432/aqb_trading -f 004_market_data_tables.sql
```

## Verify Installation

```bash
# Connect to database
psql postgresql://aqb_bot:password@localhost:5432/aqb_trading

# List all tables
\dt

# Check table counts
SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';

# Expected output: ~17 tables

# Exit
\q
```

## What Gets Created?

After running migrations, you'll have:

### 17 Tables
- **Core**: ohlcv, trades, positions, orders, account_snapshots
- **Analysis**: technical_signals, llm_analyses, ml_predictions, trading_decisions
- **Risk**: circuit_breaker_events, kill_switch_events, daily_pnl
- **Market Data**: ohlcv_data, fear_greed_index, funding_rates
- **Audit**: llm_analysis_log, system_events, daily_performance

### 9 Views
- v_active_positions
- v_recent_trades
- v_decision_analysis
- v_analysis_performance
- v_risk_summary
- v_market_data_summary
- v_funding_rate_analysis
- v_volatility_analysis
- v_market_correlation

### 7 Functions
- get_latest_price()
- calculate_position_pnl()
- is_trading_allowed()
- calculate_daily_pnl()
- get_latest_ohlcv_price()
- get_latest_fear_greed()
- get_latest_funding_rate()

### 4 Triggers
- Auto-update timestamps
- Circuit breaker cooldown calculation
- Fear & Greed classification
- Funding rate metric calculation

## Test Your Database

### Insert Sample Data

```sql
-- Insert sample OHLCV data
INSERT INTO ohlcv_data (symbol, timeframe, timestamp, open, high, low, close, volume, exchange)
VALUES ('BTC/USDT', '1h', NOW(), 50000, 50500, 49800, 50200, 100.5, 'binance');

-- Insert sample Fear & Greed reading
INSERT INTO fear_greed_index (timestamp, value)
VALUES (NOW(), 65);

-- Check if trading is allowed
SELECT * FROM is_trading_allowed();

-- Get latest market data
SELECT * FROM v_market_data_summary;
```

### Query Examples

```sql
-- Get all active positions
SELECT * FROM v_active_positions;

-- Get recent trades with P&L
SELECT symbol, side, realized_pnl, entry_time, exit_time
FROM v_recent_trades
LIMIT 10;

-- Check risk status
SELECT * FROM v_risk_summary;

-- Get latest Fear & Greed
SELECT * FROM get_latest_fear_greed();

-- Check funding rates
SELECT symbol, rate_bps, annualized_rate
FROM v_funding_rate_analysis;
```

## Common Issues

### Issue: "psql: command not found"
**Solution**: Install PostgreSQL client
```bash
# Ubuntu/Debian
sudo apt-get install postgresql-client

# macOS
brew install postgresql

# Windows
# Download from https://www.postgresql.org/download/windows/
```

### Issue: "FATAL: password authentication failed"
**Solution**: Check your connection string and ensure user exists
```bash
# Test connection
psql postgresql://username:password@localhost:5432/aqb_trading -c "SELECT 1;"
```

### Issue: "ERROR: relation already exists"
**Solution**: Database already has tables. Either:
1. Drop existing tables: `DROP SCHEMA public CASCADE; CREATE SCHEMA public;`
2. Or use a fresh database

### Issue: "Permission denied"
**Solution**: Grant proper permissions
```sql
GRANT ALL PRIVILEGES ON DATABASE aqb_trading TO aqb_bot;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO aqb_bot;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO aqb_bot;
```

## Next Steps

1. **Configure Your Bot**
   - Update your bot's database connection string
   - Test database connectivity

2. **Start Trading**
   - Bot will automatically create records in trades, positions, orders
   - Analysis tables will be populated as decisions are made
   - Risk management tables track circuit breakers and kill switches

3. **Monitor Performance**
   - Use views for real-time monitoring
   - Check daily_pnl for performance tracking
   - Review v_risk_summary regularly

4. **Set Up Monitoring** (Optional)
   - Create alerts for circuit breaker events
   - Monitor daily_pnl for losses
   - Track kill_switch_events

5. **Implement Backup Strategy**
   ```bash
   # Daily backup
   pg_dump postgresql://user:pass@localhost/aqb_trading > backup_$(date +%Y%m%d).sql

   # Restore from backup
   psql postgresql://user:pass@localhost/aqb_trading < backup_20260118.sql
   ```

## Performance Tuning

After running for a few days, optimize your database:

```sql
-- Update statistics
ANALYZE;

-- Check index usage
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan ASC;

-- Vacuum to reclaim space
VACUUM ANALYZE;
```

## Documentation

- **README.md** - Full documentation and migration details
- **SCHEMA_REFERENCE.md** - Complete schema reference with all tables/columns
- **This file (QUICKSTART.md)** - Quick setup guide

## Support

If you encounter issues:
1. Check the Common Issues section above
2. Review PostgreSQL logs: `tail -f /var/log/postgresql/postgresql-*.log`
3. Verify migration files are intact: `ls -lh *.sql`
4. Test database connection: `psql $DATABASE_URL -c "SELECT version();"`

## Security Checklist

Before going to production:

- [ ] Change default passwords
- [ ] Use strong passwords (20+ characters)
- [ ] Enable SSL/TLS for database connections
- [ ] Restrict database access to specific IPs
- [ ] Set up regular backups
- [ ] Enable query logging for audit
- [ ] Use read-only users for analytics
- [ ] Store credentials in environment variables, not code
- [ ] Regularly update PostgreSQL to latest version

## Done!

Your IFTB Trading Bot database is now ready. Start your bot and watch it populate the database with trade data, analysis results, and performance metrics.

Happy trading!
