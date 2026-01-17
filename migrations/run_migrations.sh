#!/bin/bash

# Migration runner script for IFTB Trading Bot
# Usage: ./run_migrations.sh [database_url]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Database connection
DB_URL="${1:-postgresql://localhost:5432/aqb_trading}"

echo -e "${GREEN}=== IFTB Trading Bot Database Migrations ===${NC}"
echo "Database: $DB_URL"
echo ""

# Check if psql is available
if ! command -v psql &> /dev/null; then
    echo -e "${RED}Error: psql command not found. Please install PostgreSQL client.${NC}"
    exit 1
fi

# Test database connection
echo -e "${YELLOW}Testing database connection...${NC}"
if ! psql "$DB_URL" -c "SELECT 1;" > /dev/null 2>&1; then
    echo -e "${RED}Error: Cannot connect to database.${NC}"
    echo "Please check your connection string."
    exit 1
fi
echo -e "${GREEN}Connected successfully!${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Migration files in order
MIGRATIONS=(
    "001_initial_schema.sql"
    "002_analysis_tables.sql"
    "003_risk_tables.sql"
    "004_market_data_tables.sql"
)

# Run each migration
for migration in "${MIGRATIONS[@]}"; do
    migration_path="$SCRIPT_DIR/$migration"

    if [ ! -f "$migration_path" ]; then
        echo -e "${RED}Error: Migration file not found: $migration${NC}"
        exit 1
    fi

    echo -e "${YELLOW}Running migration: $migration${NC}"

    if psql "$DB_URL" -f "$migration_path" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $migration completed successfully${NC}"
    else
        echo -e "${RED}✗ $migration failed${NC}"
        echo "Please check the error messages above."
        exit 1
    fi

    echo ""
done

echo -e "${GREEN}=== All migrations completed successfully! ===${NC}"
echo ""

# Show summary of created tables
echo -e "${YELLOW}Database summary:${NC}"
psql "$DB_URL" -c "
    SELECT
        schemaname,
        tablename,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
    FROM pg_tables
    WHERE schemaname = 'public'
    ORDER BY tablename;
"

echo ""
echo -e "${YELLOW}Created views:${NC}"
psql "$DB_URL" -c "
    SELECT table_name
    FROM information_schema.views
    WHERE table_schema = 'public'
    ORDER BY table_name;
"

echo ""
echo -e "${GREEN}Database setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Review the tables and indexes created"
echo "2. Configure your trading bot to use this database"
echo "3. Test with sample data"
echo ""
