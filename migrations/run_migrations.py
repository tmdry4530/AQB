#!/usr/bin/env python3
"""
Database migration runner for IFTB Trading Bot
Supports both asyncpg (async) and psycopg2 (sync) connections

Usage:
    python run_migrations.py --db-url postgresql://user:pass@localhost/dbname
    python run_migrations.py --async  # Use asyncpg
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List


def print_colored(text: str, color: str = "green"):
    """Print colored text to terminal"""
    colors = {
        "red": "\033[0;31m",
        "green": "\033[0;32m",
        "yellow": "\033[1;33m",
        "blue": "\033[0;34m",
        "reset": "\033[0m",
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")


def get_migration_files() -> List[Path]:
    """Get all migration SQL files in order"""
    migrations_dir = Path(__file__).parent
    migration_files = [
        migrations_dir / "001_initial_schema.sql",
        migrations_dir / "002_analysis_tables.sql",
        migrations_dir / "003_risk_tables.sql",
        migrations_dir / "004_market_data_tables.sql",
    ]

    # Verify all files exist
    for file_path in migration_files:
        if not file_path.exists():
            print_colored(f"Error: Migration file not found: {file_path.name}", "red")
            sys.exit(1)

    return migration_files


def run_migrations_sync(db_url: str):
    """Run migrations using psycopg2 (synchronous)"""
    try:
        import psycopg2
    except ImportError:
        print_colored("Error: psycopg2 not installed. Install with: pip install psycopg2-binary", "red")
        sys.exit(1)

    print_colored("=== IFTB Trading Bot Database Migrations (Sync) ===", "green")
    print(f"Database: {db_url}")
    print()

    # Connect to database
    print_colored("Connecting to database...", "yellow")
    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        cursor = conn.cursor()
        print_colored("Connected successfully!", "green")
        print()
    except Exception as e:
        print_colored(f"Error connecting to database: {e}", "red")
        sys.exit(1)

    # Run migrations
    migration_files = get_migration_files()

    for migration_file in migration_files:
        print_colored(f"Running migration: {migration_file.name}", "yellow")

        try:
            sql = migration_file.read_text()
            cursor.execute(sql)
            print_colored(f"✓ {migration_file.name} completed successfully", "green")
        except Exception as e:
            print_colored(f"✗ {migration_file.name} failed", "red")
            print_colored(f"Error: {e}", "red")
            cursor.close()
            conn.close()
            sys.exit(1)

        print()

    # Show summary
    print_colored("=== All migrations completed successfully! ===", "green")
    print()

    print_colored("Database summary:", "yellow")
    cursor.execute("""
        SELECT
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
        FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY tablename;
    """)
    for row in cursor.fetchall():
        print(f"  {row[1]}: {row[2]}")

    print()
    print_colored("Created views:", "yellow")
    cursor.execute("""
        SELECT table_name
        FROM information_schema.views
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    for row in cursor.fetchall():
        print(f"  - {row[0]}")

    cursor.close()
    conn.close()

    print()
    print_colored("Database setup complete!", "green")


async def run_migrations_async(db_url: str):
    """Run migrations using asyncpg (asynchronous)"""
    try:
        import asyncpg
    except ImportError:
        print_colored("Error: asyncpg not installed. Install with: pip install asyncpg", "red")
        sys.exit(1)

    print_colored("=== IFTB Trading Bot Database Migrations (Async) ===", "green")
    print(f"Database: {db_url}")
    print()

    # Connect to database
    print_colored("Connecting to database...", "yellow")
    try:
        conn = await asyncpg.connect(db_url)
        print_colored("Connected successfully!", "green")
        print()
    except Exception as e:
        print_colored(f"Error connecting to database: {e}", "red")
        sys.exit(1)

    # Run migrations
    migration_files = get_migration_files()

    for migration_file in migration_files:
        print_colored(f"Running migration: {migration_file.name}", "yellow")

        try:
            sql = migration_file.read_text()
            await conn.execute(sql)
            print_colored(f"✓ {migration_file.name} completed successfully", "green")
        except Exception as e:
            print_colored(f"✗ {migration_file.name} failed", "red")
            print_colored(f"Error: {e}", "red")
            await conn.close()
            sys.exit(1)

        print()

    # Show summary
    print_colored("=== All migrations completed successfully! ===", "green")
    print()

    print_colored("Database summary:", "yellow")
    rows = await conn.fetch("""
        SELECT
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
        FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY tablename;
    """)
    for row in rows:
        print(f"  {row['tablename']}: {row['size']}")

    print()
    print_colored("Created views:", "yellow")
    rows = await conn.fetch("""
        SELECT table_name
        FROM information_schema.views
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    for row in rows:
        print(f"  - {row['table_name']}")

    await conn.close()

    print()
    print_colored("Database setup complete!", "green")


def main():
    parser = argparse.ArgumentParser(
        description="Run database migrations for IFTB Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using environment variable
  export DATABASE_URL="postgresql://user:pass@localhost/aqb"
  python run_migrations.py

  # Using command line argument
  python run_migrations.py --db-url postgresql://user:pass@localhost/aqb

  # Using async mode (asyncpg)
  python run_migrations.py --async --db-url postgresql://user:pass@localhost/aqb

  # Using sync mode (psycopg2) - default
  python run_migrations.py --db-url postgresql://user:pass@localhost/aqb
        """
    )

    parser.add_argument(
        "--db-url",
        help="Database connection URL (default: from DATABASE_URL env var)",
        default=None
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use asyncpg instead of psycopg2"
    )

    args = parser.parse_args()

    # Get database URL
    db_url = args.db_url
    if not db_url:
        import os
        db_url = os.getenv("DATABASE_URL")

    if not db_url:
        print_colored("Error: No database URL provided", "red")
        print("Either set DATABASE_URL environment variable or use --db-url argument")
        sys.exit(1)

    # Run migrations
    if args.use_async:
        asyncio.run(run_migrations_async(db_url))
    else:
        run_migrations_sync(db_url)


if __name__ == "__main__":
    main()
