"""
Backtest Report Generator for IFTB Trading Bot.

Generates comprehensive reports from backtest results including:
- Text summaries
- JSON exports
- Performance visualizations
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from iftb.backtest.engine import BacktestResult, BacktestMetrics
from iftb.utils import get_logger

logger = get_logger(__name__)


class BacktestReporter:
    """
    Generate reports from backtest results.

    Supports multiple output formats:
    - Console summary
    - JSON export
    - Markdown report
    """

    def __init__(self, result: BacktestResult):
        """
        Initialize reporter with backtest result.

        Args:
            result: BacktestResult from backtest engine
        """
        self.result = result

    def print_summary(self):
        """Print formatted summary to console."""
        m = self.result.metrics
        c = self.result.config

        print("\n" + "=" * 60)
        print("BACKTEST RESULTS SUMMARY")
        print("=" * 60)

        print(f"\n{'Configuration':}")
        print(f"  Symbol: {c.symbol}")
        print(f"  Period: {c.start_date.date()} to {c.end_date.date()}")
        print(f"  Initial Capital: ${c.initial_capital:,.2f}")
        print(f"  Timeframe: {c.timeframe}")

        print(f"\n{'Performance':}")
        print(f"  Final Balance: ${self.result.final_balance:,.2f}")
        print(f"  Total Return: ${m.total_return:,.2f} ({m.total_return_pct:.2%})")
        print(f"  Annualized Return: {m.annualized_return:.2%}")

        print(f"\n{'Risk Metrics':}")
        print(f"  Max Drawdown: ${m.max_drawdown:,.2f} ({m.max_drawdown_pct:.2%})")
        print(f"  Volatility (Ann.): {m.volatility:.2%}")

        print(f"\n{'Risk-Adjusted Returns':}")
        print(f"  Sharpe Ratio: {m.sharpe_ratio:.4f}")
        print(f"  Sortino Ratio: {m.sortino_ratio:.4f}")
        print(f"  Calmar Ratio: {m.calmar_ratio:.4f}")

        print(f"\n{'Trade Statistics':}")
        print(f"  Total Trades: {m.total_trades}")
        print(f"  Win Rate: {m.win_rate:.2%}")
        print(f"  Profit Factor: {m.profit_factor:.2f}")
        print(f"  Avg Win: ${m.avg_win:,.2f}")
        print(f"  Avg Loss: ${m.avg_loss:,.2f}")
        print(f"  Largest Win: ${m.largest_win:,.2f}")
        print(f"  Largest Loss: ${m.largest_loss:,.2f}")

        print(f"\n{'Streaks':}")
        print(f"  Max Consecutive Wins: {m.max_consecutive_wins}")
        print(f"  Max Consecutive Losses: {m.max_consecutive_losses}")

        print(f"\n{'Costs & Exposure':}")
        print(f"  Total Fees: ${m.total_fees:,.2f}")
        print(f"  Time in Market: {m.time_in_market_pct:.2%}")
        print(f"  Avg Holding Period: {m.avg_holding_period}")

        # Sharpe ratio assessment
        print(f"\n{'Assessment':}")
        if m.sharpe_ratio >= 1.5:
            print(f"  ✅ Sharpe Ratio meets target (≥1.5)")
        else:
            print(f"  ⚠️ Sharpe Ratio below target ({m.sharpe_ratio:.2f} < 1.5)")

        if m.max_drawdown_pct <= 0.15:
            print(f"  ✅ Max Drawdown acceptable (≤15%)")
        else:
            print(f"  ⚠️ Max Drawdown high ({m.max_drawdown_pct:.2%} > 15%)")

        print("\n" + "=" * 60)

    def to_json(self) -> str:
        """
        Export results to JSON string.

        Returns:
            JSON string of results
        """
        data = {
            "config": {
                "symbol": self.result.config.symbol,
                "start_date": self.result.config.start_date.isoformat(),
                "end_date": self.result.config.end_date.isoformat(),
                "initial_capital": self.result.config.initial_capital,
                "timeframe": self.result.config.timeframe,
                "maker_fee": self.result.config.maker_fee,
                "taker_fee": self.result.config.taker_fee,
            },
            "metrics": self.result.metrics.to_dict(),
            "summary": {
                "final_balance": self.result.final_balance,
                "total_trades": len(self.result.trades),
                "meets_sharpe_target": self.result.metrics.sharpe_ratio >= 1.5,
                "meets_drawdown_target": self.result.metrics.max_drawdown_pct <= 0.15,
            },
            "trades": [
                {
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "amount": t.amount,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "exit_reason": t.exit_reason,
                }
                for t in self.result.trades
            ],
            "generated_at": datetime.utcnow().isoformat(),
        }

        return json.dumps(data, indent=2)

    def save_json(self, path: str | Path):
        """
        Save results to JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(self.to_json())

        logger.info("backtest_report_saved", path=str(path))

    def to_markdown(self) -> str:
        """
        Generate markdown report.

        Returns:
            Markdown formatted report string
        """
        m = self.result.metrics
        c = self.result.config

        report = []
        report.append("# Backtest Report")
        report.append(f"\n**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")

        report.append("## Configuration\n")
        report.append(f"| Parameter | Value |")
        report.append(f"|-----------|-------|")
        report.append(f"| Symbol | {c.symbol} |")
        report.append(f"| Period | {c.start_date.date()} to {c.end_date.date()} |")
        report.append(f"| Initial Capital | ${c.initial_capital:,.2f} |")
        report.append(f"| Timeframe | {c.timeframe} |")
        report.append(f"| Maker Fee | {c.maker_fee:.4%} |")
        report.append(f"| Taker Fee | {c.taker_fee:.4%} |")

        report.append("\n## Performance Summary\n")
        report.append(f"| Metric | Value |")
        report.append(f"|--------|-------|")
        report.append(f"| Final Balance | ${self.result.final_balance:,.2f} |")
        report.append(f"| Total Return | ${m.total_return:,.2f} ({m.total_return_pct:.2%}) |")
        report.append(f"| Annualized Return | {m.annualized_return:.2%} |")
        report.append(f"| Max Drawdown | {m.max_drawdown_pct:.2%} |")
        report.append(f"| Volatility | {m.volatility:.2%} |")

        report.append("\n## Risk-Adjusted Returns\n")
        report.append(f"| Ratio | Value | Target | Status |")
        report.append(f"|-------|-------|--------|--------|")
        sharpe_status = "✅" if m.sharpe_ratio >= 1.5 else "⚠️"
        report.append(f"| Sharpe Ratio | {m.sharpe_ratio:.4f} | ≥1.5 | {sharpe_status} |")
        report.append(f"| Sortino Ratio | {m.sortino_ratio:.4f} | - | - |")
        report.append(f"| Calmar Ratio | {m.calmar_ratio:.4f} | - | - |")

        report.append("\n## Trade Statistics\n")
        report.append(f"| Metric | Value |")
        report.append(f"|--------|-------|")
        report.append(f"| Total Trades | {m.total_trades} |")
        report.append(f"| Win Rate | {m.win_rate:.2%} |")
        report.append(f"| Profit Factor | {m.profit_factor:.2f} |")
        report.append(f"| Avg Win | ${m.avg_win:,.2f} |")
        report.append(f"| Avg Loss | ${m.avg_loss:,.2f} |")
        report.append(f"| Largest Win | ${m.largest_win:,.2f} |")
        report.append(f"| Largest Loss | ${m.largest_loss:,.2f} |")
        report.append(f"| Max Consecutive Wins | {m.max_consecutive_wins} |")
        report.append(f"| Max Consecutive Losses | {m.max_consecutive_losses} |")

        report.append("\n## Recent Trades\n")
        report.append("| Entry | Exit | Side | Entry Price | Exit Price | PnL | Reason |")
        report.append("|-------|------|------|-------------|------------|-----|--------|")

        for trade in self.result.trades[-10:]:  # Last 10 trades
            pnl_str = f"${trade.pnl:,.2f}" if trade.pnl >= 0 else f"-${abs(trade.pnl):,.2f}"
            report.append(
                f"| {trade.entry_time.strftime('%Y-%m-%d %H:%M')} "
                f"| {trade.exit_time.strftime('%Y-%m-%d %H:%M')} "
                f"| {trade.side} "
                f"| ${trade.entry_price:,.2f} "
                f"| ${trade.exit_price:,.2f} "
                f"| {pnl_str} "
                f"| {trade.exit_reason} |"
            )

        return "\n".join(report)

    def save_markdown(self, path: str | Path):
        """
        Save markdown report to file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(self.to_markdown())

        logger.info("backtest_markdown_saved", path=str(path))

    def save_equity_curve(self, path: str | Path):
        """
        Save equity curve to CSV.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.result.equity_curve.to_csv(path, index=False)
        logger.info("equity_curve_saved", path=str(path))
