"""
Example script demonstrating the order executor functionality.

This script shows how to use the OrderExecutor for both paper trading
and live trading modes.
"""

import asyncio

from iftb.trading.executor import ExecutionRequest, OrderExecutor


async def test_paper_trading():
    """Test paper trading execution."""
    print("=" * 60)
    print("PAPER TRADING TEST")
    print("=" * 60)

    # Initialize paper trading executor
    executor = OrderExecutor(
        paper_mode=True,
        initial_balance=10000.0,
    )

    # Create a long position request
    long_request = ExecutionRequest(
        action="long",
        symbol="BTC/USDT",
        amount=0.1,
        entry_price=50000.0,
        stop_loss=48000.0,
        take_profit=55000.0,
        leverage=2,
        order_type="market",
        reason="Test long position",
    )

    print("\nExecuting long position...")
    order = await executor.execute_decision(long_request)

    print("\nOrder Result:")
    print(f"  ID: {order.id}")
    print(f"  Status: {order.status}")
    print(f"  Filled Price: {order.filled_price}")
    print(f"  Filled Amount: {order.filled_amount}")
    print(f"  Fee: {order.fee}")

    # Get account status
    account = await executor.get_account_status()
    print("\nAccount Status:")
    print(f"  Balance: ${account['balance']:.2f}")
    print(f"  Positions: {account['positions_count']}")
    print(f"  Total Equity: ${account['total_equity']:.2f}")

    if account["positions_count"] > 0:
        print("\n  Position Details:")
        for pos in account["positions"]:
            print(f"    Symbol: {pos['symbol']}")
            print(f"    Side: {pos['side']}")
            print(f"    Entry: ${pos['entry_price']:.2f}")
            print(f"    Amount: {pos['amount']}")
            print(f"    Unrealized PnL: ${pos['unrealized_pnl']:.2f}")

    # Test position close
    print("\n" + "-" * 60)
    print("Testing position close...")

    close_request = ExecutionRequest(
        action="close",
        symbol="BTC/USDT",
        amount=0.1,
        order_type="market",
    )

    close_order = await executor.execute_decision(close_request)
    print(f"Close Order Status: {close_order.status}")

    # Final account status
    final_account = await executor.get_account_status()
    print(f"\nFinal Balance: ${final_account['balance']:.2f}")
    print(f"Open Positions: {final_account['positions_count']}")


async def test_position_management():
    """Test position management features."""
    print("\n" + "=" * 60)
    print("POSITION MANAGEMENT TEST")
    print("=" * 60)

    executor = OrderExecutor(
        paper_mode=True,
        initial_balance=10000.0,
    )

    # Open position
    request = ExecutionRequest(
        action="long",
        symbol="ETH/USDT",
        amount=1.0,
        entry_price=3000.0,
        stop_loss=2800.0,
        take_profit=3500.0,
        leverage=1,
    )

    print("\nOpening ETH long position...")
    order = await executor.execute_decision(request)
    print(f"Order Status: {order.status}")

    # Update stop-loss
    print("\nUpdating stop-loss to $2900...")
    success = await executor.update_stop_loss("ETH/USDT", 2900.0)
    print(f"Stop-loss update: {'Success' if success else 'Failed'}")

    # Update take-profit
    print("\nUpdating take-profit to $3600...")
    success = await executor.update_take_profit("ETH/USDT", 3600.0)
    print(f"Take-profit update: {'Success' if success else 'Failed'}")

    # Get final account status
    account = await executor.get_account_status()
    print("\nFinal Account Status:")
    print(f"  Balance: ${account['balance']:.2f}")
    print(f"  Open Positions: {account['positions_count']}")


async def test_validation():
    """Test order validation."""
    print("\n" + "=" * 60)
    print("VALIDATION TEST")
    print("=" * 60)

    executor = OrderExecutor(
        paper_mode=True,
        initial_balance=10000.0,
        max_leverage=5,
    )

    # Test invalid leverage
    print("\nTesting invalid leverage (exceeds max)...")
    try:
        bad_request = ExecutionRequest(
            action="long",
            symbol="BTC/USDT",
            amount=0.1,
            entry_price=50000.0,
            leverage=10,  # Exceeds max of 5
        )
        await executor.execute_decision(bad_request)
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"Validation correctly rejected: {e}")

    # Test invalid stop-loss
    print("\nTesting invalid stop-loss (above entry for long)...")
    try:
        bad_request = ExecutionRequest(
            action="long",
            symbol="BTC/USDT",
            amount=0.1,
            entry_price=50000.0,
            stop_loss=51000.0,  # Stop above entry for long
        )
        await executor.execute_decision(bad_request)
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"Validation correctly rejected: {e}")

    # Test invalid take-profit
    print("\nTesting invalid take-profit (below entry for long)...")
    try:
        bad_request = ExecutionRequest(
            action="long",
            symbol="BTC/USDT",
            amount=0.1,
            entry_price=50000.0,
            take_profit=49000.0,  # TP below entry for long
        )
        await executor.execute_decision(bad_request)
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"Validation correctly rejected: {e}")

    print("\nAll validations passed!")


async def main():
    """Run all tests."""
    try:
        await test_paper_trading()
        await test_position_management()
        await test_validation()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
