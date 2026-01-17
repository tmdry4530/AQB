"""
Order Execution Module for IFTB Trading Bot.

This module handles order execution with support for both paper trading and live
trading modes. Provides unified interfaces for market, limit, stop-loss, and
take-profit orders with comprehensive position management.

Key Features:
- Paper trading simulation with realistic order fills
- Live trading via CCXT integration
- Multiple order types (market, limit, stop-loss, take-profit)
- Position tracking and management
- Slippage estimation for market orders
- Retry logic and error handling
- Order validation and risk checks
- Detailed execution logging

Example Usage:
    ```python
    from iftb.trading.executor import OrderExecutor, ExecutionRequest
    from iftb.data import ExchangeClient

    # Paper trading mode
    executor = OrderExecutor(paper_mode=True, initial_balance=10000.0)

    request = ExecutionRequest(
        action="long",
        symbol="BTC/USDT",
        amount=0.1,
        entry_price=50000.0,
        stop_loss=48000.0,
        take_profit=55000.0,
        leverage=1
    )

    order = await executor.execute_decision(request)

    # Live trading mode
    async with ExchangeClient("binance", api_key, api_secret) as client:
        executor = OrderExecutor(paper_mode=False, exchange_client=client)
        order = await executor.execute_decision(request)
    ```
"""

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Literal
import uuid

from ccxt.base.errors import (
    ExchangeError,
    InsufficientFunds,
    InvalidOrder,
    NetworkError,
    OrderNotFound,
)

from iftb.data import ExchangeClient
from iftb.utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class Order:
    """
    Order representation with status tracking.

    Attributes:
        id: Unique order identifier
        symbol: Trading pair symbol (e.g., "BTC/USDT")
        side: Order side (buy/sell)
        type: Order type (market/limit/stop_loss/take_profit)
        amount: Order quantity in base currency
        price: Limit price for limit orders (optional)
        stop_price: Trigger price for stop orders (optional)
        status: Current order status
        filled_price: Actual execution price (optional)
        filled_amount: Actual filled quantity (optional)
        fee: Trading fee paid (optional)
        timestamp: Order creation timestamp
        exchange_order_id: Exchange-specific order ID (optional)
        error_message: Error message if order failed (optional)
    """

    id: str
    symbol: str
    side: Literal["buy", "sell"]
    type: Literal["market", "limit", "stop_loss", "take_profit"]
    amount: float
    price: float | None = None
    stop_price: float | None = None
    status: Literal["pending", "filled", "cancelled", "failed"] = "pending"
    filled_price: float | None = None
    filled_amount: float | None = None
    fee: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    exchange_order_id: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict:
        """Convert order to dictionary representation."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass
class PositionState:
    """
    Current position state representation.

    Attributes:
        symbol: Trading pair symbol
        side: Position side (long/short)
        entry_price: Average entry price
        current_price: Current market price
        amount: Position size in base currency
        leverage: Position leverage multiplier
        unrealized_pnl: Unrealized profit/loss
        liquidation_price: Estimated liquidation price
        stop_loss: Stop-loss price (optional)
        take_profit: Take-profit price (optional)
        opened_at: Position open timestamp
    """

    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    current_price: float
    amount: float
    leverage: int
    unrealized_pnl: float
    liquidation_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    opened_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert position to dictionary representation."""
        data = asdict(self)
        data["opened_at"] = self.opened_at.isoformat()
        return data


@dataclass
class ExecutionRequest:
    """
    Order execution request with detailed parameters.

    Attributes:
        action: Trading action (long/short/close)
        symbol: Trading pair symbol
        amount: Order quantity
        entry_price: Desired entry price (optional, uses market if None)
        stop_loss: Stop-loss price (optional)
        take_profit: Take-profit price (optional)
        leverage: Position leverage (default: 1)
        order_type: Order type (default: market)
        reason: Decision rationale (optional)
    """

    action: Literal["long", "short", "close"]
    symbol: str
    amount: float
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    leverage: int = 1
    order_type: Literal["market", "limit"] = "market"
    reason: str | None = None


# =============================================================================
# Utility Functions
# =============================================================================


def convert_decision_to_request(
    decision,
    amount: float,
    order_type: Literal["market", "limit"] = "market",
) -> ExecutionRequest:
    """
    Convert TradingDecision from decision engine to ExecutionRequest.

    Args:
        decision: TradingDecision from decision engine
        amount: Order amount in base currency
        order_type: Order type (market or limit)

    Returns:
        ExecutionRequest for order executor

    Example:
        ```python
        from iftb.trading import DecisionEngine
        from iftb.trading.executor import convert_decision_to_request, OrderExecutor

        # Get decision from engine
        engine = DecisionEngine(...)
        decision = await engine.make_decision(signal, context)

        # Convert to execution request
        request = convert_decision_to_request(
            decision,
            amount=0.1,  # BTC amount
            order_type="market"
        )

        # Execute
        executor = OrderExecutor(paper_mode=True)
        order = await executor.execute_decision(request)
        ```
    """
    # Map action from decision engine format (LONG/SHORT/HOLD) to executor format
    action_map = {
        "LONG": "long",
        "SHORT": "short",
        "HOLD": "close",  # HOLD means close existing positions
    }

    action = action_map.get(decision.action, "close")

    # Skip HOLD decisions unless we explicitly want to close
    if decision.action == "HOLD" and not decision.vetoed:
        # For HOLD, we don't create an execution request
        # Caller should check decision.action before converting
        pass

    return ExecutionRequest(
        action=action,
        symbol=decision.symbol,
        amount=amount,
        entry_price=decision.entry_price,
        stop_loss=decision.stop_loss,
        take_profit=decision.take_profit,
        leverage=decision.leverage,
        order_type=order_type,
        reason=decision.veto_reason if decision.vetoed else ", ".join(decision.reasons),
    )


# =============================================================================
# Paper Trading Implementation
# =============================================================================


class PaperTrader:
    """
    Simulated trading for backtesting and testing.

    Provides realistic order simulation with slippage, fees, and position
    tracking without executing real trades.

    Example:
        ```python
        trader = PaperTrader(initial_balance=10000.0)

        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=0.1,
            price=50000.0
        )

        filled_order = await trader.place_order(order)
        print(f"Order filled at: {filled_order.filled_price}")
        ```
    """

    def __init__(
        self, initial_balance: float = 10000.0, maker_fee: float = 0.0002, taker_fee: float = 0.0004
    ):
        """
        Initialize paper trader.

        Args:
            initial_balance: Starting account balance in quote currency
            maker_fee: Maker fee rate (default: 0.02%)
            taker_fee: Taker fee rate (default: 0.04%)
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.positions: dict[str, PositionState] = {}
        self.orders: list[Order] = []
        self.order_history: list[Order] = []

        logger.info(
            "paper_trader_initialized",
            initial_balance=initial_balance,
            maker_fee=maker_fee,
            taker_fee=taker_fee,
        )

    async def place_order(self, order: Order, current_price: float | None = None) -> Order:
        """
        Simulate order placement and execution.

        Args:
            order: Order to place
            current_price: Current market price (uses order.price if None)

        Returns:
            Filled order with execution details
        """
        logger.info(
            "paper_placing_order",
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            type=order.type,
            amount=order.amount,
        )

        try:
            # Determine execution price
            if order.type == "market":
                # Use current price with slippage
                exec_price = current_price or order.price
                if exec_price is None:
                    raise ValueError("Current price required for market orders")

                # Add slippage (0.05% for buy, -0.05% for sell)
                slippage = 0.0005 if order.side == "buy" else -0.0005
                exec_price = exec_price * (1 + slippage)
                fee_rate = self.taker_fee

            elif order.type == "limit":
                # Limit orders fill at limit price
                exec_price = order.price
                if exec_price is None:
                    raise ValueError("Price required for limit orders")
                fee_rate = self.maker_fee

            else:  # stop_loss or take_profit
                # Stop/TP orders fill at trigger price
                exec_price = order.stop_price or order.price
                if exec_price is None:
                    raise ValueError("Stop price required for stop orders")
                fee_rate = self.taker_fee

            # Calculate costs
            order_value = order.amount * exec_price
            fee = order_value * fee_rate
            total_cost = order_value + fee if order.side == "buy" else order_value - fee

            # Check balance
            if order.side == "buy" and total_cost > self.balance:
                order.status = "failed"
                order.error_message = "Insufficient balance"
                logger.warning(
                    "paper_order_failed_insufficient_balance",
                    order_id=order.id,
                    required=total_cost,
                    available=self.balance,
                )
                self.order_history.append(order)
                return order

            # Update balance
            if order.side == "buy":
                self.balance -= total_cost
            else:
                self.balance += total_cost

            # Fill order
            order.status = "filled"
            order.filled_price = exec_price
            order.filled_amount = order.amount
            order.fee = fee

            # Update position
            await self._update_position(order)

            self.order_history.append(order)

            logger.info(
                "paper_order_filled",
                order_id=order.id,
                filled_price=exec_price,
                filled_amount=order.amount,
                fee=fee,
                balance=self.balance,
            )

            return order

        except Exception as e:
            order.status = "failed"
            order.error_message = str(e)
            self.order_history.append(order)

            logger.error(
                "paper_order_failed",
                order_id=order.id,
                error=str(e),
            )

            return order

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if order was cancelled, False otherwise
        """
        for order in self.orders:
            if order.id == order_id and order.status == "pending":
                order.status = "cancelled"
                self.orders.remove(order)
                self.order_history.append(order)

                logger.info(
                    "paper_order_cancelled",
                    order_id=order_id,
                )
                return True

        logger.warning(
            "paper_order_not_found_for_cancel",
            order_id=order_id,
        )
        return False

    async def _update_position(self, order: Order) -> None:
        """
        Update position state after order fill.

        Args:
            order: Filled order
        """
        symbol = order.symbol

        if symbol not in self.positions:
            # Open new position
            if order.side == "buy":
                side = "long"
            else:
                side = "short"

            self.positions[symbol] = PositionState(
                symbol=symbol,
                side=side,
                entry_price=order.filled_price,
                current_price=order.filled_price,
                amount=order.filled_amount,
                leverage=1,
                unrealized_pnl=0.0,
                liquidation_price=0.0,  # Calculate based on leverage
            )

            logger.info(
                "paper_position_opened",
                symbol=symbol,
                side=side,
                entry_price=order.filled_price,
                amount=order.filled_amount,
            )
        else:
            # Update existing position or close
            position = self.positions[symbol]

            if (position.side == "long" and order.side == "sell") or (
                position.side == "short" and order.side == "buy"
            ):
                # Closing position
                if order.filled_amount >= position.amount:
                    # Full close
                    del self.positions[symbol]
                    logger.info(
                        "paper_position_closed",
                        symbol=symbol,
                        side=position.side,
                    )
                else:
                    # Partial close
                    position.amount -= order.filled_amount
                    logger.info(
                        "paper_position_reduced",
                        symbol=symbol,
                        remaining_amount=position.amount,
                    )
            else:
                # Adding to position (average entry price)
                total_amount = position.amount + order.filled_amount
                position.entry_price = (
                    position.entry_price * position.amount
                    + order.filled_price * order.filled_amount
                ) / total_amount
                position.amount = total_amount

                logger.info(
                    "paper_position_increased",
                    symbol=symbol,
                    new_entry_price=position.entry_price,
                    total_amount=total_amount,
                )

    def get_balance(self) -> float:
        """
        Get current account balance.

        Returns:
            Current balance in quote currency
        """
        return self.balance

    def get_positions(self) -> list[PositionState]:
        """
        Get all open positions.

        Returns:
            List of open positions
        """
        return list(self.positions.values())

    def get_order_history(self) -> list[Order]:
        """
        Get complete order history.

        Returns:
            List of all historical orders
        """
        return self.order_history

    def update_position_prices(self, symbol: str, current_price: float) -> None:
        """
        Update position with current market price and calculate PnL.

        Args:
            symbol: Trading pair symbol
            current_price: Current market price
        """
        if symbol in self.positions:
            position = self.positions[symbol]
            position.current_price = current_price

            # Calculate unrealized PnL
            if position.side == "long":
                position.unrealized_pnl = (current_price - position.entry_price) * position.amount
            else:  # short
                position.unrealized_pnl = (position.entry_price - current_price) * position.amount


# =============================================================================
# Live Order Execution
# =============================================================================


class LiveExecutor:
    """
    Real order execution via CCXT.

    Handles live order placement, cancellation, and position management
    through exchange APIs.

    Example:
        ```python
        async with ExchangeClient("binance", api_key, api_secret) as client:
            executor = LiveExecutor(client)

            order = await executor.place_market_order(
                symbol="BTC/USDT",
                side="buy",
                amount=0.1
            )

            position = await executor.get_position("BTC/USDT")
        ```
    """

    def __init__(self, exchange_client: ExchangeClient, max_retries: int = 3):
        """
        Initialize live executor.

        Args:
            exchange_client: Connected ExchangeClient instance
            max_retries: Maximum retry attempts for failed orders
        """
        self.client = exchange_client
        self.max_retries = max_retries

        logger.info(
            "live_executor_initialized",
            exchange=exchange_client.exchange_id,
            max_retries=max_retries,
        )

    async def place_market_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
    ) -> Order:
        """
        Place market order with immediate execution.

        Args:
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            amount: Order quantity

        Returns:
            Filled order
        """
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            type="market",
            amount=amount,
        )

        logger.info(
            "placing_market_order",
            order_id=order.id,
            symbol=symbol,
            side=side,
            amount=amount,
        )

        return await self._execute_order_with_retry(order)

    async def place_limit_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
        price: float,
    ) -> Order:
        """
        Place limit order at specified price.

        Args:
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            amount: Order quantity
            price: Limit price

        Returns:
            Placed order (may be pending)
        """
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            type="limit",
            amount=amount,
            price=price,
        )

        logger.info(
            "placing_limit_order",
            order_id=order.id,
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
        )

        return await self._execute_order_with_retry(order)

    async def set_stop_loss(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
        stop_price: float,
    ) -> Order:
        """
        Set stop-loss order.

        Args:
            symbol: Trading pair symbol
            side: Order side (opposite of position)
            amount: Order quantity
            stop_price: Stop trigger price

        Returns:
            Placed stop order
        """
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            type="stop_loss",
            amount=amount,
            stop_price=stop_price,
        )

        logger.info(
            "placing_stop_loss",
            order_id=order.id,
            symbol=symbol,
            stop_price=stop_price,
        )

        return await self._execute_order_with_retry(order)

    async def set_take_profit(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
        price: float,
    ) -> Order:
        """
        Set take-profit order.

        Args:
            symbol: Trading pair symbol
            side: Order side (opposite of position)
            amount: Order quantity
            price: Take-profit price

        Returns:
            Placed take-profit order
        """
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            type="take_profit",
            amount=amount,
            price=price,
        )

        logger.info(
            "placing_take_profit",
            order_id=order.id,
            symbol=symbol,
            price=price,
        )

        return await self._execute_order_with_retry(order)

    async def close_position(self, symbol: str) -> Order:
        """
        Close entire position for symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Market order closing the position

        Raises:
            ValueError: If no position exists
        """
        position = await self.get_position(symbol)
        if not position:
            raise ValueError(f"No position found for {symbol}")

        # Determine close side (opposite of position)
        close_side = "sell" if position.side == "long" else "buy"

        logger.info(
            "closing_position",
            symbol=symbol,
            position_side=position.side,
            amount=position.amount,
        )

        return await self.place_market_order(
            symbol=symbol,
            side=close_side,
            amount=position.amount,
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel open order.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol

        Returns:
            True if cancelled successfully
        """
        try:
            await self.client.exchange.cancel_order(order_id, symbol)

            logger.info(
                "order_cancelled",
                order_id=order_id,
                symbol=symbol,
            )
            return True

        except OrderNotFound:
            logger.warning(
                "order_not_found_for_cancel",
                order_id=order_id,
                symbol=symbol,
            )
            return False
        except Exception as e:
            logger.error(
                "order_cancel_failed",
                order_id=order_id,
                symbol=symbol,
                error=str(e),
            )
            return False

    async def get_position(self, symbol: str) -> PositionState | None:
        """
        Get current position for symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Position state or None if no position
        """
        try:
            positions = await self.client.exchange.fetch_positions([symbol])

            for pos in positions:
                if pos.get("symbol") == symbol:
                    contracts = float(pos.get("contracts", 0))
                    if contracts == 0:
                        return None

                    side = "long" if pos.get("side") == "long" else "short"
                    entry_price = float(pos.get("entryPrice", 0))
                    current_price = float(pos.get("markPrice", entry_price))
                    leverage = int(pos.get("leverage", 1))
                    unrealized_pnl = float(pos.get("unrealizedPnl", 0))
                    liquidation_price = float(pos.get("liquidationPrice", 0))

                    return PositionState(
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                        current_price=current_price,
                        amount=contracts,
                        leverage=leverage,
                        unrealized_pnl=unrealized_pnl,
                        liquidation_price=liquidation_price,
                    )

            return None

        except Exception as e:
            logger.error(
                "failed_to_fetch_position",
                symbol=symbol,
                error=str(e),
            )
            return None

    async def get_open_orders(self, symbol: str) -> list[Order]:
        """
        Get all open orders for symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            List of open orders
        """
        try:
            open_orders = await self.client.exchange.fetch_open_orders(symbol)

            orders = []
            for order_data in open_orders:
                order = Order(
                    id=str(uuid.uuid4()),
                    symbol=order_data["symbol"],
                    side=order_data["side"],
                    type=order_data["type"],
                    amount=float(order_data["amount"]),
                    price=float(order_data.get("price", 0)) if order_data.get("price") else None,
                    status="pending",
                    exchange_order_id=order_data["id"],
                    timestamp=datetime.fromtimestamp(order_data["timestamp"] / 1000),
                )
                orders.append(order)

            return orders

        except Exception as e:
            logger.error(
                "failed_to_fetch_open_orders",
                symbol=symbol,
                error=str(e),
            )
            return []

    async def _execute_order_with_retry(self, order: Order) -> Order:
        """
        Execute order with retry logic.

        Args:
            order: Order to execute

        Returns:
            Executed order with status
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Validate order
                await self._validate_order(order)

                # Create order parameters
                order_params = {
                    "symbol": order.symbol,
                    "type": order.type,
                    "side": order.side,
                    "amount": order.amount,
                }

                if order.type == "limit":
                    order_params["price"] = order.price
                elif order.type in ["stop_loss", "take_profit"]:
                    order_params["stopPrice"] = order.stop_price

                # Place order via CCXT
                result = await self.client.exchange.create_order(**order_params)

                # Update order with result
                order.exchange_order_id = result["id"]
                order.status = "filled" if result.get("status") == "closed" else "pending"
                order.filled_price = (
                    float(result.get("average", 0)) if result.get("average") else None
                )
                order.filled_amount = float(result.get("filled", 0))
                order.fee = float(result.get("fee", {}).get("cost", 0))

                logger.info(
                    "order_executed",
                    order_id=order.id,
                    exchange_order_id=order.exchange_order_id,
                    status=order.status,
                    filled_price=order.filled_price,
                )

                return order

            except (InvalidOrder, InsufficientFunds) as e:
                # Don't retry these errors
                order.status = "failed"
                order.error_message = str(e)
                logger.error(
                    "order_execution_failed",
                    order_id=order.id,
                    error=str(e),
                )
                return order

            except (NetworkError, ExchangeError) as e:
                last_error = e
                logger.warning(
                    "order_execution_retry",
                    order_id=order.id,
                    attempt=attempt + 1,
                    error=str(e),
                )

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1 * (2**attempt))  # Exponential backoff
                    continue

            except Exception as e:
                last_error = e
                logger.error(
                    "order_execution_unexpected_error",
                    order_id=order.id,
                    error=str(e),
                )
                break

        # All retries failed
        order.status = "failed"
        order.error_message = str(last_error)

        logger.error(
            "order_execution_failed_all_retries",
            order_id=order.id,
            error=str(last_error),
        )

        return order

    async def _validate_order(self, order: Order) -> None:
        """
        Validate order parameters before execution.

        Args:
            order: Order to validate

        Raises:
            ValueError: If order validation fails
        """
        # Check amount
        if order.amount <= 0:
            raise ValueError(f"Invalid order amount: {order.amount}")

        # Check prices
        if order.type == "limit" and (order.price is None or order.price <= 0):
            raise ValueError("Limit orders require valid price")

        if order.type in ["stop_loss", "take_profit"] and (
            order.stop_price is None or order.stop_price <= 0
        ):
            raise ValueError("Stop orders require valid stop price")

        # Get market limits
        if self.client.exchange.markets:
            market = self.client.exchange.markets.get(order.symbol)
            if market:
                # Check minimum amount
                min_amount = market.get("limits", {}).get("amount", {}).get("min", 0)
                if min_amount and order.amount < min_amount:
                    raise ValueError(f"Order amount {order.amount} below minimum {min_amount}")

                # Check maximum amount
                max_amount = market.get("limits", {}).get("amount", {}).get("max")
                if max_amount and order.amount > max_amount:
                    raise ValueError(f"Order amount {order.amount} exceeds maximum {max_amount}")


# =============================================================================
# Unified Order Executor
# =============================================================================


class OrderExecutor:
    """
    Unified order executor with paper/live mode support.

    Provides high-level interface for executing trading decisions with
    automatic mode switching and comprehensive error handling.

    Example:
        ```python
        # Paper trading
        executor = OrderExecutor(paper_mode=True, initial_balance=10000.0)

        # Live trading
        async with ExchangeClient("binance", api_key, api_secret) as client:
            executor = OrderExecutor(paper_mode=False, exchange_client=client)

        # Execute decision
        request = ExecutionRequest(
            action="long",
            symbol="BTC/USDT",
            amount=0.1,
            stop_loss=48000.0
        )
        order = await executor.execute_decision(request)
        ```
    """

    def __init__(
        self,
        paper_mode: bool = True,
        exchange_client: ExchangeClient | None = None,
        initial_balance: float = 10000.0,
        max_position_size: float = 0.1,  # 10% of balance
        max_leverage: int = 5,
    ):
        """
        Initialize order executor.

        Args:
            paper_mode: Use paper trading mode
            exchange_client: Exchange client for live trading (required if not paper mode)
            initial_balance: Initial balance for paper trading
            max_position_size: Maximum position size as fraction of balance
            max_leverage: Maximum allowed leverage
        """
        self.paper_mode = paper_mode
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage

        if paper_mode:
            self.paper_trader = PaperTrader(initial_balance=initial_balance)
            self.live_executor = None
        else:
            if exchange_client is None:
                raise ValueError("Exchange client required for live trading mode")
            self.paper_trader = None
            self.live_executor = LiveExecutor(exchange_client)

        logger.info(
            "order_executor_initialized",
            paper_mode=paper_mode,
            max_position_size=max_position_size,
            max_leverage=max_leverage,
        )

    async def execute_decision(self, decision: ExecutionRequest) -> Order:
        """
        Execute trading decision.

        Args:
            decision: Execution request to execute

        Returns:
            Executed order

        Raises:
            ValueError: If decision validation fails
        """
        logger.info(
            "executing_decision",
            action=decision.action,
            symbol=decision.symbol,
            amount=decision.amount,
            paper_mode=self.paper_mode,
        )

        # Validate decision
        self._validate_decision(decision)

        # Determine order side
        if decision.action == "long":
            side = "buy"
        elif decision.action == "short":
            side = "sell"
        elif decision.action == "close":
            # Get current position to determine close side
            position = await self._get_current_position(decision.symbol)
            if not position:
                raise ValueError(f"No position to close for {decision.symbol}")
            side = "sell" if position.side == "long" else "buy"
        else:
            raise ValueError(f"Invalid action: {decision.action}")

        # Execute order
        if decision.order_type == "market" or decision.entry_price is None:
            order = await self._execute_market_order(
                symbol=decision.symbol,
                side=side,
                amount=decision.amount,
            )
        else:
            order = await self._execute_limit_order(
                symbol=decision.symbol,
                side=side,
                amount=decision.amount,
                price=decision.entry_price,
            )

        # Set stop-loss and take-profit if specified
        if order.status == "filled" and decision.action != "close":
            if decision.stop_loss:
                await self._set_stop_loss_order(
                    symbol=decision.symbol,
                    side="sell" if side == "buy" else "buy",
                    amount=decision.amount,
                    stop_price=decision.stop_loss,
                )

            if decision.take_profit:
                await self._set_take_profit_order(
                    symbol=decision.symbol,
                    side="sell" if side == "buy" else "buy",
                    amount=decision.amount,
                    price=decision.take_profit,
                )

        return order

    async def update_stop_loss(self, symbol: str, new_stop: float) -> bool:
        """
        Update stop-loss for existing position.

        Args:
            symbol: Trading pair symbol
            new_stop: New stop-loss price

        Returns:
            True if updated successfully
        """
        logger.info(
            "updating_stop_loss",
            symbol=symbol,
            new_stop=new_stop,
        )

        try:
            position = await self._get_current_position(symbol)
            if not position:
                logger.warning("no_position_for_stop_update", symbol=symbol)
                return False

            # Cancel existing stop orders
            await self._cancel_stop_orders(symbol)

            # Place new stop order
            side = "sell" if position.side == "long" else "buy"
            await self._set_stop_loss_order(
                symbol=symbol,
                side=side,
                amount=position.amount,
                stop_price=new_stop,
            )

            return True

        except Exception as e:
            logger.error(
                "stop_loss_update_failed",
                symbol=symbol,
                error=str(e),
            )
            return False

    async def update_take_profit(self, symbol: str, new_tp: float) -> bool:
        """
        Update take-profit for existing position.

        Args:
            symbol: Trading pair symbol
            new_tp: New take-profit price

        Returns:
            True if updated successfully
        """
        logger.info(
            "updating_take_profit",
            symbol=symbol,
            new_tp=new_tp,
        )

        try:
            position = await self._get_current_position(symbol)
            if not position:
                logger.warning("no_position_for_tp_update", symbol=symbol)
                return False

            # Cancel existing TP orders
            await self._cancel_tp_orders(symbol)

            # Place new TP order
            side = "sell" if position.side == "long" else "buy"
            await self._set_take_profit_order(
                symbol=symbol,
                side=side,
                amount=position.amount,
                price=new_tp,
            )

            return True

        except Exception as e:
            logger.error(
                "take_profit_update_failed",
                symbol=symbol,
                error=str(e),
            )
            return False

    async def emergency_close_all(self) -> list[Order]:
        """
        Emergency close all open positions.

        Returns:
            List of closing orders
        """
        logger.warning("emergency_close_all_initiated")

        positions = await self._get_all_positions()
        closing_orders = []

        for position in positions:
            try:
                close_side = "sell" if position.side == "long" else "buy"
                order = await self._execute_market_order(
                    symbol=position.symbol,
                    side=close_side,
                    amount=position.amount,
                )
                closing_orders.append(order)

                logger.info(
                    "emergency_position_closed",
                    symbol=position.symbol,
                    order_id=order.id,
                )

            except Exception as e:
                logger.error(
                    "emergency_close_failed",
                    symbol=position.symbol,
                    error=str(e),
                )

        logger.info(
            "emergency_close_completed",
            positions_closed=len(closing_orders),
        )

        return closing_orders

    async def get_account_status(self) -> dict:
        """
        Get comprehensive account status.

        Returns:
            Dictionary with balance, positions, and equity
        """
        balance = await self._get_balance()
        positions = await self._get_all_positions()

        total_position_value = sum(pos.amount * pos.current_price for pos in positions)
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
        total_equity = balance + total_unrealized_pnl

        return {
            "balance": balance,
            "positions_count": len(positions),
            "positions": [pos.to_dict() for pos in positions],
            "total_position_value": total_position_value,
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_equity": total_equity,
            "paper_mode": self.paper_mode,
        }

    def _validate_decision(self, decision: ExecutionRequest) -> None:
        """
        Validate execution request parameters.

        Args:
            decision: Execution request to validate

        Raises:
            ValueError: If validation fails
        """
        if decision.amount <= 0:
            raise ValueError(f"Invalid amount: {decision.amount}")

        if decision.leverage > self.max_leverage:
            raise ValueError(f"Leverage {decision.leverage} exceeds maximum {self.max_leverage}")

        if decision.stop_loss and decision.entry_price:
            if decision.action == "long" and decision.stop_loss >= decision.entry_price:
                raise ValueError("Stop loss must be below entry for long positions")
            if decision.action == "short" and decision.stop_loss <= decision.entry_price:
                raise ValueError("Stop loss must be above entry for short positions")

        if decision.take_profit and decision.entry_price:
            if decision.action == "long" and decision.take_profit <= decision.entry_price:
                raise ValueError("Take profit must be above entry for long positions")
            if decision.action == "short" and decision.take_profit >= decision.entry_price:
                raise ValueError("Take profit must be below entry for short positions")

    async def _execute_market_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
    ) -> Order:
        """Execute market order in current mode."""
        if self.paper_mode:
            # Get current price for paper trading
            from iftb.data import fetch_latest_ticker

            ticker = await fetch_latest_ticker(symbol)

            order = Order(
                id=str(uuid.uuid4()),
                symbol=symbol,
                side=side,
                type="market",
                amount=amount,
                price=ticker.last,
            )
            return await self.paper_trader.place_order(order, ticker.last)
        return await self.live_executor.place_market_order(symbol, side, amount)

    async def _execute_limit_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
        price: float,
    ) -> Order:
        """Execute limit order in current mode."""
        if self.paper_mode:
            order = Order(
                id=str(uuid.uuid4()),
                symbol=symbol,
                side=side,
                type="limit",
                amount=amount,
                price=price,
            )
            return await self.paper_trader.place_order(order, price)
        return await self.live_executor.place_limit_order(symbol, side, amount, price)

    async def _set_stop_loss_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
        stop_price: float,
    ) -> Order:
        """Set stop-loss order in current mode."""
        if self.paper_mode:
            # Paper trading: just log, don't execute until triggered
            logger.info(
                "paper_stop_loss_set",
                symbol=symbol,
                stop_price=stop_price,
            )
            order = Order(
                id=str(uuid.uuid4()),
                symbol=symbol,
                side=side,
                type="stop_loss",
                amount=amount,
                stop_price=stop_price,
                status="pending",
            )
            return order
        return await self.live_executor.set_stop_loss(symbol, side, amount, stop_price)

    async def _set_take_profit_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
        price: float,
    ) -> Order:
        """Set take-profit order in current mode."""
        if self.paper_mode:
            # Paper trading: just log, don't execute until triggered
            logger.info(
                "paper_take_profit_set",
                symbol=symbol,
                price=price,
            )
            order = Order(
                id=str(uuid.uuid4()),
                symbol=symbol,
                side=side,
                type="take_profit",
                amount=amount,
                price=price,
                status="pending",
            )
            return order
        return await self.live_executor.set_take_profit(symbol, side, amount, price)

    async def _get_current_position(self, symbol: str) -> PositionState | None:
        """Get current position for symbol."""
        if self.paper_mode:
            return self.paper_trader.positions.get(symbol)
        return await self.live_executor.get_position(symbol)

    async def _get_all_positions(self) -> list[PositionState]:
        """Get all open positions."""
        if self.paper_mode:
            return self.paper_trader.get_positions()
        # Get all positions from exchange
        positions = []
        try:
            all_positions = await self.live_executor.client.exchange.fetch_positions()
            for pos_data in all_positions:
                contracts = float(pos_data.get("contracts", 0))
                if contracts > 0:
                    position = await self.live_executor.get_position(pos_data["symbol"])
                    if position:
                        positions.append(position)
        except Exception as e:
            logger.error("failed_to_fetch_all_positions", error=str(e))

        return positions

    async def _get_balance(self) -> float:
        """Get current account balance."""
        if self.paper_mode:
            return self.paper_trader.get_balance()
        try:
            balance = await self.live_executor.client.exchange.fetch_balance()
            return float(balance.get("USDT", {}).get("free", 0))
        except Exception as e:
            logger.error("failed_to_fetch_balance", error=str(e))
            return 0.0

    async def _cancel_stop_orders(self, symbol: str) -> None:
        """Cancel all stop-loss orders for symbol."""
        if not self.paper_mode:
            orders = await self.live_executor.get_open_orders(symbol)
            for order in orders:
                if order.type == "stop_loss":
                    await self.live_executor.cancel_order(order.exchange_order_id, symbol)

    async def _cancel_tp_orders(self, symbol: str) -> None:
        """Cancel all take-profit orders for symbol."""
        if not self.paper_mode:
            orders = await self.live_executor.get_open_orders(symbol)
            for order in orders:
                if order.type == "take_profit":
                    await self.live_executor.cancel_order(order.exchange_order_id, symbol)
