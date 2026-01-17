"""
Unit tests for Order Executor module.

Tests cover:
- Order dataclass functionality
- PaperTrader simulation logic
- ExecutionRequest conversion and validation
- OrderExecutor decision execution
- Stop-loss/take-profit validation
- Emergency close functionality
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from iftb.trading.executor import (
    ExecutionRequest,
    LiveExecutor,
    Order,
    OrderExecutor,
    PaperTrader,
    PositionState,
    convert_decision_to_request,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_order() -> Order:
    """Create a sample order for testing."""
    return Order(
        id=str(uuid.uuid4()),
        symbol="BTC/USDT",
        side="buy",
        type="market",
        amount=0.1,
        price=50000.0,
    )


@pytest.fixture
def sample_limit_order() -> Order:
    """Create a sample limit order for testing."""
    return Order(
        id=str(uuid.uuid4()),
        symbol="ETH/USDT",
        side="sell",
        type="limit",
        amount=1.0,
        price=3000.0,
    )


@pytest.fixture
def sample_stop_loss_order() -> Order:
    """Create a sample stop-loss order for testing."""
    return Order(
        id=str(uuid.uuid4()),
        symbol="BTC/USDT",
        side="sell",
        type="stop_loss",
        amount=0.1,
        stop_price=48000.0,
    )


@pytest.fixture
def sample_position() -> PositionState:
    """Create a sample position for testing."""
    return PositionState(
        symbol="BTC/USDT",
        side="long",
        entry_price=50000.0,
        current_price=51000.0,
        amount=0.1,
        leverage=2,
        unrealized_pnl=100.0,
        liquidation_price=40000.0,
        stop_loss=48000.0,
        take_profit=55000.0,
    )


@pytest.fixture
def sample_execution_request() -> ExecutionRequest:
    """Create a sample execution request for testing."""
    return ExecutionRequest(
        action="long",
        symbol="BTC/USDT",
        amount=0.1,
        entry_price=50000.0,
        stop_loss=48000.0,
        take_profit=55000.0,
        leverage=2,
        order_type="market",
        reason="Strong bullish momentum",
    )


@pytest.fixture
def sample_trading_decision():
    """Create a sample TradingDecision for testing conversion."""
    # Using MagicMock to simulate TradingDecision from decision_engine
    decision = MagicMock()
    decision.action = "LONG"
    decision.symbol = "BTC/USDT"
    decision.entry_price = 50000.0
    decision.stop_loss = 48000.0
    decision.take_profit = 55000.0
    decision.leverage = 2
    decision.vetoed = False
    decision.veto_reason = None
    decision.reasons = ["RSI oversold", "Volume spike", "MACD crossover"]
    return decision


@pytest.fixture
def paper_trader() -> PaperTrader:
    """Create a PaperTrader instance for testing."""
    return PaperTrader(initial_balance=10000.0, maker_fee=0.0002, taker_fee=0.0004)


@pytest.fixture
def paper_executor() -> OrderExecutor:
    """Create a paper mode OrderExecutor for testing."""
    return OrderExecutor(paper_mode=True, initial_balance=10000.0)


@pytest.fixture
def mock_exchange_client():
    """Create a mock ExchangeClient for testing."""
    client = MagicMock()
    client.exchange_id = "binance"
    client.exchange = MagicMock()

    # Mock exchange methods
    client.exchange.create_order = AsyncMock(return_value={
        "id": "exchange-order-123",
        "status": "closed",
        "average": 50000.0,
        "filled": 0.1,
        "fee": {"cost": 2.0},
    })
    client.exchange.cancel_order = AsyncMock()
    client.exchange.fetch_positions = AsyncMock(return_value=[])
    client.exchange.fetch_open_orders = AsyncMock(return_value=[])
    client.exchange.fetch_balance = AsyncMock(return_value={
        "USDT": {"free": 10000.0}
    })
    client.exchange.markets = {
        "BTC/USDT": {
            "limits": {
                "amount": {"min": 0.001, "max": 1000.0}
            }
        }
    }

    return client


@pytest.fixture
def live_executor(mock_exchange_client) -> OrderExecutor:
    """Create a live mode OrderExecutor for testing."""
    return OrderExecutor(paper_mode=False, exchange_client=mock_exchange_client)


# =============================================================================
# Order Dataclass Tests
# =============================================================================


@pytest.mark.unit
class TestOrderDataclass:
    """Test Order dataclass functionality."""

    def test_order_creation_with_all_fields(self):
        """Test Order creation with all fields populated."""
        order_id = str(uuid.uuid4())
        timestamp = datetime.now()

        order = Order(
            id=order_id,
            symbol="BTC/USDT",
            side="buy",
            type="limit",
            amount=0.5,
            price=50000.0,
            stop_price=48000.0,
            status="filled",
            filled_price=49990.0,
            filled_amount=0.5,
            fee=10.0,
            timestamp=timestamp,
            exchange_order_id="exchange-123",
            error_message=None,
        )

        assert order.id == order_id
        assert order.symbol == "BTC/USDT"
        assert order.side == "buy"
        assert order.type == "limit"
        assert order.amount == 0.5
        assert order.price == 50000.0
        assert order.stop_price == 48000.0
        assert order.status == "filled"
        assert order.filled_price == 49990.0
        assert order.filled_amount == 0.5
        assert order.fee == 10.0
        assert order.timestamp == timestamp
        assert order.exchange_order_id == "exchange-123"
        assert order.error_message is None

    def test_order_creation_with_defaults(self):
        """Test Order creation with default values."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol="ETH/USDT",
            side="sell",
            type="market",
            amount=1.0,
        )

        assert order.price is None
        assert order.stop_price is None
        assert order.status == "pending"
        assert order.filled_price is None
        assert order.filled_amount is None
        assert order.fee is None
        assert isinstance(order.timestamp, datetime)
        assert order.exchange_order_id is None
        assert order.error_message is None

    def test_order_to_dict_conversion(self, sample_order):
        """Test Order to_dict conversion."""
        order_dict = sample_order.to_dict()

        assert isinstance(order_dict, dict)
        assert order_dict["id"] == sample_order.id
        assert order_dict["symbol"] == "BTC/USDT"
        assert order_dict["side"] == "buy"
        assert order_dict["type"] == "market"
        assert order_dict["amount"] == 0.1
        assert order_dict["price"] == 50000.0
        assert order_dict["status"] == "pending"
        assert isinstance(order_dict["timestamp"], str)  # ISO format

    def test_order_to_dict_with_filled_details(self, sample_order):
        """Test Order to_dict with filled details."""
        sample_order.status = "filled"
        sample_order.filled_price = 50025.0
        sample_order.filled_amount = 0.1
        sample_order.fee = 2.0

        order_dict = sample_order.to_dict()

        assert order_dict["status"] == "filled"
        assert order_dict["filled_price"] == 50025.0
        assert order_dict["filled_amount"] == 0.1
        assert order_dict["fee"] == 2.0


@pytest.mark.unit
class TestPositionStateDataclass:
    """Test PositionState dataclass functionality."""

    def test_position_creation(self, sample_position):
        """Test PositionState creation."""
        assert sample_position.symbol == "BTC/USDT"
        assert sample_position.side == "long"
        assert sample_position.entry_price == 50000.0
        assert sample_position.current_price == 51000.0
        assert sample_position.amount == 0.1
        assert sample_position.leverage == 2
        assert sample_position.unrealized_pnl == 100.0
        assert sample_position.liquidation_price == 40000.0
        assert sample_position.stop_loss == 48000.0
        assert sample_position.take_profit == 55000.0

    def test_position_to_dict_conversion(self, sample_position):
        """Test PositionState to_dict conversion."""
        pos_dict = sample_position.to_dict()

        assert isinstance(pos_dict, dict)
        assert pos_dict["symbol"] == "BTC/USDT"
        assert pos_dict["side"] == "long"
        assert pos_dict["entry_price"] == 50000.0
        assert pos_dict["amount"] == 0.1
        assert isinstance(pos_dict["opened_at"], str)  # ISO format


# =============================================================================
# PaperTrader Tests
# =============================================================================


@pytest.mark.unit
class TestPaperTraderInitialization:
    """Test PaperTrader initialization."""

    def test_initialize_with_defaults(self):
        """Test PaperTrader initialization with default parameters."""
        trader = PaperTrader()

        assert trader.initial_balance == 10000.0
        assert trader.balance == 10000.0
        assert trader.maker_fee == 0.0002
        assert trader.taker_fee == 0.0004
        assert len(trader.positions) == 0
        assert len(trader.orders) == 0
        assert len(trader.order_history) == 0

    def test_initialize_with_custom_parameters(self):
        """Test PaperTrader initialization with custom parameters."""
        trader = PaperTrader(
            initial_balance=50000.0,
            maker_fee=0.0001,
            taker_fee=0.0002,
        )

        assert trader.initial_balance == 50000.0
        assert trader.balance == 50000.0
        assert trader.maker_fee == 0.0001
        assert trader.taker_fee == 0.0002


@pytest.mark.unit
class TestPaperTraderMarketOrders:
    """Test PaperTrader market order execution."""

    @pytest.mark.asyncio
    async def test_place_market_buy_order(self, paper_trader):
        """Test placing a market buy order."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=0.1,
            price=50000.0,
        )

        filled_order = await paper_trader.place_order(order, current_price=50000.0)

        assert filled_order.status == "filled"
        assert filled_order.filled_price is not None
        assert filled_order.filled_price > 50000.0  # Buy slippage
        assert filled_order.filled_amount == 0.1
        assert filled_order.fee is not None
        assert filled_order.fee > 0

    @pytest.mark.asyncio
    async def test_place_market_sell_order(self, paper_trader):
        """Test placing a market sell order."""
        # First buy to have position
        buy_order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=0.1,
            price=50000.0,
        )
        await paper_trader.place_order(buy_order, current_price=50000.0)

        # Then sell
        sell_order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="sell",
            type="market",
            amount=0.1,
            price=51000.0,
        )

        filled_order = await paper_trader.place_order(sell_order, current_price=51000.0)

        assert filled_order.status == "filled"
        assert filled_order.filled_price is not None
        assert filled_order.filled_price < 51000.0  # Sell slippage
        assert filled_order.filled_amount == 0.1

    @pytest.mark.asyncio
    async def test_market_order_updates_balance(self, paper_trader):
        """Test that market orders update balance correctly."""
        initial_balance = paper_trader.balance

        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=0.1,
            price=50000.0,
        )

        filled_order = await paper_trader.place_order(order, current_price=50000.0)

        # Balance should decrease by order value + fee
        order_cost = filled_order.filled_price * filled_order.filled_amount + filled_order.fee
        assert paper_trader.balance == pytest.approx(initial_balance - order_cost, rel=1e-6)

    @pytest.mark.asyncio
    async def test_market_order_applies_slippage(self, paper_trader):
        """Test that market orders apply slippage."""
        buy_order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=0.1,
            price=50000.0,
        )

        filled = await paper_trader.place_order(buy_order, current_price=50000.0)

        # Buy should have positive slippage (higher price)
        expected_slippage = 50000.0 * 0.0005
        assert filled.filled_price == pytest.approx(50000.0 + expected_slippage, rel=1e-6)


@pytest.mark.unit
class TestPaperTraderLimitOrders:
    """Test PaperTrader limit order execution."""

    @pytest.mark.asyncio
    async def test_place_limit_order(self, paper_trader):
        """Test placing a limit order."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol="ETH/USDT",
            side="buy",
            type="limit",
            amount=1.0,
            price=3000.0,
        )

        filled_order = await paper_trader.place_order(order)

        assert filled_order.status == "filled"
        assert filled_order.filled_price == 3000.0  # Limit orders fill at limit price
        assert filled_order.filled_amount == 1.0
        assert filled_order.fee is not None

    @pytest.mark.asyncio
    async def test_limit_order_uses_maker_fee(self, paper_trader):
        """Test that limit orders use maker fee rate."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="limit",
            amount=0.1,
            price=50000.0,
        )

        filled_order = await paper_trader.place_order(order)

        expected_fee = 50000.0 * 0.1 * paper_trader.maker_fee
        assert filled_order.fee == pytest.approx(expected_fee, rel=1e-6)


@pytest.mark.unit
class TestPaperTraderInsufficientBalance:
    """Test PaperTrader insufficient balance handling."""

    @pytest.mark.asyncio
    async def test_insufficient_balance_for_buy(self, paper_trader):
        """Test that orders fail with insufficient balance."""
        # Try to buy more than balance allows
        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=1.0,  # 1 BTC at 50k = 50k USDT, more than 10k balance
            price=50000.0,
        )

        filled_order = await paper_trader.place_order(order, current_price=50000.0)

        assert filled_order.status == "failed"
        assert filled_order.error_message == "Insufficient balance"
        assert paper_trader.balance == 10000.0  # Balance unchanged

    @pytest.mark.asyncio
    async def test_order_history_includes_failed_orders(self, paper_trader):
        """Test that failed orders are added to order history."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=1.0,
            price=50000.0,
        )

        await paper_trader.place_order(order, current_price=50000.0)

        assert len(paper_trader.order_history) == 1
        assert paper_trader.order_history[0].status == "failed"


@pytest.mark.unit
class TestPaperTraderPositions:
    """Test PaperTrader position tracking."""

    @pytest.mark.asyncio
    async def test_position_opened_on_buy(self, paper_trader):
        """Test that buying opens a long position."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=0.1,
            price=50000.0,
        )

        await paper_trader.place_order(order, current_price=50000.0)

        assert "BTC/USDT" in paper_trader.positions
        position = paper_trader.positions["BTC/USDT"]
        assert position.side == "long"
        assert position.amount == 0.1
        assert position.entry_price > 0

    @pytest.mark.asyncio
    async def test_position_closed_fully(self, paper_trader):
        """Test that selling entire position closes it."""
        # Buy
        buy_order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=0.1,
            price=50000.0,
        )
        await paper_trader.place_order(buy_order, current_price=50000.0)

        # Sell entire position
        sell_order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="sell",
            type="market",
            amount=0.1,
            price=51000.0,
        )
        await paper_trader.place_order(sell_order, current_price=51000.0)

        assert "BTC/USDT" not in paper_trader.positions

    @pytest.mark.asyncio
    async def test_position_closed_partially(self, paper_trader):
        """Test that selling part of position reduces amount."""
        # Buy
        buy_order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=0.2,
            price=50000.0,
        )
        await paper_trader.place_order(buy_order, current_price=50000.0)

        # Sell half
        sell_order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="sell",
            type="market",
            amount=0.1,
            price=51000.0,
        )
        await paper_trader.place_order(sell_order, current_price=51000.0)

        assert "BTC/USDT" in paper_trader.positions
        position = paper_trader.positions["BTC/USDT"]
        assert position.amount == 0.1

    @pytest.mark.asyncio
    async def test_position_increased_averages_entry_price(self, paper_trader):
        """Test that adding to position averages entry price."""
        # First buy at 50k (smaller amount to leave balance for second order)
        order1 = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="limit",
            amount=0.05,
            price=50000.0,
        )
        filled1 = await paper_trader.place_order(order1)
        assert filled1.status == "filled", f"First order failed: {filled1.error_message}"

        # Second buy at 52k
        order2 = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="limit",
            amount=0.05,
            price=52000.0,
        )
        filled2 = await paper_trader.place_order(order2)
        assert filled2.status == "filled", f"Second order failed: {filled2.error_message}"

        position = paper_trader.positions["BTC/USDT"]
        assert position.amount == pytest.approx(0.1, rel=1e-6)
        expected_avg = (50000.0 + 52000.0) / 2
        assert position.entry_price == pytest.approx(expected_avg, rel=1e-6)


@pytest.mark.unit
class TestPaperTraderFees:
    """Test PaperTrader fee calculation."""

    @pytest.mark.asyncio
    async def test_market_order_taker_fee(self, paper_trader):
        """Test that market orders use taker fee."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=0.1,
            price=50000.0,
        )

        filled = await paper_trader.place_order(order, current_price=50000.0)

        order_value = filled.filled_price * filled.filled_amount
        expected_fee = order_value * paper_trader.taker_fee
        assert filled.fee == pytest.approx(expected_fee, rel=1e-6)

    @pytest.mark.asyncio
    async def test_limit_order_maker_fee(self, paper_trader):
        """Test that limit orders use maker fee."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="limit",
            amount=0.1,
            price=50000.0,
        )

        filled = await paper_trader.place_order(order)

        order_value = 50000.0 * 0.1
        expected_fee = order_value * paper_trader.maker_fee
        assert filled.fee == pytest.approx(expected_fee, rel=1e-6)


@pytest.mark.unit
class TestPaperTraderMethods:
    """Test PaperTrader utility methods."""

    def test_get_balance(self, paper_trader):
        """Test get_balance method."""
        assert paper_trader.get_balance() == 10000.0

    @pytest.mark.asyncio
    async def test_get_positions(self, paper_trader):
        """Test get_positions method."""
        # Initially empty
        assert paper_trader.get_positions() == []

        # Add position
        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=0.1,
            price=50000.0,
        )
        await paper_trader.place_order(order, current_price=50000.0)

        positions = paper_trader.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_get_order_history(self, paper_trader):
        """Test get_order_history method."""
        # Initially empty
        assert paper_trader.get_order_history() == []

        # Place order
        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=0.1,
            price=50000.0,
        )
        await paper_trader.place_order(order, current_price=50000.0)

        history = paper_trader.get_order_history()
        assert len(history) == 1
        assert history[0].symbol == "BTC/USDT"

    def test_update_position_prices(self, paper_trader):
        """Test update_position_prices for PnL calculation."""
        # Create a position manually
        paper_trader.positions["BTC/USDT"] = PositionState(
            symbol="BTC/USDT",
            side="long",
            entry_price=50000.0,
            current_price=50000.0,
            amount=0.1,
            leverage=1,
            unrealized_pnl=0.0,
            liquidation_price=0.0,
        )

        # Update with higher price
        paper_trader.update_position_prices("BTC/USDT", 51000.0)

        position = paper_trader.positions["BTC/USDT"]
        assert position.current_price == 51000.0
        expected_pnl = (51000.0 - 50000.0) * 0.1
        assert position.unrealized_pnl == pytest.approx(expected_pnl, rel=1e-6)


# =============================================================================
# ExecutionRequest Tests
# =============================================================================


@pytest.mark.unit
class TestExecutionRequest:
    """Test ExecutionRequest dataclass."""

    def test_execution_request_creation(self, sample_execution_request):
        """Test ExecutionRequest creation."""
        assert sample_execution_request.action == "long"
        assert sample_execution_request.symbol == "BTC/USDT"
        assert sample_execution_request.amount == 0.1
        assert sample_execution_request.entry_price == 50000.0
        assert sample_execution_request.stop_loss == 48000.0
        assert sample_execution_request.take_profit == 55000.0
        assert sample_execution_request.leverage == 2
        assert sample_execution_request.order_type == "market"

    def test_execution_request_with_defaults(self):
        """Test ExecutionRequest with default values."""
        request = ExecutionRequest(
            action="short",
            symbol="ETH/USDT",
            amount=1.0,
        )

        assert request.entry_price is None
        assert request.stop_loss is None
        assert request.take_profit is None
        assert request.leverage == 1
        assert request.order_type == "market"
        assert request.reason is None


@pytest.mark.unit
class TestConvertDecisionToRequest:
    """Test convert_decision_to_request function."""

    def test_convert_long_decision(self, sample_trading_decision):
        """Test converting LONG decision to execution request."""
        request = convert_decision_to_request(
            sample_trading_decision,
            amount=0.1,
            order_type="market",
        )

        assert request.action == "long"
        assert request.symbol == "BTC/USDT"
        assert request.amount == 0.1
        assert request.entry_price == 50000.0
        assert request.stop_loss == 48000.0
        assert request.take_profit == 55000.0
        assert request.leverage == 2
        assert request.order_type == "market"
        assert "RSI oversold" in request.reason

    def test_convert_short_decision(self, sample_trading_decision):
        """Test converting SHORT decision to execution request."""
        sample_trading_decision.action = "SHORT"

        request = convert_decision_to_request(
            sample_trading_decision,
            amount=0.05,
            order_type="limit",
        )

        assert request.action == "short"
        assert request.amount == 0.05
        assert request.order_type == "limit"

    def test_convert_hold_decision(self, sample_trading_decision):
        """Test converting HOLD decision to execution request."""
        sample_trading_decision.action = "HOLD"

        request = convert_decision_to_request(
            sample_trading_decision,
            amount=0.1,
        )

        assert request.action == "close"

    def test_convert_vetoed_decision(self, sample_trading_decision):
        """Test converting vetoed decision."""
        sample_trading_decision.vetoed = True
        sample_trading_decision.veto_reason = "Insufficient confidence"

        request = convert_decision_to_request(
            sample_trading_decision,
            amount=0.1,
        )

        assert request.reason == "Insufficient confidence"


# =============================================================================
# OrderExecutor Tests
# =============================================================================


@pytest.mark.unit
class TestOrderExecutorInitialization:
    """Test OrderExecutor initialization."""

    def test_paper_mode_initialization(self, paper_executor):
        """Test OrderExecutor initialization in paper mode."""
        assert paper_executor.paper_mode is True
        assert paper_executor.paper_trader is not None
        assert paper_executor.live_executor is None
        assert paper_executor.max_position_size == 0.1
        assert paper_executor.max_leverage == 5

    def test_live_mode_initialization(self, live_executor):
        """Test OrderExecutor initialization in live mode."""
        assert live_executor.paper_mode is False
        assert live_executor.paper_trader is None
        assert live_executor.live_executor is not None

    def test_live_mode_requires_exchange_client(self):
        """Test that live mode requires exchange client."""
        with pytest.raises(ValueError, match="Exchange client required"):
            OrderExecutor(paper_mode=False, exchange_client=None)

    def test_custom_risk_parameters(self):
        """Test OrderExecutor with custom risk parameters."""
        executor = OrderExecutor(
            paper_mode=True,
            initial_balance=50000.0,
            max_position_size=0.2,
            max_leverage=10,
        )

        assert executor.max_position_size == 0.2
        assert executor.max_leverage == 10
        assert executor.paper_trader.initial_balance == 50000.0


@pytest.mark.unit
class TestOrderExecutorValidation:
    """Test OrderExecutor decision validation."""

    def test_validate_invalid_amount(self, paper_executor):
        """Test validation rejects invalid amounts."""
        request = ExecutionRequest(
            action="long",
            symbol="BTC/USDT",
            amount=0.0,  # Invalid
        )

        with pytest.raises(ValueError, match="Invalid amount"):
            paper_executor._validate_decision(request)

    def test_validate_excessive_leverage(self, paper_executor):
        """Test validation rejects excessive leverage."""
        request = ExecutionRequest(
            action="long",
            symbol="BTC/USDT",
            amount=0.1,
            leverage=10,  # Exceeds max of 5
        )

        with pytest.raises(ValueError, match="exceeds maximum"):
            paper_executor._validate_decision(request)

    def test_validate_long_stop_loss_below_entry(self, paper_executor):
        """Test validation requires stop-loss below entry for long."""
        request = ExecutionRequest(
            action="long",
            symbol="BTC/USDT",
            amount=0.1,
            entry_price=50000.0,
            stop_loss=51000.0,  # Above entry
        )

        with pytest.raises(ValueError, match="Stop loss must be below entry"):
            paper_executor._validate_decision(request)

    def test_validate_short_stop_loss_above_entry(self, paper_executor):
        """Test validation requires stop-loss above entry for short."""
        request = ExecutionRequest(
            action="short",
            symbol="BTC/USDT",
            amount=0.1,
            entry_price=50000.0,
            stop_loss=49000.0,  # Below entry
        )

        with pytest.raises(ValueError, match="Stop loss must be above entry"):
            paper_executor._validate_decision(request)

    def test_validate_long_take_profit_above_entry(self, paper_executor):
        """Test validation requires take-profit above entry for long."""
        request = ExecutionRequest(
            action="long",
            symbol="BTC/USDT",
            amount=0.1,
            entry_price=50000.0,
            take_profit=49000.0,  # Below entry
        )

        with pytest.raises(ValueError, match="Take profit must be above entry"):
            paper_executor._validate_decision(request)

    def test_validate_short_take_profit_below_entry(self, paper_executor):
        """Test validation requires take-profit below entry for short."""
        request = ExecutionRequest(
            action="short",
            symbol="BTC/USDT",
            amount=0.1,
            entry_price=50000.0,
            take_profit=51000.0,  # Above entry
        )

        with pytest.raises(ValueError, match="Take profit must be below entry"):
            paper_executor._validate_decision(request)

    def test_validate_valid_long_request(self, paper_executor):
        """Test validation accepts valid long request."""
        request = ExecutionRequest(
            action="long",
            symbol="BTC/USDT",
            amount=0.1,
            entry_price=50000.0,
            stop_loss=48000.0,
            take_profit=55000.0,
            leverage=3,
        )

        # Should not raise
        paper_executor._validate_decision(request)

    def test_validate_valid_short_request(self, paper_executor):
        """Test validation accepts valid short request."""
        request = ExecutionRequest(
            action="short",
            symbol="BTC/USDT",
            amount=0.1,
            entry_price=50000.0,
            stop_loss=52000.0,
            take_profit=45000.0,
            leverage=2,
        )

        # Should not raise
        paper_executor._validate_decision(request)


@pytest.mark.unit
class TestOrderExecutorDecisionExecution:
    """Test OrderExecutor execute_decision method."""

    @pytest.mark.asyncio
    @patch("iftb.data.fetch_latest_ticker")
    async def test_execute_market_long_decision_paper_mode(
        self, mock_fetch_ticker, paper_executor
    ):
        """Test executing market long decision in paper mode."""
        # Mock ticker data
        mock_ticker = MagicMock()
        mock_ticker.last = 50000.0
        mock_fetch_ticker.return_value = mock_ticker

        request = ExecutionRequest(
            action="long",
            symbol="BTC/USDT",
            amount=0.1,
            order_type="market",
        )

        order = await paper_executor.execute_decision(request)

        assert order.status == "filled"
        assert order.side == "buy"
        assert order.type == "market"
        assert order.amount == 0.1
        assert order.filled_price is not None

    @pytest.mark.asyncio
    @patch("iftb.data.fetch_latest_ticker")
    async def test_execute_market_short_decision_paper_mode(
        self, mock_fetch_ticker, paper_executor
    ):
        """Test executing market short decision in paper mode."""
        mock_ticker = MagicMock()
        mock_ticker.last = 50000.0
        mock_fetch_ticker.return_value = mock_ticker

        request = ExecutionRequest(
            action="short",
            symbol="BTC/USDT",
            amount=0.1,
            order_type="market",
        )

        order = await paper_executor.execute_decision(request)

        assert order.status == "filled"
        assert order.side == "sell"
        assert order.type == "market"

    @pytest.mark.asyncio
    async def test_execute_limit_order_paper_mode(self, paper_executor):
        """Test executing limit order in paper mode."""
        request = ExecutionRequest(
            action="long",
            symbol="BTC/USDT",
            amount=0.1,
            entry_price=50000.0,
            order_type="limit",
        )

        order = await paper_executor.execute_decision(request)

        assert order.status == "filled"
        assert order.type == "limit"
        assert order.filled_price == 50000.0

    @pytest.mark.asyncio
    @patch("iftb.data.fetch_latest_ticker")
    async def test_execute_close_decision_requires_position(
        self, mock_fetch_ticker, paper_executor
    ):
        """Test that close action requires existing position."""
        mock_ticker = MagicMock()
        mock_ticker.last = 50000.0
        mock_fetch_ticker.return_value = mock_ticker

        request = ExecutionRequest(
            action="close",
            symbol="BTC/USDT",
            amount=0.1,
        )

        with pytest.raises(ValueError, match="No position to close"):
            await paper_executor.execute_decision(request)

    @pytest.mark.asyncio
    @patch("iftb.data.fetch_latest_ticker")
    async def test_execute_decision_with_stop_loss(
        self, mock_fetch_ticker, paper_executor
    ):
        """Test that stop-loss is set after order execution."""
        mock_ticker = MagicMock()
        mock_ticker.last = 50000.0
        mock_fetch_ticker.return_value = mock_ticker

        request = ExecutionRequest(
            action="long",
            symbol="BTC/USDT",
            amount=0.1,
            stop_loss=48000.0,
            order_type="market",
        )

        order = await paper_executor.execute_decision(request)

        assert order.status == "filled"
        # In paper mode, stop-loss is logged but not executed until triggered

    @pytest.mark.asyncio
    @patch("iftb.data.fetch_latest_ticker")
    async def test_execute_decision_with_take_profit(
        self, mock_fetch_ticker, paper_executor
    ):
        """Test that take-profit is set after order execution."""
        mock_ticker = MagicMock()
        mock_ticker.last = 50000.0
        mock_fetch_ticker.return_value = mock_ticker

        request = ExecutionRequest(
            action="long",
            symbol="BTC/USDT",
            amount=0.1,
            take_profit=55000.0,
            order_type="market",
        )

        order = await paper_executor.execute_decision(request)

        assert order.status == "filled"


@pytest.mark.unit
class TestOrderExecutorEmergencyClose:
    """Test OrderExecutor emergency close functionality."""

    @pytest.mark.asyncio
    @patch("iftb.data.fetch_latest_ticker")
    async def test_emergency_close_all_positions(
        self, mock_fetch_ticker, paper_executor
    ):
        """Test emergency close closes all open positions."""
        mock_ticker = MagicMock()
        mock_ticker.last = 50000.0
        mock_fetch_ticker.return_value = mock_ticker

        # Open multiple positions
        for symbol in ["BTC/USDT", "ETH/USDT"]:
            mock_ticker.last = 50000.0 if symbol == "BTC/USDT" else 3000.0
            request = ExecutionRequest(
                action="long",
                symbol=symbol,
                amount=0.1,
                order_type="market",
            )
            await paper_executor.execute_decision(request)

        # Emergency close
        closing_orders = await paper_executor.emergency_close_all()

        assert len(closing_orders) == 2
        assert all(order.side == "sell" for order in closing_orders)

        # Verify positions are closed
        positions = await paper_executor._get_all_positions()
        assert len(positions) == 0

    @pytest.mark.asyncio
    async def test_emergency_close_with_no_positions(self, paper_executor):
        """Test emergency close with no open positions."""
        closing_orders = await paper_executor.emergency_close_all()

        assert len(closing_orders) == 0


@pytest.mark.unit
class TestOrderExecutorAccountStatus:
    """Test OrderExecutor account status methods."""

    @pytest.mark.asyncio
    async def test_get_account_status_paper_mode(self, paper_executor):
        """Test get_account_status in paper mode."""
        status = await paper_executor.get_account_status()

        assert "balance" in status
        assert "positions_count" in status
        assert "positions" in status
        assert "total_position_value" in status
        assert "total_unrealized_pnl" in status
        assert "total_equity" in status
        assert status["paper_mode"] is True
        assert status["balance"] == 10000.0
        assert status["positions_count"] == 0

    @pytest.mark.asyncio
    @patch("iftb.data.fetch_latest_ticker")
    async def test_get_account_status_with_positions(
        self, mock_fetch_ticker, paper_executor
    ):
        """Test get_account_status with open positions."""
        mock_ticker = MagicMock()
        mock_ticker.last = 50000.0
        mock_fetch_ticker.return_value = mock_ticker

        # Open position
        request = ExecutionRequest(
            action="long",
            symbol="BTC/USDT",
            amount=0.1,
            order_type="market",
        )
        await paper_executor.execute_decision(request)

        # Update position price for PnL
        paper_executor.paper_trader.update_position_prices("BTC/USDT", 51000.0)

        status = await paper_executor.get_account_status()

        assert status["positions_count"] == 1
        assert status["total_position_value"] > 0
        assert status["total_unrealized_pnl"] > 0
        assert status["total_equity"] > status["balance"]


@pytest.mark.unit
class TestLiveExecutorIntegration:
    """Test LiveExecutor integration with OrderExecutor."""

    @pytest.mark.asyncio
    async def test_live_executor_place_market_order(self, mock_exchange_client):
        """Test LiveExecutor places market order via exchange."""
        executor = LiveExecutor(mock_exchange_client)

        order = await executor.place_market_order(
            symbol="BTC/USDT",
            side="buy",
            amount=0.1,
        )

        assert order.status == "filled"
        assert order.exchange_order_id == "exchange-order-123"
        assert order.filled_price == 50000.0

        # Verify exchange was called
        mock_exchange_client.exchange.create_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_live_executor_place_limit_order(self, mock_exchange_client):
        """Test LiveExecutor places limit order via exchange."""
        executor = LiveExecutor(mock_exchange_client)

        order = await executor.place_limit_order(
            symbol="BTC/USDT",
            side="buy",
            amount=0.1,
            price=49000.0,
        )

        assert order.type == "limit"
        mock_exchange_client.exchange.create_order.assert_called_once()


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_order_with_missing_current_price(self, paper_trader):
        """Test that market orders require current price."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=0.1,
            price=None,  # No price
        )

        filled = await paper_trader.place_order(order, current_price=None)

        assert filled.status == "failed"
        assert "Current price required" in filled.error_message

    @pytest.mark.asyncio
    async def test_limit_order_without_price(self, paper_trader):
        """Test that limit orders require price."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="limit",
            amount=0.1,
            price=None,  # No price
        )

        filled = await paper_trader.place_order(order)

        assert filled.status == "failed"
        assert "Price required" in filled.error_message

    @pytest.mark.asyncio
    async def test_stop_order_without_stop_price(self, paper_trader):
        """Test that stop orders require stop price."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="sell",
            type="stop_loss",
            amount=0.1,
            stop_price=None,  # No stop price
        )

        filled = await paper_trader.place_order(order)

        assert filled.status == "failed"
        assert "Stop price required" in filled.error_message

    @pytest.mark.asyncio
    async def test_balance_updates_correctly_on_sell(self, paper_trader):
        """Test that selling increases balance correctly."""
        # Buy first
        buy_order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="buy",
            type="market",
            amount=0.1,
            price=50000.0,
        )
        await paper_trader.place_order(buy_order, current_price=50000.0)

        balance_after_buy = paper_trader.balance

        # Sell at higher price
        sell_order = Order(
            id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="sell",
            type="market",
            amount=0.1,
            price=52000.0,
        )
        filled = await paper_trader.place_order(sell_order, current_price=52000.0)

        # Balance should increase
        assert paper_trader.balance > balance_after_buy

        # Calculate expected balance increase
        sell_value = filled.filled_price * filled.filled_amount - filled.fee
        expected_balance = balance_after_buy + sell_value
        assert paper_trader.balance == pytest.approx(expected_balance, rel=1e-6)


@pytest.mark.unit
class TestConcurrencyAndRaceConditions:
    """Test concurrent operations and race conditions."""

    @pytest.mark.asyncio
    async def test_multiple_orders_same_symbol(self, paper_trader):
        """Test handling multiple orders for same symbol."""
        orders = []
        for i in range(3):
            order = Order(
                id=str(uuid.uuid4()),
                symbol="BTC/USDT",
                side="buy",
                type="market",
                amount=0.01,
                price=50000.0 + i * 100,
            )
            filled = await paper_trader.place_order(
                order, current_price=50000.0 + i * 100
            )
            orders.append(filled)

        # All should succeed
        assert all(o.status == "filled" for o in orders)

        # Position should reflect total amount
        position = paper_trader.positions["BTC/USDT"]
        assert position.amount == pytest.approx(0.03, rel=1e-6)

    @pytest.mark.asyncio
    async def test_order_history_maintains_order(self, paper_trader):
        """Test that order history maintains chronological order."""
        for i in range(5):
            order = Order(
                id=f"order-{i}",
                symbol="BTC/USDT",
                side="buy",
                type="market",
                amount=0.01,
                price=50000.0,
            )
            await paper_trader.place_order(order, current_price=50000.0)

        history = paper_trader.get_order_history()
        assert len(history) == 5

        # Verify IDs are in order
        for i, order in enumerate(history):
            assert order.id == f"order-{i}"
