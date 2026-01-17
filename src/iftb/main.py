"""
IFTB - Intelligent Futures Trading Bot

Main entry point for the trading bot application.
Integrates all components: data fetching, technical analysis,
LLM sentiment, ML validation, decision engine, and order execution.
"""

import asyncio
from datetime import UTC, datetime
import signal
import sys
from typing import NoReturn

from iftb.analysis import (
    LLMVetoSystem,
    TechnicalAnalyzer,
    XGBoostValidator,
    create_analyzer_from_settings,
)
from iftb.config import get_settings
from iftb.data import (
    CacheManager,
    DatabaseManager,
    ExchangeClient,
    ExternalDataAggregator,
    OHLCVRepository,
    PositionRepository,
    RealTimeDataManager,
    TradeRepository,
    create_market_streamer,
    fetch_latest_ohlcv,
    fetch_latest_ticker,
)
from iftb.trading import (
    OrderExecutor,
    convert_decision_to_request,
    create_decision_engine,
)
from iftb.utils import LogConfig, get_logger, setup_logging

# Global shutdown flag
shutdown_event = asyncio.Event()


def setup_signal_handlers() -> None:
    """Setup graceful shutdown handlers for SIGINT and SIGTERM."""

    def signal_handler(signum: int, frame: object) -> None:
        logger = get_logger(__name__)
        logger.info("shutdown_signal_received", signal=signal.Signals(signum).name)
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def initialize_components() -> dict:
    """
    Initialize all system components.

    Returns:
        Dictionary containing initialized components.
    """
    logger = get_logger(__name__)
    settings = get_settings()

    logger.info("initializing_components")

    components: dict = {}

    # Initialize database connection
    try:
        db_url = (
            f"postgresql+asyncpg://{settings.database.user}:"
            f"{settings.database.password.get_secret_value()}@"
            f"{settings.database.host}:{settings.database.port}/"
            f"{settings.database.name}"
        )
        db_manager = DatabaseManager(
            database_url=db_url,
            pool_size=settings.database.pool_size,
        )
        await db_manager.connect()
        components["db"] = db_manager
        components["ohlcv_repo"] = OHLCVRepository(db_manager)
        components["trade_repo"] = TradeRepository(db_manager)
        components["position_repo"] = PositionRepository(db_manager)
        logger.info("database_initialized")
    except Exception as e:
        logger.warning("database_init_failed", error=str(e))
        # Continue without database - some features will be disabled

    # Initialize Redis cache
    try:
        cache_manager = CacheManager(
            host=settings.redis.host,
            port=settings.redis.port,
            password=settings.redis.password.get_secret_value()
            if settings.redis.password
            else None,
            db=settings.redis.db,
        )
        await cache_manager.connect()
        components["cache"] = cache_manager
        logger.info("redis_initialized")
    except Exception as e:
        logger.warning("redis_init_failed", error=str(e))

    # Initialize exchange client
    try:
        exchange_client = ExchangeClient(
            exchange_id="binanceusdm",
            api_key=settings.exchange.api_key.get_secret_value(),
            api_secret=settings.exchange.api_secret.get_secret_value(),
            testnet=settings.exchange.testnet,
        )
        await exchange_client.initialize()
        components["exchange"] = exchange_client
        logger.info("exchange_initialized", testnet=settings.exchange.testnet)
    except Exception as e:
        logger.error("exchange_init_failed", error=str(e))
        raise RuntimeError(f"Exchange initialization failed: {e}")

    # Initialize external data aggregator (Fear & Greed, Coinglass)
    try:
        external_data = ExternalDataAggregator()
        components["external_data"] = external_data
        logger.info("external_data_initialized")
    except Exception as e:
        logger.warning("external_data_init_failed", error=str(e))

    # Note: TechnicalAnalyzer is created per-request in process_symbol
    # because it requires OHLCV data in its constructor
    logger.info("technical_analyzer_ready")

    # Initialize LLM analyzer
    try:
        llm_analyzer = create_analyzer_from_settings()
        components["llm_analyzer"] = llm_analyzer
        components["llm_veto"] = LLMVetoSystem(llm_analyzer)
        logger.info("llm_analyzer_initialized", model=settings.llm.model)
    except Exception as e:
        logger.warning("llm_init_failed", error=str(e))

    # Initialize ML model
    try:
        ml_model = XGBoostValidator()
        # Try to load existing model
        model_path = "models/xgb_validator.json"
        try:
            ml_model.load_model(model_path)
            logger.info("ml_model_loaded", path=model_path)
        except FileNotFoundError:
            logger.warning("ml_model_not_found", path=model_path)
        components["ml_model"] = ml_model
    except Exception as e:
        logger.warning("ml_model_init_failed", error=str(e))

    # Initialize decision engine
    components["decision_engine"] = create_decision_engine()
    logger.info("decision_engine_initialized")

    # Initialize order executor
    executor = OrderExecutor(
        paper_mode=settings.trading.paper_trading,
        exchange_client=components.get("exchange"),
        initial_balance=settings.trading.initial_balance,
    )
    components["executor"] = executor
    logger.info(
        "executor_initialized",
        paper_mode=settings.trading.paper_trading,
    )

    # Initialize real-time data manager (optional)
    if settings.trading.use_websocket:
        try:
            streamer = create_market_streamer(
                symbols=settings.trading.symbols,
                timeframes=settings.trading.timeframes,
                testnet=settings.exchange.testnet,
            )
            rtm = RealTimeDataManager(
                streamer=streamer,
                cache=components.get("cache"),
                db_manager=components.get("db"),
            )
            components["realtime_manager"] = rtm
            logger.info("realtime_manager_initialized")
        except Exception as e:
            logger.warning("realtime_manager_init_failed", error=str(e))

    logger.info("components_initialized", count=len(components))
    return components


async def shutdown_components(components: dict) -> None:
    """
    Gracefully shutdown all components.

    Args:
        components: Dictionary of initialized components.
    """
    logger = get_logger(__name__)
    logger.info("shutting_down_components")

    # Stop real-time data manager
    if "realtime_manager" in components:
        try:
            await components["realtime_manager"].stop()
            logger.info("realtime_manager_stopped")
        except Exception as e:
            logger.error("realtime_manager_stop_failed", error=str(e))

    # Close exchange connection
    if "exchange" in components:
        try:
            await components["exchange"].close()
            logger.info("exchange_closed")
        except Exception as e:
            logger.error("exchange_close_failed", error=str(e))

    # Close Redis connection
    if "cache" in components:
        try:
            await components["cache"].close()
            logger.info("redis_closed")
        except Exception as e:
            logger.error("redis_close_failed", error=str(e))

    # Close database connection
    if "db" in components:
        try:
            await components["db"].close()
            logger.info("database_closed")
        except Exception as e:
            logger.error("database_close_failed", error=str(e))

    logger.info("components_shutdown_complete")


def ohlcv_bars_to_dataframe(ohlcv_bars: list) -> "pd.DataFrame":
    """Convert list of OHLCVBar to pandas DataFrame."""
    import pandas as pd

    data = {
        "timestamp": [bar.timestamp for bar in ohlcv_bars],
        "open": [bar.open for bar in ohlcv_bars],
        "high": [bar.high for bar in ohlcv_bars],
        "low": [bar.low for bar in ohlcv_bars],
        "close": [bar.close for bar in ohlcv_bars],
        "volume": [bar.volume for bar in ohlcv_bars],
    }
    return pd.DataFrame(data)


async def process_symbol(
    symbol: str,
    timeframe: str,
    components: dict,
    logger,
) -> None:
    """
    Process a single symbol for trading decision.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        timeframe: Candlestick timeframe (e.g., "1h")
        components: Dictionary of initialized components
        logger: Logger instance
    """
    settings = get_settings()

    try:
        # 1. Fetch latest OHLCV data
        exchange = components["exchange"]
        ohlcv_bars = await fetch_latest_ohlcv(
            exchange,
            symbol=symbol,
            timeframe=timeframe,
            limit=100,  # Need 100 bars for indicators
        )

        if not ohlcv_bars or len(ohlcv_bars) < 52:
            logger.warning(
                "insufficient_data",
                symbol=symbol,
                timeframe=timeframe,
                bars=len(ohlcv_bars) if ohlcv_bars else 0,
            )
            return

        # Convert to DataFrame for technical analysis
        ohlcv_df = ohlcv_bars_to_dataframe(ohlcv_bars)

        # 2. Get current ticker for entry price
        ticker = await fetch_latest_ticker(exchange, symbol)
        current_price = ticker.last if ticker else ohlcv_bars[-1].close

        # 3. Generate technical signals
        technical_analyzer = TechnicalAnalyzer(ohlcv_df)
        composite_signal = technical_analyzer.generate_composite_signal()

        logger.debug(
            "technical_signal",
            symbol=symbol,
            signal=composite_signal.overall_signal,
            confidence=composite_signal.confidence,
        )

        # 4. Get market context from external data
        market_context = None
        if "external_data" in components:
            try:
                market_context = await components["external_data"].get_market_context(symbol)
            except Exception as e:
                logger.warning("market_context_failed", error=str(e))

        # 5. Get LLM analysis (if available)
        llm_analysis = None
        if "llm_analyzer" in components:
            try:
                llm_analyzer = components["llm_analyzer"]
                llm_analysis = await llm_analyzer.analyze_market(
                    symbol=symbol,
                    ohlcv_data=ohlcv_bars[-20:],  # Last 20 bars
                    technical_signal=composite_signal,
                    market_context=market_context,
                )
            except Exception as e:
                logger.warning("llm_analysis_failed", error=str(e))

        # 6. Get ML model prediction (if available)
        ml_prediction = None
        if "ml_model" in components:
            try:
                ml_model = components["ml_model"]
                if ml_model.is_trained:
                    ml_prediction = ml_model.predict(
                        ohlcv_data=ohlcv_bars,
                        technical_signal=composite_signal,
                        market_context=market_context,
                    )
            except Exception as e:
                logger.warning("ml_prediction_failed", error=str(e))

        # 7. Get account status for balance info
        executor = components["executor"]
        account_status = executor.get_account_status()
        account_balance = account_status.get("balance", 10000.0)

        # 8. Make trading decision
        decision_engine = components["decision_engine"]
        decision = await decision_engine.make_decision(
            symbol=symbol,
            current_price=current_price,
            technical_signal=composite_signal,
            llm_analysis=llm_analysis,
            ml_prediction=ml_prediction,
            market_context=market_context,
            account_balance=account_balance,
        )

        logger.info(
            "trading_decision",
            symbol=symbol,
            action=decision.action,
            confidence=round(decision.confidence, 4),
            vetoed=decision.vetoed,
            veto_reason=decision.veto_reason,
        )

        # 9. Execute trade if not HOLD and not vetoed
        if decision.action != "HOLD" and not decision.vetoed:
            # Convert decision to execution request
            exec_request = convert_decision_to_request(decision)

            # Execute the order
            try:
                order = await executor.execute_decision(exec_request)

                logger.info(
                    "order_executed",
                    symbol=symbol,
                    action=decision.action,
                    order_id=order.id,
                    filled_price=order.filled_price,
                    amount=order.filled_amount,
                    status=order.status,
                )

                # Store trade record if database is available
                if "trade_repo" in components:
                    try:
                        await components["trade_repo"].save_trade(
                            symbol=symbol,
                            side=decision.action.lower(),
                            entry_price=order.filled_price or current_price,
                            amount=order.filled_amount or exec_request.amount,
                            leverage=decision.leverage,
                            stop_loss=decision.stop_loss,
                            take_profit=decision.take_profit,
                            timestamp=datetime.now(UTC),
                        )
                    except Exception as e:
                        logger.warning("trade_save_failed", error=str(e))

            except Exception as e:
                logger.error(
                    "order_execution_failed",
                    symbol=symbol,
                    action=decision.action,
                    error=str(e),
                )

    except Exception as e:
        logger.error(
            "symbol_processing_error",
            symbol=symbol,
            timeframe=timeframe,
            error=str(e),
            exc_info=True,
        )


async def main_loop(components: dict) -> None:
    """
    Main trading loop.

    Args:
        components: Dictionary of initialized components.
    """
    logger = get_logger(__name__)
    settings = get_settings()

    logger.info(
        "main_loop_started",
        symbols=settings.trading.symbols,
        timeframes=settings.trading.timeframes,
        paper_mode=settings.trading.paper_trading,
    )

    # Calculate loop interval based on smallest timeframe
    timeframe_seconds = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }

    # Get minimum timeframe interval
    min_interval = min(timeframe_seconds.get(tf, 3600) for tf in settings.trading.timeframes)
    # Use 80% of interval to ensure we catch the new candle
    loop_interval = max(int(min_interval * 0.8), 30)

    logger.info("loop_interval_set", interval_seconds=loop_interval)

    iteration = 0
    while not shutdown_event.is_set():
        try:
            iteration += 1
            loop_start = datetime.now(UTC)

            logger.debug("main_loop_iteration", iteration=iteration)

            # Process each symbol-timeframe combination
            for symbol in settings.trading.symbols:
                for timeframe in settings.trading.timeframes:
                    if shutdown_event.is_set():
                        break

                    await process_symbol(
                        symbol=symbol,
                        timeframe=timeframe,
                        components=components,
                        logger=logger,
                    )

                    # Small delay between symbols to avoid rate limiting
                    await asyncio.sleep(0.5)

            # Log executor status periodically
            if iteration % 10 == 0 and "executor" in components:
                executor = components["executor"]
                status = executor.get_account_status()
                logger.info(
                    "account_status",
                    balance=status.get("balance"),
                    equity=status.get("equity"),
                    open_positions=status.get("open_positions"),
                    paper_mode=status.get("paper_mode"),
                )

            # Calculate remaining time to sleep
            elapsed = (datetime.now(UTC) - loop_start).total_seconds()
            sleep_time = max(loop_interval - elapsed, 1)

            logger.debug(
                "loop_iteration_complete",
                iteration=iteration,
                elapsed_seconds=round(elapsed, 2),
                sleep_seconds=round(sleep_time, 2),
            )

            # Wait for next iteration or shutdown
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=sleep_time,
                )
            except TimeoutError:
                pass  # Normal timeout, continue loop

        except asyncio.CancelledError:
            logger.info("main_loop_cancelled")
            break
        except Exception as e:
            logger.error("main_loop_error", error=str(e), exc_info=True)
            await asyncio.sleep(5)  # Wait before retry

    logger.info("main_loop_stopped", total_iterations=iteration)


async def async_main() -> int:
    """
    Async main function.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    settings = get_settings()

    # Setup logging
    log_config = LogConfig(
        level=settings.logging.level,
        format=settings.logging.format,
        file_path=settings.logging.file_path,
        include_timestamp=True,
        include_caller_info=True,
    )
    setup_logging(log_config)

    logger = get_logger(__name__)
    logger.info(
        "iftb_starting",
        version="0.1.0",
        environment=settings.logging.level,
        paper_mode=settings.trading.paper_trading,
    )

    # Setup signal handlers
    setup_signal_handlers()

    components = {}
    try:
        # Initialize components
        components = await initialize_components()

        # Start real-time data manager if available
        if "realtime_manager" in components:
            await components["realtime_manager"].start()

        # Run main loop
        await main_loop(components)

        return 0

    except KeyboardInterrupt:
        logger.info("keyboard_interrupt")
        return 0
    except Exception as e:
        logger.critical("fatal_error", error=str(e), exc_info=True)
        return 1
    finally:
        # Cleanup
        await shutdown_components(components)
        logger.info("iftb_stopped")


def main() -> NoReturn:
    """
    Main entry point.

    This function is called when running the bot via the CLI.
    """
    exit_code = asyncio.run(async_main())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
