# LLM Market Analyzer Documentation

## Overview

The LLM Market Analyzer is an intelligent market analysis system powered by Anthropic's Claude AI. It provides sophisticated sentiment analysis, trade veto capabilities, and risk assessment based on news, market context, and technical indicators.

## Features

### Core Capabilities

1. **Sentiment Analysis**
   - 5-level sentiment classification (Very Bearish to Very Bullish)
   - Confidence scoring (0-1 scale)
   - Multi-factor analysis combining news, market data, and technical signals

2. **Veto System**
   - Automatic trade blocking based on negative sentiment
   - Direction conflict detection (e.g., bullish sentiment vs short trade)
   - Low confidence filtering
   - Position size adjustment recommendations

3. **Caching & Performance**
   - Redis-based response caching
   - Automatic cache key generation
   - Configurable TTL (default: 5 minutes)
   - Reduced API costs through intelligent caching

4. **Error Handling**
   - Exponential backoff retry logic
   - Multiple fallback modes
   - Health monitoring and status tracking
   - Graceful degradation

5. **Rate Limiting**
   - 10 requests per minute limit
   - Automatic throttling
   - Request queue management

## Architecture

### Components

```
LLMAnalyzer
├── Sentiment Analysis
├── Veto System
├── Cache Layer (Redis)
├── Rate Limiter
└── Fallback Modes

LLMVetoSystem
├── Sentiment Checks
├── Confidence Checks
├── Direction Conflict Detection
└── Position Size Calculation
```

### Data Flow

```
Input: News + Market Context + Price
    ↓
Cache Check (Redis)
    ↓ (miss)
Rate Limit Check
    ↓
Build Prompt
    ↓
Call Claude API
    ↓
Parse Response
    ↓
Cache Result
    ↓
Return Analysis
```

## Usage

### Basic Usage

```python
from iftb.analysis import LLMAnalyzer, create_analyzer_from_settings
from iftb.data import NewsMessage, MarketContext

# Create analyzer from settings
analyzer = await create_analyzer_from_settings()

# Analyze market
analysis = await analyzer.analyze_market(
    symbol="BTCUSDT",
    news_messages=recent_news,
    market_context=context,
    current_price=50000.0,
)

print(f"Sentiment: {analysis.sentiment}")
print(f"Confidence: {analysis.confidence:.2%}")
print(f"Summary: {analysis.summary}")
```

### Using the Veto System

```python
from iftb.analysis import LLMVetoSystem

veto_system = LLMVetoSystem()

# Check if trade should be blocked
should_block, reason = veto_system.should_veto_trade(
    analysis=analysis,
    trade_direction="long",
)

if should_block:
    print(f"Trade vetoed: {reason}")
else:
    # Calculate position size multiplier
    multiplier = veto_system.calculate_position_size_multiplier(analysis)
    position_size = base_position * multiplier
```

### Without Cache (Simplified)

```python
from iftb.analysis import LLMAnalyzer

analyzer = LLMAnalyzer(
    api_key="your_api_key",
    model="claude-sonnet-4-20250514",
)

analysis = await analyzer.analyze_market(
    symbol="BTCUSDT",
    news_messages=news,
    market_context=context,
    current_price=50000.0,
)
```

## Data Structures

### SentimentScore Enum

```python
class SentimentScore(Enum):
    VERY_BEARISH = -1.0   # Strong negative indicators
    BEARISH = -0.5        # Negative indicators
    NEUTRAL = 0.0         # Mixed/unclear signals
    BULLISH = 0.5         # Positive indicators
    VERY_BULLISH = 1.0    # Strong positive indicators
```

### LLMAnalysis

```python
@dataclass
class LLMAnalysis:
    sentiment: SentimentScore      # Market sentiment
    confidence: float              # 0-1 confidence score
    summary: str                   # Brief analysis summary
    key_factors: list[str]         # Key factors influencing decision
    should_veto: bool              # Whether to block trade
    veto_reason: str | None        # Reason if vetoed
    timestamp: datetime            # Analysis timestamp
    model: str                     # Claude model used
    prompt_tokens: int             # Input tokens
    completion_tokens: int         # Output tokens
    cached: bool                   # Whether from cache
```

## Configuration

### Environment Variables

```bash
# Required
LLM_ANTHROPIC_API_KEY=sk-ant-...

# Optional (defaults shown)
LLM_MODEL=claude-sonnet-4-20250514
LLM_MAX_TOKENS=1000
LLM_CACHE_TTL_SECONDS=300

# Redis (for caching)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
```

### Constants (from config/constants.py)

```python
# Veto thresholds
SENTIMENT_VETO_THRESHOLD = -0.5      # Block if sentiment <= -0.5
CONFIDENCE_VETO_THRESHOLD = 0.3      # Block if confidence < 0.3
SENTIMENT_CAUTION_THRESHOLD = -0.2   # Reduce position if sentiment <= -0.2
CONFIDENCE_CAUTION_THRESHOLD = 0.5   # Reduce position if confidence < 0.5
NEWS_CONFLICT_PENALTY = 0.5          # 50% reduction when news conflicts
```

## Fallback Modes

The analyzer automatically enters fallback mode after repeated API failures:

### 1. CONSERVATIVE Mode (default)
- Neutral sentiment (0.0)
- Low confidence (0.3)
- No veto
- Allows trading with reduced position sizing

### 2. VETO_ALL Mode
- Very bearish sentiment (-1.0)
- High confidence (1.0)
- Blocks all trades
- Activated after 5+ consecutive failures

### 3. CACHE_ONLY Mode
- Uses only cached analyses
- No new API calls
- Returns error if no cache hit
- Must be manually activated

## Rate Limiting

- **Limit**: 10 requests per minute
- **Window**: 60 seconds rolling window
- **Behavior**: Automatic sleep/wait when limit reached
- **Override**: Not recommended (respects Anthropic API limits)

## Caching Strategy

### Cache Key Generation

Cache keys are generated from:
1. Trading symbol
2. News message texts
3. Market context (Fear & Greed, funding rates)
4. Current price

**Note**: Small price movements don't invalidate cache (rounded to significant digits).

### Cache TTL

- **Default**: 300 seconds (5 minutes)
- **Rationale**: Market conditions change slowly, but news requires timely analysis
- **Configurable**: Via `LLM_CACHE_TTL_SECONDS`

### Cache Storage

```
Key Format: llm:market_sentiment:{sha256_hash}
Value: JSON serialized LLMAnalysis
TTL: 300 seconds (default)
```

## Prompt Engineering

The analyzer uses a structured prompt that includes:

1. **Trading Context**
   - Symbol
   - Current price

2. **Recent News** (up to 10 items)
   - Timestamp
   - Urgency flag
   - News text

3. **Market Context**
   - Fear & Greed Index
   - Funding rates
   - Open interest
   - Long/short ratios

4. **Response Format** (JSON)
   - Sentiment classification
   - Confidence score
   - Analysis summary
   - Key factors
   - Veto decision

## Error Handling

### Retry Logic

```python
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0 seconds
MAX_BACKOFF = 30.0 seconds
```

**Backoff Formula**: `min(INITIAL_BACKOFF * 2^attempt, MAX_BACKOFF)`

### Error Tracking

- Consecutive error counter
- Last error timestamp
- Automatic fallback mode activation
- Health status monitoring

## Health Monitoring

```python
health = await analyzer.health_check()

# Returns:
{
    "status": "healthy" | "degraded",
    "fallback_mode": "conservative" | "veto_all" | None,
    "consecutive_errors": 0,
    "last_error_time": "2024-01-17T10:30:00Z" | None,
    "cache_enabled": True | False,
    "rate_limit_usage": 5  # current usage count
}
```

## Best Practices

### 1. Always Use Caching in Production

```python
# Good: Uses cache
analyzer = await create_analyzer_from_settings()

# Avoid: No cache (expensive)
analyzer = LLMAnalyzer(api_key="...")
```

### 2. Check Veto Before Trading

```python
analysis = await analyzer.analyze_market(...)
should_veto, reason = veto_system.should_veto_trade(analysis, trade_direction)

if should_veto:
    logger.warning("trade_vetoed", reason=reason)
    return

# Proceed with trade...
```

### 3. Adjust Position Size Based on Confidence

```python
base_position = calculate_kelly_size(...)
multiplier = veto_system.calculate_position_size_multiplier(analysis)
final_position = base_position * multiplier
```

### 4. Monitor Health Status

```python
health = await analyzer.health_check()
if health["status"] == "degraded":
    logger.warning("llm_analyzer_degraded", mode=health["fallback_mode"])
    # Consider using technical-only signals
```

### 5. Handle Fallback Gracefully

```python
analysis = await analyzer.analyze_market(...)

if analysis.cached:
    logger.info("using_cached_analysis")

if analysis.prompt_tokens == 0:
    # Fallback analysis
    logger.warning("using_fallback_analysis")
    # Reduce confidence in trading decision
```

## Integration Example

```python
async def make_trading_decision(
    symbol: str,
    technical_signal: TechnicalSignal,
    news_messages: list[NewsMessage],
    market_context: MarketContext,
    current_price: float,
) -> TradingDecision:
    """Make trading decision with LLM analysis."""

    # Get LLM analysis
    analyzer = await create_analyzer_from_settings()
    llm_analysis = await analyzer.analyze_market(
        symbol=symbol,
        news_messages=news_messages,
        market_context=market_context,
        current_price=current_price,
    )

    # Check veto
    veto_system = LLMVetoSystem()
    should_veto, veto_reason = veto_system.should_veto_trade(
        analysis=llm_analysis,
        trade_direction=technical_signal.direction,
    )

    if should_veto:
        return TradingDecision(
            action="NO_TRADE",
            reason=veto_reason,
            confidence=0.0,
        )

    # Calculate position size
    base_position = calculate_kelly_position(technical_signal)
    sentiment_multiplier = veto_system.calculate_position_size_multiplier(llm_analysis)

    # Combine technical and sentiment confidence
    combined_confidence = (
        technical_signal.confidence * 0.6 +
        llm_analysis.confidence * 0.4
    )

    return TradingDecision(
        action="TRADE",
        direction=technical_signal.direction,
        position_size=base_position * sentiment_multiplier,
        confidence=combined_confidence,
        sentiment=llm_analysis.sentiment,
        summary=llm_analysis.summary,
    )
```

## Cost Optimization

### Token Usage

- **Typical prompt**: 100-200 tokens
- **Typical response**: 50-100 tokens
- **Cost per analysis**: ~$0.001-0.002 (Claude Sonnet 4)

### Optimization Strategies

1. **Enable Caching**: Reduces API calls by 80-90%
2. **Limit News Items**: Only include 10 most recent/relevant
3. **Batch Analysis**: Group similar requests when possible
4. **Increase Cache TTL**: For slower-moving markets
5. **Use Conservative Fallback**: Reduces API dependency

## Security Considerations

1. **API Key Protection**
   - Store in environment variables
   - Never commit to version control
   - Use secret management in production

2. **Rate Limiting**
   - Respects Anthropic API limits
   - Prevents quota exhaustion
   - Protects against runaway costs

3. **Input Validation**
   - News text length limits
   - Price range validation
   - Timestamp verification

4. **Output Validation**
   - JSON schema enforcement
   - Confidence score validation
   - Sentiment enum validation

## Troubleshooting

### Issue: API Timeout

**Symptoms**: Requests taking >30 seconds, timeout errors

**Solutions**:
1. Check network connectivity
2. Verify API key is valid
3. Check Anthropic service status
4. Increase timeout in client initialization

### Issue: High Cache Miss Rate

**Symptoms**: Most requests call API, high costs

**Solutions**:
1. Increase cache TTL
2. Round price values to reduce key variation
3. Check Redis connectivity
4. Monitor cache storage limits

### Issue: Fallback Mode Stuck

**Symptoms**: Analyzer stays in fallback mode after API recovery

**Solutions**:
1. Restart analyzer instance
2. Check consecutive error counter
3. Manually reset error tracking
4. Verify API credentials

### Issue: Inconsistent Sentiment

**Symptoms**: Similar inputs produce different sentiments

**Solutions**:
1. Check if cache is disabled
2. Verify cache key generation
3. Review news content for variations
4. Monitor Claude model updates

## Performance Metrics

### Expected Performance

- **Cache Hit Rate**: 80-90% in production
- **API Response Time**: 1-3 seconds (without cache)
- **Cache Response Time**: <50ms
- **Rate Limit Usage**: ~10-20% under normal load
- **Error Rate**: <1% (with healthy API connection)

## Future Enhancements

1. **Multi-Model Support**
   - Support for GPT-4, Gemini
   - Model ensemble voting
   - Cost-performance optimization

2. **Advanced Prompt Engineering**
   - Few-shot learning examples
   - Domain-specific fine-tuning
   - Dynamic prompt generation

3. **Enhanced Caching**
   - Semantic similarity cache matching
   - Predictive cache warming
   - Distributed cache for scaling

4. **Risk Scoring**
   - Quantitative risk scores
   - Factor importance weighting
   - Correlation analysis

5. **Backtesting Integration**
   - Historical sentiment analysis
   - Performance attribution
   - Strategy optimization

## References

- [Anthropic Claude API Documentation](https://docs.anthropic.com/)
- [IFTB Risk Management Constants](../src/iftb/config/constants.py)
- [Example Usage](../examples/llm_analyzer_example.py)
- [Unit Tests](../tests/unit/test_llm_analyzer.py)
