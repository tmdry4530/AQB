# LLM Analyzer Quick Reference

## Quick Start

```python
from iftb.analysis import create_analyzer_from_settings, LLMVetoSystem

# Initialize
analyzer = await create_analyzer_from_settings()
veto_system = LLMVetoSystem()

# Analyze
analysis = await analyzer.analyze_market(
    symbol="BTCUSDT",
    news_messages=recent_news,
    market_context=context,
    current_price=50000.0,
)

# Check veto
should_block, reason = veto_system.should_veto_trade(analysis, "long")

# Position sizing
multiplier = veto_system.calculate_position_size_multiplier(analysis)
```

## Sentiment Scores

| Score | Value | Meaning |
|-------|-------|---------|
| VERY_BEARISH | -1.0 | Strong sell signals |
| BEARISH | -0.5 | Negative indicators |
| NEUTRAL | 0.0 | Mixed/unclear |
| BULLISH | 0.5 | Positive indicators |
| VERY_BULLISH | 1.0 | Strong buy signals |

## Confidence Levels

| Range | Interpretation |
|-------|---------------|
| 0.0-0.3 | Low confidence - insufficient data |
| 0.4-0.6 | Moderate - some clear signals |
| 0.7-0.9 | High - strong aligned signals |
| 0.9-1.0 | Very high - overwhelming evidence |

## Veto Thresholds

| Threshold | Value | Action |
|-----------|-------|--------|
| SENTIMENT_VETO | -0.5 | Block trade if sentiment ≤ -0.5 |
| CONFIDENCE_VETO | 0.3 | Block trade if confidence < 0.3 |
| SENTIMENT_CAUTION | -0.2 | Reduce position if sentiment ≤ -0.2 |
| CONFIDENCE_CAUTION | 0.5 | Reduce position if confidence < 0.5 |

## Fallback Modes

| Mode | Sentiment | Confidence | Veto | Use Case |
|------|-----------|------------|------|----------|
| CONSERVATIVE | NEUTRAL | 0.3 | No | Default fallback |
| VETO_ALL | VERY_BEARISH | 1.0 | Yes | After 5+ API errors |
| CACHE_ONLY | N/A | N/A | N/A | Manual activation only |

## Rate Limiting

- **Limit**: 10 requests/minute
- **Window**: 60 seconds rolling
- **Action**: Automatic sleep when exceeded

## Caching

- **Default TTL**: 300 seconds (5 minutes)
- **Key Format**: `llm:market_sentiment:{hash}`
- **Storage**: Redis
- **Hit Rate**: 80-90% expected

## Health Check

```python
health = await analyzer.health_check()
```

Returns:
```json
{
    "status": "healthy",
    "fallback_mode": null,
    "consecutive_errors": 0,
    "last_error_time": null,
    "cache_enabled": true,
    "rate_limit_usage": 5
}
```

## Common Patterns

### Pattern 1: Basic Trade Decision

```python
analysis = await analyzer.analyze_market(...)

if analysis.should_veto:
    return "NO_TRADE"

should_block, _ = veto_system.should_veto_trade(analysis, direction)
if should_block:
    return "NO_TRADE"

return "TRADE"
```

### Pattern 2: Position Sizing

```python
base_size = calculate_kelly_size(signal)
multiplier = veto_system.calculate_position_size_multiplier(analysis)
final_size = base_size * multiplier
```

### Pattern 3: Confidence Combination

```python
technical_weight = 0.6
sentiment_weight = 0.4

combined_confidence = (
    technical_signal.confidence * technical_weight +
    analysis.confidence * sentiment_weight
)
```

### Pattern 4: Health Monitoring

```python
health = await analyzer.health_check()

if health["status"] == "degraded":
    # Fall back to technical-only signals
    use_technical_only_mode()
```

## Error Handling

```python
try:
    analysis = await analyzer.analyze_market(...)
except Exception as e:
    logger.error("llm_analysis_failed", error=str(e))
    # Use fallback or technical-only mode
    analysis = create_fallback_analysis()
```

## Configuration

### Minimal (.env)

```bash
LLM_ANTHROPIC_API_KEY=sk-ant-...
```

### Full Configuration

```bash
# Required
LLM_ANTHROPIC_API_KEY=sk-ant-...

# Optional
LLM_MODEL=claude-sonnet-4-20250514
LLM_MAX_TOKENS=1000
LLM_CACHE_TTL_SECONDS=300

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
```

## Cost Estimation

| Volume | API Calls/Day | Cost/Day (USD) |
|--------|---------------|----------------|
| Low | 100 | $0.10-0.20 |
| Medium | 500 | $0.50-1.00 |
| High | 2000 | $2.00-4.00 |

**Note**: With 80% cache hit rate, costs reduced by 5x.

## Key Imports

```python
from iftb.analysis import (
    LLMAnalyzer,
    LLMVetoSystem,
    LLMAnalysis,
    SentimentScore,
    FallbackMode,
    create_analyzer_from_settings,
)

from iftb.data import (
    NewsMessage,
    MarketContext,
)
```

## Testing

```python
# Run tests
pytest tests/unit/test_llm_analyzer.py -v

# Check syntax
python -m py_compile src/iftb/analysis/llm_analyzer.py

# Run example
python examples/llm_analyzer_example.py
```

## Troubleshooting

| Issue | Quick Fix |
|-------|-----------|
| API timeout | Check network, verify API key |
| High costs | Enable caching, increase TTL |
| Cache misses | Check Redis connection |
| Stuck fallback | Restart analyzer instance |
| Rate limit | Wait 60 seconds or reduce frequency |

## Best Practices Checklist

- [ ] Always use `create_analyzer_from_settings()` for cache
- [ ] Check veto before executing trades
- [ ] Adjust position size based on confidence
- [ ] Monitor health status regularly
- [ ] Handle fallback mode gracefully
- [ ] Log analysis results for debugging
- [ ] Use cached results when available
- [ ] Protect API keys in environment variables

## Example Decision Flow

```
Input: News + Market Data + Price
    ↓
[Analyze Market]
    ↓
Check should_veto from analysis
    ↓ Yes → Block Trade
    ↓ No
Check veto_system.should_veto_trade()
    ↓ Yes → Block Trade
    ↓ No
Calculate position_size_multiplier
    ↓
Adjust position size
    ↓
Execute trade
```

## Support

- **Documentation**: `/docs/LLM_ANALYZER.md`
- **Examples**: `/examples/llm_analyzer_example.py`
- **Tests**: `/tests/unit/test_llm_analyzer.py`
- **Source**: `/src/iftb/analysis/llm_analyzer.py`
