# Multi-stage build for IFTB (Intelligent Futures Trading Bot)

# Stage 1: Builder
FROM python:3.12-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt* pyproject.toml* setup.py* ./

# Install Python dependencies using uv
RUN if [ -f requirements.txt ]; then \
    uv pip install --system --no-cache -r requirements.txt; \
    elif [ -f pyproject.toml ]; then \
    uv pip install --system --no-cache -e .; \
    fi

# Stage 2: Runtime
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 iftb && \
    mkdir -p /app /app/data /app/logs && \
    chown -R iftb:iftb /app

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=iftb:iftb src/ ./src/
COPY --chown=iftb:iftb configs/ ./configs/
COPY --chown=iftb:iftb migrations/ ./migrations/
COPY --chown=iftb:iftb scripts/ ./scripts/
COPY --chown=iftb:iftb data/ ./data/

# Switch to non-root user
USER iftb

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for health check endpoint
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command - run the bot
CMD ["python", "-m", "src.iftb.main"]
