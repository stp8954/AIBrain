# NyRAG Docker Image
# Multi-stage build for smaller final image

FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
WORKDIR /build
COPY pyproject.toml ./
COPY src/ ./src/

# Install the package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# ============================================
# Production stage
# ============================================
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy application source
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/

# Create data directory for SQLite, uploads, and output
RUN mkdir -p /app/data /app/data/output /app/data/uploads && \
    chown -R appuser:appuser /app/data

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV VESPA_URL=http://vespa:8080
ENV VESPA_CONFIG_URL=http://vespa:19071
ENV NYRAG_DATA_DIR=/app/data
ENV NYRAG_AUTO_DEPLOY=true

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Run the application
# Uses config from mounted volume or default
CMD ["python", "-m", "uvicorn", "nyrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
