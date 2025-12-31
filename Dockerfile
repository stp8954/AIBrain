# NyRAG Docker Image
# Multi-stage build for smaller final image

FROM python:3.11-slim as builder

# Install build dependencies
# Note: libgl1-mesa-dev and poppler are needed for Docling PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1-mesa-dev \
    libglib2.0-0 \
    poppler-utils \
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

# Pre-download the embedding model for faster first-run
# This caches the model in the Docker image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# ============================================
# Production stage
# ============================================
FROM python:3.11-slim

# Install runtime dependencies
# Note: libgl1, libglib2.0-0, and poppler-utils are needed for Docling PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy pre-downloaded HuggingFace models from builder
COPY --from=builder --chown=appuser:appuser /root/.cache/huggingface /home/appuser/.cache/huggingface

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
ENV HF_HOME=/home/appuser/.cache/huggingface

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Run the application
# Uses config from mounted volume or default
CMD ["python", "-m", "uvicorn", "nyrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
