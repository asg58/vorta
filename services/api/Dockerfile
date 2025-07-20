# 🚀 VORTA ULTRA API - Enterprise Multi-Stage Build
# ULTRA HIGH GRADE - Revolutionary Computing Infrastructure

# Stage 1: Build Dependencies
FROM python:3.12.9-slim AS builder

# Ultra professional metadata
LABEL maintainer="VORTA ULTRA Team"
LABEL version="2.0.0-ultra"
LABEL description="Enterprise-grade AI API with revolutionary performance"

# Enterprise build arguments
ARG BUILD_ENV=production
ARG OPTIMIZATION_LEVEL=ultra

# Set enterprise environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for ultra performance
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    g++ \
    git \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create ultra-optimized virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Ultra professional working directory
WORKDIR /build

# ULTRA OPTIMIZATION: Copy requirements FIRST for Docker layer caching
# This ensures downloads are cached even if source code changes
COPY services/api/requirements.txt .

# Install pip upgrades in separate cached layer
RUN pip install --upgrade pip setuptools wheel

# 🚀 ULTRA CACHE OPTIMIZATION: Install ALL dependencies with PERSISTENT CACHE MOUNT
# This prevents re-downloading ALL large packages (>5MB):
# - PyTorch (821MB) 
# - Transformers (~400MB)
# - NumPy/SciPy (~100MB each)
# - Pandas (~50MB)
# - Librosa (~30MB)
# - OpenAI Whisper (~20MB)
# - Scikit-learn (~40MB)
# - And many more large ML packages!
# BuildKit will cache ALL downloads between builds for ULTRA speed!
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Stage 2: Runtime Environment - ULTRA LIGHTWEIGHT
FROM python:3.12.9-slim AS runtime

# Copy optimized virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Enterprise runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create enterprise user (security best practice)
RUN groupadd -r vorta && useradd -r -g vorta -d /app -s /bin/bash vorta

# Ultra professional working directory
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=vorta:vorta services/api/*.py ./

# Create logs directory with proper permissions
RUN mkdir -p logs && chown -R vorta:vorta logs

# Switch to non-root user for security
USER vorta

# Expose API port
EXPOSE 8000

# ULTRA FAST health check - optimized for rapid startup
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=2 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Ultra professional startup with enterprise-grade process management
CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--access-log", \
     "--log-level", "info", \
     "--loop", "asyncio"]
