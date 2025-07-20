# 🚀 VORTA ULTRA Dashboard - ULTIMATE SPEED MULTI-STAGE BUILD
# Stage 1: Build dependencies (CACHED)
FROM python:3.12.9-slim as builder

# Set build working directory
WORKDIR /build

# Install build dependencies in one optimized layer
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    g++ \
    gcc \
    make \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create virtual environment for isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CRITICAL CACHE OPTIMIZATION: Copy pip config and requirements FIRST
COPY pip.conf /etc/pip.conf
COPY requirements.txt .

# SEPARATE CACHED LAYER: Upgrade pip tools (rarely changes)
RUN pip install --upgrade pip setuptools wheel

# SEPARATE CACHED LAYER: Install dependencies (PERSISTENT PIP CACHE)
# This layer will be cached until requirements.txt changes
# BuildKit cache mount prevents re-downloading PyTorch (821MB) and other large packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --only-binary=all \
    --prefer-binary \
    --no-compile \
    --disable-pip-version-check \
    --timeout=300 \
    -r requirements.txt

# Stage 2: Runtime (MINIMAL)
FROM python:3.12.9-slim as runtime

# Set runtime working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only runtime essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r vorta && useradd -r -g vorta -d /app -s /bin/bash vorta

# Copy application code with proper ownership
COPY --chown=vorta:vorta . .

# Create logs directory
RUN mkdir -p logs && chown -R vorta:vorta logs

# Switch to non-root user
USER vorta

# Expose Streamlit port
EXPOSE 8501

# Health check for enterprise deployment
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Ultra professional Streamlit startup
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
