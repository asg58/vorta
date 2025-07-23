# VORTA Ultra High-Grade Dashboard - Docker Configuration
FROM python:3.11.7-slim

# Set working directory
WORKDIR /app

# Install system dependencies (sorted alphabetically)
RUN apt-get update && apt-get install -y \
    curl \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash vorta && \
    chown -R vorta:vorta /app
USER vorta

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start command - Use the latest Ultra AGI integrated dashboard server
CMD ["python", "-m", "uvicorn", "ultra_dashboard_server:app", "--host", "0.0.0.0", "--port", "8080"]
