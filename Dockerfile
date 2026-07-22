# ==============================================================================
# AI Stock Advisor - Multi-stage Production Dockerfile
# ==============================================================================

# --- Stage 1: Build & Dependency compilation ---
FROM python:3.12-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build dependencies for C-compiled modules (like scikit-learn, lightgbm if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements and build packages wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Stage 2: Final Slim Release image ---
FROM python:3.12-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app:/app/src

# Install system runtime libraries (LightGBM requires libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application source directories
COPY . .

# Expose API port (8000) and Streamlit port (8501)
EXPOSE 8000
EXPOSE 8501

# Default runtime startup launch wrapper (Can be overridden in docker-compose)
CMD ["python", "src/ai_stock_advisor/api/main.py"]
