FROM python:3.10-slim

ARG TARGETARCH

# Install system dependencies with cache mount
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root user
RUN useradd -m appuser

# Copy requirements first
COPY requirements.txt .

# Optimized Torch installation with pip cache mount:
# 1. Using cache mount to persist downloads across builds
# 2. Prefer CPU-only index for amd64
# 3. For arm64, we use default PyPI (which contains CPU-compatible wheels)
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$TARGETARCH" = "amd64" ]; then \
        pip install --timeout 1000 torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --timeout 1000 torch torchvision; \
    fi

# Install other dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 1000 --upgrade -r requirements.txt

# Copy weights separately first (they change less often than code)
COPY --chown=appuser weights/ ./weights/

# Copy the rest of application files
COPY --chown=appuser . .

# Switch to non-root user
USER appuser
ENV PATH="/home/appuser/.local/bin:$PATH"

ENV PORT=8080
EXPOSE $PORT

CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port $PORT"]
