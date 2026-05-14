# Stage 1: Build & Dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim AS runner

WORKDIR /app

# Install runtime libs for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from builder
COPY --from=builder /root/.local /root/.local
COPY . .

# Ensure scripts in .local/bin are usable and modules in /app are discoverable
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Metadata
LABEL maintainer="Google Senior SWE & Lead AI Architect"
LABEL project="conformal-cashflow"
LABEL version="1.0"

CMD ["python", "src/main.py"]
