# Stage 1: Build & Dependencies
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim as runner

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /root/.local /root/.local
COPY . .

# Ensure scripts in .local/bin are usable
ENV PATH=/root/.local/bin:$PATH

# Metadata
LABEL project="conformal-cashflow"
LABEL version="1.0"

CMD ["python", "src/main.py"]
