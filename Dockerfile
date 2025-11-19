# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory for Redis
RUN mkdir -p data

# Make entrypoint script executable
RUN chmod +x docker-entrypoint.sh

# Expose ports
EXPOSE 6379 8000

# Set entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]