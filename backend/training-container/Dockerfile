# Training container for Vertex AI
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-13

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY training/ ./training/
COPY scripts/ ./scripts/

# Make scripts executable
RUN chmod +x scripts/*.sh

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Entry point
ENTRYPOINT ["python", "training/main.py"]