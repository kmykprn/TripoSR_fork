FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Install system packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file first (Docker cache optimization)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY tsr/ /app/tsr/
COPY tsr_pipeline/ /app/tsr_pipeline/
COPY handler.py /app/handler.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV MODEL_NAME="stabilityai/TripoSR"
ENV MODEL_CACHE_DIR="/runpod-volume/models"

# Run handler for Runpod
CMD ["python", "-u", "/app/handler.py"]