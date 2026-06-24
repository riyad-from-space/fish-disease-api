FROM python:3.11-slim

WORKDIR /app

# Memory optimization environment variables
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV CUDA_VISIBLE_DEVICES=-1
ENV TF_NUM_INTEROP_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1
ENV MALLOC_TRIM_THRESHOLD_=65536
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code (atf_layers.py defines the custom layers the model
# needs to load — the app will NOT start without it)
COPY main.py .
COPY atf_layers.py .
COPY static/ static/

# Copy the inference model (optimizer state stripped, ~146MB)
COPY ATF_Net_Fusion_Model_inference.h5* ./

# Expose port
EXPOSE 8000

# Start server with single worker for memory efficiency
CMD ["python", "main.py"]
