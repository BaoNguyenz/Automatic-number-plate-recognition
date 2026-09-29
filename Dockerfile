# ==============================================================================
# ANPR Sentry Tactical v4.2 - Production Dockerfile
# Base: PyTorch with CUDA 12.4 & cuDNN 9 (NVIDIA GPU Accelerated)
# ==============================================================================
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

LABEL maintainer="BaoNguyen <baohuulenguyen@gmail.com>"
LABEL description="Automatic Number Plate Recognition (ANPR) System with TensorRT, ByteTrack, PaddleOCR & Qwen2-VL"

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (OpenCV headless runtime, FFmpeg, Curl, Git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-cache PaddleOCR lightweight models into image (offline-ready)
RUN python3 -c "from paddleocr import PaddleOCR; PaddleOCR(lang='en', use_angle_cls=False)" || true

# Copy project source code
COPY . /app

# Ensure scripts and entrypoint have Unix LF endings and are executable
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Expose FastAPI HTTP Web Dashboard Port
EXPOSE 8000

# Health check to ensure FastAPI is online
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/api/system/status || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
