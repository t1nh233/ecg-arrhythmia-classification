# Dockerfile for ECG Arrhythmia Classification System
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install CPU version of PyTorch first to reduce image size, then install other dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and other files
COPY src/ ./src/
COPY static/ ./static/
COPY checkpoints/ ./checkpoints/
COPY dataraw/raw/ ./dataraw/raw/
COPY main.py .

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
