# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7861
ENV HF_HOME=/tmp/huggingface_cache

# Set working directory in container
WORKDIR /app

# Install system build dependencies for potential compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the SentenceTransformer model to speed up container startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the application code
COPY . /app/

# Expose port (default fallback, but PaaS overrides this)
EXPOSE 7861

# Start the application
CMD ["python", "app.py"]
