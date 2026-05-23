# Use an official Python runtime as a parent image
FROM python:3.11

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7861
ENV HF_HOME=/tmp/huggingface_cache

# Install system build dependencies for potential compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user with UID 1000 (standard for Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Set up working directory in home directory
WORKDIR /home/user/app

# Install python dependencies to user directory
COPY --chown=user requirements.txt /home/user/app/

# Switch to the non-root user
USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download the SentenceTransformer model to speed up container startup
# Pass the HF_HOME cache path explicitly to make sure it is writeable
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/tmp/huggingface_cache')"

# Copy the rest of the application code with user ownership
COPY --chown=user . /home/user/app

# Expose port (default fallback, but PaaS overrides this)
EXPOSE 7861

# Start the application
CMD ["python", "app.py"]
