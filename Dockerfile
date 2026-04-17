# ============================================================
# Dockerfile for E-Commerce Churn AI Platform
# ============================================================
# This packages the entire application into a container that
# runs the same way on any machine (Mac, Linux, Windows, cloud).
#
# HOW DOCKER WORKS (simplified):
# 1. Start from a base image (Python 3.11)
# 2. Copy our code into the container
# 3. Install dependencies
# 4. Define the command to run
#
# BUILD:   docker build -t churn-ai .
# RUN:     docker run -p 8501:8501 -p 8000:8000 --env-file .env churn-ai
# ============================================================

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caches this layer if requirements don't change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Generate data and train model
RUN python data/generate_data.py && \
    python src/data_pipeline.py && \
    python src/feature_engineering.py && \
    python src/model_training.py

# Expose ports: 8501 for Streamlit, 8000 for FastAPI
EXPOSE 8501 8000

# Run both services using a simple shell script
COPY start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]
