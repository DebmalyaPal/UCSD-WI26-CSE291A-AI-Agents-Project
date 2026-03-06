# Dockerfile for CREW agents
# Added by UCSD-WI26-CSE291A-AI-Agents-Project team
# Original CREW code licensed under Apache 2.0

FROM python:3.11-slim

WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libxcb1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy your project files
COPY . /app

RUN pip install --upgrade pip && \
    pip install -r crew_algorithms/wildfire_alg/req.txt

# Set environment
ENV PYTHONPATH=/app

# Default command
CMD ["python", "crew_algorithms/wildfire_alg/algorithms/CAMON/__main__.py"]
