FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for some Python ML packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Hugging Face Spaces require port 7860
EXPOSE 7860

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
