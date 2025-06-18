FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Expose port and run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
