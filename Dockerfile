# Use Python 3.9 as base image
FROM python:3.9-alpine

# Set working directory
WORKDIR /app

# Install required build dependencies
RUN apk add --no-cache build-base gcc g++ musl-dev python3-dev libffi-dev openssl-dev

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory if it doesn't exist
RUN mkdir -p data models

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=api/app.py
ENV FLASK_ENV=production

# Start the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api.app:app"]
