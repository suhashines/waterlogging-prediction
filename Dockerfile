# Use Python 3.8 as base image
FROM python:3.9-alpine

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

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