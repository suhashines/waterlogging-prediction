#!/bin/bash

# Waterlogging Prediction System Setup Script
# This script sets up the environment, trains the model, and starts the API

# Exit on any error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print colored message
function print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

function print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

function print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d "." -f 1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d "." -f 2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python version $PYTHON_VERSION is not supported. Please install Python 3.8 or higher."
    exit 1
fi

print_message "Using Python $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    print_message "Creating virtual environment..."
    python3 -m venv venv
else
    print_warning "Virtual environment already exists. Skipping creation."
fi

# Activate virtual environment
print_message "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
print_message "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create directories if they don't exist
print_message "Setting up directories..."
mkdir -p data
mkdir -p models

# Check if any CSV files exist in the data directory
CSV_COUNT=$(find data -name "*.csv" | wc -l)
if [ "$CSV_COUNT" -eq 0 ]; then
    print_warning "No CSV files found in the data directory. Please add your data files to the 'data' directory."
    exit 1
fi

# Train the model
print_message "Training the model..."
python train.py --data-dir data --model-type rf

# Check if model files were created
if [ ! -f "models/waterlogging_model.joblib" ] || [ ! -f "models/risk_config.joblib" ]; then
    print_error "Model training failed. Model files not found."
    exit 1
fi

print_message "Model trained successfully!"

# Start the API (in the background)
print_message "Starting the API server..."
cd api
python app.py &
API_PID=$!

# Wait for the API to start
sleep 3

# Check if the API is running
if curl -s http://localhost:5000/model/weights > /dev/null; then
    print_message "API server is running at http://localhost:5000/"
    print_message "Press Ctrl+C to stop the server."
    
    # Wait for user to press Ctrl+C
    trap "kill $API_PID; print_message 'API server stopped.'; exit 0" INT
    wait $API_PID
else
    print_error "API server failed to start."
    kill $API_PID 2>/dev/null || true
    exit 1
fi