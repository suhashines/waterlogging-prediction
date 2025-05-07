# Waterlogging Prediction and Risk Analysis System

This system predicts waterlogging depth and analyzes risk based on rainfall data and geographical features.

## Project Structure

```
waterlogging_prediction/
│
├── data/                   # CSV data files for stations
│   ├── 1.csv
│   ├── 2.csv
│   └── ...
│
├── models/                 # Model implementation and saved models
│   ├── __init__.py
│   ├── data_processor.py
│   ├── feature_engineer.py
│   ├── waterlogging_predictor.py
│   ├── risk_predictor.py
│   ├── waterlogging_model.joblib  # Saved model (after training)
│   └── risk_config.joblib         # Saved risk config (after training)
│
├── utils/                  # Utility functions
│   └── __init__.py
│
├── api/                    # Flask API implementation
│   ├── __init__.py
│   └── app.py
    └── ...
│
├── config.py               # Configuration settings
├── train.py                # Training script
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Features

1. **Data Processing**: 
   - Handles missing values
   - Interpolates time series data
   - Creates sliding windows for time series modeling

2. **Feature Engineering**:
   - Extracts statistical features from rainfall time series
   - Constructs domain-specific features (seasonality, rainfall interval, wetting coefficient, etc.)
   - Incorporates weather and wind speed data when available

3. **Machine Learning Models**:
   - Random Forest (default)
   - Gradient Boosting Decision Trees
   - AdaBoost

4. **Risk Analysis**:
   - Calculates amplification factors
   - Incorporates geographical features
   - Produces risk scores and levels (low, moderate, high)



## Setup Instructions

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/waterlogging_prediction.git
cd waterlogging_prediction
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

### Data Preparation

Place your CSV data files in the `data/` directory. Each file should correspond to one station, with the filename format `<station_id>.csv`.

Each CSV should contain at least the following columns:
- `clctTime`: Timestamp
- `Rainfall(mm)`: Rainfall in millimeters
- `Waterdepth(meters)`: Water depth in meters

Optional columns:
- `Weather`: Weather condition code
- `Wind Speed`: Wind speed

### Model Training

To train the model using the default settings:

```bash
python train.py --data-dir data --model-type rf --window-size 6 --save-dir models
```

Options:
- `--data-dir`: Directory containing CSV files (default: 'data')
- `--model-type`: Type of model ('rf', 'gbdt', 'adaboost') (default: 'rf')
- `--window-size`: Size of sliding window (default: 6)
- `--tune-hyperparams`: Flag to tune hyperparameters
- `--save-dir`: Directory to save models (default: 'models')

### Running the API

To start the Flask API:

```bash
cd api
python app.py
```

The API will be available at `http://localhost:5000/`.

## API Endpoints

- [Find the api doc here](api-doc.md)

## Implementation Details

The system follows the approach from the research paper with these key components:

1. **Data Preprocessing**:
   - Interpolation for missing timestamps
   - Handling missing values
   - Time series slicing

2. **Feature Extraction**:
   - Unit rainfall calculation
   - Seasonality coefficients
   - Rainfall interval features
   - Statistical features (mean, max, std, kurtosis, skewness, AUC)

3. **Risk Analysis**:
   - Amplification factor (AF) = waterlogging depth / rainfall
   - Risk formula: r = w1*AF + w2*(1-elevation_normalized) + w3*impervious_cover + w4*(1-drainage_normalized) + w5*(1-slope_normalized) + w6*proximity_norm
   - Risk levels: low (<0.3), moderate (0.3-0.6), high (>0.6)

4. **Machine Learning Pipeline**:
   - Data processing
   - Feature engineering
   - Model training and prediction

5. **Feedback Loop**:
   - Continuous model improvement with new data
   - Updating amplification factors

## Requirements

The following packages are required:

- Flask==2.0.1
- pandas==1.3.3
- numpy==1.20.3
- scikit-learn==0.24.2
- joblib==1.0.1

To install all requirements:

```bash
pip install -r requirements.txt
```