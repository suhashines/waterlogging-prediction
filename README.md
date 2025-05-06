# Waterlogging Prediction and Risk Analysis System

This system predicts waterlogging depth and analyzes risk based on rainfall data and geographical features. It follows the methodology outlined in the research paper and implements a Flask API for predictions and feedback.

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

5. **API Endpoints**:
   - `/model/predict/`: Predict waterlogging depth and risk
   - `/model/feedback`: Provide feedback for model improvement
   - `/model/weights`: Get or update risk factor weights
   - `/model/station-data`: Get or update station geographical data

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

## API Documentation

### 1. Predict Waterlogging Depth and Risk

**Endpoint**: `POST /model/predict/`

**Request Body**:
```json
{
    "station_code": "1",
    "rainfall": 10.5,
    "timestamp": "2023-05-01T12:30:00",
    "weather": 2,  
    "windspeed": 5.2  
}
```

**Response**:
```json
{
    "prediction": {
        "waterlogging_depth": 0.25,
        "risk_factor": {
            "risk_score": 0.45,
            "risk_level": "moderate",
            "amplification_factor": 0.23,
            "factors": {
                "elevation": 0.3,
                "impervious_cover": 0.8,
                "drainage": 0.4,
                "slope": 0.2,
                "proximity_to_water": 0.5
            }
        }
    },
    "message": "Prediction successful",
    "status": "success"
}
```

### 2. Provide Feedback

**Endpoint**: `POST /model/feedback`

**Request Body**:
```json
{
    "station_code": "1",
    "rainfall": 10.5,
    "timestamp": "2023-05-01T12:30:00",
    "weather": 2,
    "windspeed": 5.2,
    "actual_waterdepth": 0.27
}
```

**Response**:
```json
{
    "message": "Feedback processed successfully",
    "status": "success",
    "details": {
        "previous_prediction": 0.25,
        "actual_value": 0.27,
        "error": 0.02
    }
}
```

### 3. Get or Update Risk Weights

**GET /model/weights**

**Response**:
```json
{
    "weights": {
        "amplification_factor": 0.4,
        "elevation": 0.2,
        "impervious_cover": 0.1,
        "drainage": 0.15,
        "slope": 0.1,
        "proximity_to_water": 0.05
    },
    "message": "Current weights retrieved successfully",
    "status": "success"
}
```

**POST /model/weights**

**Request Body**:
```json
{
    "weights": {
        "amplification_factor": 0.5,
        "elevation": 0.2,
        "impervious_cover": 0.1,
        "drainage": 0.1,
        "slope": 0.05,
        "proximity_to_water": 0.05
    }
}
```

**Response**: Same as GET response but with updated weights.

### 4. Get or Update Station Data

**GET /model/station-data?station_id=1**

**Response**:
```json
{
    "station_data": {
        "amplification_factor": 0.23,
        "elevation": 25.5,
        "impervious_cover": 0.85,
        "drainage_area": 200,
        "drainage_volume": 10000,
        "slope": 0.05,
        "proximity_to_water": 300
    },
    "message": "Station data retrieved successfully",
    "status": "success"
}
```

**POST /model/station-data**

**Request Body**:
```json
{
    "station_id": "1",
    "data": {
        "elevation": 25.5,
        "impervious_cover": 0.85,
        "drainage_area": 200,
        "drainage_volume": 10000,
        "slope": 0.05,
        "proximity_to_water": 300
    }
}
```

**Response**: Same as GET response but with updated station data.

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