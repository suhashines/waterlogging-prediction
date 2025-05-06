"""
Configuration settings for the waterlogging prediction system
"""

import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directory
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Models directory
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Model files
WATERLOGGING_MODEL_PATH = os.path.join(MODELS_DIR, 'waterlogging_model.joblib')
RISK_CONFIG_PATH = os.path.join(MODELS_DIR, 'risk_config.joblib')

# Default model parameters
DEFAULT_MODEL_TYPE = 'rf'  # Random Forest
DEFAULT_WINDOW_SIZE = 6
DEFAULT_SEASONALITY = {
    'rainy_months': [6, 7, 8, 9],  # June to September
    'transition_months': [3, 4, 5, 10],  # March to May, October
    'dry_months': [1, 2, 11, 12],  # November to February
    'coefficients': {
        'rainy': 10,      # Strong coefficient for rainy season
        'transition': 6,  # Medium coefficient for transition season
        'dry': 2          # Low coefficient for dry season
    }
}

# Default risk predictor weights
DEFAULT_RISK_WEIGHTS = {
    'amplification_factor': 0.4,
    'elevation': 0.2,
    'impervious_cover': 0.1,
    'drainage': 0.15,
    'slope': 0.1,
    'proximity_to_water': 0.05
}

# Risk levels
RISK_LEVELS = {
    'low': {'min': 0, 'max': 0.3},
    'moderate': {'min': 0.3, 'max': 0.6},
    'high': {'min': 0.6, 'max': 1.0}
}

# Flask API settings
API_HOST = '0.0.0.0'
API_PORT = 5000
API_DEBUG = False  # Set to False in production

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(BASE_DIR, 'app.log')

# Dummy geo-spatial data for stations
# This would be replaced with real data in a production environment
DUMMY_STATION_DATA = {
    '1': {
        'elevation': 33,
        'impervious_cover': 0.9,
        'drainage_area': 145.46,
        'drainage_volume': 7210.38,
        'slope': 0.04,
        'proximity_to_water': 300
    },
    '2': {
        'elevation': 18,
        'impervious_cover': 0.85,
        'drainage_area': 236.93,
        'drainage_volume': 9448.44,
        'slope': 0.06,
        'proximity_to_water': 250
    },
    '3': {
        'elevation': 5,
        'impervious_cover': 0.88,
        'drainage_area': 198.45,
        'drainage_volume': 14863.07,
        'slope': 0.03,
        'proximity_to_water': 180
    },
    '4': {
        'elevation': 11,
        'impervious_cover': 0.92,
        'drainage_area': 335.14,
        'drainage_volume': 11961.56,
        'slope': 0.05,
        'proximity_to_water': 220
    },
    '5': {
        'elevation': 25,
        'impervious_cover': 0.8,
        'drainage_area': 220.5,
        'drainage_volume': 9800.75,
        'slope': 0.045,
        'proximity_to_water': 280
    },
    '6': {
        'elevation': 15,
        'impervious_cover': 0.87,
        'drainage_area': 195.8,
        'drainage_volume': 8500.3,
        'slope': 0.035,
        'proximity_to_water': 320
    },
    '7': {
        'elevation': 28,
        'impervious_cover': 0.83,
        'drainage_area': 275.6,
        'drainage_volume': 10450.2,
        'slope': 0.055,
        'proximity_to_water': 190
    }
}