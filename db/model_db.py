# Model data module
from . import db
import math
import random
from datetime import datetime

# Define namespaces
STATION_DATA = 'station_data'
RISK_WEIGHTS = 'risk_weights'

# Initialize station data
def init_station_data():
    return {
        '1': {
            'amplification_factor': 0.23,
            'elevation': 25.5,
            'impervious_cover': 0.85,
            'drainage_area': 200,
            'drainage_volume': 10000,
            'slope': 0.05,
            'proximity_to_water': 300
        },
        '2': {
            'amplification_factor': 0.19,
            'elevation': 18.3,
            'impervious_cover': 0.78,
            'drainage_area': 180,
            'drainage_volume': 8500,
            'slope': 0.04,
            'proximity_to_water': 450
        },
        '3': {
            'amplification_factor': 0.27,
            'elevation': 12.1,
            'impervious_cover': 0.92,
            'drainage_area': 250,
            'drainage_volume': 12000,
            'slope': 0.02,
            'proximity_to_water': 200
        }
    }

# Initialize risk weights
def init_risk_weights():
    return {
        'amplification_factor': 0.4,
        'elevation': 0.2,
        'impervious_cover': 0.1,
        'drainage': 0.15,
        'slope': 0.1,
        'proximity_to_water': 0.05
    }

# Register with the database
db.register(STATION_DATA, init_station_data)
db.register(RISK_WEIGHTS, init_risk_weights)

# Initialize data
db.initialize(STATION_DATA, init_station_data)
db.initialize(RISK_WEIGHTS, init_risk_weights)

# Model functions
def get_station_data(station_id=None):
    """Get station data for a specific station or all stations"""
    if station_id:
        return db.get(STATION_DATA, station_id)
    return db.get(STATION_DATA)

def update_station_data(station_id, data):
    """Update station data for a specific station"""
    if db.get(STATION_DATA, station_id):
        return db.update(STATION_DATA, station_id, data)
    return db.set(STATION_DATA, station_id, data)

def get_risk_weights():
    """Get current risk weights"""
    return db.get(RISK_WEIGHTS)

def update_risk_weights(weights):
    """Update risk weights"""
    current_weights = db.get(RISK_WEIGHTS)
    current_weights.update(weights)
    
    # Normalize weights to sum to 1
    weight_sum = sum(current_weights.values())
    normalized_weights = {k: v / weight_sum for k, v in current_weights.items()}
    
    for k, v in normalized_weights.items():
        db.update(RISK_WEIGHTS, k, v)
    
    return normalized_weights

def predict_waterlogging(station_code, rainfall, timestamp=None, weather=None, windspeed=None):
    """Predict waterlogging based on input parameters"""
    station_data = get_station_data(station_code)
    if not station_data:
        station_data = get_station_data('1')  # Default to station 1
    
    # More rainfall and higher amplification factor leads to deeper waterlogging
    waterlogging_depth = rainfall * station_data['amplification_factor'] / 1000.0
    
    # Add some realistic variation based on other factors
    elevation_factor = 1.0 - (station_data['elevation'] / 50.0)  # Lower elevation = more water
    impervious_factor = station_data['impervious_cover']  # More impervious = more water
    slope_factor = 1.0 - (station_data['slope'] * 10)  # Lower slope = more water
    
    # Apply factors
    waterlogging_depth = waterlogging_depth * (1 + elevation_factor) * (1 + impervious_factor) * (1 + slope_factor)
    
    # Apply weather factor if provided
    if weather is not None:
        weather_factor = 1.0 + (weather / 10.0)
        waterlogging_depth *= weather_factor
        
    # Apply windspeed factor if provided
    if windspeed is not None:
        wind_factor = 1.0 - (min(windspeed, 50) / 100.0)  # Higher wind can reduce pooling
        waterlogging_depth *= wind_factor
    
    # Set a minimum depth for realism
    waterlogging_depth = max(0.01, waterlogging_depth)
    
    # Round to realistic precision
    waterlogging_depth = round(waterlogging_depth, 2)
    
    # Calculate risk
    risk_weights = get_risk_weights()
    risk_score = (
        risk_weights['amplification_factor'] * station_data['amplification_factor'] +
        risk_weights['elevation'] * (1 - station_data['elevation'] / 50.0) +
        risk_weights['impervious_cover'] * station_data['impervious_cover'] +
        risk_weights['drainage'] * (1 - station_data['drainage_area'] / 300.0) +
        risk_weights['slope'] * (1 - station_data['slope'] * 10) +
        risk_weights['proximity_to_water'] * (1 - station_data['proximity_to_water'] / 500.0)
    )
    
    # Ensure score is between 0 and 1
    risk_score = max(0, min(risk_score, 1))
    risk_score = round(risk_score, 2)
    
    # Determine risk level
    if risk_score < 0.3:
        risk_level = 'low'
    elif risk_score < 0.6:
        risk_level = 'moderate'
    else:
        risk_level = 'high'
        
    return {
        'waterlogging_depth': waterlogging_depth,
        'risk_factor': {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'amplification_factor': station_data['amplification_factor'],
            'factors': {
                'elevation': round(1 - station_data['elevation'] / 50.0, 2),
                'impervious_cover': station_data['impervious_cover'],
                'drainage': round(1 - station_data['drainage_area'] / 300.0, 2),
                'slope': round(1 - station_data['slope'] * 10, 2),
                'proximity_to_water': round(1 - station_data['proximity_to_water'] / 500.0, 2)
            }
        }
    }

def process_feedback(station_code, rainfall, actual_waterdepth, timestamp=None, weather=None, windspeed=None):
    """Process feedback to improve the model"""
    # Get the predicted depth
    prediction = predict_waterlogging(station_code, rainfall, timestamp, weather, windspeed)
    waterlogging_depth = prediction['waterlogging_depth']
    
    # Calculate error
    error = abs(waterlogging_depth - actual_waterdepth)
    
    # In a real implementation, we'd update the model
    # For our mock implementation, we'll just simulate it by slightly
    # adjusting the amplification factor for the station
    station_data = get_station_data(station_code)
    if station_data:
        # Calculate a new amplification factor based on the actual depth
        new_af = actual_waterdepth / (rainfall / 1000.0)
        
        # Use an exponential moving average to update the factor
        alpha = 0.3  # Smoothing factor
        updated_af = alpha * new_af + (1 - alpha) * station_data['amplification_factor']
        
        # Update the station data
        station_data['amplification_factor'] = round(updated_af, 2)
        update_station_data(station_code, station_data)
    
    return {
        'previous_prediction': waterlogging_depth,
        'actual_value': actual_waterdepth,
        'error': round(error, 2)
    }