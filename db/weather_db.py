# Weather data module
from . import db
import random
from datetime import datetime, timedelta

# Define namespaces
LOCATIONS = 'locations'
RAINFALL_INTENSITIES = 'rainfall_intensities'
WEATHER_CONDITIONS = 'weather_conditions'

# Initialize location data
def init_locations():
    return {
        '23.8103,90.4125': 'Dhaka, Bangladesh',
        '22.8456,89.5403': 'Khulna, Bangladesh',
        '22.3569,91.7832': 'Chittagong, Bangladesh',
    }

# Initialize rainfall intensities
def init_rainfall_intensities():
    return {
        'none': [0, 0.5],
        'light': [0.5, 5],
        'medium': [5, 10],
        'heavy': [10, 20],
        'extreme': [20, 50]
    }

# Initialize weather conditions
def init_weather_conditions():
    return {
        'none': ['Clear', 'Sunny', 'Partly Cloudy'],
        'light': ['Light Rain', 'Drizzle', 'Scattered Showers'],
        'medium': ['Moderate Rain', 'Steady Rain'],
        'heavy': ['Heavy Rain', 'Downpour', 'Thunderstorm'],
        'extreme': ['Extreme Rain', 'Severe Thunderstorm', 'Torrential Rain']
    }

# Register with the database
db.register(LOCATIONS, init_locations)
db.register(RAINFALL_INTENSITIES, init_rainfall_intensities)
db.register(WEATHER_CONDITIONS, init_weather_conditions)

# Initialize data
db.initialize(LOCATIONS, init_locations)
db.initialize(RAINFALL_INTENSITIES, init_rainfall_intensities)
db.initialize(WEATHER_CONDITIONS, init_weather_conditions)

# Helper functions
def get_location_name(lat, lon):
    """Get the name of a location based on coordinates"""
    key = f"{lat},{lon}"
    locations = db.get(LOCATIONS)
    return locations.get(key, f"Location at {lat}, {lon}")

def generate_rainfall_data(intensity=None):
    """Generate random rainfall data"""
    intensities = list(db.get(RAINFALL_INTENSITIES).keys())
    
    if not intensity:
        intensity = random.choice(intensities)
        
    rate_range = db.get(RAINFALL_INTENSITIES, intensity)
    return {
        'intensity': intensity,
        'rate': round(random.uniform(rate_range[0], rate_range[1]), 1)
    }

def generate_weather_condition(intensity):
    """Generate a random weather condition based on rainfall intensity"""
    conditions = db.get(WEATHER_CONDITIONS, intensity)
    return random.choice(conditions)

# Weather API functions
def get_current_weather(lat, lon):
    """Get current weather for specified coordinates"""
    # Generate random rainfall intensity with weighted probabilities
    intensities = list(db.get(RAINFALL_INTENSITIES).keys())
    intensity = random.choices(
        intensities,
        weights=[0.1, 0.2, 0.3, 0.3, 0.1],
        k=1
    )[0]
    
    rainfall_data = generate_rainfall_data(intensity)
    condition = generate_weather_condition(intensity)
    
    # Get location name
    location = get_location_name(lat, lon)
    
    # Generate random temperature (Celsius for Bangladesh)
    temperature = round(random.uniform(25, 32), 1)
    
    return {
        'location': location,
        'currentCondition': condition,
        'temperature': temperature,
        'rainfall': rainfall_data,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }

def get_weather_forecast(lat, lon):
    """Get weather forecast for specified coordinates"""
    # Get location name
    location = get_location_name(lat, lon)
    
    # Generate forecast for 3 hours
    forecast = []
    base_temperature = round(random.uniform(25, 32), 1)
    
    # Start with random intensity
    intensities = list(db.get(RAINFALL_INTENSITIES).keys())
    intensity_idx = random.randint(0, len(intensities)-1)
    
    for i in range(3):
        # Weather trend tends to improve or worsen gradually
        intensity_change = random.choice([-1, 0, 0, 1])
        intensity_idx = max(0, min(len(intensities)-1, intensity_idx + intensity_change))
        
        intensity = intensities[intensity_idx]
        rainfall_data = generate_rainfall_data(intensity)
        condition = generate_weather_condition(intensity)
        
        # Temperature slightly changes over time
        temperature = round(base_temperature - (i * random.uniform(0.5, 1.5)), 1)
        
        forecast.append({
            'time': f"{i+1}h",
            'condition': condition,
            'temperature': temperature,
            'rainfall': rainfall_data
        })
    
    return {
        'location': location,
        'forecast': forecast
    }

def get_weather_alerts(lat, lon):
    """Get weather alerts for specified coordinates"""
    # Get location name
    location = get_location_name(lat, lon)
    
    # Generate alerts based on random intensity
    alerts = []
    intensities = list(db.get(RAINFALL_INTENSITIES).keys())
    intensity = random.choices(
        intensities,
        weights=[0.3, 0.2, 0.2, 0.2, 0.1],
        k=1
    )[0]
    
    # Only generate alerts for medium, heavy, or extreme rainfall
    if intensity in ['medium', 'heavy', 'extreme']:
        alert_types = {
            'medium': {
                'type': 'Rainfall Advisory',
                'severity': 'info',
                'title': 'Rainfall Advisory',
                'description': 'Moderate rainfall expected. Be aware of possible minor waterlogging in low-lying areas.',
            },
            'heavy': {
                'type': 'waterlogging Watch',
                'severity': 'warning',
                'title': 'waterlogging Watch',
                'description': 'Heavy rainfall may cause waterlogging in prone areas. Take precautionary measures.',
            },
            'extreme': {
                'type': 'Flash waterlogging',
                'severity': 'danger',
                'title': 'Flash waterlogging Warning',
                'description': 'Flash waterlogging is occurring or imminent in the warned area. Move to higher ground immediately.',
            }
        }
        
        alert = alert_types[intensity]
        now = datetime.now()
        
        alerts.append({
            **alert,
            'issued': now.isoformat() + 'Z',
            'expires': (now + timedelta(hours=6)).isoformat() + 'Z'
        })
    
    return {
        'location': location,
        'alerts': alerts
    }