# Authority dashboard data module
from . import db
import random
import math
from datetime import datetime

# Define namespaces
STATIONS = 'authority_stations'
STATION_FEEDBACK = 'station_feedback'

# Initialize stations
def init_stations():
    return [
        {
            'id': 1,
            'name': 'Station A',
            'lat': 23.8103,
            'lon': 90.4125,
            'details': {
                'elevation': '11.2m',
                'landCover': 'Urban/Impervious',
                'drainage': 'Low',
                'slope': '1.8%',
                'proximity': '400m to Dhanmondi Lake'
            }
        },
        {
            'id': 2,
            'name': 'Station B',
            'lat': 23.7000,
            'lon': 90.3750,
            'details': {
                'elevation': '9.6m',
                'landCover': 'Mixed Residential',
                'drainage': 'Moderate',
                'slope': '2.5%',
                'proximity': '700m to Buriganga River'
            }
        },
        {
            'id': 3,
            'name': 'Station C',
            'lat': 23.7800,
            'lon': 90.4200,
            'details': {
                'elevation': '10.4m',
                'landCover': 'Vegetation/Suburban',
                'drainage': 'High',
                'slope': '3.2%',
                'proximity': '120m to Canal X'
            }
        }
    ]

# Initialize station feedback
def init_station_feedback():
    return {
        1: [
            {'user': 'Hasan', 'comment': 'Water levels rise quickly here after heavy rain.'},
            {'user': 'Farzana', 'comment': 'Drainage improvements are working recently.'}
        ],
        2: [
            {'user': 'Tariq', 'comment': 'Area remains flooded during monsoon.'},
            {'user': 'Mitu', 'comment': 'A key intersection affected by waterlogging.'}
        ],
        3: [
            {'user': 'Rayhan', 'comment': 'No major issue unless there\'s a storm.'}
        ]
    }

# Register with the database
db.register(STATIONS, init_stations)
db.register(STATION_FEEDBACK, init_station_feedback)

# Initialize data
db.initialize(STATIONS, init_stations)
db.initialize(STATION_FEEDBACK, init_station_feedback)

# Helper functions
def generate_timestamps(hours=6):
    """Generate timestamps for half-hour intervals"""
    return [f"{(6 + h // 2):02d}:{(h % 2) * 30:02d}" for h in range(hours * 2)]

def generate_station_data(station_id, hours=3):
    """Generate station data for monitoring"""
    timestamps = generate_timestamps(hours)
    data = []
    
    # Base values that vary by station
    base_waterlogging = 0.8 + station_id * 0.2
    base_rainfall = 5 + station_id * 2
    base_riskfactor = 1.0 + station_id * 0.2
    
    for i, timestamp in enumerate(timestamps):
        # Add some variation over time
        variation = (i / len(timestamps)) * random.uniform(0.8, 1.2)
        
        data.append({
            'timestamp': timestamp,
            'waterlogging': round(base_waterlogging + math.sin(i / 2) * 0.3 * variation, 1),
            'rainfall': round(base_rainfall + math.cos(i / 3) * 2 * variation, 1),
            'riskfactor': round(base_riskfactor + math.sin(i / 3) * 0.5 * variation, 1)
        })
    
    return data

# Authority dashboard functions
def get_stations():
    """Get all monitoring stations"""
    return db.get(STATIONS)

def get_station(station_id):
    """Get a specific station by ID"""
    stations = db.get(STATIONS)
    return next((s for s in stations if s['id'] == station_id), None)

def update_station_location(station_id, lat, lon):
    """Update the location of a station"""
    stations = db.get(STATIONS)
    station = next((s for s in stations if s['id'] == station_id), None)
    
    if not station:
        return None
    
    # Update the location
    station['lat'] = lat
    station['lon'] = lon
    
    # Update the database
    db.initialize(STATIONS, lambda: stations)
    
    return station

def get_station_data(station_id, hours=3):
    """Get monitoring data for a station"""
    station = get_station(station_id)
    if not station:
        return None
    
    # Generate data for the requested hours
    data = generate_station_data(station_id, hours)
    
    # Get feedback for this station
    feedback = db.get(STATION_FEEDBACK, station_id) or []
    
    return {
        'station': station,
        'data': data,
        'feedback': feedback
    }