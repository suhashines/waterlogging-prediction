# Route planning data module
from . import db
import random
import math
from datetime import datetime

# Define namespaces
LOCATIONS = 'route_locations'
ROUTE_TYPES = 'route_types'

# Initialize locations
def init_locations():
    return {
        'dhaka': {'lat': 23.8103, 'lon': 90.4125, 'name': 'Dhaka'},
        'khulna': {'lat': 22.8456, 'lon': 89.5403, 'name': 'Khulna'},
        'chittagong': {'lat': 22.3569, 'lon': 91.7832, 'name': 'Chittagong'},
        'rajshahi': {'lat': 24.3745, 'lon': 88.6042, 'name': 'Rajshahi'},
        'sylhet': {'lat': 24.8949, 'lon': 91.8687, 'name': 'Sylhet'},
    }

# Initialize route types
def init_route_types():
    return ['safe', 'balanced', 'shortest']

# Register with the database
db.register(LOCATIONS, init_locations)
db.register(ROUTE_TYPES, init_route_types)

# Initialize data
db.initialize(LOCATIONS, init_locations)
db.initialize(ROUTE_TYPES, init_route_types)

# Helper functions
def get_location_coordinates(location_name):
    """Get coordinates for a location name"""
    # Case-insensitive match
    location_key = location_name.lower()
    locations = db.get(LOCATIONS)
    
    if location_key in locations:
        return locations[location_key]
    else:
        # Generate consistent coordinates for unknown locations
        hash_value = sum(ord(c) for c in location_name)
        return {
            'lat': 23.685 + (hash_value % 10) * 0.02,
            'lon': 90.356 + (hash_value % 7) * 0.02,
            'name': location_name
        }

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points in meters"""
    R = 6371e3  # Earth radius in meters
    φ1 = math.radians(lat1)
    φ2 = math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    
    a = math.sin(Δφ/2) * math.sin(Δφ/2) + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ/2) * math.sin(Δλ/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def generate_segments(start, end, route_type, total_distance):
    """Generate route segments"""
    segments = []
    
    # Number of segments depends on route type
    num_segments = 2 if route_type == 'safe' else 3 if route_type == 'balanced' else 4
    segment_distance = total_distance / num_segments
    
    # Define risk profiles for each route type
    risk_profiles = {
        'safe': ['none', 'none', 'low'],
        'balanced': ['low', 'medium', 'low'],
        'shortest': ['medium', 'high', 'extreme', 'high']
    }
    
    # Define road types
    road_types = ['highway', 'major', 'local', 'bridge']
    
    current_point = start
    
    for i in range(num_segments):
        ratio = (i + 1) / num_segments
        next_point = end if i == num_segments - 1 else {
            'lat': start['lat'] + (end['lat'] - start['lat']) * ratio,
            'lon': start['lon'] + (end['lon'] - start['lon']) * ratio
        }
        
        # Get waterlogging risk for this segment
        flood_risk = risk_profiles[route_type][i % len(risk_profiles[route_type])]
        
        # Get road type
        road_type = road_types[i % len(road_types)]
        
        # Calculate realistic duration based on distance and road type
        speed_factors = {'highway': 80, 'major': 60, 'local': 40, 'bridge': 70}
        speed = speed_factors[road_type]  # km/h
        duration = int((segment_distance / 1000) / speed * 3600)  # seconds
        
        segments.append({
            'startPoint': current_point,
            'endPoint': next_point,
            'distance': int(segment_distance),
            'duration': duration,
            'floodRisk': flood_risk,
            'roadType': road_type
        })
        
        current_point = next_point
    
    return segments

def generate_safety_issues(route_type, start, end):
    """Generate safety issues based on route type"""
    issues = []
    
    if route_type == 'balanced':
        # One moderate issue for balanced routes
        issues.append({
            'type': 'waterlogging',
            'description': 'Moderate waterlogging reported along this route',
            'severity': 'warning',
            'location': {
                'lat': start['lat'] + (end['lat'] - start['lat']) * 0.4,
                'lon': start['lon'] + (end['lon'] - start['lon']) * 0.4
            }
        })
    elif route_type == 'shortest':
        # Multiple severe issues for shortest routes
        issues.append({
            'type': 'waterlogging',
            'description': 'Severe waterlogging reported on multiple segments',
            'severity': 'danger',
            'location': {
                'lat': start['lat'] + (end['lat'] - start['lat']) * 0.3,
                'lon': start['lon'] + (end['lon'] - start['lon']) * 0.3
            }
        })
        issues.append({
            'type': 'closure',
            'description': 'Road closure due to high water levels',
            'severity': 'danger',
            'location': {
                'lat': start['lat'] + (end['lat'] - start['lat']) * 0.7,
                'lon': start['lon'] + (end['lon'] - start['lon']) * 0.7
            }
        })
    
    return issues

# Route planning functions
def find_routes(start_location, end_location, timestamp=None):
    """Find safe routes between two locations"""
    # Get coordinates
    start_coords = get_location_coordinates(start_location)
    end_coords = get_location_coordinates(end_location)
    
    # Calculate straight-line distance (in meters)
    straight_line_distance = haversine(
        start_coords['lat'], start_coords['lon'],
        end_coords['lat'], end_coords['lon']
    )
    
    # Road distance is typically longer than straight-line distance
    road_distance = straight_line_distance * 1.3
    
    # Get route types
    route_types = db.get(ROUTE_TYPES)
    
    # Generate routes
    routes = []
    for i, route_type in enumerate(route_types):
        # Adjust distance based on route type
        distance_factor = 1.15 if route_type == 'safe' else 1.0 if route_type == 'balanced' else 0.9
        total_distance = road_distance * distance_factor
        
        segments = generate_segments(start_coords, end_coords, route_type, total_distance)
        
        # Calculate total distance and duration
        total_distance = sum(segment['distance'] for segment in segments)
        total_duration = sum(segment['duration'] for segment in segments)
        
        # Set safety score based on route type
        safety_scores = {'safe': 92, 'balanced': 75, 'shortest': 45}
        
        # Generate safety issues
        safety_issues = generate_safety_issues(route_type, start_coords, end_coords)
        
        routes.append({
            'id': f'route-{i+1}',
            'name': f"{route_type.capitalize()} Route",
            'startLocation': start_coords,
            'endLocation': end_coords,
            'segments': segments,
            'totalDistance': int(total_distance),
            'totalDuration': int(total_duration),
            'safetyScore': safety_scores[route_type],
            'safetyIssues': safety_issues
        })
    
    return routes