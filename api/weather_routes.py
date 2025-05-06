from flask import Blueprint, jsonify, request
from db import db, weather_db

# Create a Blueprint for weather routes
weather_bp = Blueprint('weather', __name__, url_prefix='/weather')

@weather_bp.route('/current', methods=['GET'])
def get_current_weather():
    # Get coordinates from query parameters
    lat = request.args.get('lat', '23.8103')
    lon = request.args.get('lon', '90.4125')
    
    # Get weather data from the database module
    weather_data = weather_db.get_current_weather(lat, lon)
    
    return jsonify({
        **weather_data,
        'message': 'Weather data retrieved successfully',
        'status': 'success'
    })

@weather_bp.route('/forecast', methods=['GET'])
def get_weather_forecast():
    # Get coordinates from query parameters
    lat = request.args.get('lat', '23.8103')
    lon = request.args.get('lon', '90.4125')
    
    # Get forecast data from the database module
    forecast_data = weather_db.get_weather_forecast(lat, lon)
    
    return jsonify({
        **forecast_data,
        'message': 'Weather forecast retrieved successfully',
        'status': 'success'
    })

@weather_bp.route('/alerts', methods=['GET'])
def get_weather_alerts():
    # Get coordinates from query parameters
    lat = request.args.get('lat', '23.8103')
    lon = request.args.get('lon', '90.4125')
    
    # Get alerts data from the database module
    alerts_data = weather_db.get_weather_alerts(lat, lon)
    
    return jsonify({
        **alerts_data,
        'message': 'Weather alerts retrieved successfully',
        'status': 'success'
    })

def init_app(app):
    app.register_blueprint(weather_bp)