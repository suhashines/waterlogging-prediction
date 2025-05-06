from flask import Blueprint, jsonify, request
from db import db, authority_db

# Create a Blueprint for authority dashboard routes
authority_bp = Blueprint('authority', __name__, url_prefix='/authority')

@authority_bp.route('/stations', methods=['GET'])
def get_stations():
    # Get stations from the database module
    stations = authority_db.get_stations()
    
    return jsonify({
        'stations': stations,
        'message': 'Stations retrieved successfully',
        'status': 'success'
    })

@authority_bp.route('/stations/<int:station_id>/data', methods=['GET'])
def get_station_data(station_id):
    # Get requested hours from query parameter, default to 3
    hours = int(request.args.get('hours', 3))
    
    # Get station data from the database module
    result = authority_db.get_station_data(station_id, hours)
    
    if not result:
        return jsonify({
            'message': 'Station not found',
            'status': 'error'
        }), 404
    
    return jsonify({
        **result,
        'message': 'Station data retrieved successfully',
        'status': 'success'
    })

@authority_bp.route('/stations/<int:station_id>/location', methods=['POST'])
def update_station_location(station_id):
    data = request.get_json()
    lat = data.get('lat')
    lon = data.get('lon')
    
    if lat is None or lon is None:
        return jsonify({
            'message': 'Latitude and longitude are required',
            'status': 'error'
        }), 400
    
    # Update station location with the database module
    station = authority_db.update_station_location(station_id, lat, lon)
    
    if not station:
        return jsonify({
            'message': 'Station not found',
            'status': 'error'
        }), 404
    
    return jsonify({
        'station': station,
        'message': 'Station location updated successfully',
        'status': 'success'
    })

def init_app(app):
    app.register_blueprint(authority_bp)