from flask import Blueprint, jsonify, request
from db import db, route_db

# Create a Blueprint for route planning routes
route_bp = Blueprint('routes', __name__, url_prefix='/routes')

@route_bp.route('/plan', methods=['POST'])
def plan_routes():
    data = request.get_json()
    
    # Extract locations
    start_location = data.get('startLocation', 'Dhaka')
    end_location = data.get('endLocation', 'Khulna')
    timestamp = data.get('timestamp', None)
    
    # Get routes from the database module
    routes = route_db.find_routes(start_location, end_location, timestamp)
    
    return jsonify({
        'routes': routes,
        'message': 'Routes retrieved successfully',
        'status': 'success'
    })

def init_app(app):
    app.register_blueprint(route_bp)