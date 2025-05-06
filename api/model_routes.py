from flask import Blueprint, jsonify, request
from db import db, model_db
import time
import random
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Create a Blueprint for model routes
model_bp = Blueprint('model', __name__, url_prefix='/model')

@model_bp.route('/predict/', methods=['POST'])
def predict():
    # Log request received time
    logger.info("Received prediction request")
    start_time = time.time()
    
    # Simulate model loading time
    time.sleep(0.5)
    logger.info("Model loaded for inference")
    
    data = request.get_json()
    
    # Extract data from request
    station_code = data.get('station_code', '1')
    rainfall = float(data.get('rainfall', 10.0))
    timestamp = data.get('timestamp', None)
    weather = data.get('weather', None)
    windspeed = data.get('windspeed', None)
    
    # Simulate data preprocessing time
    time.sleep(0.3)
    logger.info(f"Processing inputs: station={station_code}, rainfall={rainfall}mm")
    
    # Calculate a variable inference time based on input complexity
    # More rainfall = more computation time
    inference_time = min(2.0, max(0.8, rainfall / 15))
    
    # Add some randomness to make it more realistic
    inference_time *= random.uniform(0.9, 1.1)
    
    # Simulate the actual model inference
    logger.info("Running model inference...")
    time.sleep(inference_time)
    
    # Get prediction from the model module
    prediction = model_db.predict_waterlogging(
        station_code, rainfall, timestamp, weather, windspeed
    )
    
    # Simulate post-processing time
    time.sleep(0.2)
    logger.info("Post-processing prediction results")
    
    # Calculate total processing time
    processing_time = time.time() - start_time
    logger.info(f"Prediction completed in {processing_time:.2f} seconds")
    
    return jsonify({
        'prediction': prediction,
        'processing_time_seconds': round(processing_time, 2),
        'message': 'Prediction successful',
        'status': 'success'
    })

@model_bp.route('/feedback', methods=['POST'])
def feedback():
    # Start timing
    start_time = time.time()
    
    # Simulate initial processing delay
    time.sleep(0.3)
    
    data = request.get_json()
    
    # Extract data from request
    station_code = data.get('station_code', '1')
    rainfall = float(data.get('rainfall', 10.0))
    actual_waterdepth = float(data.get('actual_waterdepth', 0.2))
    timestamp = data.get('timestamp', None)
    weather = data.get('weather', None)
    windspeed = data.get('windspeed', None)
    
    # Simulate model update time
    # More complex if there's a larger difference between prediction and actual
    update_time = 0.7 + random.uniform(0, 0.5)
    time.sleep(update_time)
    
    # Process feedback with the model module
    result = model_db.process_feedback(
        station_code, rainfall, actual_waterdepth, timestamp, weather, windspeed
    )
    
    # Simulate saving updated model
    time.sleep(0.4)
    
    # Calculate total processing time
    processing_time = time.time() - start_time
    
    return jsonify({
        'message': 'Feedback processed successfully',
        'status': 'success',
        'details': result,
        'processing_time_seconds': round(processing_time, 2)
    })

@model_bp.route('/weights', methods=['GET', 'POST'])
def weights():
    start_time = time.time()
    
    if request.method == 'GET':
        # Simulate data retrieval time
        time.sleep(random.uniform(0.3, 0.7))
        
        weights = model_db.get_risk_weights()
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        
        return jsonify({
            'weights': weights,
            'message': 'Current weights retrieved successfully',
            'status': 'success',
            'processing_time_seconds': round(processing_time, 2)
        })
    elif request.method == 'POST':
        # Simulate request processing time
        time.sleep(0.3)
        
        data = request.get_json()
        new_weights = data.get('weights', {})
        
        # Simulate weight computation time
        time.sleep(random.uniform(0.5, 0.9))
        
        # Update weights
        updated_weights = model_db.update_risk_weights(new_weights)
        
        # Simulate saving updated weights
        time.sleep(0.4)
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        
        return jsonify({
            'weights': updated_weights,
            'message': 'Weights updated successfully',
            'status': 'success',
            'processing_time_seconds': round(processing_time, 2)
        })

@model_bp.route('/station-data', methods=['GET', 'POST'])
def station_data():
    start_time = time.time()
    
    if request.method == 'GET':
        station_id = request.args.get('station_id')
        
        # Simulate database query time - longer if getting all stations
        if station_id:
            time.sleep(random.uniform(0.2, 0.5))
        else:
            time.sleep(random.uniform(0.5, 1.0))
            
        if station_id:
            # Return data for specific station
            station_data = model_db.get_station_data(station_id)
            if station_data:
                # Calculate total processing time
                processing_time = time.time() - start_time
                
                return jsonify({
                    'station_data': station_data,
                    'message': 'Station data retrieved successfully',
                    'status': 'success',
                    'processing_time_seconds': round(processing_time, 2)
                })
            else:
                # Even errors should have realistic timing
                time.sleep(0.2)
                
                # Calculate total processing time
                processing_time = time.time() - start_time
                
                return jsonify({
                    'message': 'Station not found',
                    'status': 'error',
                    'processing_time_seconds': round(processing_time, 2)
                }), 404
        else:
            # Return data for all stations
            # Calculate total processing time
            processing_time = time.time() - start_time
            
            return jsonify({
                'station_data': model_db.get_station_data(),
                'message': 'All station data retrieved successfully',
                'status': 'success',
                'processing_time_seconds': round(processing_time, 2)
            })
    elif request.method == 'POST':
        # Simulate initial request parsing
        time.sleep(0.3)
        
        data = request.get_json()
        station_id = data.get('station_id')
        new_data = data.get('data', {})
        
        if not station_id:
            # Error checking should still have some delay
            time.sleep(0.2)
            
            # Calculate total processing time
            processing_time = time.time() - start_time
            
            return jsonify({
                'message': 'Station ID is required',
                'status': 'error',
                'processing_time_seconds': round(processing_time, 2)
            }), 400
        
        # Simulate validation and computation time
        time.sleep(random.uniform(0.4, 0.8))
        
        # Update station data
        updated_data = model_db.update_station_data(station_id, new_data)
        
        # Simulate saving to database
        time.sleep(0.4)
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        
        return jsonify({
            'station_data': updated_data,
            'message': 'Station data updated successfully',
            'status': 'success',
            'processing_time_seconds': round(processing_time, 2)
        })

def init_app(app):
    app.register_blueprint(model_bp)