from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import logging
import joblib
import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.waterlogging_predictor import WaterloggingPredictor
from models.risk_predictor import RiskPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load models
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
WATERLOGGING_MODEL_PATH = os.path.join(MODEL_DIR, 'waterlogging_model.joblib')
RISK_CONFIG_PATH = os.path.join(MODEL_DIR, 'risk_config.joblib')

# Global variables for models
waterlogging_predictor = None
risk_predictor = None

def load_models():
    """Load the trained models"""
    global waterlogging_predictor, risk_predictor
    
    # Load waterlogging predictor
    if os.path.exists(WATERLOGGING_MODEL_PATH):
        logger.info(f"Loading waterlogging model from {WATERLOGGING_MODEL_PATH}")
        waterlogging_predictor = WaterloggingPredictor(model_path=WATERLOGGING_MODEL_PATH)
    else:
        logger.warning(f"Waterlogging model not found at {WATERLOGGING_MODEL_PATH}. Initializing with default settings.")
        waterlogging_predictor = WaterloggingPredictor(model_type='rf')
    
    # Load risk predictor
    if os.path.exists(RISK_CONFIG_PATH):
        logger.info(f"Loading risk predictor config from {RISK_CONFIG_PATH}")
        risk_predictor = RiskPredictor(config_path=RISK_CONFIG_PATH)
    else:
        logger.warning(f"Risk predictor config not found at {RISK_CONFIG_PATH}. Initializing with default settings.")
        risk_predictor = RiskPredictor()

with app.app_context():
    load_models()

@app.route('/model/predict/', methods=['POST'])
def predict():
    """
    Predict waterlogging depth and risk factor
    
    Request Body:
    {
        "station_code": "1",
        "rainfall": 10.5,
        "timestamp": "2023-05-01T12:30:00",
        "weather": 2,  # Optional
        "windspeed": 5.2  # Optional
    }
    
    Response:
    {
        "prediction": {
            "waterlogging_depth": 0.25,
            "risk_factor": {
                "risk_score": 0.45,
                "risk_level": "moderate",
                "amplification_factor": 0.23,
                "factors": {...}
            }
        },
        "message": "Prediction successful",
        "status": "success"
    }
    """
    try:
        # Initialize models if not already initialized
        if waterlogging_predictor is None or risk_predictor is None:
            load_models()
        
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['station_code', 'rainfall', 'timestamp']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'message': f"Missing required field: {field}",
                    'status': 'error'
                }), 400
        
        # Create a DataFrame for prediction
        df = pd.DataFrame({
            'station_code': [data['station_code']],
            'rainfall': [float(data['rainfall'])],
            'timestamp': [pd.to_datetime(data['timestamp'])]
        })
        
        # Add optional fields if present
        if 'weather' in data:
            df['weather'] = data['weather']
        if 'windspeed' in data:
            df['windspeed'] = data['windspeed']
        
        # Predict waterlogging depth
        waterlogging_depth = waterlogging_predictor.predict(df)[0]
        
        # Predict risk factor
        risk_factor = risk_predictor.predict_risk(
            station_id=data['station_code'],
            rainfall=float(data['rainfall']),
            waterdepth=float(waterlogging_depth)
        )
        
        # Return response
        return jsonify({
            'prediction': {
                'waterlogging_depth': float(waterlogging_depth),
                'risk_factor': risk_factor
            },
            'message': "Prediction successful",
            'status': 'success'
        }), 200
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return jsonify({
            'message': f"Error in prediction: {str(e)}",
            'status': 'error'
        }), 500

@app.route('/model/feedback', methods=['POST'])
def feedback():
    """
    Provide feedback to improve future predictions
    
    Request Body:
    {
        "station_code": "1",
        "rainfall": 10.5,
        "timestamp": "2023-05-01T12:30:00",
        "weather": 2,  # Optional
        "windspeed": 5.2,  # Optional
        "actual_waterdepth": 0.27
    }
    
    Response:
    {
        "message": "Feedback processed successfully",
        "status": "success",
        "details": {
            "previous_prediction": 0.25,
            "actual_value": 0.27,
            "error": 0.02
        }
    }
    """
    try:
        # Initialize models if not already initialized
        if waterlogging_predictor is None or risk_predictor is None:
            load_models()
        
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['station_code', 'rainfall', 'timestamp', 'actual_waterdepth']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'message': f"Missing required field: {field}",
                    'status': 'error'
                }), 400
        
        # Create a DataFrame for prediction
        df = pd.DataFrame({
            'station_code': [data['station_code']],
            'rainfall': [float(data['rainfall'])],
            'timestamp': [pd.to_datetime(data['timestamp'])]
        })
        
        # Add optional fields if present
        if 'weather' in data:
            df['weather'] = data['weather']
        if 'windspeed' in data:
            df['windspeed'] = data['windspeed']
        
        # Predict waterlogging depth (before update)
        previous_prediction = waterlogging_predictor.predict(df)[0]
        
        # Add actual waterdepth to the dataframe
        df['waterdepth'] = float(data['actual_waterdepth'])
        
        # Update the waterlogging predictor with new data
        waterlogging_predictor.update_model(df)
        
        # Update the risk predictor's amplification factor
        risk_predictor.calculate_amplification_factor(
            station_id=data['station_code'],
            rainfall=float(data['rainfall']),
            waterdepth=float(data['actual_waterdepth'])
        )
        
        # Save updated models
        waterlogging_predictor.save_model(WATERLOGGING_MODEL_PATH)
        risk_predictor.save_config(RISK_CONFIG_PATH)
        
        # Calculate prediction error
        error = abs(previous_prediction - float(data['actual_waterdepth']))
        
        # Return response
        return jsonify({
            'message': "Feedback processed successfully",
            'status': 'success',
            'details': {
                'previous_prediction': float(previous_prediction),
                'actual_value': float(data['actual_waterdepth']),
                'error': float(error)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}", exc_info=True)
        return jsonify({
            'message': f"Error processing feedback: {str(e)}",
            'status': 'error'
        }), 500

@app.route('/model/weights', methods=['GET', 'POST'])
def weights():
    """
    Get or update risk factor weights
    
    GET:
    Returns current weights
    
    POST:
    Request Body:
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
    
    Response:
    {
        "message": "Weights updated successfully",
        "status": "success",
        "weights": {
            "amplification_factor": 0.5,
            "elevation": 0.2,
            "impervious_cover": 0.1,
            "drainage": 0.1,
            "slope": 0.05,
            "proximity_to_water": 0.05
        }
    }
    """
    try:
        # Initialize models if not already initialized
        if risk_predictor is None:
            load_models()
        
        if request.method == 'GET':
            # Return current weights
            return jsonify({
                'weights': risk_predictor.get_weights(),
                'message': "Current weights retrieved successfully",
                'status': 'success'
            }), 200
        
        elif request.method == 'POST':
            # Get new weights from request
            data = request.get_json()
            
            if 'weights' not in data:
                return jsonify({
                    'message': "Missing required field: weights",
                    'status': 'error'
                }), 400
            
            # Update weights
            risk_predictor.update_weights(data['weights'])
            
            # Save updated config
            risk_predictor.save_config(RISK_CONFIG_PATH)
            
            # Return updated weights
            return jsonify({
                'weights': risk_predictor.get_weights(),
                'message': "Weights updated successfully",
                'status': 'success'
            }), 200
    
    except Exception as e:
        logger.error(f"Error processing weights: {str(e)}", exc_info=True)
        return jsonify({
            'message': f"Error processing weights: {str(e)}",
            'status': 'error'
        }), 500

@app.route('/model/station-data', methods=['GET', 'POST'])
def station_data():
    """
    Get or update station data
    
    GET:
    Query Parameters:
    - station_id (optional): Get data for a specific station
    
    POST:
    Request Body:
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
    
    Response:
    {
        "message": "Station data updated successfully",
        "status": "success",
        "station_data": {...}
    }
    """
    try:
        # Initialize models if not already initialized
        if risk_predictor is None:
            load_models()
        
        if request.method == 'GET':
            # Get station_id from query parameters
            station_id = request.args.get('station_id')
            
            # Return station data
            return jsonify({
                'station_data': risk_predictor.get_station_data(station_id),
                'message': "Station data retrieved successfully",
                'status': 'success'
            }), 200
        
        elif request.method == 'POST':
            # Get data from request
            data = request.get_json()
            
            if 'station_id' not in data or 'data' not in data:
                return jsonify({
                    'message': "Missing required fields: station_id and/or data",
                    'status': 'error'
                }), 400
            
            # Update station data
            risk_predictor.update_station_data(data['station_id'], data['data'])
            
            # Save updated config
            risk_predictor.save_config(RISK_CONFIG_PATH)
            
            # Return updated station data
            return jsonify({
                'station_data': risk_predictor.get_station_data(data['station_id']),
                'message': "Station data updated successfully",
                'status': 'success'
            }), 200
    
    except Exception as e:
        logger.error(f"Error processing station data: {str(e)}", exc_info=True)
        return jsonify({
            'message': f"Error processing station data: {str(e)}",
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)