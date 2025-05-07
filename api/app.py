# api/app.py
from flask import Flask
import os
import sys
import logging

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

# Add parent directory to path to ensure imports work regardless of how the app is run
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def create_app():
    app = Flask(__name__)
    
    # Load ML models before registering routes
    from api.model_loader import load_models
    logger.info("Loading machine learning models...")
    models = load_models()
    logger.info(f"Models loaded: {list(models.keys())}")
    
    # Add a health check endpoint
    @app.route('/health')
    def health_check():
        from api.model_loader import waterlogging_predictor, risk_predictor
        return {
            "status": "ok",
            "models": {
                "waterlogging_model": "loaded" if waterlogging_predictor is not None else "not loaded",
                "risk_model": "loaded" if risk_predictor is not None else "not loaded"
            }
        }
    
    # Import and register blueprints
    # Using direct imports that work both when run directly or as a module
    from api.model_routes import init_app as init_model_routes
    from api.weather_routes import init_app as init_weather_routes
    from api.route_routes import init_app as init_route_routes
    from api.forum_routes import init_app as init_forum_routes
    from api.auth_routes import init_app as init_auth_routes
    from api.authority_routes import init_app as init_authority_routes
    
    # Initialize routes
    init_model_routes(app)
    init_weather_routes(app)
    init_route_routes(app)
    init_forum_routes(app)
    init_auth_routes(app)
    init_authority_routes(app)
    
    # Add CORS headers
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
        return response
    
    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
