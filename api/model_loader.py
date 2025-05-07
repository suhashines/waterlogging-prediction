# api/model-loader.py
import os
import logging
import joblib
import sys

# Get the parent directory path and add it to sys.path if not already there
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Now you can import from the models directory
from models.waterlogging_predictor import WaterloggingPredictor
from models.risk_predictor import RiskPredictor

# Configure logging for this module
logger = logging.getLogger(__name__)

# Global variables for models
waterlogging_predictor = None
risk_predictor = None

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
WATERLOGGING_MODEL_PATH = os.path.join(MODEL_DIR, 'waterlogging_model.joblib')
RISK_CONFIG_PATH = os.path.join(MODEL_DIR, 'risk_config.joblib')

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
    
    return {
        'waterlogging_predictor': waterlogging_predictor,
        'risk_predictor': risk_predictor
    }
