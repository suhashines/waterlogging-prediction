import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
import logging
import joblib
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskPredictor:
    """
    Risk prediction module for waterlogging based on amplification factors and geospatial features
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None, config_path: Optional[str] = None):
        """
        Initialize the RiskPredictor
        
        Args:
            weights (Dict, optional): Weights for different risk factors
            config_path (str, optional): Path to config file with weights and station data
        """
        # Default weights for risk factors
        self.default_weights = {
            'amplification_factor': 0.4,
            'elevation': 0.2,
            'impervious_cover': 0.1,
            'drainage': 0.15,
            'slope': 0.1,
            'proximity_to_water': 0.05
        }
        
        # Use provided weights or defaults
        self.weights = weights or self.default_weights
        
        # Normalize weights to sum to 1
        weight_sum = sum(self.weights.values())
        self.weights = {k: v / weight_sum for k, v in self.weights.items()}
        
        # Station data (will contain amplification factors and geospatial features)
        self.station_data = {}
        
        # Load config if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            # Initialize with dummy data for the stations
            self._initialize_dummy_station_data()
            
    def _initialize_dummy_station_data(self):
        """
        Initialize dummy station data for testing purposes
        """
        # For each station (1-7), create dummy geospatial data
        for station_id in range(1, 8):
            # Random values for each feature, with realistic ranges
            self.station_data[str(station_id)] = {
                'amplification_factor': np.random.uniform(0.1, 0.5),
                'elevation': np.random.uniform(5, 50),  # meters
                'impervious_cover': np.random.uniform(0.6, 0.95),  # fraction
                'drainage_area': np.random.uniform(100, 350),  # m²
                'drainage_volume': np.random.uniform(5000, 15000),  # m³
                'slope': np.random.uniform(0.01, 0.1),  # ratio
                'proximity_to_water': np.random.uniform(0, 500)  # meters
            }
        
        # Normalize geographic features
        self._normalize_features()
        
    def _normalize_features(self):
        """
        Normalize geographic features across all stations to [0,1] range
        """
        # Features to normalize
        features = ['elevation', 'drainage_area', 'drainage_volume', 'slope', 'proximity_to_water']
        
        # Find min and max for each feature
        feature_ranges = {}
        for feature in features:
            values = [station_data[feature] for station_data in self.station_data.values() 
                      if feature in station_data]
            if values:
                feature_ranges[feature] = (min(values), max(values))
        
        # Normalize each feature
        for station_id, data in self.station_data.items():
            for feature in features:
                if feature in data and feature in feature_ranges:
                    min_val, max_val = feature_ranges[feature]
                    if max_val > min_val:  # Avoid division by zero
                        normalized_value = (data[feature] - min_val) / (max_val - min_val)
                        
                        # Store both raw and normalized values
                        data[f'{feature}_normalized'] = normalized_value
        
    def load_config(self, config_path: str):
        """
        Load configuration data (weights and station data) from file
        
        Args:
            config_path (str): Path to the config file
        """
        logger.info(f"Loading risk predictor config from {config_path}")
        
        try:
            config_data = joblib.load(config_path)
            
            # Update weights if provided
            if 'weights' in config_data:
                self.weights = config_data['weights']
                
            # Update station data if provided
            if 'station_data' in config_data:
                self.station_data = config_data['station_data']
                
            # Normalize features
            self._normalize_features()
                
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            # Fall back to dummy data
            self._initialize_dummy_station_data()
            
    def save_config(self, config_path: str):
        """
        Save configuration data to file
        
        Args:
            config_path (str): Path to save the config file
        """
        logger.info(f"Saving risk predictor config to {config_path}")
        
        config_data = {
            'weights': self.weights,
            'station_data': self.station_data
        }
        
        try:
            joblib.dump(config_data, config_path)
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
            
    def calculate_amplification_factor(self, station_id: str, rainfall: float, 
                                       waterdepth: float) -> float:
        """
        Calculate the amplification factor (ratio of waterlogging depth to rainfall)
        
        Args:
            station_id (str): Station ID
            rainfall (float): Rainfall amount (mm)
            waterdepth (float): Waterlogging depth (m)
            
        Returns:
            float: Amplification factor
        """
        # Ensure rainfall is not zero to avoid division by zero
        if rainfall <= 0:
            return 0
            
        # Convert rainfall from mm to m for consistent units
        rainfall_m = rainfall / 1000.0
        
        # Calculate amplification factor
        af = waterdepth / rainfall_m
        
        # Update station data
        if station_id in self.station_data:
            # Use exponential moving average to update the factor (smoothing)
            alpha = 0.3  # Smoothing factor
            old_af = self.station_data[station_id].get('amplification_factor', af)
            updated_af = alpha * af + (1 - alpha) * old_af
            self.station_data[station_id]['amplification_factor'] = updated_af
        else:
            # Initialize if station not in data
            self.station_data[station_id] = {'amplification_factor': af}
            
        return af
        
    def predict_risk(self, station_id: str, rainfall: float, waterdepth: float) -> Dict:
        """
        Predict the risk level based on amplification factor and geospatial features
        
        Args:
            station_id (str): Station ID
            rainfall (float): Rainfall amount (mm)
            waterdepth (float): Waterlogging depth (m)
            
        Returns:
            Dict: Risk assessment including level and score
        """
        # Calculate amplification factor
        af = self.calculate_amplification_factor(station_id, rainfall, waterdepth)
        
        # Check if station exists in data
        if station_id not in self.station_data:
            logger.warning(f"Station {station_id} not found in data. Using default values.")
            # Add station with default values if not found
            self.station_data[station_id] = {
                'amplification_factor': af,
                'elevation_normalized': 0.5,
                'impervious_cover': 0.8,
                'drainage_area_normalized': 0.5,
                'drainage_volume_normalized': 0.5,
                'slope_normalized': 0.5,
                'proximity_to_water_normalized': 0.5
            }
        
        # Get station data
        station_data = self.station_data[station_id]
        
        # Calculate risk score
        risk_score = (
            self.weights['amplification_factor'] * station_data.get('amplification_factor', 0) +
            self.weights['elevation'] * (1 - station_data.get('elevation_normalized', 0.5)) +
            self.weights['impervious_cover'] * station_data.get('impervious_cover', 0.8) +
            self.weights['drainage'] * (1 - station_data.get('drainage_volume_normalized', 0.5)) +
            self.weights['slope'] * (1 - station_data.get('slope_normalized', 0.5)) +
            self.weights['proximity_to_water'] * station_data.get('proximity_to_water_normalized', 0.5)
        )
        
        # Determine risk level
        if risk_score < 0.3:
            risk_level = 'low'
        elif risk_score < 0.6:
            risk_level = 'moderate'
        else:
            risk_level = 'high'
            
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'amplification_factor': station_data.get('amplification_factor', 0),
            'factors': {
                'elevation': 1 - station_data.get('elevation_normalized', 0.5),
                'impervious_cover': station_data.get('impervious_cover', 0.8),
                'drainage': 1 - station_data.get('drainage_volume_normalized', 0.5),
                'slope': 1 - station_data.get('slope_normalized', 0.5),
                'proximity_to_water': station_data.get('proximity_to_water_normalized', 0.5)
            }
        }
        
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update the weights for risk factors
        
        Args:
            new_weights (Dict): New weights for risk factors
        """
        # Update weights
        self.weights.update(new_weights)
        
        # Normalize weights to sum to 1
        weight_sum = sum(self.weights.values())
        self.weights = {k: v / weight_sum for k, v in self.weights.items()}
        
        logger.info(f"Updated risk factor weights: {self.weights}")
        
    def update_station_data(self, station_id: str, data: Dict):
        """
        Update geospatial data for a station
        
        Args:
            station_id (str): Station ID
            data (Dict): New geospatial data
        """
        # Check if station exists
        if station_id not in self.station_data:
            self.station_data[station_id] = {}
            
        # Update data
        self.station_data[station_id].update(data)
        
        # Re-normalize features
        self._normalize_features()
        
        logger.info(f"Updated data for station {station_id}")
        
    def get_station_data(self, station_id: str = None) -> Dict:
        """
        Get geospatial data for a station or all stations
        
        Args:
            station_id (str, optional): Station ID
            
        Returns:
            Dict: Station data
        """
        if station_id:
            return self.station_data.get(station_id, {})
        else:
            return self.station_data
            
    def get_weights(self) -> Dict[str, float]:
        """
        Get the current weights for risk factors
        
        Returns:
            Dict: Current weights
        """
        return self.weights