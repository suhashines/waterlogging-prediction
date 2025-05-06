import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import calendar
import math
from typing import List, Dict, Union, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering class for waterlogging prediction
    
    Extracts and constructs features from rainfall time series data:
    1. Unit rainfall
    2. Seasonality coefficients
    3. Rainfall interval features
    4. Statistical features
    """
    
    def __init__(self):
        """
        Initialize the FeatureEngineer
        """
        # Define rainy, transition, and dry months based on climatology
        # These can be customized based on local climate
        self.rainy_months = [6, 7, 8, 9]  # June to September
        self.transition_months = [3, 4, 5, 10]  # March to May, October
        self.dry_months = [1, 2, 11, 12]  # November to February
        
        # Define seasonality coefficients
        self.season_coeffs = {
            'rainy': 10,      # Strong coefficient for rainy season
            'transition': 6,  # Medium coefficient for transition season
            'dry': 2          # Low coefficient for dry season
        }
        
        # Feature names
        self.feature_names = [
            'unit_rainfall',
            'seasonality_coeff',
            'rainfall_interval',
            'wetting_coeff',
            'infiltration_capacity',
            'rainfall_mean',
            'rainfall_max',
            'rainfall_std',
            'rainfall_kurtosis',
            'rainfall_skewness',
            'rainfall_auc'
        ]
        
    def get_season_coefficient(self, month: int) -> float:
        """
        Determine the seasonality coefficient based on the month
        
        Args:
            month (int): Month (1-12)
            
        Returns:
            float: Seasonality coefficient
        """
        if month in self.rainy_months:
            return self.season_coeffs['rainy']
        elif month in self.transition_months:
            return self.season_coeffs['transition']
        else:
            return self.season_coeffs['dry']
    
    def calculate_unit_rainfall(self, rainfall_series: np.ndarray) -> np.ndarray:
        """
        Calculate unit rainfall from cumulative rainfall
        
        Args:
            rainfall_series (np.ndarray): Array of rainfall values
            
        Returns:
            np.ndarray: Unit rainfall array
        """
        # Initialize unit rainfall array
        unit_rainfall = np.zeros_like(rainfall_series)
        
        # First value is the same
        unit_rainfall[0] = rainfall_series[0]
        
        # Calculate differences for subsequent values
        for i in range(1, len(rainfall_series)):
            # If current value is less than previous, it's a new rainfall event
            if rainfall_series[i] < rainfall_series[i-1]:
                unit_rainfall[i] = rainfall_series[i]
            else:
                # Calculate increment
                unit_rainfall[i] = rainfall_series[i] - rainfall_series[i-1]
                
        return unit_rainfall
    
    def calculate_rainfall_interval(self, rainfall_series: np.ndarray, threshold: float = 0.1) -> Tuple[np.ndarray, List[int]]:
        """
        Calculate intervals between rainfall events
        
        Args:
            rainfall_series (np.ndarray): Array of unit rainfall values
            threshold (float): Minimum rainfall to consider as an event
            
        Returns:
            Tuple[np.ndarray, List[int]]: Rainfall intervals and event indices
        """
        # Find indices where rainfall exceeds threshold (start of events)
        event_indices = [i for i, r in enumerate(rainfall_series) if r > threshold]
        
        # Calculate intervals between events
        intervals = np.zeros_like(rainfall_series)
        
        if not event_indices:
            return intervals, []
            
        # For indices before first event, set a large interval
        intervals[:event_indices[0]] = 24  # Assuming 24 hours
        
        # Calculate intervals between events
        for i in range(len(event_indices) - 1):
            current_event = event_indices[i]
            next_event = event_indices[i+1]
            interval = next_event - current_event
            
            # Set the interval for all points between these events
            intervals[current_event:next_event] = interval
            
        # For indices after the last event, set the same interval as the last event
        if event_indices:
            intervals[event_indices[-1]:] = intervals[event_indices[-1] - 1] if event_indices[-1] > 0 else 24
            
        return intervals, event_indices
        
    def calculate_wetting_coefficient(self, rainfall_series: np.ndarray, intervals: np.ndarray) -> np.ndarray:
        """
        Calculate wetting coefficient (ratio of mean rainfall to interval)
        
        Args:
            rainfall_series (np.ndarray): Array of rainfall values
            intervals (np.ndarray): Array of rainfall intervals
            
        Returns:
            np.ndarray: Wetting coefficient array
        """
        # Calculate mean rainfall for each point (cumulative mean up to this point)
        cumulative_mean = np.zeros_like(rainfall_series)
        for i in range(len(rainfall_series)):
            if i == 0:
                cumulative_mean[i] = rainfall_series[i]
            else:
                cumulative_mean[i] = np.mean(rainfall_series[:i+1])
        
        # Calculate wetting coefficient
        wetting_coeff = np.zeros_like(rainfall_series)
        
        # Avoid division by zero
        non_zero_intervals = intervals.copy()
        non_zero_intervals[non_zero_intervals == 0] = 1
        
        wetting_coeff = cumulative_mean / non_zero_intervals
        
        return wetting_coeff
        
    def calculate_infiltration_capacity(self, rainfall_series: np.ndarray, intervals: np.ndarray, 
                                        event_indices: List[int]) -> np.ndarray:
        """
        Calculate integrated infiltration capacity
        
        Args:
            rainfall_series (np.ndarray): Array of rainfall values
            intervals (np.ndarray): Array of rainfall intervals
            event_indices (List[int]): Indices of rainfall events
            
        Returns:
            np.ndarray: Infiltration capacity array
        """
        infiltration_capacity = np.zeros_like(rainfall_series)
        
        if not event_indices:
            return infiltration_capacity
            
        # For each event, calculate infiltration capacity
        for i, event_index in enumerate(event_indices):
            # Get the interval for this event
            interval = intervals[event_index]
            
            # Calculate max rainfall for this event
            if i < len(event_indices) - 1:
                event_rainfall = rainfall_series[event_index:event_indices[i+1]]
            else:
                event_rainfall = rainfall_series[event_index:]
                
            if len(event_rainfall) == 0:
                continue
                
            max_rainfall = np.max(event_rainfall)
            
            # Calculate slopes
            slopes = np.zeros_like(event_rainfall)
            for j in range(1, len(event_rainfall)):
                slopes[j] = event_rainfall[j] - event_rainfall[j-1]
                
            # Calculate infiltration capacity
            # Using the formula C_i = e^(-lg(δ)) * R_max * ln(Σ|α|/L)
            if interval > 0 and max_rainfall > 0:
                sum_abs_slopes = np.sum(np.abs(slopes))
                if sum_abs_slopes > 0:
                    term = sum_abs_slopes / len(event_rainfall)
                    if term > 0:
                        capacity = np.exp(-np.log10(interval)) * max_rainfall * np.log(term)
                        
                        # Assign this capacity to all points in this event
                        if i < len(event_indices) - 1:
                            infiltration_capacity[event_index:event_indices[i+1]] = capacity
                        else:
                            infiltration_capacity[event_index:] = capacity
        
        return infiltration_capacity
        
    def calculate_statistical_features(self, rainfall_series: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        """
        Calculate statistical features from rainfall time series
        
        Args:
            rainfall_series (np.ndarray): Array of rainfall values
            
        Returns:
            Tuple[float, float, float, float, float, float]: 
                Mean, Max, Std, Kurtosis, Skewness, AUC
        """
        # Simple statistics
        mean = np.mean(rainfall_series)
        max_val = np.max(rainfall_series)
        std = np.std(rainfall_series)
        
        # Kurtosis
        if std > 0 and len(rainfall_series) > 3:
            n = len(rainfall_series)
            m4 = np.sum((rainfall_series - mean) ** 4) / n
            kurtosis = m4 / (std ** 4) - 3
        else:
            kurtosis = 0
            
        # Skewness
        if std > 0 and len(rainfall_series) > 2:
            n = len(rainfall_series)
            m3 = np.sum((rainfall_series - mean) ** 3) / n
            skewness = m3 / (std ** 3)
        else:
            skewness = 0
            
        # Area Under Curve (simple sum for discrete time series)
        auc = np.sum(rainfall_series)
        
        return mean, max_val, std, kurtosis, skewness, auc
        
    def extract_features_from_window(self, window: np.ndarray, timestamp: pd.Timestamp) -> np.ndarray:
        """
        Extract all features from a rainfall time window
        
        Args:
            window (np.ndarray): Window of rainfall data
            timestamp (pd.Timestamp): Timestamp for the window
            
        Returns:
            np.ndarray: Array of extracted features
        """
        # Get rainfall series from the window
        rainfall_series = window[:, 0]  # Assuming rainfall is the first column
        
        # Get month for seasonality
        month = timestamp.month
        seasonality_coeff = self.get_season_coefficient(month)
        
        # Calculate unit rainfall
        unit_rainfall = self.calculate_unit_rainfall(rainfall_series)
        
        # Calculate rainfall intervals
        intervals, event_indices = self.calculate_rainfall_interval(unit_rainfall)
        
        # Calculate wetting coefficient
        wetting_coeff = self.calculate_wetting_coefficient(rainfall_series, intervals)
        
        # Calculate infiltration capacity
        infiltration_capacity = self.calculate_infiltration_capacity(rainfall_series, intervals, event_indices)
        
        # Calculate statistical features
        mean, max_val, std, kurtosis, skewness, auc = self.calculate_statistical_features(rainfall_series)
        
        # Create feature array
        features = np.array([
            np.mean(unit_rainfall),
            seasonality_coeff,
            np.mean(intervals),
            np.mean(wetting_coeff),
            np.mean(infiltration_capacity),
            mean,
            max_val,
            std,
            kurtosis,
            skewness,
            auc
        ])
        
        return features
        
    def fit(self, X, y=None):
        """
        Fit method (required for sklearn compatibility)
        """
        return self
        
    def transform(self, X, y=None):
        """
        Transform method (required for sklearn compatibility)
        
        Args:
            X: Input data (a tuple from DataProcessor or raw windows)
            y: Target data (optional)
            
        Returns:
            Tuple: Processed X features and y targets
        """
        # Handle different input types
        X_windows = None
        y_windows = None
        timestamps = None
        
        # If X is a tuple from DataProcessor
        if isinstance(X, tuple) and len(X) >= 2:
            X_windows, y_windows = X[:2]
            timestamps = X[2] if len(X) > 2 else [pd.Timestamp.now()] * len(X_windows)
        # If X is raw window data (numpy array)
        elif isinstance(X, np.ndarray):
            if len(X.shape) == 3:  # Expected 3D array (samples, window, features)
                X_windows = X
            elif len(X.shape) == 2:  # Handle 2D array - reshape to 3D
                # Assume it's a single window or needs reshaping
                logger.warning(f"Received 2D array with shape {X.shape}, reshaping to 3D")
                try:
                    # Try to reshape: assume first dimension is samples, second is features
                    # For a single window, create a window size of 1
                    X_windows = X.reshape(X.shape[0], 1, -1)
                except Exception as e:
                    logger.error(f"Failed to reshape input: {str(e)}")
                    # Fallback: create a minimal valid structure
                    X_windows = np.zeros((X.shape[0], 1, 1))
            else:  # Handle other dimensions
                logger.warning(f"Unexpected array shape: {X.shape}, creating dummy structure")
                # Create a minimal valid structure
                X_windows = np.zeros((1, 1, 1))
                
            # Use provided y if available, otherwise create dummy
            if y is not None:
                y_windows = y
            else:
                y_windows = np.zeros(X_windows.shape[0])
            timestamps = [pd.Timestamp.now()] * len(X_windows)
        else:
            # Try to convert to array as last resort
            try:
                logger.warning(f"Received unexpected input type: {type(X)}, attempting conversion")
                X_arr = np.array(X)
                if len(X_arr.shape) >= 2:
                    X_windows = X_arr.reshape(X_arr.shape[0], 1, -1)
                    y_windows = np.zeros(X_arr.shape[0])
                    timestamps = [pd.Timestamp.now()] * X_arr.shape[0]
                else:
                    raise ValueError("Cannot convert input to appropriate format")
            except Exception as e:
                logger.error(f"Conversion failed: {str(e)}")
                raise ValueError("Input must be a tuple from DataProcessor or array data that can be reshaped")
        
        # Make sure X_windows and y_windows are not None
        if X_windows is None or y_windows is None:
            raise ValueError("Failed to process input data")
            
        # Initialize feature array - with safer dimension handling
        n_samples = len(X_windows)
        n_features = len(self.feature_names)
        
        # Determine number of additional features safely
        additional_features = 0
        if len(X_windows.shape) >= 3:
            additional_features = X_windows.shape[2] - 1  # Subtract rainfall
            if additional_features < 0:  # In case only one or zero features
                additional_features = 0
        
        # Total features = extracted rainfall features + additional features
        X_features = np.zeros((n_samples, n_features + additional_features))
        
        # Rest of the method remains the same...