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
        elif isinstance(X, np.ndarray) and len(X.shape) == 3:
            X_windows = X
            # Use provided y if available, otherwise create dummy
            if y is not None:
                y_windows = y
            else:
                y_windows = np.zeros(X_windows.shape[0])
            timestamps = [pd.Timestamp.now()] * len(X_windows)
        else:
            raise ValueError("Input must be a tuple from DataProcessor or raw window data (numpy array)")
        
        # Make sure y_windows is not None
        if y_windows is None:
            if y is not None:
                y_windows = y
            else:
                logger.warning("No target variable provided or extracted. Using dummy targets.")
                y_windows = np.zeros(len(X_windows))
                
        # Initialize feature array
        n_samples = len(X_windows)
        n_features = len(self.feature_names)
        
        # If X_windows has additional features beyond rainfall (like weather, windspeed)
        # we need to include them
        additional_features = X_windows.shape[2] - 1  # Subtract rainfall
        
        # Total features = extracted rainfall features + additional features
        X_features = np.zeros((n_samples, n_features + additional_features))
        
        # Extract features for each window
        for i in range(n_samples):
            try:
                # Extract rainfall features - ensure rainfall values are numeric
                rainfall_series = X_windows[i, :, 0].astype(float)
                rainfall_features = self.extract_features_from_window(
                    np.column_stack((rainfall_series, np.zeros((len(rainfall_series), X_windows.shape[2]-1)))), 
                    timestamps[i]
                )
                X_features[i, :n_features] = rainfall_features
                
                # Include additional features (average over the window)
                if additional_features > 0:
                    for j in range(additional_features):
                        try:
                            # Try to convert to float - if it fails, use a default value
                            values = X_windows[i, :, j+1]
                            
                            # Check if values can be converted to float
                            try:
                                numeric_values = np.array(values, dtype=float)
                                # If successful, calculate mean of numeric values
                                # Handle NaN values by replacing with 0
                                numeric_values = np.nan_to_num(numeric_values, nan=0.0)
                                mean_value = np.mean(numeric_values)
                                X_features[i, n_features + j] = mean_value
                            except (ValueError, TypeError):
                                # If values can't be converted to float, try mode for categorical
                                # First convert to strings to ensure compatibility
                                str_values = [str(v) for v in values]
                                # Find most common value
                                from collections import Counter
                                most_common = Counter(str_values).most_common(1)
                                if most_common:
                                    # Use a simple hash of the string as a numeric representation
                                    # This is a simplistic approach but should work for basic categoricals
                                    X_features[i, n_features + j] = hash(most_common[0][0]) % 1000 / 1000.0
                                else:
                                    # Default value if all else fails
                                    X_features[i, n_features + j] = 0.0
                        except Exception as e:
                            logger.warning(f"Error processing feature {j+1}: {str(e)}")
                            X_features[i, n_features + j] = 0.0  # Default value
            except Exception as e:
                logger.error(f"Error extracting features for sample {i}: {str(e)}")
                # Set default values
                X_features[i, :] = 0.0
        
        logger.info(f"Feature engineering completed: {X_features.shape}, target shape: {y_windows.shape}")
        return X_features, y_windows