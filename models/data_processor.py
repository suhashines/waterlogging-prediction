import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Union, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor(BaseEstimator, TransformerMixin):
    """
    Data processing class for waterlogging prediction
    
    This class handles:
    1. Loading and cleaning data
    2. Interpolating missing values
    3. Creating time slices for time series modeling
    """
    
    def __init__(self, window_size: int = 3, step_size: int = 1):
        """
        Initialize the DataProcessor
        
        Args:
            window_size (int): Size of the sliding window for time series
            step_size (int): Step size for sliding the window
        """
        self.window_size = window_size
        self.step_size = step_size
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file with improved error handling
        
        Args:
            file_path (str): Path to the CSV file
                
        Returns:
            pd.DataFrame: Loaded and cleaned data
        """
        try:
            # Load data
            df = pd.read_csv(file_path)
            
            # Extract station code from file name
            station_code = file_path.split('/')[-1].split('.')[0]
            df['station_code'] = station_code
            
            # Convert timestamp to datetime
            if 'clctTime' in df.columns:
                df['clctTime'] = pd.to_datetime(df['clctTime'], errors='coerce')
                    
            # Sort by timestamp
            if 'clctTime' in df.columns:
                df = df.sort_values(by='clctTime')
                
            # Rename columns for consistency
            column_mapping = {
                'clctTime': 'timestamp',
                'Rainfall(mm)': 'rainfall',
                'Waterdepth(meters)': 'waterdepth',
                'Weather': 'weather',
                'Wind Speed': 'windspeed',
                'clctCode': 'station_code'
            }
            
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            
            # Force numeric conversion for key columns
            if 'rainfall' in df.columns:
                df['rainfall'] = pd.to_numeric(df['rainfall'], errors='coerce')
                
            if 'waterdepth' in df.columns:
                df['waterdepth'] = pd.to_numeric(df['waterdepth'], errors='coerce')
                
            if 'windspeed' in df.columns:
                df['windspeed'] = pd.to_numeric(df['windspeed'], errors='coerce')
                
            if 'weather' in df.columns:
                df['weather'] = pd.to_numeric(df['weather'], errors='coerce')
            
            # Drop rows with invalid timestamps
            if 'timestamp' in df.columns:
                df = df.dropna(subset=['timestamp'])
            
            logger.info(f"Successfully loaded data from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise

    def create_sliding_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows from time series data
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (input features) and y (target) arrays
        """
        # Extract input columns
        input_cols = ['rainfall']
        if 'weather' in df.columns:
            input_cols.append('weather')
        if 'windspeed' in df.columns:
            input_cols.append('windspeed')
            
        # Also include station_code as a feature
        if 'station_code' in df.columns:
            input_cols.append('station_code')
            
        # Ensure all values are numeric
        X_data_list = []
        for col in input_cols:
            if col == 'station_code':
                # Convert station_code to numeric representation
                X_data_list.append(df[col].astype(str).apply(lambda x: float(x) if x.isdigit() else float(hash(x) % 1000) / 1000).values.reshape(-1, 1))
            else:
                # Force numeric conversion for other columns
                X_data_list.append(pd.to_numeric(df[col], errors='coerce').fillna(0).values.reshape(-1, 1))
        
        # Combine all columns
        X_data = np.hstack(X_data_list)
        
        # Check if waterdepth exists (for training) or create dummy values (for prediction)
        if 'waterdepth' in df.columns:
            # Force numeric conversion for target
            y_data = pd.to_numeric(df['waterdepth'], errors='coerce').fillna(0).values
        else:
            # During prediction, create a dummy target of zeros
            y_data = np.zeros(len(df))
            logger.info("Target column 'waterdepth' not found - using dummy values for prediction")
        
        # Create sliding windows
        X_windows = []
        y_windows = []
        
        for i in range(0, len(df) - self.window_size + 1, self.step_size):
            X_windows.append(X_data[i:i+self.window_size])
            # Target is the water depth at the end of the window
            y_windows.append(y_data[i+self.window_size-1])
            
        return np.array(X_windows), np.array(y_windows)
            
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling missing values and outliers
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Check for missing values
        missing_values = df_clean.isnull().sum()
        if missing_values.sum() > 0:
            logger.info(f"Missing values detected: {missing_values}")
        
        # Handle missing values in rainfall data
        if 'rainfall' in df_clean.columns:
            df_clean['rainfall'] = df_clean['rainfall'].fillna(0)
            
        # Handle missing values in weather data using mode
        if 'weather' in df_clean.columns:
            df_clean['weather'] = df_clean['weather'].fillna(df_clean['weather'].mode()[0])
            
        # Handle missing values in wind speed using median
        if 'windspeed' in df_clean.columns:
            df_clean['windspeed'] = df_clean['windspeed'].fillna(df_clean['windspeed'].median())
            
        # Handle missing values in water depth
        if 'waterdepth' in df_clean.columns:
            # First, forward fill to propagate last valid observation
            df_clean['waterdepth'] = df_clean['waterdepth'].ffill()
            # Then, backward fill to handle missing values at the beginning
            df_clean['waterdepth'] = df_clean['waterdepth'].bfill()
            # If there's still missing values, fill with 0
            df_clean['waterdepth'] = df_clean['waterdepth'].fillna(0)
        
        # Remove any remaining rows with missing values
        df_clean = df_clean.dropna()
        
        # Check for outliers in rainfall data (values significantly higher than normal)
        if 'rainfall' in df_clean.columns:
            # Define threshold as 3x the 99th percentile
            threshold = 3 * df_clean['rainfall'].quantile(0.99)
            outliers = df_clean[df_clean['rainfall'] > threshold]
            
            if not outliers.empty:
                logger.info(f"Found {len(outliers)} outliers in rainfall data")
                # Cap the outliers at the threshold value
                df_clean.loc[df_clean['rainfall'] > threshold, 'rainfall'] = threshold
        
        return df_clean
        
    def interpolate_time_series(self, df: pd.DataFrame, freq: str = '5min') -> pd.DataFrame:
        """
        Resample the time series to a uniform frequency and interpolate missing values
        
        Args:
            df (pd.DataFrame): Input dataframe
            freq (str): Frequency for resampling (default: '5min')
            
        Returns:
            pd.DataFrame: Resampled and interpolated dataframe
        """
        # Make a copy
        df_resampled = df.copy()
        
        # Ensure timestamp column exists
        if 'timestamp' not in df_resampled.columns:
            if 'clctTime' in df_resampled.columns:
                df_resampled['timestamp'] = pd.to_datetime(df_resampled['clctTime'])
            else:
                raise ValueError("No timestamp column found in data")
        
        # Ensure timestamp is a datetime type
        df_resampled['timestamp'] = pd.to_datetime(df_resampled['timestamp'])
        
        # Get numeric columns
        numeric_cols = df_resampled.select_dtypes(include=[np.number]).columns.tolist()
        
        # For each station, resample separately
        resampled_dfs = []
        for station_code, group in df_resampled.groupby('station_code'):
            # Make a copy of the group
            group_copy = group.copy()
            
            # Check for duplicate timestamps
            has_duplicates = group_copy.duplicated('timestamp').any()
            if has_duplicates:
                logger.warning(f"Found duplicate timestamps for station {station_code}. Handling duplicates...")
                
                # Group by timestamp and aggregate
                # For numeric columns, take the mean
                # For categorical columns, take the first value
                numeric_agg = {col: 'mean' for col in numeric_cols if col in group_copy.columns and col != 'station_code'}
                
                # For categorical columns, take the most common value
                categorical_cols = [col for col in group_copy.columns if col not in numeric_cols and col != 'timestamp']
                categorical_agg = {col: lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0] for col in categorical_cols}
                
                # Combine aggregations
                agg_dict = {**numeric_agg, **categorical_agg}
                
                # Apply aggregation to remove duplicates
                group_copy = group_copy.groupby('timestamp').agg(agg_dict).reset_index()
                
                # Add back station_code
                group_copy['station_code'] = station_code
            
            # Now we should have unique timestamps
            # Set timestamp as index
            group_copy = group_copy.set_index('timestamp')
            
            # Resample numeric columns
            try:
                # Get only numeric columns (excluding station_code which might be numeric)
                numeric_data_cols = [col for col in numeric_cols if col in group_copy.columns and col != 'station_code']
                
                if numeric_data_cols:
                    numeric_data = group_copy[numeric_data_cols].copy()
                    
                    # Resample and interpolate
                    resampled_numeric = numeric_data.resample(freq).asfreq()
                    resampled_numeric = resampled_numeric.interpolate(method='linear')
                    
                    # For categorical columns, forward fill
                    categorical_cols = [col for col in group_copy.columns if col not in numeric_cols]
                    
                    if categorical_cols:
                        categorical_data = group_copy[categorical_cols].copy()
                        resampled_categorical = categorical_data.resample(freq).ffill().bfill()
                        
                        # Combine numeric and categorical
                        resampled_group = pd.concat([resampled_numeric, resampled_categorical], axis=1)
                    else:
                        resampled_group = resampled_numeric
                    
                    # Add station_code as a column if it's not already there
                    if 'station_code' not in resampled_group.columns:
                        resampled_group['station_code'] = station_code
                    
                    resampled_dfs.append(resampled_group)
                else:
                    # If no numeric columns, just resample with forward fill
                    resampled_group = group_copy.resample(freq).ffill().bfill()
                    resampled_dfs.append(resampled_group)
            except Exception as e:
                logger.error(f"Error resampling station {station_code}: {str(e)}")
                # If resampling fails, just use the original group
                resampled_dfs.append(group_copy)
        
        # Combine all resampled dataframes
        if resampled_dfs:
            df_resampled = pd.concat(resampled_dfs)
        
        # Reset the index to get timestamp back as a column
        df_resampled = df_resampled.reset_index()
        
        return df_resampled
        
    # def create_sliding_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Create sliding windows from time series data
        
    #     Args:
    #         df (pd.DataFrame): Input dataframe
            
    #     Returns:
    #         Tuple[np.ndarray, np.ndarray]: X (input features) and y (target) arrays
    #     """
    #     # Extract input and target columns
    #     input_cols = ['rainfall']
    #     if 'weather' in df.columns:
    #         input_cols.append('weather')
    #     if 'windspeed' in df.columns:
    #         input_cols.append('windspeed')
            
    #     # Also include station_code as a feature
    #     if 'station_code' in df.columns:
    #         input_cols.append('station_code')
            
    #     X_data = df[input_cols].values
        
    #     if 'waterdepth' not in df.columns:
    #         logger.error("Target column 'waterdepth' not found in data")
    #         raise ValueError("Target column 'waterdepth' not found in data")
            
    #     y_data = df['waterdepth'].values
        
    #     # Create sliding windows
    #     X_windows = []
    #     y_windows = []
        
    #     for i in range(0, len(df) - self.window_size + 1, self.step_size):
    #         X_windows.append(X_data[i:i+self.window_size])
    #         # Target is the water depth at the end of the window
    #         y_windows.append(y_data[i+self.window_size-1])
            
    #     return np.array(X_windows), np.array(y_windows)
        
    def fit(self, X, y=None):
        """
        Fit method (required for sklearn compatibility)
        """
        return self
        
    def transform(self, X, y=None):
        """
        Transform method (required for sklearn compatibility)
        
        Args:
            X: Input data (a list of file paths, a DataFrame, or list of DataFrames)
            y: Target data (optional)
            
        Returns:
            Tuple: Processed X and y data
        """
        # Keep the original y
        original_y = y
        
        # If X is a list of file paths, load the data
        if isinstance(X, list) and all(isinstance(item, str) for item in X):
            dfs = [self.load_data(file_path) for file_path in X]
            df = pd.concat(dfs, ignore_index=True)
        elif isinstance(X, list) and all(isinstance(item, pd.DataFrame) for item in X):
            # List of DataFrames
            df = pd.concat(X, ignore_index=True)
        elif isinstance(X, pd.DataFrame):
            df = X
        elif isinstance(X, np.ndarray):
            # If X is already a processed window, just return it with the provided y or a dummy y
            if len(X.shape) == 3:  # Shape: (n_samples, window_size, n_features)
                # Use the provided y if available, otherwise create a dummy y
                if original_y is None:
                    dummy_y = np.zeros(X.shape[0])
                    return X, dummy_y
                else:
                    return X, original_y
        else:
            raise ValueError("Input must be a DataFrame, a list of file paths, or a list of DataFrames")
            
        # Clean the data
        df_clean = self.clean_data(df)
        
        # Extract target if it exists in the dataframe and wasn't provided separately
        if original_y is None and 'waterdepth' in df_clean.columns:
            extracted_y = pd.to_numeric(df_clean['waterdepth'], errors='coerce').fillna(0).values
            logger.info(f"Extracted target from DataFrame in transform method, shape: {extracted_y.shape}")
        else:
            extracted_y = original_y  # Use the provided y or keep it as None
        
        # Interpolate
        df_interpolated = self.interpolate_time_series(df_clean)
        
        # Create sliding windows
        X_windows, sliding_y = self.create_sliding_windows(df_interpolated)
        
        # If y was provided or extracted, and has the right length, use it instead
        if extracted_y is not None and len(extracted_y) == len(sliding_y):
            sliding_y = extracted_y
        
        return X_windows, sliding_y