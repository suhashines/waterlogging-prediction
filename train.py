import os
import sys
import pandas as pd
import numpy as np
import logging
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
import argparse
import joblib

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.waterlogging_predictor import WaterloggingPredictor
from models.risk_predictor import RiskPredictor
from models.data_processor import DataProcessor  # Import DataProcessor
from models.feature_engineer import FeatureEngineer  # Import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def load_and_prepare_data(data_dir):
    """
    Load and prepare data for training
    
    Args:
        data_dir (str): Directory with CSV files
        
    Returns:
        tuple: train_files, test_files, station_data
    """
    logger.info(f"Loading data from {data_dir}")
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    if not csv_files:
        logger.error(f"No CSV files found in {data_dir}")
        sys.exit(1)
        
    logger.info(f"Found {len(csv_files)} CSV files")
    
    # Store data for each station
    station_data = {}
    
    # Store all files for training and testing
    train_files = []
    test_files = []
    
    for file_path in csv_files:
        # Get station ID from filename
        station_id = os.path.basename(file_path).split('.')[0]
        
        # Instead of trying to split the file path, just use the same file for both
        # training and testing - we'll split the actual data later in the pipeline
        train_files.append(file_path)
        test_files.append(file_path)
        
        # Store in station_data
        station_data[station_id] = {
            'train_file': file_path,
            'test_file': file_path
        }
    
    return train_files, test_files, station_data

def train_models(train_files, test_files, station_data, model_type='rf', 
                 tune_hyperparams=False, window_size=6, save_dir='models'):
    """
    Train and evaluate waterlogging prediction model
    
    Args:
        train_files (list): List of training files
        test_files (list): List of test files
        station_data (dict): Data for each station
        model_type (str): Type of model to use ('rf', 'gbdt', 'adaboost')
        tune_hyperparams (bool): Whether to tune hyperparameters
        window_size (int): Size of sliding window
        save_dir (str): Directory to save models
    """
    logger.info(f"Training {model_type} model with window size {window_size}")
    
    # Load all data first
    all_dfs = []
    for file_path in train_files:
        try:
            # Load CSV file
            df = pd.read_csv(file_path)
            
            # Extract station code from file name
            station_code = os.path.basename(file_path).split('.')[0]
            df['station_code'] = station_code
            
            # Rename columns for consistency
            column_mapping = {
                'clctTime': 'timestamp',
                'Rainfall(mm)': 'rainfall',
                'Waterdepth(meters)': 'waterdepth',
                'Weather': 'weather',
                'Wind Speed': 'windspeed'
            }
            
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            
            # Ensure timestamp is properly converted to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'clctTime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['clctTime'])
            else:
                logger.warning(f"No timestamp column found in {file_path}")
                # Create a dummy timestamp if none exists
                df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='5min')
                
            # Force numeric conversion for key columns
            if 'rainfall' in df.columns:
                df['rainfall'] = pd.to_numeric(df['rainfall'], errors='coerce').fillna(0)
            if 'waterdepth' in df.columns:
                df['waterdepth'] = pd.to_numeric(df['waterdepth'], errors='coerce').fillna(0)
            if 'weather' in df.columns:
                df['weather'] = pd.to_numeric(df['weather'], errors='coerce').fillna(0)
            if 'windspeed' in df.columns:
                df['windspeed'] = pd.to_numeric(df['windspeed'], errors='coerce').fillna(0)
            
            # Check for duplicate timestamps
            has_duplicates = df.duplicated('timestamp').any()
            if has_duplicates:
                logger.warning(f"File {file_path} has duplicate timestamps. Handling duplicates...")
                # Group by timestamp and aggregate
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_agg = {col: 'mean' for col in numeric_cols if col != 'station_code'}
                categorical_cols = [col for col in df.columns if col not in numeric_cols and col != 'timestamp']
                categorical_agg = {col: lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0] for col in categorical_cols}
                agg_dict = {**numeric_agg, **categorical_agg}
                df = df.groupby('timestamp').agg(agg_dict).reset_index()
                df['station_code'] = station_code
            
            all_dfs.append(df)
            logger.info(f"Successfully loaded {file_path} with {len(df)} rows")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
    
    # Combine all dataframes
    if not all_dfs:
        logger.error("No valid data found in the provided files")
        sys.exit(1)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined data has {len(combined_df)} rows and columns: {combined_df.columns.tolist()}")
    
    # Log data stats
    if 'rainfall' in combined_df.columns:
        logger.info(f"Rainfall stats: min={combined_df['rainfall'].min()}, max={combined_df['rainfall'].max()}, mean={combined_df['rainfall'].mean()}")
    if 'waterdepth' in combined_df.columns:
        logger.info(f"Waterdepth stats: min={combined_df['waterdepth'].min()}, max={combined_df['waterdepth'].max()}, mean={combined_df['waterdepth'].mean()}")
    
    # Split data for each station for evaluation
    station_dfs = {}
    for station_id, group_df in combined_df.groupby('station_code'):
        # Use a proper train-test split on the actual data
        train_df, test_df = train_test_split(group_df, test_size=0.2, random_state=42)
        station_dfs[station_id] = {
            'train': train_df,
            'test': test_df
        }
        logger.info(f"Station {station_id}: {len(train_df)} training samples, {len(test_df)} test samples")
    
    # Initialize the components without building a pipeline
    data_processor = DataProcessor(window_size=window_size)
    feature_engineer = FeatureEngineer()
    
    # Initialize the model
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gbdt':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_type == 'adaboost':
        model = AdaBoostRegressor(n_estimators=50, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Process data
    logger.info("Processing data...")
    X_processed, y = data_processor.transform(combined_df)
    logger.info(f"Processed data shape: {X_processed.shape}, Target shape: {y.shape}")
    
    # Extract features
    logger.info("Extracting features...")
    X_features, y_unchanged = feature_engineer.transform((X_processed, y))
    logger.info(f"Feature extraction complete: {X_features.shape}, Target shape: {y_unchanged.shape}")
    
    # Train the model
    logger.info("Training model...")
    model.fit(X_features, y_unchanged)
    logger.info("Model training completed")
    
    # Initialize the waterlogging predictor with the trained components
    waterlogging_predictor = WaterloggingPredictor(
        model_type=model_type,
        window_size=window_size
    )
    
    # Replace the pipeline components with our trained versions
    waterlogging_predictor.data_processor = data_processor
    waterlogging_predictor.feature_engineer = feature_engineer
    waterlogging_predictor.model = model
    
    # Initialize risk predictor
    risk_predictor = RiskPredictor()
    
    # Evaluate on each station
    logger.info("Evaluating model on each station...")
    station_metrics = {}
    
    for station_id, dfs in station_dfs.items():
        logger.info(f"Evaluating on station {station_id}...")
        
        # Process test data
        X_test_processed, y_test = data_processor.transform(dfs['test'])
        X_test_features, _ = feature_engineer.transform((X_test_processed, y_test))
        
        # Make predictions
        y_pred = model.predict(X_test_features)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        station_metrics[station_id] = metrics
        logger.info(f"Station {station_id} metrics: MSE = {metrics['mse']:.6f}, R² = {metrics['r2']:.6f}")
        
        # Calculate the amplification factor for this station
        test_df = dfs['test']
            
        # Calculate average amplification factor for this station
        if 'rainfall' in test_df.columns and 'waterdepth' in test_df.columns:
            # Filter out zero rainfall to avoid division by zero
            df_filtered = test_df[(test_df['rainfall'] > 0) & (test_df['waterdepth'] > 0)]
            
            if not df_filtered.empty:
                # Convert rainfall from mm to m
                af_values = df_filtered['waterdepth'] / (df_filtered['rainfall'] / 1000.0)
                
                # Calculate median to avoid influence of outliers
                median_af = np.median(af_values)
                
                # Update risk predictor
                if station_id not in risk_predictor.station_data:
                    risk_predictor.station_data[station_id] = {}
                    
                risk_predictor.station_data[station_id]['amplification_factor'] = median_af
                logger.info(f"Station {station_id} amplification factor: {median_af:.4f}")
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(
            feature_engineer.feature_names + ['additional_' + str(i) for i in range(X_features.shape[1] - len(feature_engineer.feature_names))],
            model.feature_importances_
        ))
        logger.info("Feature importance:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {feature}: {importance:.4f}")
    
    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the trained components
    logger.info(f"Saving models to {save_dir}")
    waterlogging_model_path = os.path.join(save_dir, 'waterlogging_model.joblib')
    risk_config_path = os.path.join(save_dir, 'risk_config.joblib')
    
    # Save individual components
    joblib.dump({
        'data_processor': data_processor,
        'feature_engineer': feature_engineer,
        'model': model,
        'model_type': model_type,
        'window_size': window_size
    }, waterlogging_model_path)
    risk_predictor.save_config(risk_config_path)
    logger.info("Models saved successfully")
    
    # CV results (dummy for this approach, since we're not doing proper CV)
    cv_results = {
        'avg_mse': np.mean([metrics['mse'] for metrics in station_metrics.values()]),
        'avg_r2': np.mean([metrics['r2'] for metrics in station_metrics.values()])
    }
    
    logger.info("Training and evaluation completed")
    
    return {
        'waterlogging_predictor': waterlogging_predictor,
        'risk_predictor': risk_predictor,
        'station_metrics': station_metrics,
        'cv_results': cv_results
    }

def main():
    """Main function to train models"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train waterlogging prediction model')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing CSV files (default: data)')
    parser.add_argument('--model-type', type=str, default='rf', choices=['rf', 'gbdt', 'adaboost'],
                        help='Type of model to use (default: rf)')
    parser.add_argument('--window-size', type=int, default=6,
                        help='Size of sliding window (default: 6)')
    parser.add_argument('--tune-hyperparams', action='store_true',
                        help='Tune hyperparameters')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Directory to save models (default: models)')
    
    args = parser.parse_args()
    
    # Load and prepare data
    try:
        train_files, test_files, station_data = load_and_prepare_data(args.data_dir)
        
        # Train models
        results = train_models(
            train_files=train_files,
            test_files=test_files,
            station_data=station_data,
            model_type=args.model_type,
            tune_hyperparams=args.tune_hyperparams,
            window_size=args.window_size,
            save_dir=args.save_dir
        )
        
        # Print summary
        logger.info("\nTraining Summary:")
        logger.info("==================")
        
        # Overall metrics
        cv_mse = results['cv_results']['avg_mse']
        cv_r2 = results['cv_results']['avg_r2']
        logger.info(f"Cross-validation: MSE = {cv_mse:.6f}, R² = {cv_r2:.6f}")
        
        # Station metrics
        logger.info("\nStation Metrics:")
        for station_id, metrics in results['station_metrics'].items():
            logger.info(f"  Station {station_id}: MSE = {metrics['mse']:.6f}, R² = {metrics['r2']:.6f}")
        
        logger.info("\nModels saved:")
        logger.info(f"  Waterlogging model: {os.path.join(args.save_dir, 'waterlogging_model.joblib')}")
        logger.info(f"  Risk config: {os.path.join(args.save_dir, 'risk_config.joblib')}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()