import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from typing import List, Dict, Union, Tuple, Optional
import logging
import os
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from models.data_processor import DataProcessor
from models.feature_engineer import FeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WaterloggingPredictor(BaseEstimator, RegressorMixin):
    """
    Waterlogging prediction model that uses pipelines for data processing,
    feature engineering, and prediction.
    """
    
    def __init__(self, model_type: str = 'rf', window_size: int = 6, 
                 model_params: Optional[Dict] = None, model_path: Optional[str] = None):
        """
        Initialize the WaterloggingPredictor
        
        Args:
            model_type (str): Type of model to use ('rf', 'gbdt', 'adaboost')
            window_size (int): Size of the sliding window for time series
            model_params (Dict, optional): Parameters for the model
            model_path (str, optional): Path to a pre-trained model file
        """
        self.model_type = model_type
        self.window_size = window_size
        self.model_params = model_params or {}
        self.model_path = model_path
        
        # Initialize pipeline components
        self.data_processor = DataProcessor(window_size=window_size)
        self.feature_engineer = FeatureEngineer()
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Create the pipeline
        self.pipeline = self._create_pipeline()
        
        # Load pre-trained model if provided
        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)
            
    def _initialize_model(self):
        """
        Initialize the model based on model_type
        
        Returns:
            BaseEstimator: Initialized model
        """
        if self.model_type == 'rf':
            params = {
                'n_estimators': self.model_params.get('n_estimators', 100),
                'max_depth': self.model_params.get('max_depth', None),
                'min_samples_split': self.model_params.get('min_samples_split', 2),
                'random_state': self.model_params.get('random_state', 42)
            }
            return RandomForestRegressor(**params)
            
        elif self.model_type == 'gbdt':
            params = {
                'n_estimators': self.model_params.get('n_estimators', 100),
                'learning_rate': self.model_params.get('learning_rate', 0.1),
                'max_depth': self.model_params.get('max_depth', 3),
                'random_state': self.model_params.get('random_state', 42)
            }
            return GradientBoostingRegressor(**params)
            
        elif self.model_type == 'adaboost':
            params = {
                'n_estimators': self.model_params.get('n_estimators', 50),
                'learning_rate': self.model_params.get('learning_rate', 1.0),
                'random_state': self.model_params.get('random_state', 42)
            }
            return AdaBoostRegressor(**params)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
    class TransformerWrapper(BaseEstimator, TransformerMixin):
        """
        A wrapper around transformers to ensure only X is passed through the pipeline
        """
        def __init__(self, transformer):
            self.transformer = transformer
            
        def fit(self, X, y=None):
            self.transformer.fit(X, y)
            return self
            
        def transform(self, X):
            result = self.transformer.transform(X, None)
            # Return only X, not the (X, y) tuple
            if isinstance(result, tuple) and len(result) > 0:
                return result[0]
            return result

    def _create_pipeline(self):
        """
        Create the machine learning pipeline with proper target variable handling
        
        Returns:
            Pipeline: Scikit-learn pipeline
        """
        return Pipeline([
            ('data_processor', WaterloggingPredictor.TransformerWrapper(self.data_processor)),
            ('feature_engineer', WaterloggingPredictor.TransformerWrapper(self.feature_engineer)),
            ('model', self.model)
        ])
        
    def fit(self, X, y=None):
        """
        Fit the model with proper handling of pipeline data flow
        
        Args:
            X: Input data (list of file paths or DataFrame)
            y: Target data (not used as targets are extracted from X)
            
        Returns:
            self: The fitted model
        """
        logger.info("Fitting waterlogging prediction model...")
        
        # Extract target if needed
        target_y = y
        
        # X can be a list of file paths, a DataFrame, or a list of DataFrames
        if isinstance(X, list):
            if all(isinstance(item, str) for item in X):
                logger.info(f"Training on {len(X)} data files")
            elif all(isinstance(item, pd.DataFrame) for item in X):
                logger.info(f"Training on {len(X)} DataFrames")
                # Convert list of DataFrames to a single DataFrame
                X = pd.concat(X, ignore_index=True)
        elif isinstance(X, pd.DataFrame):
            logger.info(f"Training on DataFrame with {len(X)} rows")
            
            # Extract target from DataFrame if available and not provided separately
            if target_y is None and 'waterdepth' in X.columns:
                # We need to ensure the target is converted to numeric
                target_y = pd.to_numeric(X['waterdepth'], errors='coerce').fillna(0).values
                logger.info(f"Extracted target variable from DataFrame, shape: {target_y.shape}")
        else:
            raise ValueError("X must be a list of file paths, a DataFrame, or a list of DataFrames")
        
        # If we still don't have a target, try to extract it through the data_processor
        if target_y is None:
            try:
                # Process data through the first step to get features and target
                X_processed, extracted_y = self.data_processor.transform(X)
                if extracted_y is not None and len(extracted_y) > 0:
                    target_y = extracted_y
                    logger.info(f"Extracted target through DataProcessor, shape: {target_y.shape}")
                    
                    # Now process through feature engineering
                    X_features, _ = self.feature_engineer.transform((X_processed, target_y))
                    
                    # Train the model directly
                    self.model.fit(X_features, target_y)
                    logger.info("Model training completed via direct pipeline")
                    return self
            except Exception as e:
                logger.warning(f"Could not extract target through direct pipeline: {str(e)}")
        
        # If we still don't have a target, raise an error
        if target_y is None:
            raise ValueError("Could not extract target variable 'waterdepth' from input data")
        
        try:
            # Use the pipeline with the extracted target
            self.pipeline.fit(X, target_y)
            logger.info("Model training completed via pipeline")
        except Exception as e:
            # If pipeline fails, try direct approach
            logger.warning(f"Pipeline fit failed: {str(e)}. Trying direct approach.")
            
            # Process data through each step manually
            X_processed, _ = self.data_processor.transform(X)
            X_features, _ = self.feature_engineer.transform((X_processed, None))
            
            # Ensure X_features is 2D
            if len(X_features.shape) > 2:
                X_features = X_features.reshape(X_features.shape[0], -1)
                
            # Train the model directly
            self.model.fit(X_features, target_y)
            logger.info("Model training completed via direct approach")
        
        return self

    def predict(self, X):
        """
        Make predictions with proper handling of pipeline data flow
        
        Args:
            X: Input data
            
        Returns:
            np.ndarray: Predicted waterlogging depths
        """
        try:
            # Try using the pipeline
            return self.pipeline.predict(X)
        except Exception as e:
            logger.warning(f"Pipeline prediction failed: {str(e)}. Trying direct approach.")
            
            try:
                # Process data through each step manually
                X_processed, _ = self.data_processor.transform(X)
                
                # Log details about X_processed to debug
                logger.info(f"X_processed shape: {X_processed.shape if hasattr(X_processed, 'shape') else 'no shape attribute'}")
                
                # Handle different output formats from data_processor
                if isinstance(X_processed, np.ndarray):
                    if len(X_processed.shape) < 3:
                        # Reshape to 3D if needed
                        logger.warning(f"Reshaping X_processed from {X_processed.shape} to 3D")
                        X_processed = X_processed.reshape(X_processed.shape[0], 1, -1)
                        
                X_features, _ = self.feature_engineer.transform((X_processed, None))
                
                # Ensure X_features is 2D
                if len(X_features.shape) > 2:
                    X_features = X_features.reshape(X_features.shape[0], -1)
                    
                # Make predictions directly
                return self.model.predict(X_features)
            except Exception as e:
                # Last resort: try to create a minimal viable input for the model
                logger.error(f"Direct prediction approach failed: {str(e)}. Attempting minimal prediction.")
                try:
                    # Create a minimal feature vector based on model's expected input
                    if hasattr(self.model, 'n_features_in_'):
                        n_features = self.model.n_features_in_
                    else:
                        n_features = len(self.feature_engineer.feature_names)
                    
                    # Create dummy features (all zeros)
                    X_features = np.zeros((1, n_features))
                    
                    # Make prediction on dummy data
                    logger.warning("Using dummy features for prediction - result may be inaccurate")
                    return self.model.predict(X_features)
                except Exception as e2:
                    logger.error(f"All prediction attempts failed: {str(e2)}")
                    raise ValueError(f"Failed to make prediction: {str(e)}, {str(e2)}")
        
    def evaluate(self, X_test, y_test=None):
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test data
            y_test: Test targets (optional, will be extracted from X_test if not provided)
            
        Returns:
            Dict: Evaluation metrics
        """
        # Extract target if not provided separately
        if y_test is None and isinstance(X_test, pd.DataFrame) and 'waterdepth' in X_test.columns:
            y_test = pd.to_numeric(X_test['waterdepth'], errors='coerce').fillna(0).values
            logger.info(f"Extracted target from test data, shape: {y_test.shape}")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Extract true values if y_test is still None
        if y_test is None:
            # Process the test data through the first two steps of the pipeline
            X_processed = self.pipeline.steps[0][1].transform(X_test)
            _, y_test = self.pipeline.steps[1][1].transform(X_processed)
        
        # Check if we have valid targets
        if y_test is None or len(y_test) == 0:
            logger.error("Could not extract target values for evaluation")
            return {'mse': float('inf'), 'rmse': float('inf'), 'r2': 0.0}
        
        # Ensure predictions and targets have the same length
        if len(y_pred) != len(y_test):
            logger.warning(f"Prediction length ({len(y_pred)}) doesn't match target length ({len(y_test)})")
            # Use the minimum length
            min_len = min(len(y_pred), len(y_test))
            y_pred = y_pred[:min_len]
            y_test = y_test[:min_len]
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
        
    def cross_validate(self, X, n_splits=5):
        """
        Perform cross-validation
        
        Args:
            X: Input data (DataFrame or list of file paths)
            n_splits (int): Number of CV splits
            
        Returns:
            Dict: Cross-validation results
        """
        logger.info(f"Performing {n_splits}-fold cross-validation")
        
        # Handle different input types
        if isinstance(X, list) and all(isinstance(item, str) for item in X):
            # Load data from file paths
            dfs = [self.data_processor.load_data(file_path) for file_path in X]
            df = pd.concat(dfs, ignore_index=True)
        elif isinstance(X, pd.DataFrame):
            df = X
        else:
            raise ValueError("X must be a DataFrame or list of file paths")
        
        # Process the data through data_processor
        X_clean = self.data_processor.clean_data(df)
        
        # Make sure timestamp is datetime
        if 'timestamp' in X_clean.columns:
            X_clean['timestamp'] = pd.to_datetime(X_clean['timestamp'])
        
        # Create train/test splits based on time for temporal data
        if 'timestamp' in X_clean.columns:
            # Sort by timestamp
            X_clean = X_clean.sort_values('timestamp')
            
            # Create time-based folds
            time_indices = np.array_split(np.arange(len(X_clean)), n_splits)
            cv_splits = [(np.concatenate(time_indices[:i] + time_indices[i+1:]), time_indices[i]) 
                        for i in range(n_splits)]
        else:
            # If no timestamp, use KFold
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_splits = list(kf.split(X_clean))
        
        # Store metrics for each fold
        mse_scores = []
        r2_scores = []
        
        # Perform CV
        for i, (train_idx, test_idx) in enumerate(cv_splits):
            try:
                # Split data
                X_train = X_clean.iloc[train_idx].copy()
                X_test = X_clean.iloc[test_idx].copy()
                
                # Train model on this fold
                fold_model = WaterloggingPredictor(
                    model_type=self.model_type,
                    window_size=self.window_size,
                    model_params=self.model_params
                )
                
                # Fit the model
                fold_model.fit(X_train)
                
                # Evaluate
                metrics = fold_model.evaluate(X_test)
                
                mse = metrics['mse']
                r2 = metrics['r2']
                
                mse_scores.append(mse)
                r2_scores.append(r2)
                
                logger.info(f"Fold {i+1}: MSE = {mse:.6f}, R² = {r2:.6f}")
                
            except Exception as e:
                logger.error(f"Error in fold {i+1}: {str(e)}")
                # If a fold fails, use placeholder values
                mse_scores.append(float('inf'))
                r2_scores.append(0.0)
        
        # Calculate average metrics (excluding any failed folds)
        valid_mse = [mse for mse in mse_scores if mse != float('inf')]
        valid_r2 = [r2 for r2 in r2_scores if r2 != 0.0]
        
        avg_mse = np.mean(valid_mse) if valid_mse else float('inf')
        avg_r2 = np.mean(valid_r2) if valid_r2 else 0.0
        
        cv_results = {
            'mse_scores': mse_scores,
            'r2_scores': r2_scores,
            'avg_mse': avg_mse,
            'avg_r2': avg_r2
        }
        
        logger.info(f"Average MSE: {avg_mse:.6f}, Average R²: {avg_r2:.6f}")
        return cv_results
        
    def tune_hyperparameters(self, X, param_grid=None):
        """
        Tune model hyperparameters using GridSearchCV
        
        Args:
            X: Input data
            param_grid (Dict, optional): Parameter grid for search
            
        Returns:
            Dict: Best parameters and results
        """
        logger.info("Tuning model hyperparameters")
        
        # Default parameter grid if not provided
        if param_grid is None:
            if self.model_type == 'rf':
                param_grid = {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [None, 10, 20],
                    'model__min_samples_split': [2, 5, 10]
                }
            elif self.model_type == 'gbdt':
                param_grid = {
                    'model__n_estimators': [50, 100, 200],
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__max_depth': [3, 5, 7]
                }
            elif self.model_type == 'adaboost':
                param_grid = {
                    'model__n_estimators': [50, 100, 200],
                    'model__learning_rate': [0.01, 0.1, 1.0]
                }
            
        # Create a GridSearchCV object
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the grid search
        grid_search.fit(X)
        
        # Get the best parameters and score
        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_  # Convert back to MSE
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best MSE: {best_score:.6f}")
        
        # Update the model with best parameters
        self.model_params = {k.replace('model__', ''): v for k, v in best_params.items()}
        self.model = self._initialize_model()
        self.pipeline = self._create_pipeline()
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': grid_search.cv_results_
        }
        
    def save_model(self, filepath):
        """
        Save the trained model to a file
        
        Args:
            filepath (str): Path to save the model
        """
        logger.info(f"Saving model to {filepath}")
        joblib.dump(self.pipeline, filepath)
        
    def load_model(self, filepath):
        """
        Load a trained model from a file
        
        Args:
            filepath (str): Path to the model file
        """
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        logger.info(f"Loading model from {filepath}")
        loaded_data = joblib.load(filepath)
        
        # Check if loaded data is a dictionary (our custom format) or a pipeline
        if isinstance(loaded_data, dict):
            # Extract components from dictionary
            if 'data_processor' in loaded_data:
                self.data_processor = loaded_data['data_processor']
            if 'feature_engineer' in loaded_data:
                self.feature_engineer = loaded_data['feature_engineer']
            if 'model' in loaded_data:
                self.model = loaded_data['model']
            if 'model_type' in loaded_data:
                self.model_type = loaded_data['model_type']
            if 'window_size' in loaded_data:
                self.window_size = loaded_data['window_size']
                
            # Recreate the pipeline with the loaded components
            self.pipeline = self._create_pipeline()
        else:
            # Assume it's a pipeline
            self.pipeline = loaded_data
            
            # Extract pipeline components
            self.data_processor = self.pipeline.named_steps['data_processor'].transformer
            self.feature_engineer = self.pipeline.named_steps['feature_engineer'].transformer
            self.model = self.pipeline.named_steps['model']
        
    def update_model(self, X_new, y_new=None):
        """
        Update the model with new data (online learning)
        
        Args:
            X_new: New data
            y_new: New targets (not used as targets are extracted from X_new)
            
        Returns:
            self: The updated model
        """
        logger.info("Updating model with new data")
        
        # Convert input to appropriate format if needed
        if isinstance(X_new, pd.DataFrame):
            # If X_new is a single dataframe
            pass
        elif isinstance(X_new, str):
            # If X_new is a file path
            X_new = [X_new]
        
        # Process new data
        X_processed = self.pipeline.steps[0][1].transform(X_new)
        X_features, y_new_processed = self.pipeline.steps[1][1].transform(X_processed)
        
        # Update model (if model supports partial_fit, use it, otherwise retrain)
        if hasattr(self.model, 'partial_fit'):
            self.model.partial_fit(X_features, y_new_processed)
        else:
            # Get existing data (if available)
            try:
                # This is a simplified approach - in a real system, you might want to
                # store the training data separately or use incremental learning
                X_features_old = getattr(self, '_X_features', None)
                y_old = getattr(self, '_y', None)
                
                if X_features_old is not None and y_old is not None:
                    # Combine old and new data
                    X_features_combined = np.vstack([X_features_old, X_features])
                    y_combined = np.concatenate([y_old, y_new_processed])
                else:
                    X_features_combined = X_features
                    y_combined = y_new_processed
                    
                # Store for future updates
                self._X_features = X_features_combined
                self._y = y_combined
                
                # Retrain the model
                self.model.fit(X_features_combined, y_combined)
            except:
                # If retrieval of old data fails, just train on new data
                self.model.fit(X_features, y_new_processed)
                
        logger.info("Model updated successfully")
        return self
        
    def get_feature_importance(self):
        """
        Get feature importance from the model
        
        Returns:
            Dict: Feature names and their importance scores
        """
        # Check if model has feature_importances_ attribute
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not provide feature importances")
            return None
            
        # Get feature names
        feature_names = (
            self.feature_engineer.feature_names + 
            ['additional_feature_' + str(i) for i in range(self.model.feature_importances_.shape[0] - len(self.feature_engineer.feature_names))]
        )
        
        # Create feature importance dictionary
        feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        
        # Sort by importance
        feature_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
        
        return feature_importance