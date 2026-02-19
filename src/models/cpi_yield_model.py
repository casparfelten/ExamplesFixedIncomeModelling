"""CPI-Bond Yield prediction models"""

from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CPIBondYieldModel:
    """
    Model for predicting bond yield changes from CPI shocks and background variables.
    
    Supports multiple model types: linear, Ridge, Lasso.
    """
    
    def __init__(
        self,
        model_type: str = "ridge",
        alpha: float = 1.0,
        feature_names: Optional[List[str]] = None,
        scale_features: bool = True
    ):
        """
        Initialize CPI-Bond Yield model.
        
        Args:
            model_type: Type of model ('linear', 'ridge', 'lasso')
            alpha: Regularization strength (for Ridge/Lasso)
            feature_names: List of feature names to use (if None, uses default set)
            scale_features: Whether to standardize features
        """
        self.model_type = model_type.lower()
        self.alpha = alpha
        self.scale_features = scale_features
        self.feature_names = feature_names or self._get_default_features()
        
        # Initialize model
        if self.model_type == "linear":
            self.model = LinearRegression()
        elif self.model_type == "ridge":
            self.model = Ridge(alpha=alpha)
        elif self.model_type == "lasso":
            self.model = Lasso(alpha=alpha, max_iter=10000)
        elif self.model_type == "elasticnet":
            self.model = ElasticNet(alpha=alpha, max_iter=10000)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'linear', 'ridge', 'lasso', or 'elasticnet'")
        
        self.scaler = StandardScaler() if scale_features else None
        self.is_fitted = False
        
        logger.info(f"Initialized {self.model_type} model with alpha={alpha}")
        logger.info(f"Features: {self.feature_names}")
    
    def _get_default_features(self) -> List[str]:
        """Get default feature set."""
        return [
            'cpi_shock_mom',
            'cpi_shock_yoy',
            'cpi_shock_magnitude',  # Magnitude of CPI shock (helps catch large moves)
            'yield_lagged',  # Lagged yield for momentum
            'yield_volatility',  # Recent yield volatility
            'gdp',
            'unemployment',
            'fed_funds',
            'slope_10y_2y',
            'expinf_1y',
            # Interaction terms - how CPI shock interacts with market conditions
            'cpi_shock_x_fed_funds',  # CPI shock × Fed funds rate
            'cpi_shock_x_unemployment',  # CPI shock × Unemployment
            'cpi_shock_x_volatility',  # CPI shock × Yield volatility
        ]
    
    def _prepare_features(self, df: pd.DataFrame, target_yield: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target from DataFrame.
        
        Args:
            df: Event DataFrame
            target_yield: Name of target yield ('y_2y' or 'y_10y')
        
        Returns:
            Tuple of (X, y) arrays
        """
        # Build feature matrix
        feature_cols = []
        for feat in self.feature_names:
            if feat in df.columns:
                feature_cols.append(feat)
            elif feat == 'yield_lagged' and f'{target_yield}_lagged' in df.columns:
                feature_cols.append(f'{target_yield}_lagged')
            else:
                logger.warning(f"Feature '{feat}' not found in data, skipping")
        
        if not feature_cols:
            raise ValueError("No valid features found in data")
        
        X = df[feature_cols].values
        
        # Handle missing values
        if np.isnan(X).any():
            logger.warning("Features contain missing values. Filling with median.")
            X = pd.DataFrame(X, columns=feature_cols).fillna(
                pd.DataFrame(X, columns=feature_cols).median()
            ).values
        
        # Get target
        target_col = f'{target_yield}_change'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        y = df[target_col].values
        
        # Remove rows with missing target
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            raise ValueError("No valid samples after removing missing targets")
        
        return X, y
    
    def fit(
        self,
        train_df: pd.DataFrame,
        target_yield: str = "y_2y",
        cv: Optional[int] = None
    ) -> Dict:
        """
        Fit the model on training data.
        
        Args:
            train_df: Training event DataFrame
            target_yield: Name of target yield ('y_2y' or 'y_10y')
            cv: Optional number of folds for cross-validation
        
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Fitting {self.model_type} model on {len(train_df)} samples...")
        
        # Prepare features
        X_train, y_train = self._prepare_features(train_df, target_yield)
        
        # Scale features if needed
        if self.scale_features:
            X_train = self.scaler.fit_transform(X_train)
        
        # Fit model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Compute training metrics
        y_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        train_mae = mean_absolute_error(y_train, y_pred)
        train_r2 = r2_score(y_train, y_pred)
        
        metrics = {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1]
        }
        
        # Cross-validation if requested
        if cv is not None:
            logger.info(f"Performing {cv}-fold cross-validation...")
            cv_scores = cross_val_score(
                self.model, X_train, y_train,
                cv=cv, scoring='neg_mean_squared_error'
            )
            metrics['cv_rmse'] = np.sqrt(-cv_scores.mean())
            metrics['cv_rmse_std'] = np.sqrt(cv_scores.std())
        
        logger.info(f"Training RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        
        return metrics
    
    def tune_hyperparameters(
        self,
        train_df: pd.DataFrame,
        target_yield: str = "y_2y",
        param_grid: Optional[Dict] = None,
        cv_folds: int = 5
    ) -> Dict:
        """
        Tune hyperparameters using GridSearchCV with time-series split.
        
        Args:
            train_df: Training event DataFrame
            target_yield: Name of target yield
            param_grid: Parameter grid for tuning (if None, uses default)
            cv_folds: Number of CV folds
        
        Returns:
            Dictionary with best parameters and scores
        """
        logger.info(f"Tuning hyperparameters for {self.model_type} model...")
        
        # Prepare features
        X_train, y_train = self._prepare_features(train_df, target_yield)
        
        # Scale features if needed
        if self.scale_features:
            X_train = self.scaler.fit_transform(X_train)
        
        # Default parameter grid
        if param_grid is None:
            if self.model_type == "ridge":
                param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
            elif self.model_type == "lasso":
                param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
            elif self.model_type == "elasticnet":
                param_grid = {
                    'alpha': [0.1, 0.5, 1.0, 2.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                }
            else:
                logger.warning(f"Hyperparameter tuning not supported for {self.model_type}")
                return {}
        
        # Use TimeSeriesSplit for proper time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Grid search
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        # Get best score
        best_rmse = np.sqrt(-grid_search.best_score_)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_cv_rmse': best_rmse,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV RMSE: {best_rmse:.4f}")
        
        return results
    
    def plot_learning_curve(
        self,
        train_df: pd.DataFrame,
        target_yield: str = "y_2y",
        val_df: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot learning curve (train/validation error vs training set size).
        
        Args:
            train_df: Training event DataFrame
            target_yield: Name of target yield
            val_df: Optional validation DataFrame
            save_path: Optional path to save figure
        """
        # Prepare features
        X_train, y_train = self._prepare_features(train_df, target_yield)
        
        if self.scale_features:
            X_train = self.scaler.fit_transform(X_train)
        
        # Training sizes to evaluate
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes = [int(len(X_train) * s) for s in train_sizes]
        train_sizes = sorted(set(train_sizes))
        
        train_scores = []
        val_scores = []
        
        for size in train_sizes:
            # Sample training data
            X_subset = X_train[:size]
            y_subset = y_train[:size]
            
            # Fit model
            model_copy = type(self.model)(**self.model.get_params())
            model_copy.fit(X_subset, y_subset)
            
            # Training score
            train_pred = model_copy.predict(X_subset)
            train_rmse = np.sqrt(mean_squared_error(y_subset, train_pred))
            train_scores.append(train_rmse)
            
            # Validation score (if validation set provided)
            if val_df is not None:
                X_val, y_val = self._prepare_features(val_df, target_yield)
                if self.scale_features:
                    # Use same scaler fitted on training subset
                    scaler = StandardScaler()
                    scaler.fit(X_subset)
                    X_val = scaler.transform(X_val)
                val_pred = model_copy.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                val_scores.append(val_rmse)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores, 'o-', label='Training RMSE', linewidth=2)
        if val_scores:
            plt.plot(train_sizes, val_scores, 'o-', label='Validation RMSE', linewidth=2)
        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('RMSE', fontsize=12)
        plt.title(f'Learning Curve - {self.model_type.upper()} Model', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved learning curve to {save_path}")
        
        plt.show()
    
    def predict(self, test_df: pd.DataFrame, target_yield: str = "y_2y") -> np.ndarray:
        """
        Make predictions on test data.
        
        Args:
            test_df: Test event DataFrame
            target_yield: Name of target yield (used for feature selection)
        
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare features
        X_test, _ = self._prepare_features(test_df, target_yield)
        
        # Scale features if needed
        if self.scale_features:
            X_test = self.scaler.transform(X_test)
        
        # Predict
        predictions = self.model.predict(X_test)
        
        return predictions
    
    def evaluate(
        self,
        test_df: pd.DataFrame,
        target_yield: str = "y_2y"
    ) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            test_df: Test event DataFrame
            target_yield: Name of target yield
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Get predictions
        y_pred = self.predict(test_df, target_yield)
        
        # Get actual values
        target_col = f'{target_yield}_change'
        y_true = test_df[target_col].values
        
        # Remove missing values
        valid_mask = ~np.isnan(y_true)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(y_true) == 0:
            raise ValueError("No valid targets in test data")
        
        # Compute metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Directional accuracy
        direction_correct = np.sign(y_true) == np.sign(y_pred)
        directional_accuracy = direction_correct.mean()
        
        metrics = {
            'test_rmse': rmse,
            'test_mae': mae,
            'test_r2': r2,
            'directional_accuracy': directional_accuracy,
            'n_samples': len(y_true)
        }
        
        logger.info(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Directional Acc: {directional_accuracy:.4f}")
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance/coefficients.
        
        Returns:
            DataFrame with feature names and coefficients
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        coefs = self.model.coef_
        
        # Get feature names (handle lagged yield)
        feature_names_actual = []
        for feat in self.feature_names:
            if feat == 'yield_lagged':
                # This will be handled in _prepare_features
                continue
            feature_names_actual.append(feat)
        
        # Adjust for actual features used
        if len(coefs) != len(feature_names_actual):
            feature_names_actual = [f'feature_{i}' for i in range(len(coefs))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names_actual[:len(coefs)],
            'coefficient': coefs
        }).sort_values('coefficient', key=abs, ascending=False)
        
        return importance_df


class TwoStageCPIBondYieldModel:
    """
    Two-stage shock-based model for predicting bond yield changes.
    
    Stage 1 (Regime Model): Predicts baseline yield from background factors
    Stage 2 (Shock Model): Predicts yield change from CPI shock, conditioned on regime
    """
    
    def __init__(
        self,
        regime_model_type: str = "ridge",
        shock_model_type: str = "ridge",
        regime_alpha: float = 1.0,
        shock_alpha: float = 1.0,
        scale_features: bool = True
    ):
        """
        Initialize two-stage CPI-Bond Yield model.
        
        Args:
            regime_model_type: Type of model for regime stage ('linear', 'ridge', 'lasso', 'elasticnet')
            shock_model_type: Type of model for shock stage ('linear', 'ridge', 'lasso', 'elasticnet')
            regime_alpha: Regularization strength for regime model
            shock_alpha: Regularization strength for shock model
            scale_features: Whether to standardize features
        """
        self.regime_model_type = regime_model_type.lower()
        self.shock_model_type = shock_model_type.lower()
        self.regime_alpha = regime_alpha
        self.shock_alpha = shock_alpha
        self.scale_features = scale_features
        
        # Initialize regime model (Stage 1)
        if self.regime_model_type == "linear":
            self.regime_model = LinearRegression()
        elif self.regime_model_type == "ridge":
            self.regime_model = Ridge(alpha=regime_alpha)
        elif self.regime_model_type == "lasso":
            self.regime_model = Lasso(alpha=regime_alpha, max_iter=10000)
        elif self.regime_model_type == "elasticnet":
            self.regime_model = ElasticNet(alpha=regime_alpha, max_iter=10000)
        else:
            raise ValueError(f"Unknown regime_model_type: {regime_model_type}")
        
        # Initialize shock model (Stage 2)
        if self.shock_model_type == "linear":
            self.shock_model = LinearRegression()
        elif self.shock_model_type == "ridge":
            self.shock_model = Ridge(alpha=shock_alpha)
        elif self.shock_model_type == "lasso":
            self.shock_model = Lasso(alpha=shock_alpha, max_iter=10000)
        elif self.shock_model_type == "elasticnet":
            self.shock_model = ElasticNet(alpha=shock_alpha, max_iter=10000)
        else:
            raise ValueError(f"Unknown shock_model_type: {shock_model_type}")
        
        self.regime_scaler = StandardScaler() if scale_features else None
        self.shock_scaler = StandardScaler() if scale_features else None
        
        self.regime_fitted = False
        self.shock_fitted = False
        
        logger.info(f"Initialized two-stage model: regime={regime_model_type}, shock={shock_model_type}")
    
    def _get_regime_features(self) -> List[str]:
        """
        Get features for regime model (Stage 1) - background factors only.
        
        NOTE: We do NOT include yield_lagged because:
        - yield_lagged ≈ yield_before (previous day's yield)
        - Using it would make regime model trivial (just predicting previous value)
        - Regime should be learned from macroeconomic fundamentals, not previous yield
        """
        return [
            'yield_volatility',  # Recent yield volatility (market state indicator)
            'gdp',               # Economic growth
            'unemployment',      # Labor market
            'fed_funds',        # Monetary policy
            'slope_10y_2y',     # Yield curve shape (term structure)
            'expinf_1y',        # Inflation expectations
        ]
    
    def _get_shock_features(self) -> List[str]:
        """Get features for shock model (Stage 2) - CPI shock + predicted baseline."""
        return [
            'cpi_shock_mom',
            'cpi_shock_yoy',
            'cpi_shock_magnitude',
            'yield_baseline_predicted',  # From Stage 1
            # Interactions: CPI shock × baseline
            'cpi_shock_x_baseline',
            # Interactions: CPI shock × regime indicators
            'cpi_shock_x_fed_funds',
            'cpi_shock_x_unemployment',
            'cpi_shock_x_volatility',
        ]
    
    def _prepare_regime_features(self, df: pd.DataFrame, target_yield: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and target for regime model (Stage 1)."""
        feature_cols = []
        for feat in self._get_regime_features():
            if feat in df.columns:
                feature_cols.append(feat)
            elif feat == 'yield_lagged' and f'{target_yield}_lagged' in df.columns:
                feature_cols.append(f'{target_yield}_lagged')
            else:
                logger.warning(f"Regime feature '{feat}' not found in data, skipping")
        
        if not feature_cols:
            raise ValueError("No valid regime features found in data")
        
        X = df[feature_cols].values
        
        # Handle missing values
        if np.isnan(X).any():
            X = pd.DataFrame(X, columns=feature_cols).fillna(
                pd.DataFrame(X, columns=feature_cols).median()
            ).values
        
        # Target: yield_before (baseline yield)
        target_col = f'{target_yield}_before'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        y = df[target_col].values
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            raise ValueError("No valid samples for regime model")
        
        return X, y, feature_cols
    
    def _prepare_shock_features(
        self, 
        df: pd.DataFrame, 
        target_yield: str,
        baseline_predictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and target for shock model (Stage 2)."""
        # Ensure baseline_predictions align with dataframe
        if len(baseline_predictions) != len(df):
            raise ValueError(f"Baseline predictions length ({len(baseline_predictions)}) != dataframe length ({len(df)})")
        
        # Start with CPI shock features
        feature_cols = []
        shock_base_features = ['cpi_shock_mom', 'cpi_shock_yoy', 'cpi_shock_magnitude']
        
        for feat in shock_base_features:
            if feat in df.columns:
                feature_cols.append(feat)
        
        # Add predicted baseline to dataframe
        df_with_baseline = df.copy()
        df_with_baseline['yield_baseline_predicted'] = baseline_predictions
        
        # Add interaction: CPI shock × baseline
        if 'cpi_shock_mom' in df.columns:
            df_with_baseline['cpi_shock_x_baseline'] = (
                df_with_baseline['cpi_shock_mom'] * df_with_baseline['yield_baseline_predicted']
            )
            feature_cols.append('cpi_shock_x_baseline')
        
        # Add other interactions
        if 'cpi_shock_mom' in df.columns:
            if 'fed_funds' in df.columns:
                df_with_baseline['cpi_shock_x_fed_funds'] = (
                    df_with_baseline['cpi_shock_mom'] * df_with_baseline['fed_funds']
                )
                feature_cols.append('cpi_shock_x_fed_funds')
            
            if 'unemployment' in df.columns:
                df_with_baseline['cpi_shock_x_unemployment'] = (
                    df_with_baseline['cpi_shock_mom'] * df_with_baseline['unemployment']
                )
                feature_cols.append('cpi_shock_x_unemployment')
            
            if 'yield_volatility' in df.columns:
                df_with_baseline['cpi_shock_x_volatility'] = (
                    df_with_baseline['cpi_shock_mom'] * df_with_baseline['yield_volatility']
                )
                feature_cols.append('cpi_shock_x_volatility')
        
        # Add baseline prediction
        feature_cols.append('yield_baseline_predicted')
        
        X = df_with_baseline[feature_cols].values
        
        # Handle missing values
        if np.isnan(X).any():
            X = pd.DataFrame(X, columns=feature_cols).fillna(
                pd.DataFrame(X, columns=feature_cols).median()
            ).values
        
        # Target: yield_change
        target_col = f'{target_yield}_change'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        y = df[target_col].values
        valid_mask = ~np.isnan(y)
        
        # Filter by valid target
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            raise ValueError("No valid samples for shock model")
        
        return X, y, feature_cols
    
    def fit_regime_model(
        self,
        train_df: pd.DataFrame,
        target_yield: str = "y_2y",
        cv: Optional[int] = None
    ) -> Dict:
        """Fit Stage 1: Regime model to predict baseline yield."""
        logger.info("Fitting Stage 1: Regime model (baseline yield prediction)...")
        
        X_train, y_train, feature_cols = self._prepare_regime_features(train_df, target_yield)
        
        # Scale features
        if self.scale_features:
            X_train = self.regime_scaler.fit_transform(X_train)
        
        # Fit model
        self.regime_model.fit(X_train, y_train)
        self.regime_fitted = True
        
        # Compute metrics
        y_pred = self.regime_model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        train_mae = mean_absolute_error(y_train, y_pred)
        train_r2 = r2_score(y_train, y_pred)
        
        metrics = {
            'regime_train_rmse': train_rmse,
            'regime_train_mae': train_mae,
            'regime_train_r2': train_r2,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1]
        }
        
        # Cross-validation
        if cv is not None:
            cv_scores = cross_val_score(
                self.regime_model, X_train, y_train,
                cv=cv, scoring='neg_mean_squared_error'
            )
            metrics['regime_cv_rmse'] = np.sqrt(-cv_scores.mean())
            metrics['regime_cv_rmse_std'] = np.sqrt(cv_scores.std())
        
        logger.info(f"Regime model - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        
        return metrics
    
    def fit_shock_model(
        self,
        train_df: pd.DataFrame,
        target_yield: str = "y_2y",
        cv: Optional[int] = None
    ) -> Dict:
        """Fit Stage 2: Shock model to predict yield change from CPI shock."""
        if not self.regime_fitted:
            raise ValueError("Must fit regime model first")
        
        logger.info("Fitting Stage 2: Shock model (CPI shock propagation)...")
        
        # Get baseline predictions from regime model
        baseline_predictions = self.predict_regime(train_df, target_yield)
        
        # Prepare shock features
        X_train, y_train, feature_cols = self._prepare_shock_features(
            train_df, target_yield, baseline_predictions
        )
        
        # Scale features
        if self.scale_features:
            X_train = self.shock_scaler.fit_transform(X_train)
        
        # Fit model
        self.shock_model.fit(X_train, y_train)
        self.shock_fitted = True
        
        # Compute metrics
        y_pred = self.shock_model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        train_mae = mean_absolute_error(y_train, y_pred)
        train_r2 = r2_score(y_train, y_pred)
        
        metrics = {
            'shock_train_rmse': train_rmse,
            'shock_train_mae': train_mae,
            'shock_train_r2': train_r2,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1]
        }
        
        # Cross-validation
        if cv is not None:
            cv_scores = cross_val_score(
                self.shock_model, X_train, y_train,
                cv=cv, scoring='neg_mean_squared_error'
            )
            metrics['shock_cv_rmse'] = np.sqrt(-cv_scores.mean())
            metrics['shock_cv_rmse_std'] = np.sqrt(cv_scores.std())
        
        logger.info(f"Shock model - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        
        return metrics
    
    def fit(
        self,
        train_df: pd.DataFrame,
        target_yield: str = "y_2y",
        cv: Optional[int] = None
    ) -> Dict:
        """Fit both stages sequentially."""
        # Fit Stage 1
        regime_metrics = self.fit_regime_model(train_df, target_yield, cv)
        
        # Fit Stage 2
        shock_metrics = self.fit_shock_model(train_df, target_yield, cv)
        
        # Combine metrics
        return {**regime_metrics, **shock_metrics}
    
    def predict_regime(
        self,
        df: pd.DataFrame,
        target_yield: str = "y_2y",
        return_mask: bool = False
    ):
        """
        Predict baseline yield from regime model (Stage 1).
        
        Args:
            df: DataFrame with features
            target_yield: Name of target yield
            return_mask: If True, also return valid mask
        
        Returns:
            Predictions array (and optionally valid mask)
        """
        if not self.regime_fitted:
            raise ValueError("Regime model must be fitted first")
        
        # Get regime features
        feature_cols = []
        for feat in self._get_regime_features():
            if feat in df.columns:
                feature_cols.append(feat)
            elif feat == 'yield_lagged' and f'{target_yield}_lagged' in df.columns:
                feature_cols.append(f'{target_yield}_lagged')
        
        if not feature_cols:
            raise ValueError("No valid regime features found")
        
        X = df[feature_cols].values
        
        # Handle missing values (fill with median from training - but for prediction we'll use 0 or median)
        if np.isnan(X).any():
            X = pd.DataFrame(X, columns=feature_cols).fillna(0).values
        
        if self.scale_features:
            X = self.regime_scaler.transform(X)
        
        predictions = self.regime_model.predict(X)
        
        if return_mask:
            # Valid mask: no NaN in features (after filling)
            valid_mask = ~np.isnan(predictions)
            return predictions, valid_mask
        
        return predictions
    
    def predict(
        self,
        df: pd.DataFrame,
        target_yield: str = "y_2y"
    ) -> np.ndarray:
        """Predict yield change using both stages."""
        if not self.regime_fitted or not self.shock_fitted:
            raise ValueError("Both models must be fitted first")
        
        # Stage 1: Predict baseline (for all rows)
        baseline_predictions = self.predict_regime(df, target_yield)
        
        # Stage 2: Prepare features and predict
        # Note: _prepare_shock_features will filter by valid target, so we need to handle alignment
        # For prediction, we want predictions for all rows (even if target is missing)
        # So we'll prepare features for all rows, then predict
        
        # Build features manually for all rows
        feature_cols = []
        shock_base_features = ['cpi_shock_mom', 'cpi_shock_yoy', 'cpi_shock_magnitude']
        
        for feat in shock_base_features:
            if feat in df.columns:
                feature_cols.append(feat)
        
        df_with_baseline = df.copy()
        df_with_baseline['yield_baseline_predicted'] = baseline_predictions
        
        # Add interactions
        if 'cpi_shock_mom' in df.columns:
            df_with_baseline['cpi_shock_x_baseline'] = (
                df_with_baseline['cpi_shock_mom'] * df_with_baseline['yield_baseline_predicted']
            )
            feature_cols.append('cpi_shock_x_baseline')
            
            if 'fed_funds' in df.columns:
                df_with_baseline['cpi_shock_x_fed_funds'] = (
                    df_with_baseline['cpi_shock_mom'] * df_with_baseline['fed_funds']
                )
                feature_cols.append('cpi_shock_x_fed_funds')
            
            if 'unemployment' in df.columns:
                df_with_baseline['cpi_shock_x_unemployment'] = (
                    df_with_baseline['cpi_shock_mom'] * df_with_baseline['unemployment']
                )
                feature_cols.append('cpi_shock_x_unemployment')
            
            if 'yield_volatility' in df.columns:
                df_with_baseline['cpi_shock_x_volatility'] = (
                    df_with_baseline['cpi_shock_mom'] * df_with_baseline['yield_volatility']
                )
                feature_cols.append('cpi_shock_x_volatility')
        
        feature_cols.append('yield_baseline_predicted')
        
        X = df_with_baseline[feature_cols].values
        
        # Handle missing values
        if np.isnan(X).any():
            X = pd.DataFrame(X, columns=feature_cols).fillna(
                pd.DataFrame(X, columns=feature_cols).median()
            ).values
        
        if self.scale_features:
            X = self.shock_scaler.transform(X)
        
        return self.shock_model.predict(X)
    
    def evaluate(
        self,
        test_df: pd.DataFrame,
        target_yield: str = "y_2y"
    ) -> Dict:
        """Evaluate both stages on test data."""
        # Evaluate regime model
        try:
            baseline_pred = self.predict_regime(test_df, target_yield=target_yield)
            baseline_actual = test_df[f'{target_yield}_before'].values
            valid_mask = ~np.isnan(baseline_actual)
            
            if valid_mask.sum() > 0:
                regime_rmse = np.sqrt(mean_squared_error(baseline_actual[valid_mask], baseline_pred[valid_mask]))
                regime_r2 = r2_score(baseline_actual[valid_mask], baseline_pred[valid_mask])
            else:
                regime_rmse = np.nan
                regime_r2 = np.nan
        except Exception as e:
            logger.warning(f"Could not evaluate regime model: {e}")
            regime_rmse = np.nan
            regime_r2 = np.nan
        
        # Evaluate shock model (final predictions)
        y_pred = self.predict(test_df, target_yield=target_yield)
        y_true = test_df[f'{target_yield}_change'].values
        
        # Align predictions with actuals
        valid_mask = ~np.isnan(y_true)
        if len(y_pred) == len(y_true):
            y_pred = y_pred[valid_mask]
        else:
            # If lengths don't match, try to align
            logger.warning(f"Prediction length ({len(y_pred)}) != target length ({len(y_true)})")
            if len(y_pred) >= len(y_true):
                y_pred = y_pred[:len(y_true)][valid_mask]
            else:
                y_pred = y_pred[valid_mask[:len(y_pred)]]
        
        y_true = y_true[valid_mask]
        
        if len(y_true) == 0:
            raise ValueError("No valid targets in test data")
        
        shock_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        shock_mae = mean_absolute_error(y_true, y_pred)
        shock_r2 = r2_score(y_true, y_pred)
        directional_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))
        
        return {
            'regime_rmse': regime_rmse,
            'regime_r2': regime_r2,
            'shock_rmse': shock_rmse,
            'shock_mae': shock_mae,
            'shock_r2': shock_r2,
            'directional_accuracy': directional_accuracy,
            'n_samples': len(y_true)
        }


class RegimeSwitchingCPIBondYieldModel:
    """
    Regime-switching shock-based model with non-linear models.
    
    Stage 1 (Regime Classifier): Classifies regime from background factors
    Stage 2 (Regime-Specific Shock Models): Separate non-linear models for each regime
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        regime_classifier_type: str = "random_forest",
        shock_model_type: str = "random_forest",
        regime_classifier_params: Optional[Dict] = None,
        shock_model_params: Optional[Dict] = None,
        scale_features: bool = True,
        use_classification: bool = True,
        bin_thresholds_bp: Optional[List[float]] = None
    ):
        """
        Initialize regime-switching CPI-Bond Yield model.
        
        Args:
            n_regimes: Number of regimes to classify (e.g., 3 = low/medium/high inflation)
            regime_classifier_type: Type of classifier ('random_forest', 'gradient_boosting')
            shock_model_type: Type of non-linear model for each regime ('random_forest', 'gradient_boosting')
            regime_classifier_params: Parameters for regime classifier
            shock_model_params: Parameters for shock models
            scale_features: Whether to standardize features
            use_classification: If True, classify into bins (noise/small/medium/large/very large)
            bin_thresholds_bp: Bin thresholds in basis points (default: [-30, -10, -3, 3, 10, 30])
                              Creates bins: very_large_down, large_down, medium_down, noise, medium_up, large_up, very_large_up
        """
        self.n_regimes = n_regimes
        self.regime_classifier_type = regime_classifier_type.lower()
        self.shock_model_type = shock_model_type.lower()
        self.scale_features = scale_features
        self.use_classification = use_classification
        
        # Define bin thresholds in basis points (yield changes are in percentage points, so 0.10 = 10bp)
        if bin_thresholds_bp is None:
            # Default: noise: -3 to +3bp, medium: ±3-10bp, large: ±10-30bp, very_large: >±30bp
            self.bin_thresholds = [-0.30, -0.10, -0.03, 0.03, 0.10, 0.30]  # In percentage points
            self.bin_labels = ['very_large_down', 'large_down', 'medium_down', 'noise', 'medium_up', 'large_up', 'very_large_up']
        else:
            self.bin_thresholds = [t / 100.0 for t in bin_thresholds_bp]  # Convert bp to percentage points
            # Create labels based on thresholds
            # For standard 6 thresholds: [-30, -10, -3, 3, 10, 30] bp
            # Creates 7 bins: very_large_down, large_down, medium_down, noise, medium_up, large_up, very_large_up
            n_thresholds = len(self.bin_thresholds)
            n_bins = n_thresholds + 1
            
            # For the standard 6-threshold case, use explicit labels
            if n_thresholds == 6:  # Standard case: [-30, -10, -3, 3, 10, 30]
                self.bin_labels = ['very_large_down', 'large_down', 'medium_down', 'noise', 'medium_up', 'large_up', 'very_large_up']
            else:
                # For other threshold configurations, generate labels based on position
                mid_idx = n_thresholds // 2
                self.bin_labels = []
                for i in range(n_bins):
                    if i == 0:
                        self.bin_labels.append('very_large_down')
                    elif i == 1:
                        self.bin_labels.append('large_down')
                    elif i == 2:
                        self.bin_labels.append('medium_down')
                    elif i == mid_idx:
                        self.bin_labels.append('noise')
                    elif i == mid_idx + 1:
                        self.bin_labels.append('medium_up')
                    elif i == n_bins - 2:
                        self.bin_labels.append('large_up')
                    elif i == n_bins - 1:
                        self.bin_labels.append('very_large_up')
                    else:
                        # Fallback for any remaining bins
                        if i < mid_idx:
                            self.bin_labels.append(f'medium_down')
                        else:
                            self.bin_labels.append(f'medium_up')
        
        self.n_bins = len(self.bin_thresholds) + 1
        
        # Default parameters
        regime_defaults = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
        shock_defaults = {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
        
        self.regime_classifier_params = regime_classifier_params or regime_defaults
        self.shock_model_params = shock_model_params or shock_defaults
        
        # Initialize regime classifier (Stage 1)
        if self.regime_classifier_type == "random_forest":
            self.regime_classifier = RandomForestClassifier(
                n_estimators=self.regime_classifier_params.get('n_estimators', 100),
                max_depth=self.regime_classifier_params.get('max_depth', 10),
                random_state=self.regime_classifier_params.get('random_state', 42),
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown regime_classifier_type: {regime_classifier_type}")
        
        # Initialize regime-specific shock models (Stage 2)
        # One model per regime
        self.shock_models = {}
        self.shock_scalers = {}
        
        for regime in range(n_regimes):
            if self.use_classification:
                # Classification models for bin prediction
                if self.shock_model_type == "random_forest":
                    self.shock_models[regime] = RandomForestClassifier(
                        n_estimators=self.shock_model_params.get('n_estimators', 100),
                        max_depth=self.shock_model_params.get('max_depth', 5),
                        random_state=self.shock_model_params.get('random_state', 42),
                        n_jobs=-1,
                        class_weight='balanced'  # Handle imbalanced classes
                    )
                elif self.shock_model_type == "gradient_boosting":
                    self.shock_models[regime] = GradientBoostingClassifier(
                        n_estimators=self.shock_model_params.get('n_estimators', 100),
                        max_depth=self.shock_model_params.get('max_depth', 5),
                        random_state=self.shock_model_params.get('random_state', 42)
                    )
                else:
                    raise ValueError(f"Unknown shock_model_type: {shock_model_type}")
            else:
                # Regression models for continuous prediction
                if self.shock_model_type == "random_forest":
                    self.shock_models[regime] = RandomForestRegressor(
                        n_estimators=self.shock_model_params.get('n_estimators', 100),
                        max_depth=self.shock_model_params.get('max_depth', 5),
                        random_state=self.shock_model_params.get('random_state', 42),
                        n_jobs=-1
                    )
                elif self.shock_model_type == "gradient_boosting":
                    self.shock_models[regime] = GradientBoostingRegressor(
                        n_estimators=self.shock_model_params.get('n_estimators', 100),
                        max_depth=self.shock_model_params.get('max_depth', 5),
                        random_state=self.shock_model_params.get('random_state', 42)
                    )
                else:
                    raise ValueError(f"Unknown shock_model_type: {shock_model_type}")
            
            self.shock_scalers[regime] = StandardScaler() if scale_features else None
        
        self.regime_scaler = StandardScaler() if scale_features else None
        
        self.regime_fitted = False
        self.shock_fitted = {regime: False for regime in range(n_regimes)}
        
        logger.info(f"Initialized regime-switching model: {n_regimes} regimes, classifier={regime_classifier_type}, shock={shock_model_type}")
    
    def _get_regime_features(self) -> List[str]:
        """Get features for regime classification - background factors only."""
        return [
            'yield_volatility',  # Recent yield volatility
            'gdp',               # Economic growth
            'unemployment',      # Labor market
            'fed_funds',        # Monetary policy
            'slope_10y_2y',     # Yield curve shape
            'expinf_1y',        # Inflation expectations
        ]
    
    def _get_shock_features(self) -> List[str]:
        """Get features for shock models - CPI shock variables."""
        return [
            'cpi_shock_mom',
            'cpi_shock_yoy',
            'cpi_shock_magnitude',
            'yield_volatility',  # Market volatility
            'fed_funds',         # Monetary policy context
            'unemployment',      # Economic context
            'expinf_1y',         # Inflation expectations
        ]
    
    def _create_regime_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create regime labels based on BACKGROUND factors only (not CPI shocks).
        
        Regime should be learnable from background features, so we use only:
        - fed_funds (monetary policy)
        - unemployment (economic state)
        - gdp (economic growth)
        - slope_10y_2y (yield curve shape)
        - yield_volatility (market state)
        
        This ensures the classifier can actually learn to predict regime from background features.
        """
        # Use only background factors to create regimes
        # Regime 0: Low policy rate / low volatility (low sensitivity)
        # Regime 1: Medium policy rate / medium volatility (medium sensitivity)
        # Regime 2: High policy rate / high volatility (high sensitivity)
        
        fed_funds = df['fed_funds'].fillna(df['fed_funds'].median())
        unemployment = df['unemployment'].fillna(df['unemployment'].median())
        yield_volatility = df['yield_volatility'].fillna(df['yield_volatility'].median())
        
        # Create composite score from background factors only
        # Higher fed funds + higher volatility = more sensitive regime
        policy_score = (fed_funds - fed_funds.median()) / (fed_funds.std() + 1e-6)
        volatility_score = (yield_volatility - yield_volatility.median()) / (yield_volatility.std() + 1e-6)
        composite = policy_score + volatility_score
        
        # Classify into n_regimes using quantiles
        quantiles = np.linspace(0, 1, self.n_regimes + 1)
        regime_labels = np.digitize(composite, composite.quantile(quantiles)) - 1
        regime_labels = np.clip(regime_labels, 0, self.n_regimes - 1)
        
        return regime_labels
    
    def _prepare_regime_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features for regime classification."""
        feature_cols = []
        for feat in self._get_regime_features():
            if feat in df.columns:
                feature_cols.append(feat)
            else:
                logger.warning(f"Regime feature '{feat}' not found in data, skipping")
        
        if not feature_cols:
            raise ValueError("No valid regime features found in data")
        
        X = df[feature_cols].values
        
        # Handle missing values
        if np.isnan(X).any():
            X = pd.DataFrame(X, columns=feature_cols).fillna(
                pd.DataFrame(X, columns=feature_cols).median()
            ).values
        
        # Create regime labels
        y = self._create_regime_labels(df)
        
        return X, y, feature_cols
    
    def _create_yield_bins(self, yield_changes: np.ndarray) -> np.ndarray:
        """
        Convert yield changes to bin labels.
        
        Bins are created using thresholds:
        - Bin 0: < threshold[0] (very_large_down)
        - Bin 1: threshold[0] to threshold[1] (large_down)
        - Bin 2: threshold[1] to threshold[2] (medium_down)
        - Bin 3: threshold[2] to threshold[3] (noise)
        - Bin 4: threshold[3] to threshold[4] (medium_up)
        - Bin 5: threshold[4] to threshold[5] (large_up)
        - Bin 6: >= threshold[5] (very_large_up)
        """
        # Use digitize: returns index of bin (0 = left of first threshold, n = right of last threshold)
        bins = np.digitize(yield_changes, self.bin_thresholds)
        return bins
    
    def _bins_to_representative_values(self, bins: np.ndarray) -> np.ndarray:
        """
        Convert bin labels to representative values (bin centers) for visualization.
        
        Returns approximate yield change values for each bin.
        """
        # Create bin centers
        all_thresholds = [-np.inf] + self.bin_thresholds + [np.inf]
        bin_centers = []
        
        for i in range(len(all_thresholds) - 1):
            if i == 0:
                # First bin: use threshold[0] - 0.05 as center
                center = self.bin_thresholds[0] - 0.05
            elif i == len(all_thresholds) - 2:
                # Last bin: use threshold[-1] + 0.05 as center
                center = self.bin_thresholds[-1] + 0.05
            else:
                # Middle bins: use midpoint between thresholds
                center = (all_thresholds[i] + all_thresholds[i+1]) / 2.0
            bin_centers.append(center)
        
        # Map bins to centers
        representative_values = np.array([bin_centers[int(b)] for b in bins])
        return representative_values
    
    def _prepare_shock_features(self, df: pd.DataFrame, target_yield: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features for shock models."""
        feature_cols = []
        for feat in self._get_shock_features():
            if feat in df.columns:
                feature_cols.append(feat)
            else:
                logger.warning(f"Shock feature '{feat}' not found in data, skipping")
        
        if not feature_cols:
            raise ValueError("No valid shock features found in data")
        
        X = df[feature_cols].values
        
        # Handle missing values
        if np.isnan(X).any():
            X = pd.DataFrame(X, columns=feature_cols).fillna(
                pd.DataFrame(X, columns=feature_cols).median()
            ).values
        
        # Target: yield_change
        target_col = f'{target_yield}_change'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        y_raw = df[target_col].values
        valid_mask = ~np.isnan(y_raw)
        X = X[valid_mask]
        y_raw = y_raw[valid_mask]
        
        if len(X) == 0:
            raise ValueError("No valid samples for shock model")
        
        # Convert to bins if classification
        if self.use_classification:
            y = self._create_yield_bins(y_raw)
        else:
            y = y_raw
        
        return X, y, feature_cols
    
    def fit_regime_classifier(self, train_df: pd.DataFrame) -> Dict:
        """Fit Stage 1: Regime classifier."""
        logger.info("Fitting Stage 1: Regime classifier...")
        
        X_train, y_train, feature_cols = self._prepare_regime_features(train_df)
        
        # Scale features
        if self.scale_features:
            X_train = self.regime_scaler.fit_transform(X_train)
        
        # Fit classifier
        self.regime_classifier.fit(X_train, y_train)
        self.regime_fitted = True
        
        # Compute metrics
        y_pred = self.regime_classifier.predict(X_train)
        accuracy = (y_pred == y_train).mean()
        
        # Get regime distribution
        regime_counts = pd.Series(y_train).value_counts().sort_index()
        
        metrics = {
            'regime_train_accuracy': accuracy,
            'regime_distribution': regime_counts.to_dict(),
            'n_samples': len(X_train),
            'n_features': X_train.shape[1]
        }
        
        logger.info(f"Regime classifier - Accuracy: {accuracy:.4f}")
        logger.info(f"Regime distribution: {regime_counts.to_dict()}")
        
        return metrics
    
    def fit_shock_models(self, train_df: pd.DataFrame, target_yield: str = "y_2y") -> Dict:
        """Fit Stage 2: Regime-specific shock models."""
        if not self.regime_fitted:
            raise ValueError("Must fit regime classifier first")
        
        logger.info("Fitting Stage 2: Regime-specific shock models...")
        
        # Prepare shock features first (this filters by valid target)
        X_shock, y_shock, feature_cols = self._prepare_shock_features(train_df, target_yield)
        
        # Get regime predictions for training data
        # Need to align with filtered shock features
        target_col = f'{target_yield}_change'
        valid_mask = ~train_df[target_col].isna()
        train_df_filtered = train_df[valid_mask].reset_index(drop=True)
        
        X_regime, _, _ = self._prepare_regime_features(train_df_filtered)
        if self.scale_features:
            X_regime = self.regime_scaler.transform(X_regime)
        regime_predictions = self.regime_classifier.predict(X_regime)
        
        # Now regime_predictions and X_shock/y_shock should be aligned
        if len(regime_predictions) != len(X_shock):
            raise ValueError(f"Alignment error: regime_predictions length ({len(regime_predictions)}) != X_shock length ({len(X_shock)})")
        
        # Fit separate model for each regime
        all_metrics = {}
        
        for regime in range(self.n_regimes):
            # Get samples for this regime
            regime_mask = regime_predictions == regime
            if regime_mask.sum() == 0:
                logger.warning(f"No samples for regime {regime}, skipping")
                continue
            
            X_regime_shock = X_shock[regime_mask]
            y_regime_shock = y_shock[regime_mask]
            
            if len(X_regime_shock) < 5:
                logger.warning(f"Too few samples ({len(X_regime_shock)}) for regime {regime}, skipping")
                continue
            
            # Scale features
            if self.scale_features:
                X_regime_shock = self.shock_scalers[regime].fit_transform(X_regime_shock)
            
            # Fit model
            self.shock_models[regime].fit(X_regime_shock, y_regime_shock)
            self.shock_fitted[regime] = True
            
            # Compute metrics
            y_pred = self.shock_models[regime].predict(X_regime_shock)
            
            if self.use_classification:
                # Classification metrics
                accuracy = (y_pred == y_regime_shock).mean()
                all_metrics[f'regime_{regime}_accuracy'] = accuracy
                all_metrics[f'regime_{regime}_n_samples'] = len(X_regime_shock)
                
                # Bin distribution
                bin_dist = pd.Series(y_regime_shock).value_counts().sort_index().to_dict()
                all_metrics[f'regime_{regime}_bin_distribution'] = bin_dist
                
                logger.info(f"Regime {regime} shock model - Accuracy: {accuracy:.4f}, n={len(X_regime_shock)}")
                logger.info(f"  Bin distribution: {bin_dist}")
            else:
                # Regression metrics
                rmse = np.sqrt(mean_squared_error(y_regime_shock, y_pred))
                mae = mean_absolute_error(y_regime_shock, y_pred)
                r2 = r2_score(y_regime_shock, y_pred)
                
                all_metrics[f'regime_{regime}_rmse'] = rmse
                all_metrics[f'regime_{regime}_mae'] = mae
                all_metrics[f'regime_{regime}_r2'] = r2
                all_metrics[f'regime_{regime}_n_samples'] = len(X_regime_shock)
                
                logger.info(f"Regime {regime} shock model - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, n={len(X_regime_shock)}")
        
        return all_metrics
    
    def fit(self, train_df: pd.DataFrame, target_yield: str = "y_2y") -> Dict:
        """Fit both stages sequentially."""
        # Fit Stage 1
        regime_metrics = self.fit_regime_classifier(train_df)
        
        # Fit Stage 2
        shock_metrics = self.fit_shock_models(train_df, target_yield)
        
        return {**regime_metrics, **shock_metrics}
    
    def predict_regime(self, df: pd.DataFrame) -> np.ndarray:
        """Predict regime for given data."""
        if not self.regime_fitted:
            raise ValueError("Regime classifier must be fitted first")
        
        X, _, _ = self._prepare_regime_features(df)
        
        if self.scale_features:
            X = self.regime_scaler.transform(X)
        
        return self.regime_classifier.predict(X)
    
    def predict(self, df: pd.DataFrame, target_yield: str = "y_2y") -> np.ndarray:
        """Predict yield change bins using regime-switching model."""
        if not self.regime_fitted:
            raise ValueError("Regime classifier must be fitted first")
        
        # Predict regimes (for all rows)
        regimes = self.predict_regime(df)
        
        # Prepare shock features manually (don't filter by target)
        feature_cols = []
        for feat in self._get_shock_features():
            if feat in df.columns:
                feature_cols.append(feat)
        
        if not feature_cols:
            raise ValueError("No valid shock features found in data")
        
        X_shock = df[feature_cols].values
        
        # Handle missing values
        if np.isnan(X_shock).any():
            X_shock = pd.DataFrame(X_shock, columns=feature_cols).fillna(
                pd.DataFrame(X_shock, columns=feature_cols).median()
            ).values
        
        # Predict using regime-specific models
        if self.use_classification:
            # Classification: predict bin labels
            predictions = np.zeros(len(df), dtype=int)
        else:
            # Regression: predict continuous values
            predictions = np.zeros(len(df))
        
        for regime in range(self.n_regimes):
            regime_mask = regimes == regime
            if regime_mask.sum() == 0:
                continue
            
            if not self.shock_fitted.get(regime, False):
                # If regime model not fitted, predict noise bin (middle bin) or 0
                if self.use_classification:
                    predictions[regime_mask] = len(self.bin_thresholds) // 2  # Middle bin (noise)
                else:
                    predictions[regime_mask] = 0.0
                continue
            
            X_regime_shock = X_shock[regime_mask]
            
            if self.scale_features:
                X_regime_shock = self.shock_scalers[regime].transform(X_regime_shock)
            
            predictions[regime_mask] = self.shock_models[regime].predict(X_regime_shock)
        
        return predictions
    
    def predict_proba(self, df: pd.DataFrame, target_yield: str = "y_2y") -> np.ndarray:
        """Predict probability distribution over bins."""
        if not self.use_classification:
            raise ValueError("predict_proba only available for classification models")
        
        if not self.regime_fitted:
            raise ValueError("Regime classifier must be fitted first")
        
        # Predict regimes
        regimes = self.predict_regime(df)
        
        # Prepare shock features
        feature_cols = []
        for feat in self._get_shock_features():
            if feat in df.columns:
                feature_cols.append(feat)
        
        X_shock = df[feature_cols].values
        
        # Handle missing values
        if np.isnan(X_shock).any():
            X_shock = pd.DataFrame(X_shock, columns=feature_cols).fillna(
                pd.DataFrame(X_shock, columns=feature_cols).median()
            ).values
        
        # Initialize probability matrix
        proba = np.zeros((len(df), self.n_bins))
        
        for regime in range(self.n_regimes):
            regime_mask = regimes == regime
            if regime_mask.sum() == 0:
                continue
            
            if not self.shock_fitted.get(regime, False):
                # Default: uniform distribution
                proba[regime_mask] = 1.0 / self.n_bins
                continue
            
            X_regime_shock = X_shock[regime_mask]
            
            if self.scale_features:
                X_regime_shock = self.shock_scalers[regime].transform(X_regime_shock)
            
            proba[regime_mask] = self.shock_models[regime].predict_proba(X_regime_shock)
        
        return proba
    
    def evaluate(self, test_df: pd.DataFrame, target_yield: str = "y_2y") -> Dict:
        """Evaluate on test data with detailed bin-wise metrics."""
        # Predict regimes
        regimes = self.predict_regime(test_df)
        regime_distribution = pd.Series(regimes).value_counts().sort_index().to_dict()
        
        # Get actual yield changes
        y_true_raw = test_df[f'{target_yield}_change'].values
        valid_mask = ~np.isnan(y_true_raw)
        y_true_raw = y_true_raw[valid_mask]
        regimes = regimes[valid_mask]
        
        if len(y_true_raw) == 0:
            raise ValueError("No valid targets in test data")
        
        # Convert actual to bins
        if self.use_classification:
            y_true_bins = self._create_yield_bins(y_true_raw)
            y_pred_bins = self.predict(test_df[valid_mask], target_yield)
            
            # Overall classification accuracy
            accuracy = (y_pred_bins == y_true_bins).mean()
            
            # Bin-wise accuracy
            bin_accuracy = {}
            for bin_idx in range(self.n_bins):
                bin_mask = y_true_bins == bin_idx
                if bin_mask.sum() > 0:
                    bin_accuracy[f'bin_{bin_idx}_accuracy'] = (y_pred_bins[bin_mask] == y_true_bins[bin_mask]).mean()
                    bin_accuracy[f'bin_{bin_idx}_n_samples'] = bin_mask.sum()
            
            # Large move detection (large_up, very_large_up, large_down, very_large_down)
            large_bins = [i for i, label in enumerate(self.bin_labels) if 'large' in label or 'very_large' in label]
            large_mask = np.isin(y_true_bins, large_bins)
            
            if large_mask.sum() > 0:
                # True positives: correctly identified large moves
                tp = np.sum((large_mask) & (np.isin(y_pred_bins, large_bins)))
                # False negatives: missed large moves (actual large, predicted not large)
                fn = np.sum((large_mask) & (~np.isin(y_pred_bins, large_bins)))
                # False positives: predicted large but wasn't (predicted large, actual not large)
                fp = np.sum((~large_mask) & (np.isin(y_pred_bins, large_bins)))
                # True negatives: correctly identified non-large moves
                tn = np.sum((~large_mask) & (~np.isin(y_pred_bins, large_bins)))
                
                large_detection_rate = tp / large_mask.sum() if large_mask.sum() > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            else:
                tp = fn = fp = tn = 0
                large_detection_rate = precision = recall = f1 = 0.0
            
            # Per-regime metrics
            regime_metrics = {}
            for regime in range(self.n_regimes):
                regime_mask = regimes == regime
                if regime_mask.sum() > 0:
                    regime_accuracy = (y_pred_bins[regime_mask] == y_true_bins[regime_mask]).mean()
                    regime_metrics[f'regime_{regime}_accuracy'] = regime_accuracy
                    regime_metrics[f'regime_{regime}_n_samples'] = regime_mask.sum()
            
            return {
                'overall_accuracy': accuracy,
                'bin_accuracy': bin_accuracy,
                'large_move_detection': {
                    'detection_rate': large_detection_rate,  # % of large moves correctly identified
                    'false_negatives': int(fn),  # Missed large moves
                    'false_positives': int(fp),  # Predicted large but wasn't
                    'true_positives': int(tp),
                    'true_negatives': int(tn),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'n_large_moves': int(large_mask.sum())
                },
                'regime_distribution': regime_distribution,
                'n_samples': len(y_true_raw),
                **regime_metrics
            }
        else:
            # Regression evaluation
            y_pred = self.predict(test_df[valid_mask], target_yield)
            
            # Overall metrics
            rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred))
            mae = mean_absolute_error(y_true_raw, y_pred)
            r2 = r2_score(y_true_raw, y_pred)
            directional_accuracy = np.mean(np.sign(y_true_raw) == np.sign(y_pred))
            
            # Per-regime metrics
            regime_metrics = {}
            for regime in range(self.n_regimes):
                regime_mask = regimes == regime
                if regime_mask.sum() > 0:
                    regime_rmse = np.sqrt(mean_squared_error(y_true_raw[regime_mask], y_pred[regime_mask]))
                    regime_r2 = r2_score(y_true_raw[regime_mask], y_pred[regime_mask])
                    regime_metrics[f'regime_{regime}_rmse'] = regime_rmse
                    regime_metrics[f'regime_{regime}_r2'] = regime_r2
                    regime_metrics[f'regime_{regime}_n_samples'] = regime_mask.sum()
            
            return {
                'shock_rmse': rmse,
                'shock_mae': mae,
                'shock_r2': r2,
                'directional_accuracy': directional_accuracy,
                'regime_distribution': regime_distribution,
                'n_samples': len(y_true_raw),
                **regime_metrics
            }
