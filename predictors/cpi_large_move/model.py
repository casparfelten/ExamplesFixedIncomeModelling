"""
CPI Large Move Predictor
========================

A cost-sensitive Random Forest model for detecting large yield moves
around CPI announcements.

Model Details:
- Algorithm: Random Forest with class_weight={False: 1, True: 50}
- Threshold: Selected via Time Series CV for 0% False Negative
- Features: yield_volatility, cpi_shock_mom, cpi_abs, fed_funds, slope_10y_2y, unemployment

Performance (Test Set 2003-2025):
- False Negative Rate: 0.0% (catches ALL large moves)
- False Positive Rate: 66.1%
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ..base import BasePredictor, PredictorOutput


class CPILargeMovePredictor(BasePredictor):
    """
    Detects large yield moves (|change| > 10bp) around CPI announcements.
    
    This predictor prioritizes catching ALL large moves (0% False Negative)
    at the cost of more false alarms (66% False Positive).
    
    Use Case:
    - Risk management: Never miss a large move
    - Event preparation: Flag potential high-impact CPI releases
    
    Example:
        predictor = CPILargeMovePredictor.load()
        result = predictor.predict({
            'yield_volatility': 0.05,
            'cpi_shock_mom': 0.1,
            'fed_funds': 2.5,
            'slope_10y_2y': 1.0,
            'unemployment': 4.0,
        })
        if result.prediction:
            print("Warning: High probability of large yield move!")
    """
    
    name = "CPILargeMovePredictor"
    version = "1.0.0"
    description = "Detects large yield moves (>10bp) around CPI announcements"
    
    required_inputs = [
        'yield_volatility',  # 20-day trailing volatility of 10Y yield
        'cpi_shock_mom',     # MoM CPI surprise (actual - expected)
        'fed_funds',         # Fed Funds rate
        'slope_10y_2y',      # Yield curve slope (10Y - 2Y)
        'unemployment',      # Unemployment rate
    ]
    
    # Default model path
    DEFAULT_MODEL_PATH = Path(__file__).parent / "trained_model.pkl"
    
    def __init__(
        self,
        model: Optional[RandomForestClassifier] = None,
        scaler: Optional[StandardScaler] = None,
        medians: Optional[Dict[str, float]] = None,
        threshold: float = 0.099,
    ):
        """
        Initialize the predictor.
        
        Args:
            model: Trained RandomForestClassifier
            scaler: Fitted StandardScaler for features
            medians: Median values for missing feature imputation
            threshold: Probability threshold for prediction (default: 0.099)
        """
        self.model = model
        self.scaler = scaler
        self.medians = medians or {}
        self.threshold = threshold
        self._feature_order = [
            'yield_volatility',
            'cpi_shock_mom',
            'cpi_abs',
            'fed_funds',
            'slope_10y_2y',
            'unemployment',
        ]
    
    def predict(self, inputs: Dict[str, Any]) -> PredictorOutput:
        """
        Predict probability of a large yield move.
        
        Args:
            inputs: Dictionary with required features
            
        Returns:
            PredictorOutput with probability, prediction, and confidence
        """
        self.validate_inputs(inputs)
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")
        
        # Prepare features
        features = self._prepare_features(inputs)
        
        # Get probability
        proba = self.model.predict_proba(features)[0, 1]
        
        # Make prediction
        prediction = proba >= self.threshold
        
        # Determine confidence
        if proba < 0.05:
            confidence = 'high'  # High confidence it's normal
        elif proba > 0.30:
            confidence = 'high'  # High confidence it's abnormal
        elif proba > 0.15:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return PredictorOutput(
            probability=float(proba),
            prediction=prediction,
            confidence=confidence,
            threshold=self.threshold,
            features_used={k: inputs.get(k) for k in self.required_inputs},
        )
    
    def predict_batch(self, inputs: List[Dict[str, Any]]) -> List[PredictorOutput]:
        """
        Make predictions for multiple input sets.
        
        Args:
            inputs: List of input dictionaries
            
        Returns:
            List of PredictorOutput objects
        """
        return [self.predict(inp) for inp in inputs]
    
    def _prepare_features(self, inputs: Dict[str, Any]) -> np.ndarray:
        """Prepare feature vector from inputs."""
        # Extract and order features
        raw = {
            'yield_volatility': inputs.get('yield_volatility', self.medians.get('yield_volatility', 0.05)),
            'cpi_shock_mom': inputs.get('cpi_shock_mom', self.medians.get('cpi_shock_mom', 0.0)),
            'cpi_abs': abs(inputs.get('cpi_shock_mom', self.medians.get('cpi_shock_mom', 0.0))),
            'fed_funds': inputs.get('fed_funds', self.medians.get('fed_funds', 2.5)),
            'slope_10y_2y': inputs.get('slope_10y_2y', self.medians.get('slope_10y_2y', 1.0)),
            'unemployment': inputs.get('unemployment', self.medians.get('unemployment', 5.0)),
        }
        
        # Handle missing values
        for key in raw:
            if raw[key] is None or np.isnan(raw[key]):
                raw[key] = self.medians.get(key, 0.0)
        
        # Create feature vector
        X = np.array([[raw[f] for f in self._feature_order]])
        
        # Scale features
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X
    
    @classmethod
    def load(cls, path: Optional[str] = None) -> 'CPILargeMovePredictor':
        """
        Load a trained predictor from disk.
        
        Args:
            path: Path to the saved model. If None, uses default path.
            
        Returns:
            Loaded CPILargeMovePredictor instance
        """
        if path is None:
            path = cls.DEFAULT_MODEL_PATH
        
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model found at {path}. "
                f"Run training first or specify a valid path."
            )
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        return cls(
            model=data['model'],
            scaler=data['scaler'],
            medians=data['medians'],
            threshold=data['threshold'],
        )
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the predictor to disk.
        
        Args:
            path: Path to save the model. If None, uses default path.
        """
        if path is None:
            path = self.DEFAULT_MODEL_PATH
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'medians': self.medians,
            'threshold': self.threshold,
            'version': self.version,
            'feature_order': self._feature_order,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Model saved to {path}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about this predictor."""
        base = super().get_metadata()
        base.update({
            'threshold': self.threshold,
            'feature_order': self._feature_order,
            'model_type': 'RandomForestClassifier',
            'class_weight': {False: 1, True: 50},
            'performance': {
                'false_negative_rate': 0.0,
                'false_positive_rate': 0.661,
                'test_period': '2003-2025',
            },
        })
        return base
    
    @classmethod
    def train(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        threshold: float = 0.099,
    ) -> 'CPILargeMovePredictor':
        """
        Train a new predictor.
        
        Args:
            X_train: Training features (n_samples, 6)
            y_train: Training labels (n_samples,)
            threshold: Probability threshold for prediction
            
        Returns:
            Trained CPILargeMovePredictor instance
        """
        # Fit scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight={False: 1, True: 50},
            random_state=42,
        )
        model.fit(X_scaled, y_train)
        
        # Compute medians for imputation
        feature_names = [
            'yield_volatility', 'cpi_shock_mom', 'cpi_abs',
            'fed_funds', 'slope_10y_2y', 'unemployment'
        ]
        medians = {name: float(np.nanmedian(X_train[:, i])) 
                   for i, name in enumerate(feature_names)}
        
        return cls(
            model=model,
            scaler=scaler,
            medians=medians,
            threshold=threshold,
        )

