"""
Base Predictor Interface
========================

All predictors in this module inherit from BasePredictor and implement:
- predict(inputs) -> probability or prediction
- get_metadata() -> dict of model info

Design Philosophy:
- Predictors are BLACK BOXES: you give inputs, you get outputs
- Each predictor handles its own preprocessing internally
- Predictors can be composed as edges in larger Markov/Bayesian networks
- All predictors expose probability outputs for downstream composition
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np


@dataclass
class PredictorOutput:
    """Standard output from a predictor."""
    
    probability: float  # P(event) in [0, 1]
    prediction: bool    # Binary classification (above threshold)
    confidence: str     # 'low', 'medium', 'high'
    
    # Optional metadata
    threshold: Optional[float] = None
    features_used: Optional[Dict[str, float]] = None
    
    def __repr__(self):
        return f"PredictorOutput(p={self.probability:.3f}, pred={self.prediction}, conf={self.confidence})"


class BasePredictor(ABC):
    """
    Abstract base class for all predictors.
    
    Predictors are designed to be used as edges in a Markov/Bayesian network:
    
        [Node A] --[Predictor]--> [Node B]
    
    Where the predictor computes P(B | A) based on complex logic.
    
    Example Usage:
        predictor = SomePredictor.load()
        result = predictor.predict({
            'feature1': 0.5,
            'feature2': 1.2,
        })
        print(result.probability)  # 0.73
        print(result.prediction)   # True
    """
    
    # Subclasses should override these
    name: str = "BasePredictor"
    version: str = "0.0.0"
    description: str = "Abstract base predictor"
    
    # Required inputs for this predictor
    required_inputs: List[str] = []
    
    @abstractmethod
    def predict(self, inputs: Dict[str, Any]) -> PredictorOutput:
        """
        Make a prediction given inputs.
        
        Args:
            inputs: Dictionary of feature_name -> value
            
        Returns:
            PredictorOutput with probability, prediction, and confidence
        """
        pass
    
    @abstractmethod
    def predict_batch(self, inputs: List[Dict[str, Any]]) -> List[PredictorOutput]:
        """
        Make predictions for multiple input sets.
        
        Args:
            inputs: List of input dictionaries
            
        Returns:
            List of PredictorOutput objects
        """
        pass
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate that all required inputs are present."""
        missing = [k for k in self.required_inputs if k not in inputs]
        if missing:
            raise ValueError(f"Missing required inputs: {missing}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about this predictor."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'required_inputs': self.required_inputs,
        }
    
    @classmethod
    @abstractmethod
    def load(cls, path: Optional[str] = None) -> 'BasePredictor':
        """Load a trained predictor from disk."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the predictor to disk."""
        pass
    
    def __repr__(self):
        return f"{self.name}(v{self.version})"


class PredictorGraph:
    """
    A graph of predictors that can be composed together.
    
    This is a placeholder for future Markov/Bayesian network composition.
    Each predictor acts as a complex edge in the graph.
    
    Future Features:
    - Define nodes (random variables)
    - Connect nodes via predictors
    - Propagate beliefs through the network
    - Handle uncertainty composition
    """
    
    def __init__(self):
        self.predictors: Dict[str, BasePredictor] = {}
        self.edges: List[tuple] = []  # (from_node, to_node, predictor_name)
    
    def add_predictor(self, name: str, predictor: BasePredictor):
        """Add a predictor to the graph."""
        self.predictors[name] = predictor
    
    def add_edge(self, from_node: str, to_node: str, predictor_name: str):
        """Connect two nodes via a predictor."""
        if predictor_name not in self.predictors:
            raise ValueError(f"Unknown predictor: {predictor_name}")
        self.edges.append((from_node, to_node, predictor_name))
    
    def __repr__(self):
        return f"PredictorGraph({len(self.predictors)} predictors, {len(self.edges)} edges)"

