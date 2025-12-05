"""
Predictors Module
=================

Black-box predictors for financial events. Each predictor:
- Takes inputs (features known at prediction time)
- Returns a probability or prediction
- Can be composed in Markov/Bayesian networks as complex edges

Available Predictors:
- CPILargeMovePredictor: Detects large yield moves around CPI announcements
"""

from .base import BasePredictor
from .cpi_large_move import CPILargeMovePredictor

__all__ = [
    'BasePredictor',
    'CPILargeMovePredictor',
]

