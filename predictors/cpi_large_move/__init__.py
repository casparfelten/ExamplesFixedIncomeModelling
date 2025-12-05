"""
CPI Large Move Predictor
========================

Detects large yield moves (|change| > 10bp) around CPI announcements.

Usage:
    from predictors import CPILargeMovePredictor
    
    predictor = CPILargeMovePredictor.load()
    result = predictor.predict({
        'yield_volatility': 0.05,
        'cpi_shock_mom': 0.1,
        'fed_funds': 2.5,
        'slope_10y_2y': 1.0,
        'unemployment': 4.0,
    })
    
    print(result.probability)   # 0.23
    print(result.prediction)    # True (if > threshold)
    print(result.confidence)    # 'high'
"""

from .model import CPILargeMovePredictor

__all__ = ['CPILargeMovePredictor']

