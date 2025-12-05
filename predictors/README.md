# Predictors

Black-box predictors for financial events. Each predictor is a self-contained module
that can be composed into larger Markov/Bayesian networks as "complex edges."

## Philosophy

```
[Market Conditions] --[CPILargeMovePredictor]--> [Yield Move Risk]
        |                                              |
        |                                              v
        +------------[FedMeetingPredictor]-------> [Rate Decision]
```

Each predictor:
- **Takes inputs:** Features known at prediction time
- **Returns outputs:** Probability + prediction + confidence
- **Is a black box:** Handles its own preprocessing, scaling, etc.
- **Can be composed:** Acts as an edge in a probabilistic graph

## Available Predictors

| Predictor | Description | FN Rate | FP Rate |
|-----------|-------------|---------|---------|
| `CPILargeMovePredictor` | Large yield moves around CPI | 0% | 72% |

## Quick Start

```python
from predictors import CPILargeMovePredictor

# Load trained model
predictor = CPILargeMovePredictor.load()

# Make prediction
result = predictor.predict({
    'yield_volatility': 0.05,
    'cpi_shock_mom': 0.1,
    'fed_funds': 2.5,
    'slope_10y_2y': 1.0,
    'unemployment': 4.0,
})

print(result.probability)   # 0.23
print(result.prediction)    # True (above threshold)
print(result.confidence)    # 'high'
```

## Creating New Predictors

See `docs/building_predictors_guide.md` for the full methodology. Quick steps:

### 1. Define the Problem

```python
# Be specific:
# - What: |yield_change| > 10bp
# - When: Within 1 day of CPI announcement
# - Cost: FN is 50x worse than FP
```

### 2. Create Folder Structure

```
predictors/
└── my_predictor/
    ├── __init__.py
    ├── model.py      # Predictor class
    ├── train.py      # Training script
    └── README.md     # Documentation
```

### 3. Implement the Predictor

```python
from predictors.base import BasePredictor, PredictorOutput

class MyPredictor(BasePredictor):
    name = "MyPredictor"
    version = "1.0.0"
    description = "What this predicts"
    required_inputs = ['feature1', 'feature2']
    
    def predict(self, inputs):
        self.validate_inputs(inputs)
        features = self._prepare_features(inputs)
        proba = self.model.predict_proba(features)[0, 1]
        return PredictorOutput(
            probability=float(proba),
            prediction=proba >= self.threshold,
            confidence='high' if proba > 0.3 or proba < 0.05 else 'medium'
        )
    
    def predict_batch(self, inputs):
        return [self.predict(inp) for inp in inputs]
    
    @classmethod
    def load(cls, path=None):
        # Load from disk
        ...
    
    def save(self, path):
        # Save to disk
        ...
```

### 4. Key Training Principles

- **Chronological split:** Never shuffle time series
- **Fit scaler on train only:** `scaler.fit(X_train)`
- **Select threshold via CV:** Use TimeSeriesSplit on train+val
- **Use class_weight:** For asymmetric costs, not oversampling
- **Evaluate test once:** Never tune on test data

### 5. Export and Document

```python
# In predictors/__init__.py
from .my_predictor import MyPredictor
__all__ = [..., 'MyPredictor']
```

## Future: Network Composition

```python
from predictors import PredictorGraph

graph = PredictorGraph()
graph.add_predictor('cpi_move', CPILargeMovePredictor.load())
graph.add_predictor('fed_decision', FedMeetingPredictor.load())  # Future

# Define edges
graph.add_edge('MarketConditions', 'YieldMoveRisk', 'cpi_move')
graph.add_edge('MarketConditions', 'RateDecision', 'fed_decision')

# Propagate beliefs (future feature)
beliefs = graph.propagate({'MarketConditions': current_state})
```

## Folder Structure

```
predictors/
├── __init__.py           # Exports all predictors
├── base.py               # BasePredictor and PredictorOutput classes
├── README.md             # This file
│
└── cpi_large_move/       # CPI Large Move Predictor
    ├── __init__.py       # Exports CPILargeMovePredictor
    ├── model.py          # Predictor implementation
    ├── train.py          # Training script
    ├── trained_model.pkl # Saved model weights
    └── README.md         # Documentation
```

## Design Principles

1. **Modularity:** Each predictor is independent
2. **Auditability:** Every predictor has a clear train/test split and metrics
3. **Composability:** Predictors can be combined in larger networks
4. **Reproducibility:** Training is deterministic (random_state=42)
5. **No Leakage:** All preprocessing uses train-only statistics

## Related Docs

- `docs/building_predictors_guide.md` - **Comprehensive guide for building new predictors**
- `docs/anomaly_detection_strategy.md` - Strategy for anomaly detection
- `docs/audit_report.md` - Full audit of methodology and data leakage checks

