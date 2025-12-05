# CPI Large Move Predictor

Detects large yield moves (|change| > 10bp) around CPI announcements.

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
print(result.prediction)    # True
print(result.confidence)    # 'high'
```

## Performance

| Metric | Value |
|--------|-------|
| False Negative Rate | **0.0%** |
| False Positive Rate | 66.1% |
| Test Period | 2003-2025 |

**Interpretation:** Never misses a large move, but flags ~2/3 of normal events.

## Required Inputs

| Feature | Description | Typical Range |
|---------|-------------|---------------|
| `yield_volatility` | 20-day trailing std of 10Y yield | 0.02 - 0.15 |
| `cpi_shock_mom` | MoM CPI surprise (actual - expected) | -0.5 to +0.5 |
| `fed_funds` | Current Fed Funds rate | 0 - 20 |
| `slope_10y_2y` | Yield curve slope (10Y - 2Y) | -1 to +3 |
| `unemployment` | Unemployment rate | 3 - 12 |

## Model Details

- **Algorithm:** Cost-Sensitive Random Forest
- **Class Weight:** `{False: 1, True: 50}` (heavily penalize missing abnormals)
- **Threshold:** 0.099 (selected via Time Series CV for FN=0)
- **Features:** 6 total (includes derived `cpi_abs`)

## Use Cases

### Risk Management
```python
if result.prediction:
    reduce_position()
    increase_hedges()
```

### Pre-Event Alert
```python
# Before each CPI release
result = predictor.predict(current_conditions)
if result.probability > 0.15:
    alert("High-impact CPI release expected")
```

### Network Composition
```python
# This predictor can be an edge in a larger network
graph.add_edge('MarketConditions', 'LargeMoveRisk', predictor)
```

## Files

- `model.py` - Predictor implementation
- `trained_model.pkl` - Trained model weights (after training)
- `README.md` - This file

## Training

To retrain the model:

```python
from predictors.cpi_large_move import CPILargeMovePredictor

# Prepare your data
X_train = ...  # (n_samples, 6) array
y_train = ...  # (n_samples,) boolean array

# Train and save
predictor = CPILargeMovePredictor.train(X_train, y_train)
predictor.save()
```

## Caveats

1. **High False Positive Rate:** By design, to achieve 0% FN
2. **Feature Distribution Shift:** Trained on 1976-2003, tested on 2003-2025
3. **Missing Features:** If any feature unavailable, uses training median

## Audit

This model has been audited for:
- ✅ No lookahead bias in features
- ✅ No data leakage in threshold selection
- ✅ Clean train/test separation

See `docs/audit_report.md` for full details.

