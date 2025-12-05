# Building Event Predictors: A Comprehensive Guide

This document captures everything we learned building the CPI Large Move predictor,
including what worked, what failed, and how to approach similar problems.

---

## Table of Contents

1. [Philosophy](#philosophy)
2. [Problem Framing](#problem-framing)
3. [Data Preparation](#data-preparation)
4. [Model Selection](#model-selection)
5. [Evaluation Methodology](#evaluation-methodology)
6. [Avoiding Data Leakage](#avoiding-data-leakage)
7. [Threshold Selection](#threshold-selection)
8. [What We Tried and Learned](#what-we-tried-and-learned)
9. [Implementation Checklist](#implementation-checklist)

---

## Philosophy

### Predictors as Black Boxes

Each predictor should be:
- **Self-contained:** All preprocessing, scaling, and logic is internal
- **Auditable:** Clear train/test separation with documented metrics
- **Composable:** Can be used as an edge in a larger probabilistic network
- **Reproducible:** Deterministic training (use `random_state=42`)

### The Asymmetric Cost Problem

Many financial prediction problems have asymmetric costs:
- Missing a market crash is catastrophic
- False alarms are annoying but survivable

This means **optimizing for accuracy is wrong**. Instead, optimize for:
- Zero false negatives (catch ALL important events)
- Minimum false positives (reduce noise)

---

## Problem Framing

### Step 1: Define the Target Precisely

Bad: "Predict if CPI will move markets"
Good: "Predict if |yield change| > 10bp within 1 day of CPI announcement"

Be specific about:
- **What** you're predicting (yield change, price move, volatility spike)
- **Threshold** for "abnormal" (10bp, 2 standard deviations, etc.)
- **Time window** (intraday, 1-day, 1-week)
- **Baseline** (what counts as "normal")

### Step 2: Define Your Cost Function

Ask: "What's worse - a false positive or false negative?"

| Use Case | Priority |
|----------|----------|
| Risk management | FN=0 (never miss a risk event) |
| Trading signals | Balance FP/FN (precision matters) |
| Anomaly detection | FN=0 (catch all anomalies) |

### Step 3: Choose Appropriate Features

Features must be **known at prediction time**:

✅ Good features:
- Yesterday's volatility
- Current Fed Funds rate
- Previous CPI reading
- Yield curve slope (as of yesterday)

❌ Bad features (lookahead bias):
- Today's closing price
- This month's CPI reading (if predicting CPI event)
- "Average volatility for this month"

---

## Data Preparation

### Event-Based vs Time-Series

For event prediction, convert to **event-based format**:

```python
# Time-series format (bad for events)
# date, yield, cpi, fed_funds
# 2024-01-01, 4.5, 302.1, 5.25
# 2024-01-02, 4.6, 302.1, 5.25
# ...

# Event-based format (good)
# event_date, cpi_shock, yield_change, yield_volatility, fed_funds
# 2024-01-11, 0.1, 0.12, 0.05, 5.25  # CPI announcement
# 2024-02-13, -0.05, 0.03, 0.04, 5.25
# ...
```

### Feature Engineering

**Trailing indicators** (safe):
```python
# Volatility as of day BEFORE event
df['yield_volatility'] = df['yield'].rolling(20).std().shift(1)
```

**Shock/surprise** (be careful):
```python
# CPI shock = actual - previous
# Both values are known at announcement time
df['cpi_shock'] = df['cpi'] - df['cpi'].shift(1)
```

**Derived features**:
```python
df['cpi_abs'] = df['cpi_shock'].abs()  # Magnitude matters more than direction
```

### Handling Missing Values

Compute medians **from training data only**:

```python
# During training
train_medians = X_train.median()

# For prediction
X_test = X_test.fillna(train_medians)  # Use TRAIN medians
```

---

## Model Selection

### What We Tried (Ranked by Performance)

| Model | FP Rate | Why It Worked/Failed |
|-------|---------|---------------------|
| **Cost-Sensitive Random Forest** | 66-72% | ✅ Best. class_weight optimizes splits directly |
| Gradient Boosted Trees (balanced) | 68% | Good, but oversampling less effective than class_weight |
| Rule-Based Thresholds | 91% | Too permissive; OR logic catches everything |
| Polynomial Logistic Regression | 95% | Smooth boundaries don't match tabular data |
| Neural Network (MLP) | 100% | Too few samples; NNs need thousands |
| One-Class Isolation Forest | 99% | Abnormals don't look anomalous in feature space |
| GMM Hidden Variables | 100% | Clusters by features, not by abnormality |

### Key Insight: Trees Beat Neural Networks on Small Tabular Data

Why? 
- Trees create axis-aligned splits (matches how features separate classes)
- Trees handle feature interactions naturally
- Trees work with hundreds of samples; NNs need thousands
- `class_weight` integrates cost into every split decision

### Recommended Model for Asymmetric Costs

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,           # Prevent overfitting
    class_weight={
        False: 1,          # Normal events
        True: 50,          # Abnormal events (50x penalty for missing)
    },
    random_state=42,
)
```

The `class_weight` parameter tells the model: "A false negative is 50x worse than a false positive."

### Alternative: Gradient Boosted Trees

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42,
)
# Note: GBM doesn't have class_weight, so use oversampling instead
```

---

## Evaluation Methodology

### The Golden Rule

> **Never touch test data until final evaluation.**

This means:
- No peeking at test distributions
- No tuning hyperparameters on test data
- No selecting thresholds using test data
- Test data is used ONCE, at the very end

### Chronological Splitting

Time-series data requires **chronological** splits:

```
|-------- Train --------|--- Val ---|---- Test ----|
     1976-2000              2000-2003     2003-2025
```

Never shuffle time-series data! Shuffling creates lookahead bias.

```python
def create_train_test_split(df, test_size=0.30):
    df = df.sort_values('date')  # Chronological order
    n_test = int(len(df) * test_size)
    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]
    return train, test
```

### Time Series Cross-Validation

For hyperparameter tuning and threshold selection, use `TimeSeriesSplit`:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X_train):
    # Each fold: train on past, validate on future
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    # Train and evaluate...
```

This simulates real-world usage where you only have past data.

### Metrics for Asymmetric Problems

Don't use accuracy! Use:

```python
# Confusion matrix terms
FN = ((pred == False) & (actual == True)).sum()  # Missed abnormals
FP = ((pred == True) & (actual == False)).sum()  # False alarms
TP = ((pred == True) & (actual == True)).sum()   # Caught abnormals
TN = ((pred == False) & (actual == False)).sum() # Correct normals

# Key metrics
fn_rate = FN / (FN + TP)  # Miss rate (want 0%)
fp_rate = FP / (FP + TN)  # False alarm rate (minimize)
recall = TP / (TP + FN)   # Same as 1 - fn_rate
precision = TP / (TP + FP)  # Proportion of flags that are real
```

---

## Avoiding Data Leakage

### Common Leakage Sources

| Source | Example | Fix |
|--------|---------|-----|
| Feature computation | Using today's close to predict today's move | Use `.shift(1)` |
| Scaling | Fitting scaler on all data | Fit on train only |
| Missing value imputation | Using global median | Use train median |
| Threshold selection | Picking threshold that works on test | Use CV on train |
| Model selection | Choosing model based on test performance | Use validation set |

### The Leakage Checklist

Before final evaluation, verify:

1. **Features:** All computed from data known BEFORE the event
2. **Scaler:** Fitted on train+val only, then `.transform()` test
3. **Medians:** Computed from train+val only
4. **Threshold:** Selected via Time Series CV on train+val
5. **Model:** Never saw test data during training
6. **Evaluation:** Test data used exactly ONCE

### Verification Code

```python
# Verify scaler was fit on train only
print("Scaler mean:", scaler.mean_)
print("Train mean:", X_train.mean(axis=0))
print("Test mean:", X_test.mean(axis=0))
# Scaler mean should match train mean, NOT test mean
```

---

## Threshold Selection

### The Problem

Model outputs probabilities. You need a threshold to convert to predictions:
- `threshold = 0.5`: Standard, but may miss rare events
- `threshold = 0.1`: Catches more abnormals, but more false alarms

### The Solution: Time Series CV

```python
from sklearn.model_selection import TimeSeriesSplit

thresholds_per_fold = []
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X_train_val):
    X_tr, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_tr, y_val = y_train_val[train_idx], y_train_val[val_idx]
    
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_val)[:, 1]
    
    # Find minimum probability among abnormals
    if y_val.sum() > 0:
        min_abnormal_prob = proba[y_val].min()
        thresholds_per_fold.append(min_abnormal_prob)

# Use MAX threshold (most conservative for FN=0)
threshold = max(thresholds_per_fold)
```

Why MAX? Because if a fold needed a low threshold to catch its abnormals, that means the model was uncertain. Using the maximum ensures we're conservative across all folds.

---

## What We Tried and Learned

### Lesson 1: Regime Discovery Doesn't Help

We tried:
- GMM clustering on features → regimes
- Error-based clustering (where model failed)
- Sensitivity-based clustering

**Why it failed:** Regimes defined by feature clusters don't align with "sensitivity to events." A high-volatility cluster might contain both large and small yield moves.

**What works instead:** Let the tree model learn interactions implicitly through splits.

### Lesson 2: Distribution Shift is Real

| Feature | Train Period (1976-2003) | Test Period (2003-2025) |
|---------|--------------------------|-------------------------|
| fed_funds | 5.2% avg | 1.8% avg |
| yield_vol | 0.08 avg | 0.05 avg |

The model was trained in a high-rate era but tested in a low-rate era. This is why:
- Test abnormal probabilities are lower than train
- Threshold selection via CV is crucial

### Lesson 3: Feature Overlap Limits Performance

```
Feature overlap between normal and abnormal events:
  - yield_vol:  54% of normals in abnormal range
  - |cpi_shock|: 98% overlap
  - fed_funds:   96% overlap
```

With this much overlap, achieving FN=0 **requires** high FP. This is a fundamental limit of these features, not a model failure.

### Lesson 4: class_weight > Oversampling

Both handle class imbalance, but:
- `class_weight` integrates cost into the loss function at every split
- Oversampling (SMOTE) just changes the data distribution

`class_weight` is more effective because it directly optimizes for asymmetric costs.

### Lesson 5: Multiplicative Models Fail

We tried:
```
expected_move = P(high_sensitivity) × |CPI_shock| × avg_sensitivity
```

This assumes the relationship is multiplicative, but it's not. Market sensitivity to CPI depends on complex interactions that can't be captured in a product form.

---

## Implementation Checklist

### Before Training

- [ ] Define target variable precisely (what, threshold, time window)
- [ ] Define cost function (FN vs FP priority)
- [ ] Create event-based dataset
- [ ] Verify all features are known BEFORE prediction time
- [ ] Create chronological train/val/test splits
- [ ] Document date ranges for each split

### During Training

- [ ] Compute medians from train only
- [ ] Fit scaler on train only
- [ ] Use Time Series CV for threshold selection
- [ ] Use `class_weight` for asymmetric costs
- [ ] Set `random_state=42` for reproducibility

### Before Final Evaluation

- [ ] Verify scaler parameters match train statistics
- [ ] Verify threshold was selected without using test data
- [ ] Confirm test data has never been seen by model

### Final Evaluation

- [ ] Evaluate on test set ONCE
- [ ] Report FN rate, FP rate, confusion matrix
- [ ] Document any distribution shift between train and test
- [ ] Save model with version number

### Documentation

- [ ] Record all approaches tried and why they failed
- [ ] Document feature definitions and computation
- [ ] Explain threshold selection methodology
- [ ] Provide usage examples
- [ ] Include audit trail for data leakage checks

---

## Quick Reference: Model Template

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# 1. Prepare data (chronological split)
train_val, test = chronological_split(df, test_size=0.30)

# 2. Compute statistics from train+val only
medians = train_val[features].median()
scaler = StandardScaler()
X_train_val_scaled = scaler.fit_transform(train_val[features].fillna(medians))
y_train_val = train_val['target'].values

# 3. Select threshold via Time Series CV
tscv = TimeSeriesSplit(n_splits=5)
thresholds = []
for train_idx, val_idx in tscv.split(X_train_val_scaled):
    model = RandomForestClassifier(
        n_estimators=100, max_depth=5,
        class_weight={False: 1, True: 50},
        random_state=42
    )
    model.fit(X_train_val_scaled[train_idx], y_train_val[train_idx])
    proba = model.predict_proba(X_train_val_scaled[val_idx])[:, 1]
    if y_train_val[val_idx].sum() > 0:
        thresholds.append(proba[y_train_val[val_idx]].min())
threshold = max(thresholds)

# 4. Train final model on all train+val
final_model = RandomForestClassifier(
    n_estimators=100, max_depth=5,
    class_weight={False: 1, True: 50},
    random_state=42
)
final_model.fit(X_train_val_scaled, y_train_val)

# 5. Evaluate on test (ONCE)
X_test_scaled = scaler.transform(test[features].fillna(medians))
y_test = test['target'].values
proba_test = final_model.predict_proba(X_test_scaled)[:, 1]
pred_test = proba_test >= threshold

fn_rate = ((pred_test == False) & (y_test == True)).sum() / y_test.sum()
fp_rate = ((pred_test == True) & (y_test == False)).sum() / (y_test == False).sum()
print(f"FN Rate: {fn_rate:.1%}, FP Rate: {fp_rate:.1%}")
```

---

## Files in This Project

| File | Purpose |
|------|---------|
| `predictors/base.py` | BasePredictor interface and PredictorOutput |
| `predictors/cpi_large_move/model.py` | CPI predictor implementation |
| `predictors/cpi_large_move/train.py` | Training script |
| `docs/audit_report.md` | Full audit of methodology |
| `docs/anomaly_detection_strategy.md` | Strategy overview |
| `docs/building_predictors_guide.md` | This guide |

---

## Summary

1. **Frame the problem** with precise target and cost function
2. **Prepare data** in event-based format with chronological splits
3. **Use tree models** with `class_weight` for asymmetric costs
4. **Select threshold** via Time Series CV on train+val only
5. **Evaluate once** on held-out test set
6. **Document everything** including what failed

The goal is a predictor that:
- Never misses important events (FN = 0)
- Minimizes false alarms (FP as low as possible)
- Can be audited and trusted
- Can be composed into larger systems

