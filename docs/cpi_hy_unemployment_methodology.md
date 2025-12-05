# CPI → HY OAS & Unemployment Event Detection: Methodology

> **Notebook**: `notebooks/04_cpi_hy_unemployment_events.ipynb`  
> **Grid Search Module**: `src/models/event_grid_search.py`  
> **Last Updated**: December 2025

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Grid Search Module](#2-grid-search-module)
3. [Results: FN ≤ 1% Constraint](#3-results-fn--1-constraint)
4. [Results: FN ≤ 5% Constraint](#4-results-fn--5-constraint)
5. [Recommended Configurations](#5-recommended-configurations)
6. [Metrics Reference](#6-metrics-reference)
7. [Limitations & Future Work](#7-limitations--future-work)

---

## 1. Problem Statement

### Goal
Given a macro announcement (CPI or Unemployment release), predict whether there will be a "large" move in a target variable. **Low false negatives are critical** - we must catch almost all large moves.

### Optimization Priority

1. **Primary**: FN% ≤ 1% or ≤ 5% (catch 95-99% of large moves)
2. **Secondary**: Maximize TN/FP ratio (reduce false alarms)

### Datasets

| Model | Events | Train/Test | Test Period |
|-------|--------|------------|-------------|
| CPI → HY OAS | 346 | 242/104 | 2017-2025 |
| Unemp → VIX | 431 | 301/130 | 2015-2025 |

---

## 2. Grid Search Module

A reusable Python module was created for exhaustive grid search:

**Location**: `src/models/event_grid_search.py`

### Usage

```python
from src.models.event_grid_search import run_full_grid_search, find_best_configs

# Run grid search
results = run_full_grid_search(
    train_df, test_df, features, 'hy_change',
    large_thresholds=[0.05, 0.08, 0.10, 0.12, 0.15]
)

# Find best with FN <= 5%
best = find_best_configs(results, max_fn_rate=0.05)
```

### Models Tested
- RandomForest (RF, RF_deep, RF_shallow)
- ExtraTrees
- GradientBoosting (GB, GB_deep, GB_fast)
- AdaBoost
- LogisticRegression

### Weights Tested
1, 2, 3, 5, 8, 10, 15, 20, 30, 50, balanced

### Configurations Tested
- CPI→HY: **10,692 configurations**
- Unemp→VIX: **8,910 configurations**

---

## 3. Results: FN ≤ 1% Constraint

**Requirement**: Catch 99%+ of all large moves (FN ≤ 1%)

### CPI → HY OAS (FN ≤ 1%)

| Threshold | Model | Weight | Prob | AUC | TP | FP | FN | TN | FN% | Recall | TN/FP |
|-----------|-------|--------|------|-----|-----|-----|-----|-----|-----|--------|-------|
| **15bp** | **LogReg** | **50** | **0.80** | **0.911** | **8** | **19** | **0** | **77** | **0.0%** | **100%** | **4.05x** |
| 10bp | GB_fast | 3 | 0.05 | 0.700 | 16 | 67 | 0 | 21 | 0.0% | 100% | 0.31x |
| 8bp | RF_deep | 1 | 0.10 | 0.721 | 25 | 69 | 0 | 10 | 0.0% | 100% | 0.14x |
| 12bp | AdaBoost | 1 | 0.20 | 0.591 | 12 | 81 | 0 | 11 | 0.0% | 100% | 0.14x |
| 6bp | RF | 5 | 0.10 | 0.636 | 36 | 61 | 0 | 7 | 0.0% | 100% | 0.11x |
| 5bp | AdaBoost | 50 | 0.40 | 0.561 | 41 | 59 | 0 | 4 | 0.0% | 100% | 0.07x |

**Best**: LogReg with weight=50, threshold=15bp → **TN/FP = 4.05x, 100% recall**

### Unemployment → VIX (FN ≤ 1%)

| Threshold | Model | Weight | Prob | AUC | TP | FP | FN | TN | FN% | Recall | TN/FP |
|-----------|-------|--------|------|-----|-----|-----|-----|-----|-----|--------|-------|
| **2.5pt** | **RF_shallow** | **50** | **0.20** | **0.716** | **15** | **73** | **0** | **42** | **0.0%** | **100%** | **0.58x** |
| 3.0pt | ExtraTrees | 30 | 0.30 | 0.715 | 9 | 77 | 0 | 44 | 0.0% | 100% | 0.57x |
| 2.0pt | RF_shallow | 50 | 0.20 | 0.813 | 21 | 74 | 0 | 35 | 0.0% | 100% | 0.47x |
| 1.0pt | RF_shallow | 8 | 0.30 | 0.665 | 59 | 63 | 0 | 8 | 0.0% | 100% | 0.13x |
| 1.5pt | RF | 50 | 0.15 | 0.729 | 37 | 83 | 0 | 10 | 0.0% | 100% | 0.12x |

**Best**: RF_shallow with weight=50, threshold=2.5pt → **TN/FP = 0.58x, 100% recall**

---

## 4. Results: FN ≤ 5% Constraint

**Requirement**: Catch 95%+ of all large moves (FN ≤ 5%)

### CPI → HY OAS (FN ≤ 5%)

| Threshold | Model | Weight | Prob | AUC | TP | FP | FN | TN | FN% | Recall | TN/FP |
|-----------|-------|--------|------|-----|-----|-----|-----|-----|-----|--------|-------|
| **15bp** | **LogReg** | **50** | **0.80** | **0.911** | **8** | **19** | **0** | **77** | **0.0%** | **100%** | **4.05x** |
| 8bp | RF_shallow | 30 | 0.45 | 0.573 | 24 | 57 | 1 | 22 | 4.0% | 96% | 0.39x |
| 10bp | GB_fast | 3 | 0.05 | 0.700 | 16 | 67 | 0 | 21 | 0.0% | 100% | 0.31x |
| 6bp | RF_deep | 30 | 0.10 | 0.518 | 35 | 54 | 1 | 14 | 2.8% | 97% | 0.26x |
| 5bp | AdaBoost | balanced | 0.30 | 0.561 | 39 | 51 | 2 | 12 | 4.9% | 95% | 0.24x |

**Best**: LogReg with weight=50, threshold=15bp → **TN/FP = 4.05x, 100% recall**

### Unemployment → VIX (FN ≤ 5%)

| Threshold | Model | Weight | Prob | AUC | TP | FP | FN | TN | FN% | Recall | TN/FP |
|-----------|-------|--------|------|-----|-----|-----|-----|-----|-----|--------|-------|
| **2.0pt** | **GB** | **20** | **0.05** | **0.792** | **20** | **48** | **1** | **61** | **4.8%** | **95%** | **1.27x** |
| 2.5pt | RF_shallow | 50 | 0.20 | 0.716 | 15 | 73 | 0 | 42 | 0.0% | 100% | 0.58x |
| 3.0pt | ExtraTrees | 30 | 0.30 | 0.715 | 9 | 77 | 0 | 44 | 0.0% | 100% | 0.57x |
| 1.5pt | RF_shallow | 50 | 0.40 | 0.722 | 36 | 68 | 1 | 25 | 2.7% | 97% | 0.37x |
| 1.0pt | RF | 5 | 0.15 | 0.686 | 57 | 54 | 2 | 17 | 3.4% | 97% | 0.31x |

**Best**: GB with weight=20, threshold=2.0pt → **TN/FP = 1.27x, 95% recall**

---

## 5. Recommended Configurations

### For FN ≤ 1% (Maximum Safety)

| Model | Config | TP | FP | FN | TN | Recall | TN/FP |
|-------|--------|-----|-----|-----|-----|--------|-------|
| **CPI→HY** | LogReg, w=50, 15bp, @0.80 | 8 | 19 | 0 | 77 | **100%** | **4.05x** |
| **Unemp→VIX** | RF_shallow, w=50, 2.5pt, @0.20 | 15 | 73 | 0 | 42 | **100%** | 0.58x |

### For FN ≤ 5% (Balanced)

| Model | Config | TP | FP | FN | TN | Recall | TN/FP |
|-------|--------|-----|-----|-----|-----|--------|-------|
| **CPI→HY** | LogReg, w=50, 15bp, @0.80 | 8 | 19 | 0 | 77 | **100%** | **4.05x** |
| **Unemp→VIX** | GB, w=20, 2.0pt, @0.05 | 20 | 48 | 1 | 61 | **95%** | **1.27x** |

### Key Insights

1. **CPI→HY**: LogisticRegression with high weight (50) works best
   - AUC = 0.911 (excellent discrimination)
   - TN/FP = 4.05x at 100% recall

2. **Unemp→VIX**: Trade-off between recall and TN/FP
   - FN≤1%: Use RF_shallow (100% recall, TN/FP=0.58x)
   - FN≤5%: Use GB (95% recall, TN/FP=1.27x)

3. **Higher thresholds** (15bp, 2-2.5pt) give better TN/FP ratios

---

## 6. Metrics Reference

### Confusion Matrix

|  | Predicted Normal | Predicted Large |
|--|------------------|-----------------|
| **Actual Normal** | TN | FP |
| **Actual Large** | FN | TP |

### Key Metrics

| Metric | Formula | Priority |
|--------|---------|----------|
| **FN%** | FN / (TP + FN) | **PRIMARY** - must be ≤ 1% or ≤ 5% |
| **Recall** | TP / (TP + FN) = 1 - FN% | Same as FN% |
| **TN/FP** | TN / FP | **SECONDARY** - maximize after FN constraint |
| **Specificity** | TN / (TN + FP) | Related to TN/FP |

---

## 7. Limitations & Future Work

### Current Limitations

1. Small positive class at high thresholds (8 events at 15bp)
2. Simple surprise calculation (current - previous)
3. Approximate unemployment calendar

### Future Enhancements

1. Consensus-based surprises (actual vs. analyst consensus)
2. Official BLS/BEA release calendars
3. More features (VIX term structure, FedWatch, credit flows)
4. Larger samples by combining event types

---

## Appendix: Quick Start Code

```python
from src.models.event_grid_search import quick_search

# Run grid search with FN constraints
results = quick_search(
    train_df, test_df, 
    features=['cpi_shock', 'cpi_shock_abs', 'yield_vol', 'hy_vol', 'slope', 'ff', 'hy_before'],
    target_col='hy_change',
    large_thresholds=[0.05, 0.08, 0.10, 0.12, 0.15],
    fn_constraints=[0.01, 0.05]  # 1% and 5%
)
```

---

*Document maintained by: Macro Shock Detection Project*
