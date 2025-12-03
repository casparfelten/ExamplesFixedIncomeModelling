# Markov Regime-Switching Classifier: Implementation Plan

## Overview

This document describes the design and implementation plan for a **Markov-augmented regime-switching classifier** that predicts binned yield changes around CPI announcements. The key innovation is discovering regimes from **model error patterns** rather than just background features.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 1: Global Model                                                        │
│  ┌────────────────────┐                                                      │
│  │ All Training Data  │ ──► GradientBoostingClassifier ──► Global Model     │
│  │ (CPI events)       │         f_global(X)                                  │
│  └────────────────────┘                                                      │
│           │                                                                  │
│           ▼                                                                  │
│  Step 2: Error Analysis                                                      │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ For each event t:                                                   │     │
│  │   - Run global model → get prediction                               │     │
│  │   - Compare to actual → compute error flags                         │     │
│  │   - Extract background features (slope, VIX, HY OAS, STLFSI)       │     │
│  │   - Create composite vector: Z_t = [background, error_flags]        │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│           │                                                                  │
│           ▼                                                                  │
│  Step 3: Regime Discovery                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ Cluster Z_t vectors using K-Means or GMM                           │     │
│  │   - K = 3 or 4 regimes                                              │     │
│  │   - Each cluster = a "regime" where model behaves differently       │     │
│  │   - Assign regime labels R_t to each event                          │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│           │                                                                  │
│           ▼                                                                  │
│  Step 4: Transition Matrix                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ Sort events by date                                                 │     │
│  │ Count transitions: N_ij = #{t : R_{t-1}=i, R_t=j}                  │     │
│  │ Row-normalize → Π (K×K transition probability matrix)              │     │
│  │ Compute stationary distribution π_stationary                        │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│           │                                                                  │
│           ▼                                                                  │
│  Step 5: Regime-Specific Models                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ For each regime k = 0, 1, ..., K-1:                                 │     │
│  │   - Filter events where R_t = k                                     │     │
│  │   - Train GradientBoostingClassifier f_k(X)                        │     │
│  │   - These specialize in predicting yields in their regime           │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREDICTION PHASE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Given: New CPI event with features X_t, previous regime R_{t-1}            │
│                                                                              │
│  1. Regime Prior:                                                            │
│     P(R_t = k) = Π[R_{t-1}, k]  (row of transition matrix)                  │
│                                                                              │
│  2. Per-Regime Predictions:                                                  │
│     For each k: p^(k) = f_k(X_t)  (bin probabilities from regime-k model)   │
│                                                                              │
│  3. Mixture:                                                                 │
│     p_final = Σ_k P(R_t = k) × p^(k)                                        │
│                                                                              │
│  4. Output: Probability distribution over yield bins                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Requirements

### Existing Data (from `prepare_event_data`)
| Feature | Description | Source |
|---------|-------------|--------|
| `cpi_shock_mom` | Month-over-month CPI change | CPI releases |
| `cpi_shock_yoy` | Year-over-year CPI change | CPI releases |
| `y_2y_change` | 2-year yield change on CPI day | Treasury data |
| `slope_10y_2y` | Yield curve slope (10Y - 2Y) | FRED |
| `fed_funds` | Federal funds rate | FRED |
| `unemployment` | Unemployment rate | FRED |
| `gdp` | GDP | FRED |

### New Background Features (to integrate)
| Feature | FRED Series | Description | Frequency |
|---------|-------------|-------------|-----------|
| `hy_oas` | BAMLH0A0HYM2 | High-yield option-adjusted spread | Daily |
| `vix` | VIXCLS | CBOE Volatility Index | Daily |
| `stlfsi` | STLFSI4 | St. Louis Fed Financial Stress Index | Weekly |

### Derived Features
| Feature | Formula | Description |
|---------|---------|-------------|
| `S_CPI` | `(cpi_shock_mom - mean) / std` | Standardized CPI surprise |
| `S_CPI_abs` | `|S_CPI|` | Absolute CPI surprise magnitude |
| `yield_vol_20d` | Rolling 20-day std of yield changes | Recent yield volatility |

---

## Error Flags for Regime Discovery

When running the global model on training data, classify each prediction error:

```python
error_flags = {
    'correct': prediction == actual,
    'missed_large_down': actual == 'Large Down' and prediction != 'Large Down',
    'missed_large_up': actual == 'Large Up' and prediction != 'Large Up',
    'false_alarm_large': prediction in ['Large Down', 'Large Up'] and actual == 'Neutral',
    'wrong_direction': (actual in ['Down', 'Large Down'] and prediction in ['Up', 'Large Up']) or vice versa,
    'magnitude_error': signed difference between predicted and actual bins
}
```

### Composite Regime-Discovery Vector

For each event t:
```
Z_t = [
    # Background features (normalized)
    slope_10y_2y,
    vix,
    hy_oas,
    stlfsi,
    
    # Error indicators (from global model)
    missed_large_move,      # binary: 1 if actual was large and prediction wasn't
    wrong_direction,        # binary: 1 if predicted opposite direction
    magnitude_error,        # integer: actual_bin - predicted_bin
]
```

---

## Implementation Plan

### Phase 1: Data Integration

**File: `src/data/fred_loader.py`**
- Ensure new series (VIX, HY OAS, STLFSI) are loaded and merged
- Handle weekly STLFSI by forward-filling to daily
- Add to the merged panel

**File: `src/models/prepare_data.py`**
- Update `prepare_event_data()` to include new background features
- Add derived features (S_CPI, yield_vol_20d)

### Phase 2: Core Model Classes

**File: `src/models/markov_regime_classifier.py`** (NEW)

```python
class MarkovRegimeClassifier:
    """
    Markov-augmented regime-switching classifier for CPI-yield prediction.
    
    Architecture:
    1. Global model: baseline classifier trained on all data
    2. Regime discovery: cluster events by (background, model_error)
    3. Transition matrix: Markov chain over regimes
    4. Regime models: specialized classifiers per regime
    5. Mixture prediction: combine regime priors with per-regime predictions
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        global_model_params: dict = None,
        regime_model_params: dict = None,
        background_features: list = None,
        shock_features: list = None,
        bin_edges: list = None,
    ):
        pass
    
    # ========== TRAINING ==========
    
    def fit(self, train_df: pd.DataFrame, target_yield: str = 'y_2y') -> dict:
        """Full training pipeline."""
        self._fit_global_model(train_df, target_yield)
        self._compute_error_flags(train_df, target_yield)
        self._discover_regimes(train_df)
        self._fit_transition_matrix(train_df)
        self._fit_regime_models(train_df, target_yield)
        return self.get_training_summary()
    
    def _fit_global_model(self, train_df, target_yield):
        """Step 1: Train global baseline classifier."""
        pass
    
    def _compute_error_flags(self, train_df, target_yield):
        """Step 2a: Run global model, compute prediction errors."""
        pass
    
    def _discover_regimes(self, train_df):
        """Step 2b & 3: Cluster (background, errors) to find regimes."""
        pass
    
    def _fit_transition_matrix(self, train_df):
        """Step 4: Build Markov transition matrix from regime sequence."""
        pass
    
    def _fit_regime_models(self, train_df, target_yield):
        """Step 5: Train per-regime classifiers."""
        pass
    
    # ========== PREDICTION ==========
    
    def predict_proba(
        self, 
        X: pd.DataFrame, 
        prev_regime: int = None,
        return_regime_info: bool = False
    ) -> np.ndarray:
        """
        Predict probability distribution over yield bins.
        
        Args:
            X: Features for event(s)
            prev_regime: Previous regime (for Markov prior). 
                        If None, use stationary distribution.
            return_regime_info: If True, also return regime probabilities
        
        Returns:
            Array of shape (n_events, n_bins) with bin probabilities
        """
        pass
    
    def predict(self, X: pd.DataFrame, prev_regime: int = None) -> np.ndarray:
        """Predict most likely bin."""
        proba = self.predict_proba(X, prev_regime)
        return np.argmax(proba, axis=1)
    
    def infer_regime(self, X: pd.DataFrame) -> np.ndarray:
        """
        Infer most likely regime from background features only.
        Used at test time when we don't have error flags.
        """
        pass
    
    # ========== EVALUATION ==========
    
    def evaluate(self, test_df: pd.DataFrame, target_yield: str = 'y_2y') -> dict:
        """Full evaluation with regime-aware metrics."""
        pass
    
    def evaluate_per_regime(self, test_df, target_yield) -> dict:
        """Breakdown of performance by regime."""
        pass
    
    def evaluate_large_moves(self, test_df, target_yield) -> dict:
        """Detailed large-move detection metrics."""
        pass
    
    # ========== ANALYSIS ==========
    
    def get_regime_characteristics(self) -> pd.DataFrame:
        """Describe each regime (mean background, error patterns)."""
        pass
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """Return formatted transition matrix."""
        pass
    
    def plot_regime_distribution(self):
        """Visualize regime assignments over time."""
        pass
    
    def plot_confusion_by_regime(self):
        """Per-regime confusion matrices."""
        pass
```

### Phase 3: Notebook Implementation

**File: `notebooks/04_markov_regime_classifier.ipynb`** (NEW)

```
Cell structure:

1. Introduction & Setup
   - Imports, logging, configuration
   - Explanation of the architecture

2. Load and Prepare Data
   - Load event data with new background features
   - Show data summary and new features
   - Train/test split (70/30, chronological)

3. Step 1: Global Baseline Model
   - Train GradientBoostingClassifier on all training data
   - Evaluate: accuracy, confusion matrix, large-move detection
   - This is our reference point

4. Step 2: Error Analysis
   - Run global model on training data
   - Compute error flags for each event
   - Visualize: which events are hard? any patterns?

5. Step 3: Regime Discovery
   - Build Z_t vectors (background + error flags)
   - Cluster with K-Means (K=3)
   - Analyze clusters: what makes each regime distinct?
   - Visualize regime characteristics

6. Step 4: Markov Transition Matrix
   - Build transition matrix from regime sequence
   - Check: diagonal dominance (persistence)
   - Compute stationary distribution
   - Visualize transitions

7. Step 5: Train Regime-Specific Models
   - Train per-regime classifiers
   - Compare to global model within each regime
   - Check: do some regimes have better/worse accuracy?

8. Step 6: Combined Predictor
   - Implement mixture prediction
   - Walk through example predictions

9. Full Evaluation
   - Overall accuracy comparison: global vs Markov-regime
   - Per-regime performance
   - Large-move detection: precision, recall, F1
   - False negative analysis
   - Statistical significance test

10. Analysis & Insights
    - What do regimes represent economically?
    - When does Markov-regime beat global?
    - Feature importance by regime

11. Production Interface
    - Function: predict_next_cpi_reaction(cpi_surprise, background, prev_regime)
    - Example usage
```

---

## Key Design Decisions

### 1. Number of Regimes (K)
**Choice: K = 3**

Rationale:
- ~600 training events / 3 = ~200 per regime (adequate sample size)
- Interpretable: "calm", "nervous", "stressed" markets
- Can increase to 4 if cluster analysis suggests it

### 2. Error Flags Definition

**Primary flags:**
| Flag | Definition |
|------|------------|
| `missed_large` | Actual was Large Up/Down, predicted wasn't |
| `false_alarm` | Predicted Large, actual was Neutral |
| `wrong_direction` | Predicted Up, actual Down (or vice versa) |

**Numeric:**
| Flag | Definition |
|------|------------|
| `magnitude_error` | actual_bin_idx - predicted_bin_idx |

### 3. Background Features for Regime Discovery

```python
BACKGROUND_FEATURES = [
    'slope_10y_2y',   # Yield curve shape
    'vix',            # Equity volatility (risk sentiment)
    'hy_oas',         # Credit spreads (credit stress)
    'stlfsi',         # Financial stress composite
]
```

### 4. Shock Features for Prediction

```python
SHOCK_FEATURES = [
    'S_CPI',          # Standardized CPI surprise
    'S_CPI_abs',      # Magnitude of surprise
    'cpi_shock_yoy',  # YoY context
    'fed_funds',      # Policy rate context
    'slope_10y_2y',   # Curve shape
]
```

### 5. Bin Definition

```python
BIN_EDGES = [-np.inf, -0.10, -0.03, 0.03, 0.10, np.inf]
BIN_LABELS = ['Large Down', 'Small Down', 'Neutral', 'Small Up', 'Large Up']
```

### 6. Regime Inference at Test Time

At test time, we don't have error flags (we don't know if model will be wrong). Two options:

**Option A (Simple):** Use previous regime + Markov transition
- P(R_t | R_{t-1}) = Π[R_{t-1}, :]
- Works if regimes are persistent

**Option B (Background-based):** Train a regime classifier
- Separate model: background → regime
- Apply at test time to infer regime from background only
- More robust if regimes correlate with background

**Decision: Implement both, compare performance.**

---

## Evaluation Metrics

### Primary Metrics
| Metric | Description |
|--------|-------------|
| Overall Accuracy | % correct bin predictions |
| Large Move Precision | When we predict large, how often correct? |
| Large Move Recall | Of actual large moves, how many did we catch? |
| Large Move F1 | Harmonic mean of precision and recall |

### Comparison Metrics
| Comparison | Question |
|------------|----------|
| Markov-Regime vs Global | Does regime-switching improve overall? |
| Per-Regime Accuracy | Which regimes are easier/harder? |
| Transition Persistence | How stable are regimes? |

### Diagnostic Metrics
| Metric | Purpose |
|--------|---------|
| Regime Distribution | Are regimes balanced? |
| Within-Regime Error Patterns | Do regimes capture error types? |
| Lift over Random | Statistical significance |

---

## File Structure After Implementation

```
ExamplesFixedIncomeModelling/
├── src/
│   ├── models/
│   │   ├── markov_regime_classifier.py    # NEW: Core model class
│   │   ├── prepare_data.py                # MODIFIED: Add new features
│   │   └── ...
│   └── data/
│       └── fred_loader.py                 # MODIFIED: Load new series
├── notebooks/
│   ├── 04_markov_regime_classifier.ipynb  # NEW: Full implementation
│   └── ...
├── docs/
│   └── 04_markov_regime_classifier_plan.md  # THIS FILE
└── data/
    └── raw/fred/
        ├── BAMLH0A0HYM2.csv               # HY OAS
        ├── VIXCLS.csv                     # VIX
        └── STLFSI4.csv                    # STLFSI
```

---

## Implementation Order

1. **Data Integration** (30 min)
   - Update `fred_loader.py` to include new series in merge
   - Update `prepare_data.py` to add features to event dataframe
   - Test: verify new features appear in `prepare_event_data()` output

2. **Notebook Skeleton** (20 min)
   - Create `04_markov_regime_classifier.ipynb`
   - Set up imports, config, data loading
   - Train/test split

3. **Global Model** (20 min)
   - Train baseline GradientBoosting
   - Evaluate with existing metrics
   - Store as reference

4. **Error Analysis** (30 min)
   - Run global model on training data
   - Compute error flags
   - Build Z_t vectors
   - Visualize

5. **Regime Discovery** (30 min)
   - K-Means clustering on Z_t
   - Analyze cluster characteristics
   - Assign regime labels

6. **Transition Matrix** (20 min)
   - Count transitions
   - Build Π matrix
   - Compute stationary distribution

7. **Regime Models** (30 min)
   - Train K classifiers
   - Per-regime evaluation

8. **Mixture Predictor** (30 min)
   - Implement mixture logic
   - Test on held-out data

9. **Full Evaluation** (30 min)
   - Compare to global baseline
   - Large-move metrics
   - Statistical tests

10. **Extract to Module** (30 min)
    - Create `MarkovRegimeClassifier` class
    - Move notebook code to module
    - Clean up notebook to use module

---

## Expected Outcomes

### Performance Targets
| Metric | Global Baseline | Target with Markov-Regime |
|--------|-----------------|---------------------------|
| Overall Accuracy | 40.5% | 45-50% |
| Large Move Recall | 74% (at threshold 0.1) | 80%+ |
| Large Move Precision | 11% | 20%+ |

### Interpretability Gains
- Clear regime definitions (e.g., "high VIX + inverted curve = nervous regime")
- Understanding of when model works / fails
- Temporal dynamics of market regimes

---

## Open Questions (To Resolve During Implementation)

1. **Regime stability**: How many regimes change between consecutive CPI events?
2. **Feature importance by regime**: Does CPI matter more in some regimes?
3. **Optimal K**: Should we use 3 or 4 regimes?
4. **Error flag weighting**: Should `missed_large` count more than `wrong_direction` in clustering?
5. **Test-time regime inference**: Markov-only vs background-classifier—which works better?

---

## Next Steps

1. Review this plan
2. Confirm design decisions
3. Begin Phase 1: Data Integration
4. Proceed through implementation order

