# QSIG Macro Graph: Edge Design Specification

> **Implementation**: `src/graph/`  
> **Build Script**: `scripts/build_macro_graph.py`  
> **Last Updated**: December 2025

---

## Table of Contents

1. [Overview](#1-overview)
2. [Core Concepts](#2-core-concepts)
3. [Data Model](#3-data-model)
4. [Runtime Flow](#4-runtime-flow)
5. [Integration with Grid Search](#5-integration-with-grid-search)
6. [Usage Examples](#6-usage-examples)
7. [API Reference](#7-api-reference)

---

## 1. Overview

### Goals

We want a **graph of macro-event → market-move edges**, where:

* Each **edge** is a *specific, trained model configuration* (model + features + threshold + backtest stats)
* Multiple edges exist for the **same mapping** (e.g. CPI→HY with FN≤1% and FN≤5% variants)
* Edges are **swappable** at runtime for policy reasons (safety vs noise)
* Each edge carries enough **backtest metadata** to later:
  * Compute confidence intervals
  * Build Bayesian reliability priors
  * Do network-level risk aggregation

### Currently Implemented Edges

| Slot ID | From | To | FN Constraints |
|---------|------|-----|----------------|
| `CPI->HY_OAS` | CPI Release | HY OAS Spread | 1%, 5% |
| `UNEMP->VIX` | Jobs Report | VIX | 1%, 5% |

---

## 2. Core Concepts

### 2.1 Nodes

**Node** = conceptual entity in the macro graph.

Two types:
- `event`: CPI, UNEMPLOYMENT, FOMC, GDP
- `instrument`: HY_OAS, VIX, YIELD_2Y, YIELD_10Y

```python
from src.graph import NodeId, NodeType

cpi_node = NodeId(
    type=NodeType.EVENT,
    name="CPI",
    description="Consumer Price Index release"
)
```

### 2.2 Edge Slots vs Edges

We distinguish:
- **EdgeSlot** = "this mapping exists in the graph" (e.g. `"CPI->HY_OAS"`)
- **Edge** = "one specific model configuration implementing this mapping"

Each EdgeSlot can have many Edges (different model types, different FN constraints).

At runtime, you choose which Edge is *active* for that slot.

### 2.3 Feature Architecture

- **Global feature set** is computed once per event-date (from FRED, calendars, etc.)
- Each Edge explicitly declares:
  - `active_factor`: primary event variable (e.g. `cpi_shock_abs`)
  - `background_features`: conditioning factors (e.g. `yield_vol_10y`, `slope_10y_2y`)

---

## 3. Data Model

### 3.1 EventContext

Represents a single macro event instance with precomputed features:

```python
from src.graph import EventContext, NodeId, NodeType

ctx = EventContext(
    node=NodeId(NodeType.EVENT, "CPI"),
    event_date="2024-03-12",
    features={
        "cpi_shock_abs": 0.3,
        "yield_vol_10y": 0.05,
        "slope_10y_2y": 1.2,
        "fed_funds": 5.25,
        "hy_oas_before": 4.0,
        "vix_before": 15.0,
        "stlfsi": 0.2,
    },
    meta={
        "release_time_utc": "13:30:00",
        "source_calendar": "BLS",
    }
)
```

### 3.2 EdgeStats

Encapsulates empirical knowledge about an Edge:

```python
EdgeStats(
    train_period=("1997-01-01", "2017-12-31"),
    test_period=("2018-01-01", "2025-12-01"),
    n_test_events=104,
    n_test_pos=8,
    n_test_neg=96,
    tp=8, fp=19, fn=0, tn=77,
    auc=0.911,
    base_rate=0.077,
    fn_rate=0.0,
    fp_rate=0.198,
    precision=0.296,
    tn_fp_ratio=4.05,
    # Optional: per-event predictions for bootstrapping
    test_predictions=[...],
)
```

### 3.3 Edge

Self-contained model configuration:

```python
Edge(
    edge_id="CPI->HY__FN1pct__LogReg_w50_thr15bp",
    slot_id="CPI->HY_OAS",
    from_node=NodeId(NodeType.EVENT, "CPI"),
    to_node=NodeId(NodeType.INSTRUMENT, "HY_OAS"),
    model_type="LogReg",
    model_location="registry/models/CPI->HY__FN1pct__LogReg.pkl",
    active_factor="cpi_shock_abs",
    background_features=["yield_vol_10y", "slope_10y_2y", ...],
    target_series="HY_OAS",
    target_unit="bp",
    large_move_threshold=LargeMoveThreshold(0.15, "85th percentile"),
    fn_constraint=0.01,
    prob_cutoff=0.80,
    threshold_selection_method="grid_search + FN<=1%",
    stats=EdgeStats(...),
    version="2025-12-05_cpi_hy_v1",
    created_at="2025-12-05T10:30:00Z",
    created_by="build_macro_graph.py",
)
```

### 3.4 EdgeResult

Runtime output from applying an Edge:

```python
EdgeResult(
    edge_id="CPI->HY__FN1pct__LogReg_w50_thr15bp",
    slot_id="CPI->HY_OAS",
    from_node=...,
    to_node=...,
    event_date="2024-03-12",
    prob_large_move=0.85,
    flag_large_move=True,
    fn_constraint=0.01,
    large_move_threshold=0.15,
    target_series="HY_OAS",
    target_unit="bp",
    backtest={
        "auc": 0.911,
        "base_rate": 0.077,
        "tp": 8, "fp": 19, "fn": 0, "tn": 77,
        "fn_rate": 0.0, "fp_rate": 0.198,
        "tn_fp_ratio": 4.05,
    }
)
```

---

## 4. Runtime Flow

### 4.1 Feature Generation

1. For each upcoming event date:
   - Identify event type (`CPI`, `UNEMPLOYMENT`)
   - Compute global features (all known as of t–1)
2. Package into `EventContext`

### 4.2 Edge Application

```python
from src.graph import Graph, EdgeRunner, load_graph

# Load graph
graph = load_graph("registry/macro_graph.json")

# Create runner
runner = EdgeRunner(graph, model_base_path=".")

# Apply edge
result = runner.apply("CPI->HY_OAS", ctx)
print(result.prob_large_move)  # 0.85
print(result.flag_large_move)  # True
```

### 4.3 Policy Selection

```python
# Default edge
result = runner.apply("CPI->HY_OAS", ctx)

# Safe policy (lowest FN constraint)
result = runner.apply("CPI->HY_OAS", ctx, policy="safe")

# Balanced policy (best TN/FP among FN≤5%)
result = runner.apply("CPI->HY_OAS", ctx, policy="balanced")

# Specific edge
result = runner.apply("CPI->HY_OAS", ctx, edge_id="CPI->HY__FN5pct__GB")
```

---

## 5. Integration with Grid Search

### 5.1 EdgeSpec for Search

```python
from src.graph import EdgeSpec, NodeId, NodeType

spec = EdgeSpec(
    slot_id="CPI->HY_OAS",
    from_node=NodeId(NodeType.EVENT, "CPI"),
    to_node=NodeId(NodeType.INSTRUMENT, "HY_OAS"),
    active_factor="cpi_shock_abs",
    background_features=["yield_vol_10y", "slope_10y_2y", ...],
    target_col="hy_change",
    large_threshold_candidates=[0.05, 0.08, 0.10, 0.12, 0.15],
    fn_constraints=[0.01, 0.05],
)
```

### 5.2 Building Edges from Grid Search

```python
from src.graph import EdgeBuilder
from src.models.event_grid_search import run_full_grid_search

# Run grid search
results = run_full_grid_search(train_df, test_df, features, ...)

# Build edges
builder = EdgeBuilder(model_output_dir="models/edges")
edges = builder.build_from_grid_search(
    spec=spec,
    grid_results=results,
    train_df=train_df,
    test_df=test_df,
)
```

---

## 6. Usage Examples

### Build the Graph

```bash
# With grid search (slow)
python scripts/build_macro_graph.py

# With known best configs (fast)
python scripts/build_macro_graph.py --skip-search
```

### Load and Use

```python
from src.graph import load_graph, EdgeRunner, EventContext, NodeId, NodeType

# Load
graph = load_graph("registry/macro_graph.json")

# Create context
ctx = EventContext(
    node=NodeId(NodeType.EVENT, "CPI"),
    event_date="2024-03-12",
    features={"cpi_shock_abs": 0.3, ...}
)

# Run
runner = EdgeRunner(graph)
result = runner.apply("CPI->HY_OAS", ctx)

if result.flag_large_move:
    print("⚠️ Large HY OAS move expected!")
```

### Inspect the Graph

```python
from src.graph import load_graph

graph = load_graph("registry/macro_graph.json")

# List slots
print(graph.list_slots())  # ['CPI->HY_OAS', 'UNEMP->VIX']

# Get edge details
edge = graph.get_edge("CPI->HY_OAS")
print(edge.summary())
```

---

## 7. API Reference

### Module: `src.graph`

#### Types

| Type | Description |
|------|-------------|
| `NodeType` | Enum: `EVENT`, `INSTRUMENT` |
| `NodeId` | Node identifier (type + name + description) |
| `LargeMoveThreshold` | Threshold value + definition |
| `CVFoldMetric` | Per-fold CV metrics |
| `TestPrediction` | Single test prediction |
| `EdgeStats` | Backtest statistics |
| `Edge` | Model configuration |
| `EdgeSlot` | Logical mapping with multiple edges |
| `Graph` | Top-level container |
| `EventContext` | Runtime input |
| `EdgeResult` | Runtime output |
| `EdgeSpec` | Grid search specification |

#### Classes

| Class | Description |
|-------|-------------|
| `EdgeRunner` | Execute edges against contexts |
| `BatchEdgeRunner` | Batch execution |
| `EdgeBuilder` | Build edges from grid search |
| `GraphRegistry` | Version-controlled storage |

#### Functions

| Function | Description |
|----------|-------------|
| `save_graph(graph, path)` | Save graph to JSON |
| `load_graph(path)` | Load graph from JSON |
| `save_edge(edge, path)` | Save single edge |
| `load_edge(path)` | Load single edge |
| `export_graph_summary(graph, path)` | Export markdown summary |

---

## 8. Extensibility

### Adding New Event Types

1. Define new `NodeId` (e.g. `"FOMC"`, `"GDP"`)
2. Define feature spec (e.g. `fomc_surprise`, `dotplot_changes`)
3. Run grid search to populate `FOMC->2Y`, `FOMC->VIX` EdgeSlots

### Adding New Instruments

1. Add new instrument nodes
2. Add target series to event-level data
3. Define EdgeSpecs for new mappings

### Future: Bayesian Reliability

Because each Edge stores:
- `tp, fp, fn, tn` on test
- Optional per-fold CV metrics
- Optional per-event test predictions

You can later:
- Derive **TPR/FPR posteriors** (Beta(tp+1, fn+1) and Beta(fp+1, tn+1))
- Build a **Bayesian layer** for network-level inference

---

*Document maintained by: QSIG Macro Shock Detection Project*

