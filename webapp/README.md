# QSIG Macro Graph Explorer

Interactive webapp for exploring and analyzing the Bayesian macro graph models.

## Features

### 🗺️ Graph Overview
- **Interactive graph visualization** showing events (left) and instruments (right)
- **Node details** with descriptions and types
- **Edge information** showing model connections and descriptions

### 🔍 Edge Details  
- **Model specifications** for each edge (CPI→HY_OAS, UNEMPLOYMENT→VIX)
- **Performance metrics** including AUC, FN/FP rates, confusion matrices
- **Feature specifications** and median values for imputation
- **Multiple model variants** per edge slot with different FN constraints

### 🔬 Parameter Perturbation
- **"What if" scenarios** - adjust event parameters and see predictions
- **Interactive parameter controls** for active factors and background features
- **Quick preset shocks** (small/medium/large) for rapid testing
- **Real-time prediction updates** showing probability of large moves
- **Policy comparison** between "safe" (low FN) and "balanced" models

### 📈 Model Performance
- **Cross-model comparison** of AUC scores across all edge slots
- **FN vs FP tradeoff analysis** scatter plots
- **Detailed performance tables** with all metrics

## Getting Started

### Prerequisites
- Python 3.8+ with virtual environment support
- Existing QSIG graph structure in `/registry/macro_graph.json`
- Trained models in `/registry/models/`

### Installation & Launch

```bash
cd webapp
./start_webapp.sh
```

The webapp will launch at `http://localhost:8501`

### Manual Installation

```bash
# Create and activate virtual environment
python -m venv ../webapp_env
source ../webapp_env/bin/activate

# Install dependencies
pip install streamlit plotly networkx pandas numpy scikit-learn

# Launch webapp
streamlit run app.py
```

## Usage Examples

### 1. Exploring Graph Structure
1. Go to "Graph Overview" 
2. View the interactive network diagram
3. Click on nodes/edges for details
4. Check node descriptions and connections

### 2. Analyzing Model Performance
1. Select "Edge Details"
2. Choose an edge slot (e.g., "CPI->HY_OAS")
3. Expand model variants to see:
   - Performance metrics (AUC, FN/FP rates)
   - Confusion matrices
   - Feature medians for imputation

### 3. Running Parameter Perturbation
1. Go to "Parameter Perturbation"
2. Select event type (CPI or UNEMPLOYMENT)
3. Adjust parameters:
   - **Active Factor**: Main event shock (e.g., `cpi_shock_abs`)
   - **Background Features**: Market conditions (`yield_vol_10y`, `fed_funds`, etc.)
4. Use quick presets or manual values
5. View predictions for all connected instruments

### 4. Scenario Analysis Examples

**Example: CPI Inflation Shock**
```
Event: CPI
Active Factor: cpi_shock_abs = 0.8 (large shock)
Background: 
  - yield_vol_10y = 0.06 (elevated volatility)
  - fed_funds = 5.5 (tight policy)
  - hy_oas_before = 4.0 (normal spreads)

Expected Result: High probability of large HY_OAS move
```

**Example: Employment Surprise**
```
Event: UNEMPLOYMENT  
Active Factor: unemp_surprise_abs = 0.3 (big miss)
Background:
  - vix_before = 25 (elevated fear)
  - slope_10y_2y = 0.5 (flattening curve)

Expected Result: Elevated VIX large move probability
```

## Architecture

### Backend Integration
- Uses existing `/src/graph/` modules:
  - `types.py` - Data structures (Graph, Edge, EventContext)
  - `edge_runner.py` - Model execution and prediction
- Loads graph from `/registry/macro_graph.json`
- Accesses trained models from `/registry/models/`

### Frontend Components
- **Streamlit** - Web framework and UI components
- **Plotly** - Interactive visualizations (network graph, charts)
- **NetworkX** - Graph layout algorithms
- **Pandas** - Data manipulation and display

### Key Features
- **Model caching** - EdgeRunner caches loaded models for performance
- **Policy selection** - Choose between "safe" (low FN) vs "balanced" models
- **Real-time updates** - Parameter changes instantly update predictions
- **Error handling** - Graceful failure with informative messages

## Model Integration

The webapp integrates directly with your existing model infrastructure:

- **Graph Definition**: `/registry/macro_graph.json`
- **Trained Models**: `/registry/models/models/*/model.pkl`
- **Feature Medians**: Used for missing value imputation
- **Multiple Policies**: Safe vs balanced model selection

### Model Types Supported
- Random Forest (RF_shallow)
- Extra Trees (ExtraTrees) 
- Logistic Regression (LogReg)
- Gradient Boosting (GB)

### Performance Metrics
- AUC (Area Under Curve)
- False Negative Rate (FN Rate)
- False Positive Rate (FP Rate)
- True Negative/False Positive Ratio (TN/FP)
- Precision, Base Rate
- Confusion Matrix (TP, FP, FN, TN)

## Troubleshooting

### Common Issues

**Graph not loading:**
- Verify `/registry/macro_graph.json` exists and is valid JSON
- Check file permissions

**Models not found:**
- Ensure `/registry/models/` contains trained model files
- Verify model paths in macro_graph.json match actual files

**Prediction errors:**
- Check that feature names match between webapp and models
- Verify all required features are provided
- Look for missing model dependencies

**Performance issues:**
- Models are cached after first load
- Large graphs may take time to render initially
- Consider reducing visualization complexity for very large graphs

### Debug Mode

To enable debug information:

```bash
streamlit run app.py --logger.level=debug
```

## Future Enhancements

Potential extensions (not implemented):

- **Time series visualization** of historical predictions
- **Model comparison tools** for A/B testing different edges  
- **Batch scenario analysis** for multiple parameter combinations
- **Export functionality** for prediction results
- **Real-time data integration** with live market feeds
- **Model confidence intervals** using bootstrap methods
- **Network propagation effects** showing cascading predictions