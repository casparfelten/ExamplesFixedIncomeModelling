# QSIG Webapp Comparison

## Two Interfaces Available

### 1. Original Dash Interface
**File:** `app.py` | **Script:** `./start_webapp_dash.sh` | **Port:** 8050

- Framework: Dash with dash-cytoscape
- Features: Interactive network graph, floating side panels
- Styling: Professional financial theme
- Functionality: Full parameter perturbation, real-time predictions

### 2. Clean Flask Interface (Webapp Two)  
**File:** `app_flask.py` | **Script:** `./start_webapp_two.sh` | **Port:** 5000

- Framework: Pure Flask with Jinja2 templates
- Features: Clean minimal design, SVG-based graph visualization
- Styling: Dark theme with monospace fonts, tasteful and clean
- Functionality: Graph overview, edge details, parameter perturbation

## Quick Start

```bash
# Start original Dash interface (port 8050)
./start_webapp_dash.sh

# Start clean Flask interface (port 5000) 
./start_webapp_two.sh
```

## Key Differences

| Feature | Dash Interface | Flask Interface |
|---------|---------------|----------------|
| **Design** | Professional financial | Clean minimal dark |
| **Graph** | Interactive Cytoscape | Simple SVG |
| **Navigation** | Modal panels | Page-based |
| **Dependencies** | Dash, Plotly, Cytoscape | Flask only |
| **Complexity** | High interactivity | Minimal simplicity |
| **Performance** | Heavier | Lighter |

## Recommendation

- **Dash Interface**: If you need rich interactivity and don't mind the visual complexity
- **Flask Interface**: If you prefer clean, minimal, and fast UI with tasteful dark styling

Both interfaces provide the same core functionality for viewing graph models and running parameter perturbations.