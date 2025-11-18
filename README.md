# Fixed Income Event Modelling - Data Setup

This project provides a clean Python environment and data infrastructure for fixed-income event modelling. The focus is on **data preparation only** - no statistical models, regressions, or forecasting are implemented at this stage.

## Project Goal

The project is designed to model how macro regime and "opinion" factors affect the U.S. 2-year Treasury yield (daily, from FRED series `DGS2`). The infrastructure supports:

- Pulling macro/economic regime variables (unemployment, CPI, GDP, Fed funds)
- Pulling opinion/expectation variables (curve slope: 10Y - 2Y)
- Preparing for future integration with external probability series:
  - CME FedWatch (Fed-meeting outcome probabilities)
  - Polymarket (binary event odds)

## Project Structure

```
.
├── .venv/                    # Virtual environment (created by make venv)
├── data/
│   ├── raw/
│   │   ├── fred/            # FRED data (CSV files)
│   │   ├── fedwatch/        # FedWatch Excel files (manual download)
│   │   └── polymarket/      # Polymarket JSON/CSV files
│   └── processed/           # Processed/merged datasets
├── notebooks/
│   └── 00_data_overview.ipynb  # EDA and sanity checks
├── src/
│   ├── config.py            # Configuration settings
│   ├── data/
│   │   ├── fred_loader.py       # FRED data download/loading
│   │   ├── fedwatch_loader.py   # FedWatch Excel parsing
│   │   ├── polymarket_loader.py # Polymarket API fetching
│   │   └── merge_panel.py       # Panel merging utilities
│   └── utils/
│       ├── logging_utils.py     # Logging setup
│       └── paths.py             # Path utilities
├── Makefile                 # Common tasks
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Environment Setup

### Prerequisites

- Python >= 3.10
- Make (for using Makefile commands)

### Installation

1. **Create virtual environment and install dependencies:**
   ```bash
   make venv
   make install
   ```

   This will:
   - Create a virtual environment in `.venv/`
   - Install all required packages from `requirements.txt`

2. **Set up environment variables:**
   
   Create a `.env` file in the project root with your FRED API key:
   ```
   FRED_API_KEY=your_fred_api_key_here
   ```
   
   To get a FRED API key (free):
   - Visit https://fred.stlouisfed.org/docs/api/api_key.html
   - Sign up and request an API key

### Dependencies

The project uses:
- `pandas` >= 2.0.0 - Data manipulation
- `numpy` >= 1.24.0 - Numerical operations
- `matplotlib` >= 3.7.0 - Plotting
- `requests` >= 2.31.0 - HTTP requests
- `python-dotenv` >= 1.0.0 - Environment variable management
- `openpyxl` >= 3.1.0 - Excel file reading (for FedWatch)
- `pyarrow` >= 12.0.0 - Fast data I/O (optional)
- `jupyterlab` >= 4.0.0 - Jupyter notebooks
- `fredapi` >= 0.5.0 - FRED API wrapper

## Data Sources

### 1. FRED (St. Louis Fed)

The project downloads the following FRED series (all free, no login required beyond API key):

**Underlying:**
- `DGS2` - 2-Year Treasury Constant Maturity Rate (daily)

**Curve component:**
- `DGS10` - 10-Year Treasury Constant Maturity Rate (daily)
- Computed: `slope_10y_2y = DGS10 - DGS2`

**Macro regime:**
- `UNRATE` - Unemployment Rate (monthly)
- `CPIAUCSL` - Consumer Price Index (monthly, index)
- `FEDFUNDS` - Effective Federal Funds Rate (daily/monthly)
- `GDPC1` - Real Gross Domestic Product (quarterly)

**Downloading FRED data:**
```bash
make download-data
```

Or programmatically:
```python
from src.data.fred_loader import load_all_fred_data
from src.config import FRED_SERIES
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("FRED_API_KEY")
load_all_fred_data(FRED_SERIES, api_key)
```

Data is saved as CSV files in `data/raw/fred/{SERIES_ID}.csv` with columns:
- `date` - Datetime index
- `value` - Series value

### 2. CME FedWatch

FedWatch data must be **manually downloaded** from the CME FedWatch tool website and placed in `data/raw/fedwatch/`.

**File naming convention:**
- Files should be named like `fedwatch_meeting_YYYYMMDD.xlsx` (e.g., `fedwatch_meeting_20240320.xlsx`)

**Expected file structure:**
Each Excel file should contain:
- Meeting date information
- Target rate ranges (low, high) in basis points
- Probabilities for each range

The `fedwatch_loader.py` module will:
- Scan for all Excel files in `data/raw/fedwatch/`
- Parse each file into a tidy DataFrame with columns:
  - `as_of_date` - Date the probabilities refer to
  - `meeting_date` - Date of the FOMC meeting
  - `target_range_low` - Lower bound of target rate (in bps)
  - `target_range_high` - Upper bound of target rate (in bps)
  - `probability` - Probability as fraction (0-1)

**Note:** The parser is flexible and attempts to handle common FedWatch formats. You may need to adjust `parse_fedwatch_file()` in `src/data/fedwatch_loader.py` based on the actual file structure you download.

### 3. Polymarket

Polymarket data is fetched via their public API (no authentication required).

**Fetching data:**
```python
from src.data.polymarket_loader import fetch_market_history

# Fetch data for a specific market
market_id = "your-market-id-or-slug"
fetch_market_history(market_id)
```

Data is saved as JSON in `data/raw/polymarket/{market_id}.json` and can be loaded into a DataFrame with columns:
- `datetime` - UTC timestamp
- `price` - Market price (~probability for "YES" outcome)
- `volume` - Trading volume (if available)
- `liquidity` - Market liquidity (if available)

**Note:** The Polymarket API structure may vary. The loader attempts multiple endpoints and formats. You may need to customize `fetch_market_history()` and `load_polymarket_data()` based on the actual API structure.

## Usage

### Building a Unified Panel

The main workflow is to build a unified daily panel that merges FRED data with FedWatch probabilities:

```python
from src.data.merge_panel import build_fed_panel
import pandas as pd

# Build panel with all available data
df = build_fed_panel()

# Or specify date range and FedWatch filters
df = build_fed_panel(
    start_date=pd.Timestamp("2023-01-01"),
    end_date=pd.Timestamp("2024-12-31"),
    meeting_date=pd.Timestamp("2024-03-20"),
    target_range=(425, 450)  # 25bp hike in bps
)
```

The resulting DataFrame has columns:
- `date` - Daily date index
- `y_2y` - 2-Year Treasury yield
- `y_10y` - 10-Year Treasury yield
- `slope_10y_2y` - 10Y - 2Y spread
- `unemployment` - Unemployment rate (forward-filled to daily)
- `cpi` - CPI index (forward-filled to daily)
- `fed_funds` - Fed funds rate (forward-filled to daily)
- `gdp` - Real GDP (forward-filled to daily, quarterly data)
- `p_fed_outcome` - FedWatch probability for chosen outcome (forward-filled)

### Running the Data Overview Notebook

The notebook `notebooks/00_data_overview.ipynb` provides basic sanity checks:

1. Loads the Fed panel
2. Displays basic info (`head()`, `info()`, `describe()`)
3. Visualizes:
   - 2-Year Treasury yield over time
   - (TODO: FedWatch probability over time)
   - (TODO: Scatter plot of 2Y yield vs FedWatch probability)

To run:
```bash
# Activate virtual environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows

# Start Jupyter
jupyter lab
```

## Makefile Commands

```bash
make venv          # Create virtual environment
make install       # Install dependencies (also creates venv if needed)
make download-data # Download/refresh FRED data
make clean         # Clean cache and temporary files
make help          # Show available commands
```

## Configuration

Configuration is centralized in `src/config.py`:

- **Data directories:** Paths to raw and processed data
- **FRED series:** List of series IDs to download
- **Column mappings:** Standardized column names for merged panel
- **FedWatch settings:** File patterns and outcome mappings (placeholder)

## Alignment with Original Design

This implementation aligns with the original design specification:

✅ **Environment Setup:**
- Python >= 3.10 support
- Virtual environment with all required packages
- Makefile with common tasks
- README documentation

✅ **Project Structure:**
- Matches specified directory layout
- Clean separation of concerns (data loaders, utils, config)

✅ **FRED Data:**
- Downloads all specified series (DGS2, DGS10, UNRATE, CPIAUCSL, FEDFUNDS, GDPC1)
- Saves as CSV with standardized format (date, value)
- Merges into daily panel with forward-fill for monthly/quarterly data
- Computes derived features (slope_10y_2y)

✅ **FedWatch:**
- Scans for Excel files
- Parses to tidy DataFrame with required columns
- Extracts daily probability series for chosen meeting/outcome

✅ **Polymarket:**
- Fetches data for single market via API
- Saves raw JSON
- Normalizes to DataFrame with datetime, price, volume, liquidity

✅ **Panel Merging:**
- `build_fed_panel()` creates unified daily panel
- Handles frequency alignment (forward-fill)
- Placeholder for `build_polymarket_panel()`

✅ **Utilities:**
- Config module with all settings
- Path utilities for data directories
- Logging setup

⚠️ **Notebook:**
- Basic structure in place
- Loads panel and shows basic stats
- Has 2Y yield plot
- **Missing:** FedWatch probability plot and scatter plot (2Y vs probability)

## Next Steps

This project is **data preparation only**. The next phase would involve:

1. Completing notebook visualizations (FedWatch probability plots)
2. Adding more robust error handling for FedWatch parser
3. Testing with actual FedWatch Excel files
4. Validating Polymarket API endpoints
5. **Then:** Plugging in statistical models (regressions, ML, etc.)

## Notes

- **No models implemented:** As per design, this project focuses solely on data infrastructure
- **FedWatch parser:** May need customization based on actual file format
- **Polymarket API:** Endpoints may need updates based on actual API structure
- **Data freshness:** FRED data can be refreshed with `make download-data`
- **Environment variables:** `.env` file is gitignored - create your own with `FRED_API_KEY`

## License

[Add your license here]

