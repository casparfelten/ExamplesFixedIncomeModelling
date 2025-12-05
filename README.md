# Fixed Income Event Modelling - Data Infrastructure

A clean Python environment for downloading, cleaning, and aligning fixed-income data from multiple sources (FRED, CME FedWatch, Polymarket) to support event-modelling research.

**Note:** This project focuses on data infrastructure only. No statistical models, regressions, or forecasting are implemented yet.

## Project Goal

Model how macro regime and "opinion" factors affect U.S. 2-year Treasury yields, with the ability to compare against external probability series from:
- CME FedWatch (Fed meeting outcome probabilities)
- Polymarket (binary event odds)

## Project Structure

```
ExamplesFixedIncomeModelling/
├── .venv/                    # Virtual environment
├── data/
│   ├── raw/
│   │   ├── fred/            # FRED CSV files
│   │   ├── fedwatch/        # FedWatch Excel files
│   │   └── polymarket/      # Polymarket JSON/CSV files
│   └── processed/           # Merged panels
├── docs/
│   ├── anomaly_detection_strategy.md  # Strategy for event detection
│   └── audit_report.md               # Methodology audit
├── notebooks/
│   ├── 00_data_overview.ipynb        # Basic EDA
│   ├── 01_datagetter.ipynb           # Data fetching
│   ├── 02_cpi_yield_model.ipynb      # CPI-yield regime model
│   └── 03_regime_switching_classifier.ipynb  # Anomaly detection
├── predictors/                        # BLACK-BOX PREDICTORS
│   ├── base.py                       # BasePredictor interface
│   └── cpi_large_move/               # CPI Large Move detector
│       ├── model.py                  # Predictor implementation
│       ├── train.py                  # Training script
│       └── trained_model.pkl         # Saved weights
├── src/
│   ├── config.py            # Configuration settings
│   ├── data/
│   │   ├── fred_loader.py   # FRED data download and loading
│   │   ├── fedwatch_loader.py  # FedWatch Excel parsing
│   │   ├── polymarket_loader.py # Polymarket API fetching
│   │   └── merge_panel.py   # Panel builder for unified datasets
│   ├── models/
│   │   ├── cpi_yield_model.py  # Regime-switching model
│   │   ├── backtest.py         # Walk-forward backtesting
│   │   └── prepare_data.py     # Event data preparation
│   └── utils/
│       ├── paths.py         # Path utility functions
│       └── logging_utils.py # Logging setup
├── Makefile                 # Common tasks
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Setup Instructions

### 1. Prerequisites

- Python >= 3.10
- `make` (optional, for using Makefile commands)

### 2. Create Virtual Environment

```bash
make venv
```

Or manually:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
make install
```

Or manually:
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your FRED API key:

```
FRED_API_KEY=your_fred_api_key_here
```

**Getting a FRED API Key:**
1. Go to https://fred.stlouisfed.org/docs/api/api_key.html
2. Sign up for a free account
3. Request an API key (free, no credit card required)
4. Copy the key to your `.env` file

## Data Sources

### FRED (St. Louis Fed)

The project downloads the following FRED series:

- **Underlying:**
  - `DGS2` - 2-Year Treasury Constant Maturity Rate (daily)

- **Curve component:**
  - `DGS10` - 10-Year Treasury Constant Maturity Rate (daily)
  - Computed: `slope_10y_2y = DGS10 - DGS2`

- **Macro regime:**
  - `UNRATE` - Unemployment Rate (monthly)
  - `CPIAUCSL` - Consumer Price Index (monthly)
  - `FEDFUNDS` - Effective Federal Funds Rate (daily/monthly)
  - `GDPC1` - Real Gross Domestic Product (quarterly)

- **H.4.1 / Liquidity-related:**
  - `WRESBAL` - Reserve balances of depository institutions (weekly)
  - `WTREGEN` - Treasury General Account (daily)
  - `RRPONTSYD` - Overnight Reverse Repurchase Agreements (daily)
  - `WALCL` - Total assets of the Federal Reserve (weekly)

- **Market stress indicators:**
  - `BAMLH0A0HYM2` - ICE BofA US High Yield Index Option-Adjusted Spread (daily) — credit stress indicator
  - `VIXCLS` - CBOE Volatility Index: VIX (daily) — equity market volatility
  - `STLFSI4` - St. Louis Fed Financial Stress Index (weekly) — comprehensive stress measure

### CME FedWatch

FedWatch data must be manually downloaded from the CME FedWatch tool website and placed in `data/raw/fedwatch/`.

**File naming convention:** `fedwatch_meeting_YYYYMMDD.xlsx`

The loader parses Excel files to extract:
- Meeting date
- Target rate ranges (low, high in basis points)
- Probabilities for each outcome

### Polymarket

Polymarket data is fetched via their public API. The loader supports fetching historical data for binary contracts by market ID.

## Usage

### Data Fetching Workflow

**Important:** All bulk data fetching should be done through the datagetter notebook (`notebooks/01_datagetter.ipynb`). This notebook includes smart caching that checks date ranges and avoids unnecessary re-fetching.

1. **Run the datagetter notebook first:**
   ```bash
   jupyter lab notebooks/01_datagetter.ipynb
   ```
   
   The notebook will:
   - Check if FRED data exists and covers the required date range
   - Download missing or outdated FRED series automatically
   - Check for FedWatch Excel files (manually downloaded)
   - Fetch Polymarket data for configured markets
   - Skip data that already exists (unless `FORCE_RELOAD=True`)

2. **Use other notebooks for analysis:**
   - Other notebooks should only READ already-fetched data
   - They will not trigger downloads automatically

### Download Data (Alternative Methods)

For programmatic access, you can also download FRED data:

```bash
make download-data
```

Or manually:
```bash
python -c "from src.data.fred_loader import load_all_fred_data; from src.config import FRED_SERIES; import os; from dotenv import load_dotenv; load_dotenv(); api_key = os.getenv('FRED_API_KEY'); load_all_fred_data(FRED_SERIES, api_key)"
```

**Note:** The datagetter notebook is the recommended approach as it includes date range checking and avoids unnecessary downloads.

### Using the Data Loaders

#### FRED Data

```python
from src.data.fred_loader import download_series, load_series, merge_fred_panel

# Download a single series
download_series('DGS2')

# Load a cached series
df = load_series('DGS2')

# Merge all series into a daily panel
panel = merge_fred_panel()
```

#### FedWatch Data

```python
from src.data.fedwatch_loader import load_all_fedwatch, extract_daily_probability_series

# Load all FedWatch files
all_data = load_all_fedwatch()

# Extract daily probability series for a specific meeting/outcome
prob_series = extract_daily_probability_series(
    meeting_date=pd.Timestamp('2024-03-20'),
    target_range=(425, 450)  # 25bp hike
)
```

#### Polymarket Data

```python
from src.data.polymarket_loader import fetch_market_history, load_polymarket_data

# Fetch and save market data
fetch_market_history('market_id_here')

# Load cached data
df = load_polymarket_data('market_id_here')
```

#### Build Unified Panel

```python
from src.data.merge_panel import build_fed_panel

# Build daily panel with FRED + FedWatch data
panel = build_fed_panel(
    start_date=pd.Timestamp('2020-01-01'),
    end_date=pd.Timestamp('2024-12-31')
)
```

### Notebooks

#### 01_datagetter.ipynb - Data Fetching

**Run this notebook first** to fetch all required data:

```bash
jupyter lab notebooks/01_datagetter.ipynb
```

This notebook handles all bulk data fetching with smart caching:
- Checks date ranges to avoid unnecessary downloads
- Respects `FORCE_RELOAD` flag for complete refresh
- Configurable date ranges and market lists
- Summary of what was downloaded vs skipped

**Configuration:** Edit the configuration section at the top of the notebook to:
- Set `FORCE_RELOAD = True` to force re-download everything
- Configure FRED date ranges (`FRED_START_DATE`, `FRED_END_DATE`)
- Add Polymarket market IDs to `POLYMARKET_MARKETS`
- Specify expected FedWatch filenames (optional)

#### 00_data_overview.ipynb - Data Exploration

Run the data overview notebook for analysis:

```bash
jupyter lab notebooks/00_data_overview.ipynb
```

Or:

```bash
jupyter notebook notebooks/00_data_overview.ipynb
```

The notebook includes:
- Basic data inspection (`head()`, `info()`, `describe()`)
- Visualizations:
  - 2Y yield over time
  - FedWatch probability over time
  - 2Y yield vs FedWatch probability (scatter)

**Note:** This notebook only reads data - it does not fetch data. Run `01_datagetter.ipynb` first.

## Makefile Commands

- `make venv` - Create virtual environment
- `make install` - Install dependencies
- `make download-data` - Download/refresh FRED data
- `make clean` - Clean cache and temporary files
- `make help` - Show available commands

## Data Processing Notes

### Frequency Alignment

- Monthly series (unemployment, CPI) are forward-filled to daily frequency
- Quarterly series (GDP) are forward-filled to daily frequency
- Daily series (yields, Fed funds) are kept as-is

### Derived Features

- `slope_10y_2y`: Computed as `DGS10 - DGS2`
- `cpi_yoy`: Year-over-year CPI change (if computed)

### Column Naming

The processed FRED panel uses standardized column names:

| FRED Series | Column Name | Description |
|-------------|-------------|-------------|
| DGS2 | `y_2y` | 2-Year Treasury yield |
| DGS10 | `y_10y` | 10-Year Treasury yield |
| UNRATE | `unemployment` | Unemployment rate |
| CPIAUCSL | `cpi` | Consumer Price Index |
| FEDFUNDS | `fed_funds` | Federal Funds rate |
| GDPC1 | `gdp` | Real GDP |
| WRESBAL | `reserve_balances` | Reserve balances |
| WTREGEN | `treasury_general_account` | Treasury General Account |
| RRPONTSYD | `on_rrp_balance` | ON RRP balance |
| WALCL | `total_assets` | Fed total assets |
| BAMLH0A0HYM2 | `hy_oas` | High-yield credit spread |
| VIXCLS | `vix` | CBOE VIX |
| STLFSI4 | `stlfsi` | St. Louis Fed Financial Stress Index |

## Predictors (Black-Box Models)

The `predictors/` folder contains self-contained, audited prediction models that can be
composed into larger Markov/Bayesian networks as "complex edges."

### Available Predictors

| Predictor | Description | FN Rate | FP Rate |
|-----------|-------------|---------|---------|
| `CPILargeMovePredictor` | Large yield moves around CPI | 0% | 72% |

### Quick Usage

```python
from predictors import CPILargeMovePredictor

predictor = CPILargeMovePredictor.load()
result = predictor.predict({
    'yield_volatility': 0.05,
    'cpi_shock_mom': 0.1,
    'fed_funds': 2.5,
    'slope_10y_2y': 1.0,
    'unemployment': 4.0,
})

if result.prediction:
    print("Warning: High probability of large yield move!")
```

### Network Composition (Future)

```python
# Each predictor can be an edge in a probabilistic graph
graph.add_edge('MarketConditions', 'YieldMoveRisk', predictor)
```

See `predictors/README.md` for full documentation.

## Documentation

- `docs/anomaly_detection_strategy.md` - Strategy for anomaly detection
- `docs/audit_report.md` - Full audit of methodology and data leakage checks

## Next Steps

This infrastructure is ready for:
- Building more predictors for different event types
- Composing predictors into Markov/Bayesian networks
- Real-time event monitoring and risk flagging

## Troubleshooting

### FRED API Key Issues

If you get "FRED_API_KEY not found" errors:
1. Ensure `.env` file exists in project root
2. Check that `FRED_API_KEY=your_key` is set in `.env`
3. Verify the API key is valid at https://fred.stlouisfed.org

### FedWatch Files Not Found

If FedWatch data is missing:
1. Manually download Excel files from CME FedWatch website
2. Place files in `data/raw/fedwatch/`
3. Use naming convention: `fedwatch_meeting_YYYYMMDD.xlsx`

### Import Errors

If you get import errors:
1. Ensure virtual environment is activated
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Check that you're running from the project root directory

## License

This project is for research purposes.

## Contributing

This is a research project. Contributions and improvements are welcome.
