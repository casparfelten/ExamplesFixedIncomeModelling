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
├── notebooks/
│   └── 00_data_overview.ipynb  # Basic EDA and sanity checks
├── src/
│   ├── config.py            # Configuration settings
│   ├── data/
│   │   ├── fred_loader.py   # FRED data download and loading
│   │   ├── fedwatch_loader.py  # FedWatch Excel parsing
│   │   ├── polymarket_loader.py # Polymarket API fetching
│   │   └── merge_panel.py   # Panel builder for unified datasets
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

### Download Data

Download all FRED series:

```bash
make download-data
```

Or manually:
```bash
python -c "from src.data.fred_loader import load_all_fred_data; from src.config import FRED_SERIES; import os; from dotenv import load_dotenv; load_dotenv(); api_key = os.getenv('FRED_API_KEY'); load_all_fred_data(FRED_SERIES, api_key)"
```

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

Run the data overview notebook:

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

## Next Steps

This infrastructure is ready for:
- Statistical modelling of yield movements
- Regression analysis of macro factors
- Comparison with external probability series
- Event study analysis

**Note:** No models are implemented yet. This is purely data infrastructure.

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
