# Agent Instructions - Data Fetching Workflow

This document describes the data fetching workflow and conventions for this project.

## Overview

This project uses a **separation of concerns** approach for data management:
- **Data Fetching**: Done exclusively in `notebooks/01_datagetter.ipynb`
- **Data Reading**: All other notebooks and code only read already-fetched data

This design prevents unnecessary API calls and ensures data consistency across the project.

## Data Fetching Workflow

### Step 1: Run the Datagetter Notebook

**Always run `notebooks/01_datagetter.ipynb` first** before running any analysis notebooks.

The datagetter notebook:
1. Checks if data already exists and covers required date ranges
2. Downloads only missing or outdated data
3. Skips data that already exists (unless `FORCE_RELOAD=True`)
4. Provides a summary of what was downloaded vs skipped

### Step 2: Configure the Notebook

Edit the **Configuration section** at the top of `01_datagetter.ipynb`:

```python
# Force reload: If True, re-downloads all data regardless of what exists
FORCE_RELOAD = False

# FRED Data Configuration
FRED_START_DATE = None  # None = all available data from FRED
FRED_END_DATE = None    # None = today (checks if data is recent within threshold)
FRED_RECENT_THRESHOLD_DAYS = 7  # If end_date is None, check if data is within this many days of today

# Polymarket Configuration
POLYMARKET_MARKETS = [
    # Add market IDs here
]

# FedWatch Configuration
EXPECTED_FEDWATCH_FILES = None  # None = just check if any files exist
```

### Step 3: Run Analysis Notebooks

After data is fetched, run analysis notebooks (e.g., `00_data_overview.ipynb`). These notebooks:
- Only read existing data files
- Do not trigger downloads automatically
- Will fail gracefully if data is missing (with clear error messages)

## Force Reload Flag

The `FORCE_RELOAD` flag controls whether to bypass all caching checks:

- **`FORCE_RELOAD = False`** (default):
  - Checks if data exists and covers required date ranges
  - Only downloads missing or outdated data
  - Skips data that already exists

- **`FORCE_RELOAD = True`**:
  - Bypasses all checks
  - Re-downloads all data regardless of what exists
  - Useful for:
    - Debugging data issues
    - Getting fresh data after API changes
    - Ensuring data consistency

## Date Range Checking Logic

### FRED Data

The datagetter notebook uses `check_data_coverage()` to determine if existing data is sufficient:

1. **If `FRED_START_DATE` is specified:**
   - Checks if existing data starts on or before this date
   - Downloads if data starts later than required

2. **If `FRED_END_DATE` is specified:**
   - Checks if existing data ends on or after this date
   - Downloads if data ends earlier than required

3. **If `FRED_END_DATE` is None:**
   - Checks if existing data is recent (within `FRED_RECENT_THRESHOLD_DAYS` of today)
   - Downloads if data is older than threshold

4. **If both are None:**
   - Downloads all available data from FRED
   - On subsequent runs, checks if data is recent (within threshold)

### Polymarket Data

- Checks if JSON file exists for each market ID
- Downloads only if file doesn't exist (unless `FORCE_RELOAD=True`)
- No date range checking (markets are fetched as complete snapshots)

### FedWatch Data

- FedWatch files are manually downloaded Excel files
- Notebook only checks if files exist
- No automatic downloading (manual process)

## Code Conventions

### For Data Fetching

**DO:**
- Use `notebooks/01_datagetter.ipynb` for all bulk data fetching
- Use `check_data_coverage()` before downloading FRED data
- Use `check_market_exists()` before fetching Polymarket data
- Respect the `FORCE_RELOAD` flag

**DON'T:**
- Don't call `download_series()` or `fetch_market_history()` in analysis notebooks
- Don't set `auto_download=True` in `merge_fred_panel()` (default is False)
- Don't bypass date range checks unless `FORCE_RELOAD=True`

### For Data Reading

**DO:**
- Use `load_series()` to read cached FRED data
- Use `merge_fred_panel()` with `auto_download=False` (default)
- Use `load_polymarket_data()` to read cached Polymarket data
- Use `build_fed_panel()` which only reads existing data

**DON'T:**
- Don't call download functions in analysis code
- Don't enable auto-download in merge functions
- Don't assume data exists - handle `FileNotFoundError` gracefully

## Example Workflow

```python
# 1. In 01_datagetter.ipynb (run first)
FORCE_RELOAD = False
FRED_START_DATE = None
FRED_END_DATE = None
# ... run notebook to fetch data

# 2. In analysis notebook (run after datagetter)
from src.data.merge_panel import build_fed_panel

# This only reads existing data - no downloads
panel = build_fed_panel(
    start_date=pd.Timestamp('2020-01-01'),
    end_date=pd.Timestamp('2024-12-31')
)
```

## Troubleshooting

### Data Not Found Errors

If you get `FileNotFoundError`:
1. Run `01_datagetter.ipynb` first
2. Check that `FRED_API_KEY` is set in `.env` file
3. Verify data files exist in `data/raw/` directories

### Outdated Data

If data seems outdated:
1. Set `FORCE_RELOAD = True` in datagetter notebook
2. Or adjust `FRED_RECENT_THRESHOLD_DAYS` to be more strict
3. Or specify `FRED_END_DATE` explicitly

### API Rate Limits

If hitting API rate limits:
1. The datagetter notebook skips existing data by default
2. Only missing/outdated data is downloaded
3. Use `FORCE_RELOAD = False` to minimize API calls

## Summary

- **Fetching**: Use `01_datagetter.ipynb` only
- **Reading**: Use loader functions in other notebooks/code
- **Caching**: Smart date range checking prevents unnecessary downloads
- **Force Reload**: Set `FORCE_RELOAD=True` to bypass all checks

