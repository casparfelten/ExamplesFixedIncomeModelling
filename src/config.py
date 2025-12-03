"""Configuration settings for the project"""

from pathlib import Path

# Data directory paths (relative to project root)
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# FRED data configuration
FRED_RAW_DIR = RAW_DATA_DIR / "fred"
FRED_SERIES = [
    # Yields / Curve
    "DGS2",      # 2-Year Treasury Constant Maturity Rate (daily)
    "DGS10",     # 10-Year Treasury Constant Maturity Rate (daily)
    # Macro regime
    "UNRATE",    # Unemployment Rate (monthly)
    "CPIAUCSL",  # Consumer Price Index for All Urban Consumers: All Items (monthly)
    "FEDFUNDS",  # Effective Federal Funds Rate (daily/monthly)
    "GDPC1",     # Real Gross Domestic Product (quarterly)
    # H.4.1 / Liquidity-related
    "WRESBAL",   # Reserve balances of depository institutions (weekly)
    "WTREGEN",   # Treasury General Account (daily)
    "RRPONTSYD", # Overnight Reverse Repurchase Agreements (daily)
    "WALCL",     # Total assets of the Federal Reserve (weekly)
]

# FRED API configuration
FRED_API_BASE_URL = "https://api.stlouisfed.org/fred"

# FedWatch data configuration
FEDWATCH_RAW_DIR = RAW_DATA_DIR / "fedwatch"
# File pattern: fedwatch_meeting_YYYYMMDD.xlsx
FEDWATCH_FILE_PATTERN = "fedwatch_meeting_*.xlsx"

# FedWatch meeting/outcome mapping (placeholder - update as needed)
# Format: {meeting_date: {target_range: description}}
FEDWATCH_OUTCOMES = {
    # Example structure:
    # "2024-03-20": {
    #     (425, 450): "25bp hike",
    #     (400, 425): "Hold",
    # }
}

# Polymarket data configuration
POLYMARKET_RAW_DIR = RAW_DATA_DIR / "polymarket"
POLYMARKET_API_BASE_URL = "https://clob.polymarket.com"

# Column name mappings for standardized output
FRED_COLUMN_MAPPING = {
    "DGS2": "y_2y",
    "DGS10": "y_10y",
    "UNRATE": "unemployment",
    "CPIAUCSL": "cpi",
    "FEDFUNDS": "fed_funds",
    "GDPC1": "gdp",
    "WRESBAL": "reserve_balances",
    "WTREGEN": "treasury_general_account",
    "RRPONTSYD": "on_rrp_balance",
    "WALCL": "total_assets",
}

# FRED series metadata (units and transformation hints)
FRED_SERIES_METADATA = {
    "DGS2": {"unit": "percent", "frequency": "daily"},
    "DGS10": {"unit": "percent", "frequency": "daily"},
    "UNRATE": {"unit": "percent", "frequency": "monthly"},
    "CPIAUCSL": {"unit": "index", "frequency": "monthly"},
    "FEDFUNDS": {"unit": "percent", "frequency": "daily"},
    "GDPC1": {"unit": "billions_usd", "frequency": "quarterly"},
    "WRESBAL": {"unit": "millions_usd", "frequency": "weekly"},
    "WTREGEN": {"unit": "millions_usd", "frequency": "daily"},
    "RRPONTSYD": {"unit": "millions_usd", "frequency": "daily"},
    "WALCL": {"unit": "millions_usd", "frequency": "weekly"},
}

# CME FedWatch API configuration
CME_FEDWATCH_BASE_URL = "https://www.cmegroup.com/CmeWS/mvc/FedWatchTool/"

# Atlanta Fed Market Probability Tracker configuration
ATLANTA_MPT_BASE_URL = "https://www.atlantafed.org/cqer/research/market-probability-tracker"

# Inflation announcements data configuration
INFLATION_ANNOUNCEMENTS_RAW_DIR = RAW_DATA_DIR / "inflation_announcements"
BLS_SCHEDULE_URL = "https://www.bls.gov/schedule/news_release/cpi.htm"

# Polymarket event mapping
# Maps internal event_id to market_id and metadata
POLYMARKET_EVENT_MAPPING = {
    # Example structure:
    # "ceasefire_ukraine_by_2024Q4": {
    #     "market_id": "market-slug-here",
    #     "description": "Ceasefire in Ukraine by end of Q4 2024",
    #     "resolution_date": "2024-12-31",
    # }
}

