"""Configuration settings for the project"""

from pathlib import Path

# Data directory paths (relative to project root)
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# FRED data configuration
FRED_RAW_DIR = RAW_DATA_DIR / "fred"
FRED_SERIES = [
    "DGS2",      # 2-Year Treasury Constant Maturity Rate (daily)
    "DGS10",     # 10-Year Treasury Constant Maturity Rate (daily)
    "UNRATE",    # Unemployment Rate (monthly)
    "CPIAUCSL",  # Consumer Price Index for All Urban Consumers: All Items (monthly)
    "FEDFUNDS",  # Effective Federal Funds Rate (daily/monthly)
    "GDPC1",     # Real Gross Domestic Product (quarterly)
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
}

