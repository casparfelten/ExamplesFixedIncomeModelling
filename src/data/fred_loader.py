"""FRED data download and loading utilities"""

import os
from pathlib import Path
from typing import List, Optional
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

from src.config import FRED_SERIES, FRED_COLUMN_MAPPING
from src.utils.paths import get_raw_data_path, ensure_dir_exists
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Load environment variables
load_dotenv()


def download_series(series_id: str, api_key: Optional[str] = None) -> Path:
    """
    Download a FRED series and save to CSV.
    
    Args:
        series_id: FRED series ID (e.g., 'DGS2')
        api_key: FRED API key (if None, reads from environment)
    
    Returns:
        Path to saved CSV file
    """
    if api_key is None:
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise ValueError("FRED_API_KEY not found in environment. Set it in .env file or pass as argument.")
    
    logger.info(f"Downloading FRED series: {series_id}")
    
    # Initialize FRED client
    fred = Fred(api_key=api_key)
    
    # Download data
    try:
        df = fred.get_series(series_id)
    except Exception as e:
        logger.error(f"Failed to download {series_id}: {e}")
        raise
    
    # Convert to DataFrame if Series
    if isinstance(df, pd.Series):
        df = df.to_frame(name="value")
    
    # Ensure date index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Reset index to have 'date' column
    df = df.reset_index()
    df.columns = ["date", "value"]
    
    # Save to CSV
    raw_dir = get_raw_data_path("fred")
    ensure_dir_exists(raw_dir)
    filepath = raw_dir / f"{series_id}.csv"
    df.to_csv(filepath, index=False)
    
    logger.info(f"Saved {series_id} to {filepath}")
    return filepath


def load_series(series_id: str) -> pd.DataFrame:
    """
    Load a cached FRED series from CSV.
    
    Args:
        series_id: FRED series ID
    
    Returns:
        DataFrame with 'date' column and 'value' column, indexed by date
    """
    raw_dir = get_raw_data_path("fred")
    filepath = raw_dir / f"{series_id}.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Series {series_id} not found at {filepath}. "
            f"Run download_series('{series_id}') first."
        )
    
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.set_index("date")
    df.columns = ["value"]
    
    return df


def load_all_fred_data(series_list: List[str], api_key: Optional[str] = None) -> None:
    """
    Download all specified FRED series.
    
    Args:
        series_list: List of FRED series IDs
        api_key: FRED API key (if None, reads from environment)
    """
    logger.info(f"Downloading {len(series_list)} FRED series...")
    
    for series_id in series_list:
        try:
            download_series(series_id, api_key)
        except Exception as e:
            logger.error(f"Failed to download {series_id}: {e}")
            continue
    
    logger.info("FRED data download complete.")


def merge_fred_panel(series_list: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Merge all FRED series into a single daily panel.
    
    Args:
        series_list: List of series IDs to merge (default: all from config)
    
    Returns:
        Daily DataFrame with standardized column names
    """
    if series_list is None:
        series_list = FRED_SERIES
    
    logger.info(f"Merging {len(series_list)} FRED series into daily panel...")
    
    # Load all series
    dataframes = {}
    for series_id in series_list:
        try:
            df = load_series(series_id)
            # Get standardized column name
            col_name = FRED_COLUMN_MAPPING.get(series_id, series_id.lower())
            dataframes[col_name] = df["value"]
        except FileNotFoundError as e:
            logger.warning(f"Skipping {series_id}: {e}")
            continue
    
    if not dataframes:
        raise ValueError("No FRED series could be loaded.")
    
    # Merge into single DataFrame
    panel = pd.DataFrame(dataframes)
    
    # Create daily index (business days)
    # Start from earliest date, end at latest date
    start_date = panel.index.min()
    end_date = panel.index.max()
    daily_index = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # Reindex to daily and forward-fill
    panel = panel.reindex(daily_index)
    panel = panel.ffill()  # Forward-fill for monthly/quarterly data
    
    # Compute derived features
    if "y_2y" in panel.columns and "y_10y" in panel.columns:
        panel["slope_10y_2y"] = panel["y_10y"] - panel["y_2y"]
        logger.info("Computed slope_10y_2y = y_10y - y_2y")
    
    # Optional: Compute YoY CPI change if CPI is available
    if "cpi" in panel.columns:
        panel["cpi_yoy"] = panel["cpi"].pct_change(periods=365) * 100  # Approximate YoY
        logger.info("Computed CPI YoY change")
    
    # Reset index to have 'date' column
    panel = panel.reset_index()
    panel.columns = ["date"] + list(panel.columns[1:])
    
    logger.info(f"Merged panel shape: {panel.shape}")
    return panel

