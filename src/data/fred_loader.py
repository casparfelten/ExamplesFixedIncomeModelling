"""FRED data download and loading utilities"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
from datetime import datetime, timedelta

from src.config import FRED_SERIES, FRED_COLUMN_MAPPING
from src.utils.paths import get_raw_data_path, ensure_dir_exists
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Load environment variables
load_dotenv()


def get_series_date_range(series_id: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Get the date range (min, max) from an existing FRED series CSV.
    
    Args:
        series_id: FRED series ID
    
    Returns:
        Tuple of (min_date, max_date) if file exists, None otherwise
    """
    raw_dir = get_raw_data_path("fred")
    filepath = raw_dir / f"{series_id}.csv"
    
    if not filepath.exists():
        return None
    
    try:
        df = pd.read_csv(filepath, parse_dates=["date"])
        if df.empty:
            return None
        min_date = df["date"].min()
        max_date = df["date"].max()
        return (min_date, max_date)
    except Exception as e:
        logger.warning(f"Failed to read date range from {filepath}: {e}")
        return None


def check_data_coverage(
    series_id: str, 
    start_date: Optional[pd.Timestamp] = None, 
    end_date: Optional[pd.Timestamp] = None,
    recent_threshold_days: int = 7
) -> Tuple[bool, Optional[str]]:
    """
    Check if existing FRED series data covers the requested date range.
    
    Args:
        series_id: FRED series ID
        start_date: Required start date (None = no requirement)
        end_date: Required end date (None = check if recent, within threshold_days of today)
        recent_threshold_days: If end_date is None, check if max date is within this many days of today
    
    Returns:
        Tuple of (is_covered, reason). is_covered=True if data exists and covers the range.
    """
    date_range = get_series_date_range(series_id)
    
    if date_range is None:
        return (False, "File does not exist")
    
    min_date, max_date = date_range
    
    # Check start date requirement
    if start_date is not None:
        if min_date > start_date:
            return (False, f"Data starts at {min_date.date()}, but {start_date.date()} is required")
    
    # Check end date requirement
    if end_date is not None:
        if max_date < end_date:
            return (False, f"Data ends at {max_date.date()}, but {end_date.date()} is required")
    else:
        # Check if data is recent enough (within threshold of today)
        today = pd.Timestamp.now().normalize()
        days_old = (today - max_date).days
        if days_old > recent_threshold_days:
            return (False, f"Data is {days_old} days old (max date: {max_date.date()}), threshold: {recent_threshold_days} days")
    
    return (True, f"Data covers range: {min_date.date()} to {max_date.date()}")


def download_series(
    series_id: str, 
    api_key: Optional[str] = None,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None
) -> Path:
    """
    Download a FRED series and save to CSV.
    
    Args:
        series_id: FRED series ID (e.g., 'DGS2')
        api_key: FRED API key (if None, reads from environment)
        start_date: Optional start date for data (None = all available)
        end_date: Optional end date for data (None = today)
    
    Returns:
        Path to saved CSV file
    """
    if api_key is None:
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise ValueError("FRED_API_KEY not found in environment. Set it in .env file or pass as argument.")
    
    date_range_str = ""
    if start_date is not None or end_date is not None:
        start_str = start_date.strftime("%Y-%m-%d") if start_date else "earliest"
        end_str = end_date.strftime("%Y-%m-%d") if end_date else "today"
        date_range_str = f" ({start_str} to {end_str})"
    
    logger.info(f"Downloading FRED series: {series_id}{date_range_str}")
    
    # Initialize FRED client
    fred = Fred(api_key=api_key)
    
    # Download data with optional date range
    try:
        if start_date is not None or end_date is not None:
            # Convert to datetime for fredapi
            start_dt = start_date.to_pydatetime() if start_date else None
            end_dt = end_date.to_pydatetime() if end_date else None
            df = fred.get_series(series_id, start=start_dt, end=end_dt)
        else:
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


def load_all_fred_data(series_list: List[str], api_key: Optional[str] = None, skip_existing: bool = True) -> None:
    """
    Download all specified FRED series.
    
    Args:
        series_list: List of FRED series IDs
        api_key: FRED API key (if None, reads from environment)
        skip_existing: If True, skip series that already exist locally
    """
    logger.info(f"Downloading {len(series_list)} FRED series...")
    
    raw_dir = get_raw_data_path("fred")
    downloaded_count = 0
    skipped_count = 0
    
    for series_id in series_list:
        # Check if file already exists
        if skip_existing:
            filepath = raw_dir / f"{series_id}.csv"
            if filepath.exists():
                logger.info(f"Skipping {series_id}: already exists at {filepath}")
                skipped_count += 1
                continue
        
        try:
            download_series(series_id, api_key)
            downloaded_count += 1
        except Exception as e:
            logger.error(f"Failed to download {series_id}: {e}")
            continue
    
    logger.info(f"FRED data download complete. Downloaded: {downloaded_count}, Skipped: {skipped_count}")


def merge_fred_panel(series_list: Optional[List[str]] = None, api_key: Optional[str] = None, auto_download: bool = False) -> pd.DataFrame:
    """
    Merge all FRED series into a single daily panel.
    
    This function is for READING existing data only. For downloading data,
    use the datagetter notebook (01_datagetter.ipynb) or download_series() directly.
    
    Args:
        series_list: List of series IDs to merge (default: all from config)
        api_key: FRED API key (if None, reads from environment) - only used if auto_download=True
        auto_download: If True, automatically download missing series (default: False)
    
    Returns:
        Daily DataFrame with standardized column names
    """
    if series_list is None:
        series_list = FRED_SERIES
    
    logger.info(f"Merging {len(series_list)} FRED series into daily panel...")
    
    # Get API key if needed for auto-download
    if auto_download and api_key is None:
        api_key = os.getenv("FRED_API_KEY")
    
    # Load all series
    dataframes = {}
    for series_id in series_list:
        try:
            # Check if file exists, download if missing and auto_download is enabled
            raw_dir = get_raw_data_path("fred")
            filepath = raw_dir / f"{series_id}.csv"
            
            if not filepath.exists() and auto_download:
                if not api_key:
                    logger.warning(
                        f"Series {series_id} not found and FRED_API_KEY not available. "
                        f"Skipping {series_id}. Set FRED_API_KEY in .env file to enable auto-download."
                    )
                    continue
                try:
                    logger.info(f"Auto-downloading missing series: {series_id}")
                    download_series(series_id, api_key)
                except Exception as e:
                    logger.error(f"Failed to auto-download {series_id}: {e}. Skipping.")
                    continue
            
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

