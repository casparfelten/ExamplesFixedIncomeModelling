"""Atlanta Fed Market Probability Tracker data loading utilities"""

import os
from pathlib import Path
from typing import Optional, List
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv

from src.config import ATLANTA_MPT_BASE_URL
from src.utils.paths import get_raw_data_path, get_processed_data_path, ensure_dir_exists
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Load environment variables
load_dotenv()


def download_atlanta_mpt_data(
    save_path: Optional[Path] = None,
    force_reload: bool = False
) -> Path:
    """
    Download historical data from Atlanta Fed Market Probability Tracker.
    
    The Atlanta Fed MPT publishes policy rate probability distributions.
    This function downloads the data file (CSV or other format) from their website.
    
    Args:
        save_path: Path to save the data file (if None, uses default location)
        force_reload: If True, re-download even if file exists
    
    Returns:
        Path to saved data file
    """
    base_url = os.getenv("ATLANTA_MPT_BASE_URL", ATLANTA_MPT_BASE_URL)
    
    # Atlanta Fed MPT data download URL
    # Note: Actual URL may need adjustment based on their website structure
    # Common patterns: direct CSV download, or API endpoint
    download_url = f"{base_url}/data.csv"
    
    if save_path is None:
        raw_dir = get_raw_data_path("atlanta_mpt")
        ensure_dir_exists(raw_dir)
        # Use date-based filename
        today = datetime.now().strftime("%Y%m%d")
        save_path = raw_dir / f"atlanta_mpt_{today}.csv"
    
    if not force_reload and save_path.exists():
        logger.info(f"Atlanta MPT data already exists at {save_path}. Skipping download (use force_reload=True to override).")
        return save_path
    
    logger.info(f"Downloading Atlanta Fed MPT data from {download_url}...")
    
    try:
        response = requests.get(download_url, timeout=60)
        response.raise_for_status()
        
        # Save raw file
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Saved Atlanta MPT data to {save_path}")
        return save_path
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download Atlanta MPT data: {e}")
        logger.warning("Atlanta Fed MPT data may need to be downloaded manually from their website")
        raise
    except Exception as e:
        logger.error(f"Error saving Atlanta MPT data: {e}")
        raise


def load_atlanta_mpt_panel() -> pd.DataFrame:
    """
    Load and normalize all Atlanta MPT raw files into a tidy panel.
    
    Returns:
        DataFrame with columns: as_of_date, horizon_date, rate_bin_low_bps,
        rate_bin_high_bps, probability
    """
    raw_dir = get_raw_data_path("atlanta_mpt")
    csv_files = list(raw_dir.glob("atlanta_mpt_*.csv"))
    
    if not csv_files:
        logger.warning("No Atlanta MPT CSV files found")
        return pd.DataFrame(columns=[
            "as_of_date", "horizon_date", "rate_bin_low_bps",
            "rate_bin_high_bps", "probability"
        ])
    
    logger.info(f"Loading {len(csv_files)} Atlanta MPT CSV files...")
    
    all_data = []
    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath)
            
            # Normalize the data structure
            # Atlanta Fed MPT CSV format may vary - handle common patterns
            normalized_df = normalize_atlanta_mpt_data(df, filepath)
            
            if not normalized_df.empty:
                all_data.append(normalized_df)
        except Exception as e:
            logger.warning(f"Failed to load {filepath.name}: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame(columns=[
            "as_of_date", "horizon_date", "rate_bin_low_bps",
            "rate_bin_high_bps", "probability"
        ])
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates
    combined = combined.drop_duplicates(
        subset=["as_of_date", "horizon_date", "rate_bin_low_bps", "rate_bin_high_bps"],
        keep="last"
    )
    
    # Sort by as_of_date and horizon_date
    combined = combined.sort_values(["as_of_date", "horizon_date"])
    
    # Save processed panel
    processed_dir = get_processed_data_path()
    ensure_dir_exists(processed_dir)
    parquet_path = processed_dir / "atlanta_mpt.parquet"
    combined.to_parquet(parquet_path, index=False)
    logger.info(f"Saved processed Atlanta MPT panel to {parquet_path}")
    logger.info(f"Atlanta MPT panel shape: {combined.shape}")
    
    return combined


def normalize_atlanta_mpt_data(df: pd.DataFrame, filepath: Path) -> pd.DataFrame:
    """
    Normalize Atlanta MPT CSV data into standard format.
    
    This function handles various possible CSV formats from Atlanta Fed MPT.
    Adjust column mappings based on actual data structure.
    
    Args:
        df: Raw DataFrame from CSV
        filepath: Path to source file (for extracting as_of_date if needed)
    
    Returns:
        Normalized DataFrame with standard columns
    """
    # Try to extract as_of_date from filename or DataFrame
    as_of_date = None
    
    # Extract date from filename: atlanta_mpt_YYYYMMDD.csv
    filename = filepath.stem
    date_match = pd.to_datetime(filename.split("_")[-1], format="%Y%m%d", errors="coerce")
    if pd.notna(date_match):
        as_of_date = date_match
    
    # If not in filename, try to find in DataFrame
    if as_of_date is None:
        # Look for date columns
        date_cols = [col for col in df.columns if any(term in col.lower() for term in ["date", "as_of", "asof"])]
        if date_cols:
            as_of_date = pd.to_datetime(df[date_cols[0]].iloc[0], errors="coerce")
    
    # Default to today if still not found
    if as_of_date is None or pd.isna(as_of_date):
        as_of_date = pd.Timestamp.now().normalize()
    
    rows = []
    
    # Try different column name patterns
    # Pattern 1: Explicit columns
    if all(col in df.columns for col in ["horizon_date", "rate_low", "rate_high", "probability"]):
        for _, row in df.iterrows():
            try:
                horizon_date = pd.to_datetime(row["horizon_date"])
                rate_low = float(row["rate_low"])
                rate_high = float(row["rate_high"])
                prob = float(row["probability"])
                
                # Convert percentage to decimal if needed
                if prob > 1:
                    prob = prob / 100.0
                
                rows.append({
                    "as_of_date": as_of_date,
                    "horizon_date": horizon_date,
                    "rate_bin_low_bps": rate_low * 100,  # Convert to basis points
                    "rate_bin_high_bps": rate_high * 100,
                    "probability": prob
                })
            except Exception as e:
                logger.debug(f"Skipping row due to parsing error: {e}")
                continue
    
    # Pattern 2: Rate bins as columns, probabilities as values
    elif any("rate" in col.lower() or "horizon" in col.lower() for col in df.columns):
        # Look for rate/horizon columns
        rate_cols = [col for col in df.columns if "rate" in col.lower() or "horizon" in col.lower()]
        prob_cols = [col for col in df.columns if "prob" in col.lower() or "p(" in col.lower()]
        
        if rate_cols and prob_cols:
            for _, row in df.iterrows():
                try:
                    # Extract horizon from rate column or separate column
                    horizon_str = str(row[rate_cols[0]])
                    # Try to parse horizon (could be date, meeting identifier, or months ahead)
                    horizon_date = pd.to_datetime(horizon_str, errors="coerce")
                    
                    if pd.isna(horizon_date):
                        # If not a date, might be months ahead or meeting identifier
                        # Default to as_of_date + some period (adjust based on actual format)
                        horizon_date = as_of_date + pd.DateOffset(months=3)
                    
                    # Extract rate range from column name or value
                    # This is a simplified parser - adjust based on actual format
                    rate_low = 0.0
                    rate_high = 0.0
                    prob = float(row[prob_cols[0]])
                    
                    if prob > 1:
                        prob = prob / 100.0
                    
                    rows.append({
                        "as_of_date": as_of_date,
                        "horizon_date": horizon_date,
                        "rate_bin_low_bps": rate_low * 100,
                        "rate_bin_high_bps": rate_high * 100,
                        "probability": prob
                    })
                except Exception as e:
                    logger.debug(f"Skipping row due to parsing error: {e}")
                    continue
    
    # Pattern 3: Generic - try to infer structure
    else:
        logger.warning(f"Could not automatically detect Atlanta MPT data structure in {filepath.name}")
        logger.warning("You may need to customize the parser for this file format")
        logger.warning(f"Columns found: {list(df.columns)}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            "as_of_date", "horizon_date", "rate_bin_low_bps",
            "rate_bin_high_bps", "probability"
        ])
    
    result_df = pd.DataFrame(rows)
    
    if result_df.empty:
        logger.warning(f"No data extracted from {filepath.name}")
    
    return result_df


def extract_mpt_distribution(
    horizon_date: Optional[pd.Timestamp] = None,
    horizon_months: Optional[int] = None,
    as_of_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Extract probability distribution for a specific horizon.
    
    Args:
        horizon_date: Target date/horizon for distribution (if None, uses next FOMC meeting)
        horizon_months: Alternative: Months ahead from as_of_date (if None and horizon_date is None, uses next meeting)
        as_of_date: Date to extract distribution for (if None, uses most recent)
    
    Returns:
        DataFrame with columns: as_of_date, horizon_date, rate_bin_low_bps,
        rate_bin_high_bps, probability
    """
    df = load_atlanta_mpt_panel()
    
    if df.empty:
        logger.warning("No Atlanta MPT data available")
        return pd.DataFrame(columns=[
            "as_of_date", "horizon_date", "rate_bin_low_bps",
            "rate_bin_high_bps", "probability"
        ])
    
    # Filter by as_of_date if specified
    if as_of_date is not None:
        df = df[df["as_of_date"] == as_of_date]
    else:
        # Use most recent as_of_date
        latest_as_of = df["as_of_date"].max()
        df = df[df["as_of_date"] == latest_as_of]
        logger.info(f"Using most recent as_of_date: {latest_as_of}")
    
    # Filter by horizon
    if horizon_date is not None:
        df = df[df["horizon_date"] == horizon_date]
    elif horizon_months is not None and as_of_date is not None:
        target_date = as_of_date + pd.DateOffset(months=horizon_months)
        # Find closest horizon_date
        df["horizon_diff"] = (df["horizon_date"] - target_date).abs()
        closest_idx = df["horizon_diff"].idxmin()
        closest_horizon = df.loc[closest_idx, "horizon_date"]
        df = df[df["horizon_date"] == closest_horizon]
        logger.info(f"Using closest horizon_date to {target_date.date()}: {closest_horizon.date()}")
    else:
        # Use next FOMC meeting (approximate - use closest horizon_date)
        if not df.empty:
            next_horizon = df["horizon_date"].min()
            df = df[df["horizon_date"] == next_horizon]
            logger.info(f"Using next horizon_date: {next_horizon.date()}")
    
    if df.empty:
        logger.warning("No data matching criteria")
        return pd.DataFrame(columns=[
            "as_of_date", "horizon_date", "rate_bin_low_bps",
            "rate_bin_high_bps", "probability"
        ])
    
    # Return distribution
    result = df[[
        "as_of_date", "horizon_date", "rate_bin_low_bps",
        "rate_bin_high_bps", "probability"
    ]].copy()
    
    logger.info(f"Extracted {len(result)} rate bins for specified horizon")
    return result


