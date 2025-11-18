"""CME FedWatch EOD REST API data loading utilities"""

import os
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

from src.config import CME_FEDWATCH_BASE_URL
from src.utils.paths import get_raw_data_path, get_processed_data_path, ensure_dir_exists
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Load environment variables
load_dotenv()


def get_api_key() -> Optional[str]:
    """Get CME FedWatch API key from environment."""
    return os.getenv("CME_FEDWATCH_API_KEY")


def fetch_fedwatch_meetings(api_key: Optional[str] = None) -> List[Dict]:
    """
    Fetch list of FOMC meetings from CME FedWatch API.
    
    Args:
        api_key: CME FedWatch API key (if None, reads from environment)
    
    Returns:
        List of meeting dictionaries with meeting_id, meeting_date, target_ranges
    """
    if api_key is None:
        api_key = get_api_key()
        if not api_key:
            raise ValueError("CME_FEDWATCH_API_KEY not found in environment. Set it in .env file.")
    
    base_url = os.getenv("CME_FEDWATCH_BASE_URL", CME_FEDWATCH_BASE_URL)
    
    # CME FedWatch API endpoint for meetings
    # Note: Actual endpoint structure may vary - adjust based on API documentation
    url = f"{base_url}meetings"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        logger.info("Fetching FOMC meetings from CME FedWatch API...")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        meetings = response.json()
        
        # Normalize response structure (API may return different formats)
        if isinstance(meetings, dict):
            # If wrapped in a dict, try common keys
            meetings = meetings.get("meetings", meetings.get("data", [meetings]))
        elif not isinstance(meetings, list):
            meetings = [meetings]
        
        logger.info(f"Fetched {len(meetings)} FOMC meetings")
        return meetings
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch meetings from CME FedWatch API: {e}")
        raise
    except Exception as e:
        logger.error(f"Error parsing meetings response: {e}")
        raise


def fetch_fedwatch_probabilities(
    meeting_id: str,
    as_of_date: Optional[pd.Timestamp] = None,
    api_key: Optional[str] = None
) -> Dict:
    """
    Fetch daily probability distribution for a specific FOMC meeting.
    
    Args:
        meeting_id: FOMC meeting identifier
        as_of_date: Date to fetch probabilities for (if None, uses today)
        api_key: CME FedWatch API key (if None, reads from environment)
    
    Returns:
        Dictionary with probability distribution data
    """
    if api_key is None:
        api_key = get_api_key()
        if not api_key:
            raise ValueError("CME_FEDWATCH_API_KEY not found in environment. Set it in .env file.")
    
    if as_of_date is None:
        as_of_date = pd.Timestamp.now().normalize()
    
    base_url = os.getenv("CME_FEDWATCH_BASE_URL", CME_FEDWATCH_BASE_URL)
    
    # CME FedWatch API endpoint for probabilities
    # Note: Actual endpoint structure may vary - adjust based on API documentation
    date_str = as_of_date.strftime("%Y-%m-%d")
    url = f"{base_url}meetings/{meeting_id}/probabilities"
    params = {"as_of_date": date_str}
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        logger.info(f"Fetching probabilities for meeting {meeting_id} as of {date_str}...")
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Fetched probabilities for meeting {meeting_id}")
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch probabilities for meeting {meeting_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error parsing probabilities response: {e}")
        raise


def download_all_fedwatch_data(
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    api_key: Optional[str] = None,
    force_reload: bool = False
) -> None:
    """
    Download historical FedWatch data for all meetings in date range.
    
    Args:
        start_date: Start date for data download (if None, defaults to 2014-01-01)
        end_date: End date for data download (if None, uses today)
        api_key: CME FedWatch API key (if None, reads from environment)
        force_reload: If True, re-download even if file exists
    """
    if api_key is None:
        api_key = get_api_key()
        if not api_key:
            logger.warning("CME_FEDWATCH_API_KEY not found. Skipping FedWatch data download.")
            return
    
    if start_date is None:
        start_date = pd.Timestamp("2014-01-01")
    if end_date is None:
        end_date = pd.Timestamp.now().normalize()
    
    logger.info(f"Downloading FedWatch data from {start_date.date()} to {end_date.date()}...")
    
    # Fetch list of meetings
    try:
        meetings = fetch_fedwatch_meetings(api_key)
    except Exception as e:
        logger.error(f"Failed to fetch meetings list: {e}")
        return
    
    raw_dir = get_raw_data_path("fedwatch")
    ensure_dir_exists(raw_dir)
    
    # Download probabilities for each meeting and each day in range
    downloaded_count = 0
    skipped_count = 0
    
    current_date = start_date
    while current_date <= end_date:
        for meeting in meetings:
            meeting_id = meeting.get("meeting_id") or meeting.get("id") or str(meeting.get("meeting_date", ""))
            meeting_date = pd.to_datetime(meeting.get("meeting_date") or meeting.get("date"))
            
            # Skip if meeting is after end_date
            if meeting_date > end_date:
                continue
            
            # Save raw JSON
            filename = f"fedwatch_{meeting_id}_{current_date.strftime('%Y%m%d')}.json"
            filepath = raw_dir / filename
            
            if not force_reload and filepath.exists():
                skipped_count += 1
                continue
            
            try:
                data = fetch_fedwatch_probabilities(meeting_id, current_date, api_key)
                
                # Add metadata
                data["_metadata"] = {
                    "meeting_id": meeting_id,
                    "meeting_date": meeting_date.isoformat(),
                    "as_of_date": current_date.isoformat(),
                    "fetched_at": datetime.now().isoformat()
                }
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                
                downloaded_count += 1
            except Exception as e:
                logger.warning(f"Failed to download probabilities for meeting {meeting_id} on {current_date.date()}: {e}")
                continue
        
        current_date += timedelta(days=1)
    
    logger.info(f"FedWatch download complete. Downloaded: {downloaded_count}, Skipped: {skipped_count}")


def load_raw_fedwatch_json(filepath: Path) -> pd.DataFrame:
    """
    Load and normalize a single FedWatch JSON file into DataFrame format.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        DataFrame with columns: as_of_date, meeting_id, meeting_date, 
        target_rate_low_bps, target_rate_high_bps, probability
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract metadata
    metadata = data.get("_metadata", {})
    meeting_id = metadata.get("meeting_id")
    meeting_date = pd.to_datetime(metadata.get("meeting_date"))
    as_of_date = pd.to_datetime(metadata.get("as_of_date"))
    
    # Extract probability distribution
    # API response structure may vary - handle common patterns
    probabilities = data.get("probabilities", data.get("data", data.get("distribution", [])))
    
    if not isinstance(probabilities, list):
        # If not a list, try to extract from nested structure
        probabilities = probabilities.get("bins", []) if isinstance(probabilities, dict) else []
    
    rows = []
    for prob_item in probabilities:
        # Handle different response formats
        if isinstance(prob_item, dict):
            # Common format: {"target_range_low": 425, "target_range_high": 450, "probability": 0.75}
            low = prob_item.get("target_range_low") or prob_item.get("low") or prob_item.get("rate_low")
            high = prob_item.get("target_range_high") or prob_item.get("high") or prob_item.get("rate_high")
            prob = prob_item.get("probability") or prob_item.get("prob") or prob_item.get("p")
            
            # If rate is in percentage, convert to decimal
            if prob and prob > 1:
                prob = prob / 100.0
            
            if low is not None and high is not None and prob is not None:
                rows.append({
                    "as_of_date": as_of_date,
                    "meeting_id": meeting_id,
                    "meeting_date": meeting_date,
                    "target_rate_low_bps": float(low),
                    "target_rate_high_bps": float(high),
                    "probability": float(prob)
                })
        elif isinstance(prob_item, (list, tuple)) and len(prob_item) >= 3:
            # Tuple/list format: [low, high, probability]
            rows.append({
                "as_of_date": as_of_date,
                "meeting_id": meeting_id,
                "meeting_date": meeting_date,
                "target_rate_low_bps": float(prob_item[0]),
                "target_rate_high_bps": float(prob_item[1]),
                "probability": float(prob_item[2])
            })
    
    return pd.DataFrame(rows)


def build_fedwatch_panel() -> pd.DataFrame:
    """
    Build processed FedWatch panel from all raw JSON files.
    
    Returns:
        DataFrame with columns: as_of_date, meeting_id, meeting_date,
        target_rate_low_bps, target_rate_high_bps, probability
    """
    raw_dir = get_raw_data_path("fedwatch")
    json_files = list(raw_dir.glob("fedwatch_*.json"))
    
    if not json_files:
        logger.warning("No FedWatch JSON files found")
        return pd.DataFrame(columns=[
            "as_of_date", "meeting_id", "meeting_date",
            "target_rate_low_bps", "target_rate_high_bps", "probability"
        ])
    
    logger.info(f"Loading {len(json_files)} FedWatch JSON files...")
    
    all_data = []
    for filepath in json_files:
        try:
            df = load_raw_fedwatch_json(filepath)
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {filepath.name}: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame(columns=[
            "as_of_date", "meeting_id", "meeting_date",
            "target_rate_low_bps", "target_rate_high_bps", "probability"
        ])
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates (keep last if same as_of_date, meeting_id, target_range)
    combined = combined.drop_duplicates(
        subset=["as_of_date", "meeting_id", "target_rate_low_bps", "target_rate_high_bps"],
        keep="last"
    )
    
    # Sort by as_of_date and meeting_date
    combined = combined.sort_values(["as_of_date", "meeting_date"])
    
    # Save processed panel
    processed_dir = get_processed_data_path()
    ensure_dir_exists(processed_dir)
    parquet_path = processed_dir / "fedwatch_full.parquet"
    combined.to_parquet(parquet_path, index=False)
    logger.info(f"Saved processed FedWatch panel to {parquet_path}")
    logger.info(f"FedWatch panel shape: {combined.shape}")
    
    return combined


def load_all_fedwatch() -> pd.DataFrame:
    """
    Load processed FedWatch panel from parquet file.
    
    Returns:
        DataFrame with all FedWatch data
    """
    processed_dir = get_processed_data_path()
    parquet_path = processed_dir / "fedwatch_full.parquet"
    
    if not parquet_path.exists():
        logger.warning("Processed FedWatch panel not found. Building from raw data...")
        return build_fedwatch_panel()
    
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded FedWatch panel: {len(df)} rows")
    return df


def extract_daily_probability_series(
    meeting_id: Optional[str] = None,
    target_range: Optional[Tuple[float, float]] = None,
    meeting_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Extract a daily time series of probabilities for a specific meeting and outcome.
    
    Args:
        meeting_id: FOMC meeting identifier (if None, uses most recent)
        target_range: Tuple of (low, high) target rate range in bps (if None, uses most common)
        meeting_date: Alternative: Date of the FOMC meeting (if None and meeting_id is None, uses most recent)
    
    Returns:
        DataFrame with columns: date, p_fed_outcome
    """
    df = load_all_fedwatch()
    
    if df.empty:
        logger.warning("No FedWatch data available")
        return pd.DataFrame(columns=["date", "p_fed_outcome"])
    
    # Filter by meeting_id or meeting_date
    if meeting_id is not None:
        df = df[df["meeting_id"] == meeting_id]
    elif meeting_date is not None:
        df = df[df["meeting_date"] == meeting_date]
    
    # Filter by target range if specified
    if target_range is not None:
        low, high = target_range
        df = df[
            (df["target_rate_low_bps"] == low) & 
            (df["target_rate_high_bps"] == high)
        ]
    
    # If no filters specified, use most recent meeting and most common outcome
    if meeting_id is None and meeting_date is None:
        if "meeting_date" in df.columns and not df["meeting_date"].isna().all():
            latest_meeting_date = df["meeting_date"].max()
            df = df[df["meeting_date"] == latest_meeting_date]
            logger.info(f"Using most recent meeting date: {latest_meeting_date}")
    
    if target_range is None and not df.empty:
        # Use most common target range
        range_counts = df.groupby(["target_rate_low_bps", "target_rate_high_bps"]).size()
        if len(range_counts) > 0:
            most_common = range_counts.idxmax()
            target_range = most_common
            df = df[
                (df["target_rate_low_bps"] == most_common[0]) &
                (df["target_rate_high_bps"] == most_common[1])
            ]
            logger.info(f"Using most common target range: {target_range}")
    
    if df.empty:
        logger.warning("No data matching criteria")
        return pd.DataFrame(columns=["date", "p_fed_outcome"])
    
    # Create daily series using as_of_date
    result = df[["as_of_date", "probability"]].copy()
    result.columns = ["date", "p_fed_outcome"]
    result = result.sort_values("date")
    result = result.drop_duplicates(subset=["date"], keep="last")
    
    logger.info(f"Extracted {len(result)} daily probability points")
    return result
