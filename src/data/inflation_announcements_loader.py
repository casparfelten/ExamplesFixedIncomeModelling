"""Inflation announcement release date loading utilities"""

import os
import re
import urllib.parse
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fredapi import Fred

from src.config import BLS_SCHEDULE_URL
from src.utils.paths import get_raw_data_path, ensure_dir_exists
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Load environment variables
load_dotenv()


def get_fred_api_key() -> Optional[str]:
    """Get FRED API key from environment."""
    return os.getenv("FRED_API_KEY")


def fetch_fred_release_dates(
    series_id: str = "CPIAUCSL",
    api_key: Optional[str] = None,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Fetch CPI release dates using FRED API realtime_start dates.
    
    The FRED API provides realtime_start dates which indicate when data was first
    available. For monthly series like CPI, we can infer release dates by checking
    when each observation first appeared in the database.
    
    Args:
        series_id: FRED series ID (default: CPIAUCSL)
        api_key: FRED API key (if None, reads from environment)
        start_date: Optional start date for data
        end_date: Optional end date for data
    
    Returns:
        DataFrame with columns: data_period, release_date, source
    """
    if api_key is None:
        api_key = get_fred_api_key()
        if not api_key:
            logger.warning("FRED_API_KEY not found. Cannot fetch release dates from FRED.")
            return pd.DataFrame(columns=["data_period", "release_date", "release_time", "source"])
    
    logger.info(f"Fetching release dates from FRED API for {series_id}...")
    
    try:
        fred = Fred(api_key=api_key)
        
        # Get series observations with realtime_start
        # We'll use the FRED API directly to get realtime information
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "realtime_start": "1776-07-04",  # Earliest possible date
        }
        
        if start_date:
            params["observation_start"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["observation_end"] = end_date.strftime("%Y-%m-%d")
        
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "observations" not in data:
            logger.warning(f"No observations found in FRED API response for {series_id}")
            return pd.DataFrame(columns=["data_period", "release_date", "release_time", "source"])
        
        # Extract release dates
        release_dates = []
        for obs in data["observations"]:
            if obs.get("value") == ".":  # Skip missing values
                continue
            
            # observation_date is the data period (e.g., "2024-01-01" for January 2024)
            # realtime_start is when this observation first became available
            obs_date = pd.to_datetime(obs["date"])
            realtime_start = pd.to_datetime(obs["realtime_start"])
            
            # For monthly CPI, the observation date is the first of the month
            # The release date is when it first appeared (realtime_start)
            data_period = obs_date.strftime("%Y-%m")
            release_date = realtime_start.date()
            
            release_dates.append({
                "data_period": data_period,
                "release_date": release_date,
                "release_time": "08:30 ET",  # Standard BLS release time
                "source": "fred"
            })
        
        # Remove duplicates (same data_period might appear multiple times with different realtime_start)
        df = pd.DataFrame(release_dates)
        if not df.empty:
            # Keep the earliest release_date for each data_period
            df = df.sort_values("release_date").drop_duplicates(subset=["data_period"], keep="first")
            df = df.sort_values("data_period")
        
        logger.info(f"Fetched {len(df)} release dates from FRED API")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch release dates from FRED API: {e}")
        return pd.DataFrame(columns=["data_period", "release_date", "release_time", "source"])


def scrape_bls_schedule(url: Optional[str] = None) -> pd.DataFrame:
    """
    Scrape BLS CPI release schedule from their website.
    
    Args:
        url: BLS schedule URL (default: from config)
    
    Returns:
        DataFrame with columns: data_period, release_date, release_time, source
    """
    if url is None:
        url = BLS_SCHEDULE_URL
    
    logger.info(f"Scraping BLS schedule from {url}...")
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        release_dates = []
        
        # BLS schedule page typically has a table with release dates
        # Look for tables containing CPI release information
        tables = soup.find_all("table")
        
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) < 2:
                    continue
                
                # Look for date patterns in the cells
                text = " ".join(cell.get_text(strip=True) for cell in cells)
                
                # Pattern: "CPI for [Month Year] - [Release Date]"
                # Or: "[Month Year] - [Release Date]"
                # Or: "[Release Date] - CPI for [Month Year]"
                
                # Try to match various date patterns
                # Pattern 1: "CPI for January 2024 - December 13, 2023" (data period - release date)
                # Pattern 2: "December 13, 2023 - CPI for January 2024" (release date - data period)
                # Pattern 3: Table format with separate columns
                
                # Look for month names and dates
                month_pattern = r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})"
                date_pattern = r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})|(\w+)\s+(\d{1,2}),\s+(\d{4})"
                
                months = re.findall(month_pattern, text, re.IGNORECASE)
                dates = re.findall(date_pattern, text)
                
                if months and dates:
                    # Try to extract data period and release date
                    # This is heuristic and may need adjustment based on actual page structure
                    pass
        
        # Alternative: Look for specific table structure
        # BLS often uses a table with columns: Release Date | Data Period | Time
        for table in tables:
            # Check if this looks like a CPI release schedule table
            headers = table.find_all("th")
            header_text = " ".join(h.get_text(strip=True).lower() for h in headers)
            
            if "release" in header_text or "date" in header_text or "cpi" in header_text:
                rows = table.find_all("tr")[1:]  # Skip header row
                
                for row in rows:
                    cells = row.find_all(["td", "th"])
                    if len(cells) < 2:
                        continue
                    
                    cell_texts = [cell.get_text(strip=True) for cell in cells]
                    
                    # Try to parse release date and data period
                    release_date_str = None
                    data_period_str = None
                    release_time = "08:30 ET"  # Default
                    
                    for i, text in enumerate(cell_texts):
                        # Look for date patterns
                        date_match = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", text)
                        if date_match:
                            if release_date_str is None:
                                release_date_str = text
                            elif data_period_str is None:
                                data_period_str = text
                        
                        # Look for month-year patterns (data period)
                        month_match = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})", text, re.IGNORECASE)
                        if month_match:
                            month_name, year = month_match.groups()
                            month_num = datetime.strptime(month_name, "%B").month
                            data_period_str = f"{year}-{month_num:02d}"
                        
                        # Look for time patterns
                        time_match = re.search(r"(\d{1,2}):(\d{2})\s*(AM|PM|ET|EST|EDT)?", text, re.IGNORECASE)
                        if time_match:
                            release_time = text
                    
                    if release_date_str:
                        try:
                            # Parse release date
                            release_date = pd.to_datetime(release_date_str).date()
                            
                            # If we don't have data_period, try to infer from release date
                            # CPI is typically released mid-month for the previous month
                            if data_period_str is None:
                                # If released in month M, it's usually for month M-1
                                if release_date.month == 1:
                                    data_period = f"{release_date.year - 1}-12"
                                else:
                                    data_period = f"{release_date.year}-{release_date.month - 1:02d}"
                            else:
                                data_period = data_period_str
                            
                            release_dates.append({
                                "data_period": data_period,
                                "release_date": release_date,
                                "release_time": release_time,
                                "source": "bls"
                            })
                        except Exception as e:
                            logger.debug(f"Failed to parse date from '{release_date_str}': {e}")
                            continue
        
        df = pd.DataFrame(release_dates)
        if not df.empty:
            # Remove duplicates
            df = df.drop_duplicates(subset=["data_period"], keep="first")
            df = df.sort_values("data_period")
        
        logger.info(f"Scraped {len(df)} release dates from BLS schedule")
        return df
        
    except Exception as e:
        logger.error(f"Failed to scrape BLS schedule: {e}")
        return pd.DataFrame(columns=["data_period", "release_date", "release_time", "source"])


def download_inflation_announcements(
    api_key: Optional[str] = None,
    force_reload: bool = False,
    prefer_source: str = "fred"  # "fred" or "bls"
) -> Path:
    """
    Download inflation announcement release dates from available sources.
    
    Args:
        api_key: FRED API key (if None, reads from environment)
        force_reload: If True, re-download even if file exists
        prefer_source: Preferred data source ("fred" or "bls")
    
    Returns:
        Path to saved CSV file
    """
    raw_dir = get_raw_data_path("inflation_announcements")
    ensure_dir_exists(raw_dir)
    filepath = raw_dir / "cpi_release_dates.csv"
    
    if filepath.exists() and not force_reload:
        logger.info(f"Inflation announcements already exist at {filepath}. Use force_reload=True to re-download.")
        return filepath
    
    logger.info("Downloading inflation announcement release dates...")
    
    df_fred = pd.DataFrame()
    df_bls = pd.DataFrame()
    
    # Try FRED API first if preferred
    if prefer_source == "fred" or prefer_source == "both":
        df_fred = fetch_fred_release_dates(api_key=api_key)
    
    # Try BLS scraping as fallback or if preferred
    if prefer_source == "bls" or prefer_source == "both" or (prefer_source == "fred" and df_fred.empty):
        df_bls = scrape_bls_schedule()
    
    # Combine data sources, preferring FRED if available
    if not df_fred.empty and not df_bls.empty:
        # Merge, keeping FRED data where available, BLS as fallback
        df_combined = df_fred.copy()
        # Add BLS data for periods not in FRED
        df_bls_missing = df_bls[~df_bls["data_period"].isin(df_fred["data_period"])]
        df_combined = pd.concat([df_combined, df_bls_missing], ignore_index=True)
        df_combined = df_combined.sort_values("data_period")
    elif not df_fred.empty:
        df_combined = df_fred
    elif not df_bls.empty:
        df_combined = df_bls
    else:
        logger.warning("No release dates could be fetched from any source")
        # Create empty DataFrame with correct structure
        df_combined = pd.DataFrame(columns=["data_period", "release_date", "release_time", "source"])
    
    # Save to CSV
    df_combined.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df_combined)} release dates to {filepath}")
    
    return filepath


def load_inflation_announcements() -> pd.DataFrame:
    """
    Load cached inflation announcement release dates.
    
    Returns:
        DataFrame with columns: data_period, release_date, release_time, source
    """
    raw_dir = get_raw_data_path("inflation_announcements")
    filepath = raw_dir / "cpi_release_dates.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Inflation announcements not found at {filepath}. "
            f"Run download_inflation_announcements() first."
        )
    
    df = pd.read_csv(filepath, parse_dates=["release_date"])
    return df

