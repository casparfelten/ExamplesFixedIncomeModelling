"""CME FedWatch data loading and parsing utilities"""

from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd

from src.utils.paths import get_raw_data_path
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def scan_fedwatch_files() -> List[Path]:
    """
    Scan for all FedWatch Excel files in the raw data directory.
    
    Returns:
        List of file paths
    """
    raw_dir = get_raw_data_path("fedwatch")
    files = list(raw_dir.glob("*.xlsx")) + list(raw_dir.glob("*.xls"))
    
    logger.info(f"Found {len(files)} FedWatch Excel files")
    return files


def parse_fedwatch_file(filepath: Path) -> pd.DataFrame:
    """
    Parse a FedWatch Excel file into a tidy DataFrame.
    
    Assumes the Excel file has a structure with:
    - Meeting date information
    - Target rate ranges (low, high) in basis points
    - Probabilities for each range
    
    This is a flexible parser that attempts to handle common FedWatch formats.
    You may need to adjust based on the actual file structure.
    
    Args:
        filepath: Path to Excel file
    
    Returns:
        DataFrame with columns: as_of_date, meeting_date, target_range_low, 
        target_range_high, probability
    """
    logger.info(f"Parsing FedWatch file: {filepath.name}")
    
    try:
        # Try reading the first sheet
        df = pd.read_excel(filepath, sheet_name=0)
    except Exception as e:
        logger.error(f"Failed to read Excel file {filepath}: {e}")
        raise
    
    # This is a generic parser - actual FedWatch files may have different structures
    # Common patterns:
    # 1. First row/column may contain metadata (meeting date, as-of date)
    # 2. Rate ranges may be in rows or columns
    # 3. Probabilities may be percentages or decimals
    
    # Try to extract meeting date and as-of date from filename or sheet
    # Example filename: fedwatch_meeting_20240320.xlsx
    filename = filepath.stem
    meeting_date = None
    as_of_date = None
    
    # Try to extract date from filename
    import re
    date_match = re.search(r'(\d{8})', filename)
    if date_match:
        date_str = date_match.group(1)
        try:
            meeting_date = pd.to_datetime(date_str, format='%Y%m%d')
            as_of_date = pd.Timestamp.now().normalize()  # Default to today
        except:
            pass
    
    # Try to find rate ranges and probabilities in the DataFrame
    # This is a simplified parser - adjust based on actual file structure
    result_rows = []
    
    # Look for columns that might contain rate information
    # Common column names: "Target Rate", "Rate Range", "Probability", etc.
    rate_cols = [col for col in df.columns if any(
        term in str(col).lower() for term in ['rate', 'target', 'range']
    )]
    prob_cols = [col for col in df.columns if 'prob' in str(col).lower()]
    
    # If we can't find standard columns, try to infer structure
    if not rate_cols or not prob_cols:
        # Assume first few columns might be rates, last might be probability
        # Or rates might be in index
        logger.warning(f"Could not automatically detect rate/probability columns in {filepath.name}")
        logger.warning("You may need to customize the parser for this file format")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            "as_of_date", "meeting_date", "target_range_low", 
            "target_range_high", "probability"
        ])
    
    # Parse rows
    for idx, row in df.iterrows():
        # Try to extract rate range
        # This is a placeholder - adjust based on actual format
        try:
            # Example: if rate is in format "425-450" or separate low/high columns
            rate_str = str(row[rate_cols[0]])
            
            # Try to parse range
            if '-' in rate_str:
                low, high = rate_str.split('-')
                target_range_low = float(low.strip())
                target_range_high = float(high.strip())
            else:
                # Assume single value, use as both low and high
                target_range_low = float(rate_str)
                target_range_high = target_range_low
            
            # Get probability
            prob_str = str(row[prob_cols[0]])
            # Remove % if present and convert to fraction
            prob_str = prob_str.replace('%', '').strip()
            probability = float(prob_str) / 100.0 if float(prob_str) > 1 else float(prob_str)
            
            result_rows.append({
                "as_of_date": as_of_date,
                "meeting_date": meeting_date,
                "target_range_low": target_range_low,
                "target_range_high": target_range_high,
                "probability": probability
            })
        except Exception as e:
            logger.debug(f"Skipping row {idx} due to parsing error: {e}")
            continue
    
    result_df = pd.DataFrame(result_rows)
    
    if result_df.empty:
        logger.warning(f"No data extracted from {filepath.name}")
    
    return result_df


def load_all_fedwatch() -> pd.DataFrame:
    """
    Load and concatenate all FedWatch files.
    
    Returns:
        Combined DataFrame with all parsed FedWatch data
    """
    files = scan_fedwatch_files()
    
    if not files:
        logger.warning("No FedWatch files found")
        return pd.DataFrame(columns=[
            "as_of_date", "meeting_date", "target_range_low",
            "target_range_high", "probability"
        ])
    
    all_data = []
    for filepath in files:
        try:
            df = parse_fedwatch_file(filepath)
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            logger.error(f"Failed to parse {filepath}: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame(columns=[
            "as_of_date", "meeting_date", "target_range_low",
            "target_range_high", "probability"
        ])
    
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded {len(combined)} FedWatch records from {len(all_data)} files")
    
    return combined


def extract_daily_probability_series(
    meeting_date: Optional[pd.Timestamp] = None,
    target_range: Optional[Tuple[float, float]] = None
) -> pd.DataFrame:
    """
    Extract a daily time series of probabilities for a specific meeting and outcome.
    
    Args:
        meeting_date: Date of the FOMC meeting (if None, uses most recent)
        target_range: Tuple of (low, high) target rate range in bps (if None, uses most common)
    
    Returns:
        DataFrame with columns: date, p_fed_outcome
    """
    df = load_all_fedwatch()
    
    if df.empty:
        logger.warning("No FedWatch data available")
        return pd.DataFrame(columns=["date", "p_fed_outcome"])
    
    # Filter by meeting date if specified
    if meeting_date is not None:
        df = df[df["meeting_date"] == meeting_date]
    
    # Filter by target range if specified
    if target_range is not None:
        low, high = target_range
        df = df[
            (df["target_range_low"] == low) & 
            (df["target_range_high"] == high)
        ]
    
    # If no filters specified, use most recent meeting and most common outcome
    if meeting_date is None:
        if "meeting_date" in df.columns and not df["meeting_date"].isna().all():
            meeting_date = df["meeting_date"].max()
            df = df[df["meeting_date"] == meeting_date]
            logger.info(f"Using most recent meeting date: {meeting_date}")
    
    if target_range is None and not df.empty:
        # Use most common target range
        range_counts = df.groupby(["target_range_low", "target_range_high"]).size()
        most_common = range_counts.idxmax()
        target_range = most_common
        df = df[
            (df["target_range_low"] == most_common[0]) &
            (df["target_range_high"] == most_common[1])
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

