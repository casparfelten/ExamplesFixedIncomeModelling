"""Panel merging utilities for building unified datasets"""

from typing import Optional, Tuple
import pandas as pd

from src.data.fred_loader import merge_fred_panel
from src.data.fedwatch_loader import extract_daily_probability_series
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_fed_panel(
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    meeting_date: Optional[pd.Timestamp] = None,
    target_range: Optional[Tuple[float, float]] = None
) -> pd.DataFrame:
    """
    Build a unified daily panel for Fed scenario analysis.
    
    Merges FRED data (yields, macro variables) with FedWatch probability data.
    
    Args:
        start_date: Start date for the panel (if None, uses earliest available)
        end_date: End date for the panel (if None, uses latest available)
        meeting_date: FOMC meeting date for FedWatch probabilities (if None, uses most recent)
        target_range: Target rate range tuple (low, high) in bps for FedWatch (if None, uses most common)
    
    Returns:
        Daily DataFrame with columns:
        - date
        - y_2y (2Y Treasury yield)
        - y_10y (10Y Treasury yield)
        - slope_10y_2y (10Y - 2Y spread)
        - unemployment
        - cpi
        - fed_funds
        - gdp (optional)
        - p_fed_outcome (FedWatch probability for chosen outcome)
    """
    logger.info("Building Fed panel...")
    
    # Load FRED panel
    logger.info("Loading FRED data...")
    fred_panel = merge_fred_panel()
    
    # Load FedWatch probability series
    logger.info("Loading FedWatch probability data...")
    try:
        fedwatch_series = extract_daily_probability_series(
            meeting_date=meeting_date,
            target_range=target_range
        )
    except Exception as e:
        logger.warning(f"Could not load FedWatch data: {e}. Proceeding without FedWatch probabilities.")
        fedwatch_series = pd.DataFrame(columns=["date", "p_fed_outcome"])
    
    # Merge on date
    logger.info("Merging FRED and FedWatch data...")
    
    # Ensure both have 'date' column (not index)
    if 'date' not in fred_panel.columns:
        fred_panel = fred_panel.reset_index()
        if 'date' not in fred_panel.columns:
            fred_panel['date'] = fred_panel.index
    
    if not fedwatch_series.empty and 'date' not in fedwatch_series.columns:
        fedwatch_series = fedwatch_series.reset_index()
        if 'date' not in fedwatch_series.columns:
            fedwatch_series['date'] = fedwatch_series.index
    
    # Convert date columns to datetime
    fred_panel['date'] = pd.to_datetime(fred_panel['date'])
    if not fedwatch_series.empty:
        fedwatch_series['date'] = pd.to_datetime(fedwatch_series['date'])
    
    # Merge
    if fedwatch_series.empty:
        panel = fred_panel.copy()
        panel['p_fed_outcome'] = None
    else:
        panel = pd.merge(
            fred_panel,
            fedwatch_series[['date', 'p_fed_outcome']],
            on='date',
            how='left'
        )
        # Forward-fill FedWatch probabilities (they update periodically, not daily)
        panel['p_fed_outcome'] = panel['p_fed_outcome'].ffill()
    
    # Filter by date range if specified
    if start_date is not None:
        panel = panel[panel['date'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        panel = panel[panel['date'] <= pd.to_datetime(end_date)]
    
    # Sort by date
    panel = panel.sort_values('date').reset_index(drop=True)
    
    # Drop rows where key series are missing (optional - you may want to keep them)
    # For now, we'll keep all rows and let forward-fill handle missing values
    
    logger.info(f"Fed panel built: {len(panel)} rows, {len(panel.columns)} columns")
    logger.info(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
    
    return panel


def build_polymarket_panel(
    market_id: str,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Build a unified daily panel for Polymarket/war scenario analysis.
    
    TODO: This is a placeholder for future implementation.
    Will merge:
    - Bund yields / energy prices (from FRED or other sources)
    - Polymarket probabilities for a geopolitical contract
    
    Args:
        market_id: Polymarket market ID for the contract
        start_date: Start date for the panel
        end_date: End date for the panel
    
    Returns:
        Daily DataFrame with merged data
    """
    logger.info("Building Polymarket panel (placeholder)...")
    logger.warning("build_polymarket_panel is not yet implemented")
    
    # TODO: Load Bund yields from FRED or other source
    # TODO: Load energy prices (WTI, Brent) from FRED or other source
    # TODO: Load Polymarket data for the specified market_id
    # TODO: Merge all data on daily frequency
    # TODO: Return unified panel
    
    # Placeholder return
    return pd.DataFrame(columns=[
        'date',
        'bund_yield',  # Placeholder
        'wti_price',  # Placeholder
        'p_polymarket_outcome'  # Placeholder
    ])

