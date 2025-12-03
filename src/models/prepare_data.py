"""Data preparation utilities for CPI-Bond Yield model"""

from typing import Optional, Tuple, List
import pandas as pd
import numpy as np

from src.data.merge_panel import build_fed_panel
from src.data.inflation_announcements_loader import load_inflation_announcements
from src.data.fred_loader import load_series
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def prepare_event_data(
    target_yield: str = "y_2y",
    prediction_horizon: int = 0,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    min_yield_history: int = 30
) -> pd.DataFrame:
    """
    Prepare event-based dataset for CPI-Bond Yield modeling.
    
    IMPORTANT: This creates a dataset based on DISCRETE CPI announcement events only.
    - One row per CPI announcement date (not daily data)
    - CPI shocks are calculated from actual monthly values revealed on announcement dates
    - Yield changes are measured as reactions to CPI announcements (discrete events)
    - We do NOT include non-announcement days - this is about shocks, not daily noise
    
    Creates features based on CPI announcement dates, not daily forward-filled values.
    
    Args:
        target_yield: Which yield to predict ('y_2y' or 'y_10y')
        prediction_horizon: Days after announcement to predict (0 = same day, 1 = next day, etc.)
        start_date: Optional start date filter
        end_date: Optional end date filter
        min_yield_history: Minimum days of yield history required before announcement
    
    Returns:
        DataFrame with one row per CPI announcement event, containing:
        - date: Announcement date (discrete event date)
        - cpi_shock_mom: Month-over-month CPI change (%) - the actual shock revealed
        - cpi_shock_yoy: Year-over-year CPI change (%)
        - cpi_level: CPI index level
        - target_yield_change: Change in target yield (reaction to CPI announcement)
        - target_yield_level: Level of target yield before announcement
        - background features: gdp, unemployment, fed_funds, slope_10y_2y
        - lagged_yield: Previous day's yield (for momentum)
    """
    logger.info("Preparing event-based CPI-Bond Yield dataset...")
    
    # Load panel data
    logger.info("Loading Fed panel...")
    panel = build_fed_panel(start_date=start_date, end_date=end_date)
    
    # Ensure date is datetime
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel.sort_values('date').reset_index(drop=True)
    
    # Load raw CPI data to get actual monthly values (not forward-filled)
    logger.info("Loading raw CPI data for accurate monthly values...")
    try:
        cpi_raw = load_series("CPIAUCSL")
        cpi_raw = cpi_raw.reset_index()
        cpi_raw.columns = ['date', 'cpi_value']
        cpi_raw['date'] = pd.to_datetime(cpi_raw['date'])
        # Create a mapping of data period to CPI value
        cpi_raw['year_month'] = cpi_raw['date'].dt.to_period('M')
        # Get the CPI value for each month (use first of month or last value in month)
        cpi_monthly = cpi_raw.groupby('year_month')['cpi_value'].last().reset_index()
        cpi_monthly['data_period'] = cpi_monthly['year_month'].astype(str)
        cpi_dict = dict(zip(cpi_monthly['data_period'], cpi_monthly['cpi_value']))
        logger.info(f"Loaded {len(cpi_dict)} monthly CPI values")
    except Exception as e:
        logger.warning(f"Could not load raw CPI data: {e}. Will use panel values.")
        cpi_dict = {}
    
    # Load CPI announcement dates
    logger.info("Loading CPI announcement dates...")
    try:
        announcements = load_inflation_announcements()
        announcements['release_date'] = pd.to_datetime(announcements['release_date'])
    except FileNotFoundError:
        logger.warning("CPI announcement dates not found. Attempting to use monthly CPI changes as proxy...")
        # Fallback: use monthly CPI changes as proxy for announcements
        announcements = _create_proxy_announcements(panel)
    
    # Filter announcements to date range
    if start_date:
        announcements = announcements[announcements['release_date'] >= pd.to_datetime(start_date)]
    if end_date:
        announcements = announcements[announcements['release_date'] <= pd.to_datetime(end_date)]
    
    logger.info(f"Found {len(announcements)} CPI announcements")
    
    # Create event-based dataset
    # IMPORTANT: We ONLY create events for CPI announcement dates (discrete events)
    # We do NOT include non-announcement days - this is about shocks, not daily noise
    events = []
    
    for idx, ann in announcements.iterrows():
        ann_date = ann['release_date']
        data_period = ann['data_period']
        
        # This is a discrete CPI announcement event - not random daily variation
        
        # Find panel rows around announcement date
        # Get announcement day and previous day
        ann_day = panel[panel['date'] == ann_date]
        
        if ann_day.empty:
            # Try to find closest business day if announcement was on weekend/holiday
            # Look within 3 days
            for offset in [1, -1, 2, -2, 3, -3]:
                candidate_date = ann_date + pd.Timedelta(days=offset)
                ann_day = panel[panel['date'] == candidate_date]
                if not ann_day.empty:
                    ann_date = candidate_date
                    break
        
        if ann_day.empty:
            logger.debug(f"Skipping announcement {data_period} - no panel data for {ann['release_date']}")
            continue
        
        ann_row = ann_day.iloc[0]
        
        # Get previous trading day (before announcement)
        # This should be before the final ann_date (which may have been adjusted)
        prev_day = panel[panel['date'] < ann_date].iloc[-1:] if len(panel[panel['date'] < ann_date]) > 0 else None
        
        # Get CPI values for MoM shock calculation
        # Use raw monthly CPI values if available, otherwise fall back to panel
        current_cpi = None
        prev_cpi = None
        
        # Try to get current month's CPI from raw data
        if data_period in cpi_dict:
            current_cpi = cpi_dict[data_period]
        else:
            # Fallback to panel value
            current_cpi = ann_row.get('cpi', np.nan)
        
        # Get previous month's CPI
        try:
            prev_month_period = _get_previous_month(data_period)
            
            # Try to get previous month's CPI from raw data
            if prev_month_period in cpi_dict:
                prev_cpi = cpi_dict[prev_month_period]
            else:
                # Fallback: try to get from previous month's announcement
                prev_month_ann = announcements[announcements['data_period'] == prev_month_period]
                if not prev_month_ann.empty:
                    prev_month_date = pd.to_datetime(prev_month_ann.iloc[0]['release_date'])
                    prev_month_row = panel[panel['date'] == prev_month_date]
                    if prev_month_row.empty:
                        prev_month_row = panel[panel['date'] <= prev_month_date].iloc[-1:] if len(panel[panel['date'] <= prev_month_date]) > 0 else None
                    if prev_month_row is not None and not prev_month_row.empty:
                        prev_cpi = prev_month_row.iloc[0].get('cpi', np.nan)
        except Exception as e:
            logger.debug(f"Error getting previous month CPI for {data_period}: {e}")
        
        # Compute CPI MoM shock
        # This is the actual CPI shock revealed on the announcement date
        # Calculated from actual monthly CPI values (not forward-filled daily values)
        # This is a discrete shock event, not continuous daily variation
        if prev_cpi is not None and current_cpi is not None and not (pd.isna(prev_cpi) or pd.isna(current_cpi)) and prev_cpi > 0:
            cpi_shock_mom = ((current_cpi - prev_cpi) / prev_cpi) * 100
        else:
            cpi_shock_mom = np.nan
        
        # Get YoY change
        cpi_yoy = np.nan
        try:
            prev_year_period = _get_previous_year(data_period)
            
            # Try to get previous year's CPI from raw data
            if prev_year_period in cpi_dict:
                prev_year_cpi = cpi_dict[prev_year_period]
                if prev_year_cpi > 0 and current_cpi is not None:
                    cpi_yoy = ((current_cpi - prev_year_cpi) / prev_year_cpi) * 100
            else:
                # Fallback: use panel value
                cpi_yoy = ann_row.get('cpi_yoy', np.nan)
        except Exception:
            # Fallback: use panel value
            cpi_yoy = ann_row.get('cpi_yoy', np.nan)
        
        # Get target yield BEFORE announcement (previous trading day)
        # This is the yield before the CPI shock is revealed
        # This captures the baseline yield before the discrete CPI announcement event
        if prev_day is not None and not prev_day.empty:
            target_yield_before = prev_day.iloc[0].get(target_yield)
        else:
            target_yield_before = np.nan
        
        # Get target yield AFTER announcement (on announcement day + horizon)
        # For same-day (horizon=0), use announcement day's yield (after announcement)
        # This captures the market's reaction to the CPI shock on the announcement day
        # For multi-day horizons, use future day's yield
        if prediction_horizon == 0:
            # Same-day: use announcement day's yield (captures reaction to CPI shock on announcement day)
            # This is the yield AFTER the CPI announcement is released (discrete event)
            target_yield_after = ann_row.get(target_yield)
        else:
            # Multi-day: use yield on future date
            future_date = ann_date + pd.Timedelta(days=prediction_horizon)
            future_row = panel[panel['date'] == future_date]
            if future_row.empty:
                # Find closest date within 2 days
                for offset in [1, -1, 2, -2]:
                    candidate_date = future_date + pd.Timedelta(days=offset)
                    future_row = panel[panel['date'] == candidate_date]
                    if not future_row.empty:
                        break
            
            if not future_row.empty:
                target_yield_after = future_row.iloc[0].get(target_yield)
            else:
                target_yield_after = np.nan
        
        # Compute yield change (after - before)
        # This is the yield change SPECIFICALLY due to the CPI announcement shock
        # It's the difference between yield after announcement and yield before announcement
        # This captures the discrete event reaction, not random daily noise
        if pd.notna(target_yield_before) and pd.notna(target_yield_after):
            target_yield_change = target_yield_after - target_yield_before
        else:
            target_yield_change = np.nan
        
        # Get lagged yield (previous day) - same as before for momentum feature
        if prev_day is not None and not prev_day.empty:
            lagged_yield = prev_day.iloc[0].get(target_yield)
        else:
            lagged_yield = np.nan
        
        # Get background variables
        gdp = ann_row.get('gdp', np.nan)
        unemployment = ann_row.get('unemployment', np.nan)
        fed_funds = ann_row.get('fed_funds', np.nan)
        slope = ann_row.get('slope_10y_2y', np.nan)
        
        # Calculate recent yield volatility (rolling std of yield changes over last 20 days)
        # This helps identify when markets are volatile and more likely to have large moves
        recent_dates = panel[panel['date'] < ann_date].tail(20)
        if len(recent_dates) > 5:
            recent_yields = recent_dates[target_yield].values
            if len(recent_yields[~np.isnan(recent_yields)]) > 5:
                yield_volatility = np.nanstd(np.diff(recent_yields[~np.isnan(recent_yields)]))
            else:
                yield_volatility = np.nan
        else:
            yield_volatility = np.nan
        
        # CPI shock magnitude (absolute value) - helps identify large shocks
        cpi_shock_magnitude = abs(cpi_shock_mom) if pd.notna(cpi_shock_mom) else np.nan
        
        # Check if we have minimum history
        if pd.isna(target_yield_before) or pd.isna(lagged_yield):
            continue
        
        # Create event row
        event = {
            'date': ann_date,
            'data_period': data_period,
            'cpi_shock_mom': cpi_shock_mom,
            'cpi_shock_yoy': cpi_yoy,
            'cpi_shock_magnitude': cpi_shock_magnitude,  # New: magnitude of shock
            'cpi_level': current_cpi if current_cpi is not None else ann_row.get('cpi', np.nan),
            f'{target_yield}_before': target_yield_before,
            f'{target_yield}_change': target_yield_change,
            f'{target_yield}_lagged': lagged_yield,
            'yield_lagged': lagged_yield,  # Generic name for model feature
            'yield_volatility': yield_volatility,  # New: recent volatility
            'gdp': gdp,
            'unemployment': unemployment,
            'fed_funds': fed_funds,
            'slope_10y_2y': slope,
            # Interaction terms - how CPI shock interacts with market conditions
            'cpi_shock_x_fed_funds': cpi_shock_mom * fed_funds if pd.notna(cpi_shock_mom) and pd.notna(fed_funds) else np.nan,
            'cpi_shock_x_unemployment': cpi_shock_mom * unemployment if pd.notna(cpi_shock_mom) and pd.notna(unemployment) else np.nan,
            'cpi_shock_x_volatility': cpi_shock_mom * yield_volatility if pd.notna(cpi_shock_mom) and pd.notna(yield_volatility) else np.nan,
        }
        
        events.append(event)
    
    events_df = pd.DataFrame(events)
    
    if events_df.empty:
        logger.warning("No events created. Check data availability.")
        return events_df
    
    # Drop rows with missing target
    events_df = events_df.dropna(subset=[f'{target_yield}_change'])
    
    logger.info(f"Created {len(events_df)} events with valid target data")
    logger.info(f"Date range: {events_df['date'].min()} to {events_df['date'].max()}")
    
    return events_df


def _get_previous_month(period: str) -> str:
    """Get previous month's period string (YYYY-MM format)."""
    year, month = map(int, period.split('-'))
    if month == 1:
        return f"{year - 1}-12"
    else:
        return f"{year}-{month - 1:02d}"


def _get_previous_year(period: str) -> str:
    """Get same month previous year's period string (YYYY-MM format)."""
    year, month = period.split('-')
    return f"{int(year) - 1}-{month}"


def _create_proxy_announcements(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Create proxy announcement dates from monthly CPI changes.
    
    This is a fallback when actual announcement dates are not available.
    """
    logger.info("Creating proxy announcement dates from monthly CPI changes...")
    
    # Find dates where CPI changes (monthly data)
    panel = panel.copy()
    panel['cpi_change'] = panel['cpi'].diff()
    
    # Get first date of each month where CPI changed
    announcements = []
    panel['year_month'] = panel['date'].dt.to_period('M')
    
    for period, group in panel.groupby('year_month'):
        # Find first date in month where CPI changed
        changed = group[group['cpi_change'] != 0]
        if not changed.empty:
            first_change = changed.iloc[0]
            announcements.append({
                'data_period': str(period),
                'release_date': first_change['date'],
                'release_time': '08:30 ET',
                'source': 'proxy'
            })
    
    return pd.DataFrame(announcements)


def create_train_test_split(
    events_df: pd.DataFrame,
    test_size: float = 0.3,
    min_train_size: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create chronological train/test split for time series data.
    
    Args:
        events_df: Event-based DataFrame with 'date' column
        test_size: Proportion of data to use for testing (default: 0.3)
        min_train_size: Minimum number of events in training set
    
    Returns:
        Tuple of (train_df, test_df)
    """
    events_df = events_df.sort_values('date').reset_index(drop=True)
    
    n_total = len(events_df)
    n_test = int(n_total * test_size)
    n_train = n_total - n_test
    
    if min_train_size and n_train < min_train_size:
        logger.warning(f"Requested min_train_size={min_train_size} but only {n_train} training samples available")
        n_train = min_train_size
        n_test = n_total - n_train
    
    train_df = events_df.iloc[:n_train].copy()
    test_df = events_df.iloc[n_train:].copy()
    
    logger.info(f"Train/test split: {len(train_df)} train, {len(test_df)} test")
    logger.info(f"Train date range: {train_df['date'].min()} to {train_df['date'].max()}")
    logger.info(f"Test date range: {test_df['date'].min()} to {test_df['date'].max()}")
    
    return train_df, test_df

