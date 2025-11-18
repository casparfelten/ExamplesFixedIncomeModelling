"""Polymarket data fetching and loading utilities"""

from pathlib import Path
from typing import Optional
import json
import pandas as pd
import requests
from datetime import datetime

from src.utils.paths import get_raw_data_path, ensure_dir_exists
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def fetch_market_history(
    market_id: str,
    save_path: Optional[Path] = None
) -> Path:
    """
    Fetch historical data for a Polymarket binary contract.
    
    Uses Polymarket's public API to fetch market data.
    The API structure may vary, so this is a flexible implementation.
    
    Args:
        market_id: Polymarket market ID or slug
        save_path: Path to save the data (if None, uses default location)
    
    Returns:
        Path to saved data file
    """
    logger.info(f"Fetching Polymarket data for market: {market_id}")
    
    # Polymarket API endpoints (these may need to be updated based on actual API)
    # Common endpoints:
    # - GraphQL: https://clob.polymarket.com/graphql
    # - REST: https://clob.polymarket.com/markets/{market_id}
    
    base_url = "https://clob.polymarket.com"
    
    # Try GraphQL endpoint first
    graphql_url = f"{base_url}/graphql"
    
    # GraphQL query to fetch market data
    query = """
    query GetMarket($marketId: String!) {
      market(id: $marketId) {
        id
        question
        outcomes {
          id
          title
          price
          volume
        }
        volume
        liquidity
        createdAt
      }
    }
    """
    
    # Try REST endpoint as alternative
    rest_url = f"{base_url}/markets/{market_id}"
    
    data = None
    try:
        # Try REST endpoint first (simpler)
        response = requests.get(rest_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        logger.info("Fetched data from REST endpoint")
    except Exception as e:
        logger.warning(f"REST endpoint failed: {e}. Trying alternative methods...")
        
        # Alternative: Try to fetch from Polymarket's public data endpoints
        # This is a placeholder - adjust based on actual API structure
        try:
            # Some markets may have historical data at different endpoints
            historical_url = f"{base_url}/markets/{market_id}/history"
            response = requests.get(historical_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            logger.info("Fetched data from history endpoint")
        except Exception as e2:
            logger.error(f"Failed to fetch Polymarket data: {e2}")
            raise
    
    # Save raw data
    if save_path is None:
        raw_dir = get_raw_data_path("polymarket")
        ensure_dir_exists(raw_dir)
        save_path = raw_dir / f"{market_id}.json"
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.info(f"Saved Polymarket data to {save_path}")
    return save_path


def load_polymarket_data(market_id: str) -> pd.DataFrame:
    """
    Load cached Polymarket data and normalize into a DataFrame.
    
    Args:
        market_id: Polymarket market ID
    
    Returns:
        DataFrame with columns: datetime (UTC), price, volume (optional), liquidity (optional)
    """
    raw_dir = get_raw_data_path("polymarket")
    json_path = raw_dir / f"{market_id}.json"
    
    if not json_path.exists():
        raise FileNotFoundError(
            f"Polymarket data for {market_id} not found at {json_path}. "
            f"Run fetch_market_history('{market_id}') first."
        )
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Normalize the data structure
    # Polymarket API structure can vary, so this is a flexible parser
    rows = []
    
    # Try to extract time series data
    # Common structures:
    # 1. List of price points with timestamps
    # 2. Nested structure with market info and price history
    # 3. GraphQL response format
    
    if isinstance(data, list):
        # Assume list of price points
        for item in data:
            try:
                dt = pd.to_datetime(item.get('timestamp') or item.get('time') or item.get('date'))
                price = float(item.get('price') or item.get('lastPrice') or 0)
                volume = item.get('volume')
                liquidity = item.get('liquidity')
                
                rows.append({
                    'datetime': dt,
                    'price': price,
                    'volume': volume if volume is not None else None,
                    'liquidity': liquidity if liquidity is not None else None
                })
            except Exception as e:
                logger.debug(f"Skipping item due to parsing error: {e}")
                continue
    
    elif isinstance(data, dict):
        # Try to find price history in nested structure
        if 'priceHistory' in data:
            history = data['priceHistory']
            for item in history:
                try:
                    dt = pd.to_datetime(item.get('timestamp') or item.get('time'))
                    price = float(item.get('price') or 0)
                    rows.append({
                        'datetime': dt,
                        'price': price,
                        'volume': item.get('volume'),
                        'liquidity': item.get('liquidity')
                    })
                except Exception as e:
                    logger.debug(f"Skipping history item: {e}")
                    continue
        
        # If no price history, try to extract current market state
        elif 'outcomes' in data:
            # This is current state, not historical
            # For historical data, you may need to use a different endpoint
            logger.warning("Data appears to be current state, not historical. Historical endpoints may be needed.")
            for outcome in data.get('outcomes', []):
                try:
                    price = float(outcome.get('price', 0))
                    # Use current timestamp if no historical data
                    rows.append({
                        'datetime': pd.Timestamp.now(tz='UTC'),
                        'price': price,
                        'volume': outcome.get('volume'),
                        'liquidity': outcome.get('liquidity')
                    })
                except Exception as e:
                    logger.debug(f"Skipping outcome: {e}")
                    continue
    
    if not rows:
        logger.warning(f"Could not extract time series data from {json_path}")
        logger.warning("You may need to customize the parser for this market's data structure")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['datetime', 'price', 'volume', 'liquidity'])
    
    df = pd.DataFrame(rows)
    df = df.sort_values('datetime')
    df = df.drop_duplicates(subset=['datetime'], keep='last')
    
    # Ensure datetime is timezone-aware (UTC)
    if df['datetime'].dt.tz is None:
        df['datetime'] = df['datetime'].dt.tz_localize('UTC')
    else:
        df['datetime'] = df['datetime'].dt.tz_convert('UTC')
    
    logger.info(f"Loaded {len(df)} Polymarket data points for {market_id}")
    return df

