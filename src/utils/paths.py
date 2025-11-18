"""Path utility functions for data directories"""

from pathlib import Path

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_raw_data_path(subdir: str = "") -> Path:
    """
    Get path to raw data directory or subdirectory.
    
    Args:
        subdir: Subdirectory name (e.g., 'fred', 'fedwatch', 'polymarket')
    
    Returns:
        Path to raw data directory or subdirectory
    """
    path = PROJECT_ROOT / "data" / "raw"
    if subdir:
        path = path / subdir
    return path


def get_processed_data_path() -> Path:
    """
    Get path to processed data directory.
    
    Returns:
        Path to processed data directory
    """
    return PROJECT_ROOT / "data" / "processed"


def ensure_dir_exists(path: Path) -> None:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)

