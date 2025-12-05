#!/usr/bin/env python3
"""
Run Grid Search for Model Discovery
====================================

This script runs exhaustive grid search to find candidate model configurations.
Results are saved for human review - models are NOT automatically promoted.

Usage:
    # Run grid search for all mappings
    python scripts/run_grid_search.py
    
    # Run for specific mapping
    python scripts/run_grid_search.py --mapping CPI->HY_OAS
    
    # Force rerun (ignore cache)
    python scripts/run_grid_search.py --force

After running:
    1. Review results: python scripts/review_candidates.py
    2. Promote models: python scripts/promote_model.py <model_id>
    3. Build graph:    python scripts/build_graph.py
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from src.models.event_grid_search import run_full_grid_search, find_best_configs


# ============================================================================
# MAPPING DEFINITIONS
# ============================================================================

MAPPINGS = {
    "CPI->HY_OAS": {
        "event": "CPI",
        "instrument": "HY_OAS",
        "target_col": "hy_change",
        "active_factor": "cpi_shock_abs",
        "features": [
            "cpi_shock_abs",
            "yield_vol_10y",
            "hy_vol",
            "slope_10y_2y",
            "fed_funds",
            "hy_oas_before",
            "stlfsi",
        ],
        "thresholds": [0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
        "loader": "load_cpi_hy_dataset",
    },
    "UNEMP->VIX": {
        "event": "UNEMPLOYMENT",
        "instrument": "VIX",
        "target_col": "vix_change",
        "active_factor": "unemp_surprise_abs",
        "features": [
            "unemp_surprise_abs",
            "vix_vol",
            "yield_vol_10y",
            "slope_10y_2y",
            "fed_funds",
            "vix_before",
            "stlfsi",
        ],
        "thresholds": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        "loader": "load_unemp_vix_dataset",
    },
}


# ============================================================================
# DATA LOADERS
# ============================================================================

def load_cpi_hy_dataset():
    """Load CPI → HY OAS event dataset."""
    from src.data.fred_loader import load_series
    from src.data.merge_panel import build_fed_panel
    from src.data.inflation_announcements_loader import load_inflation_announcements
    
    panel = build_fed_panel()
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel.sort_values('date').reset_index(drop=True)
    
    cpi_dates = load_inflation_announcements()
    cpi_dates['release_date'] = pd.to_datetime(cpi_dates['release_date'])
    
    cpi_raw = load_series("CPIAUCSL").reset_index()
    cpi_raw.columns = ['date', 'cpi_value']
    cpi_raw['date'] = pd.to_datetime(cpi_raw['date'])
    cpi_raw['year_month'] = cpi_raw['date'].dt.to_period('M')
    cpi_monthly = cpi_raw.groupby('year_month')['cpi_value'].last().reset_index()
    cpi_monthly['data_period'] = cpi_monthly['year_month'].astype(str)
    cpi_dict = dict(zip(cpi_monthly['data_period'], cpi_monthly['cpi_value']))
    
    def get_prev_month(period):
        year, month = map(int, period.split('-'))
        if month == 1:
            return f"{year - 1}-12"
        return f"{year}-{month - 1:02d}"
    
    events = []
    for idx, ann in cpi_dates.iterrows():
        ann_date = ann['release_date']
        data_period = ann['data_period']
        
        ann_rows = panel[panel['date'] == ann_date]
        prev_rows = panel[panel['date'] < ann_date].tail(1)
        
        if ann_rows.empty or prev_rows.empty:
            continue
        
        ann_row = ann_rows.iloc[0]
        prev_row = prev_rows.iloc[0]
        
        hy_after = ann_row.get('hy_oas', np.nan)
        hy_before = prev_row.get('hy_oas', np.nan)
        
        if pd.isna(hy_before) or pd.isna(hy_after):
            continue
        
        hy_change = hy_after - hy_before
        
        prev_period = get_prev_month(data_period)
        cpi_current = cpi_dict.get(data_period)
        cpi_prev = cpi_dict.get(prev_period)
        cpi_shock = (cpi_current - cpi_prev) if cpi_current and cpi_prev else None
        
        recent = panel[panel['date'] < ann_date].tail(20)
        yield_vol_10y = np.nan
        if len(recent) > 5 and 'y_10y' in recent.columns:
            vals = recent['y_10y'].dropna().values
            if len(vals) > 5:
                yield_vol_10y = np.std(np.diff(vals))
        
        hy_vol = np.nan
        if len(recent) > 5 and 'hy_oas' in recent.columns:
            vals = recent['hy_oas'].dropna().values
            if len(vals) > 5:
                hy_vol = np.std(np.diff(vals))
        
        events.append({
            'date': ann_date,
            'hy_change': hy_change,
            'cpi_shock': cpi_shock,
            'cpi_shock_abs': abs(cpi_shock) if cpi_shock else np.nan,
            'yield_vol_10y': yield_vol_10y,
            'hy_vol': hy_vol,
            'slope_10y_2y': prev_row.get('slope_10y_2y', np.nan),
            'fed_funds': prev_row.get('fed_funds', np.nan),
            'hy_oas_before': hy_before,
            'stlfsi': prev_row.get('stlfsi', np.nan),
            'vix_before': prev_row.get('vix', np.nan),
        })
    
    df = pd.DataFrame(events)
    df = df.dropna(subset=['hy_change', 'cpi_shock_abs'])
    df = df.sort_values('date').reset_index(drop=True)
    
    n_test = int(len(df) * 0.30)
    train_df = df.iloc[:-n_test].copy()
    test_df = df.iloc[-n_test:].copy()
    
    return train_df, test_df


def load_unemp_vix_dataset():
    """Load Unemployment → VIX event dataset."""
    from src.data.fred_loader import load_series
    from src.data.merge_panel import build_fed_panel
    
    panel = build_fed_panel()
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel.sort_values('date').reset_index(drop=True)
    
    unemp = load_series("UNRATE").reset_index()
    unemp.columns = ['date', 'unemp_value']
    unemp['date'] = pd.to_datetime(unemp['date'])
    
    events = []
    for i in range(1, len(unemp)):
        ann_date = unemp.iloc[i]['date']
        
        ann_rows = panel[(panel['date'] >= ann_date) & (panel['date'] <= ann_date + pd.Timedelta(days=3))]
        prev_rows = panel[panel['date'] < ann_date].tail(1)
        
        if ann_rows.empty or prev_rows.empty:
            continue
        
        ann_row = ann_rows.iloc[0]
        prev_row = prev_rows.iloc[0]
        
        vix_after = ann_row.get('vix', np.nan)
        vix_before = prev_row.get('vix', np.nan)
        
        if pd.isna(vix_before) or pd.isna(vix_after):
            continue
        
        vix_change = vix_after - vix_before
        unemp_surprise = unemp.iloc[i]['unemp_value'] - unemp.iloc[i-1]['unemp_value']
        
        recent = panel[panel['date'] < ann_date].tail(20)
        
        vix_vol = np.nan
        if len(recent) > 5 and 'vix' in recent.columns:
            vals = recent['vix'].dropna().values
            if len(vals) > 5:
                vix_vol = np.std(np.diff(vals))
        
        yield_vol_10y = np.nan
        if len(recent) > 5 and 'y_10y' in recent.columns:
            vals = recent['y_10y'].dropna().values
            if len(vals) > 5:
                yield_vol_10y = np.std(np.diff(vals))
        
        events.append({
            'date': ann_date,
            'vix_change': vix_change,
            'unemp_surprise': unemp_surprise,
            'unemp_surprise_abs': abs(unemp_surprise),
            'vix_vol': vix_vol,
            'yield_vol_10y': yield_vol_10y,
            'slope_10y_2y': prev_row.get('slope_10y_2y', np.nan),
            'fed_funds': prev_row.get('fed_funds', np.nan),
            'vix_before': vix_before,
            'stlfsi': prev_row.get('stlfsi', np.nan),
            'hy_oas_before': prev_row.get('hy_oas', np.nan),
        })
    
    df = pd.DataFrame(events)
    df = df.dropna(subset=['vix_change', 'unemp_surprise_abs'])
    df = df.sort_values('date').reset_index(drop=True)
    
    n_test = int(len(df) * 0.30)
    train_df = df.iloc[:-n_test].copy()
    test_df = df.iloc[-n_test:].copy()
    
    return train_df, test_df


# ============================================================================
# MAIN
# ============================================================================

def run_search_for_mapping(
    mapping_name: str,
    mapping_config: dict,
    output_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    """Run grid search for a single mapping."""
    
    output_file = output_dir / f"grid_search_{mapping_name.replace('->', '_')}.csv"
    
    if output_file.exists() and not force:
        print(f"  Loading cached results from {output_file}")
        return pd.read_csv(output_file)
    
    # Load data
    print(f"  Loading data...")
    loader_name = mapping_config["loader"]
    if loader_name == "load_cpi_hy_dataset":
        train_df, test_df = load_cpi_hy_dataset()
    elif loader_name == "load_unemp_vix_dataset":
        train_df, test_df = load_unemp_vix_dataset()
    else:
        raise ValueError(f"Unknown loader: {loader_name}")
    
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Run grid search
    print(f"  Running grid search...")
    results = run_full_grid_search(
        train_df, test_df,
        features=mapping_config["features"],
        target_col=mapping_config["target_col"],
        large_thresholds=mapping_config["thresholds"],
        verbose=True,
    )
    
    # Add mapping info
    results['mapping'] = mapping_name
    results['event'] = mapping_config["event"]
    results['instrument'] = mapping_config["instrument"]
    
    # Add dataset info
    results['n_train'] = len(train_df)
    results['n_test'] = len(test_df)
    results['train_start'] = str(train_df['date'].min().date())
    results['train_end'] = str(train_df['date'].max().date())
    results['test_start'] = str(test_df['date'].min().date())
    results['test_end'] = str(test_df['date'].max().date())
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file, index=False)
    print(f"  Saved {len(results)} results to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run grid search for model discovery")
    parser.add_argument("--mapping", type=str, help="Specific mapping to run (e.g. CPI->HY_OAS)")
    parser.add_argument("--force", action="store_true", help="Force rerun, ignore cache")
    parser.add_argument("--output", type=str, default="registry/search_results", help="Output directory")
    args = parser.parse_args()
    
    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GRID SEARCH FOR MODEL DISCOVERY")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Force: {args.force}")
    
    # Determine which mappings to run
    if args.mapping:
        if args.mapping not in MAPPINGS:
            print(f"\nError: Unknown mapping '{args.mapping}'")
            print(f"Available: {list(MAPPINGS.keys())}")
            return 1
        mappings_to_run = {args.mapping: MAPPINGS[args.mapping]}
    else:
        mappings_to_run = MAPPINGS
    
    all_results = []
    
    for mapping_name, mapping_config in mappings_to_run.items():
        print(f"\n{'='*70}")
        print(f"MAPPING: {mapping_name}")
        print("=" * 70)
        
        results = run_search_for_mapping(
            mapping_name, mapping_config, output_dir, args.force
        )
        all_results.append(results)
        
        # Show best for each FN constraint
        print(f"\n  Best configurations:")
        for fn_max in [0.01, 0.05]:
            best = find_best_configs(results, max_fn_rate=fn_max)
            if len(best) > 0:
                top = best.iloc[0]
                print(f"    FN≤{fn_max*100:.0f}%: {top['model']} w={top['weight']}, "
                      f"thresh={top['large_threshold']:.2f}, @{top['prob_threshold']:.2f}, "
                      f"AUC={top['auc']:.3f}, TN/FP={top['tn_fp_ratio']:.2f}x")
            else:
                print(f"    FN≤{fn_max*100:.0f}%: No configs found")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to {output_dir}")
    print("\nNext steps:")
    print("  1. Review candidates: python scripts/review_candidates.py")
    print("  2. Promote models:    python scripts/promote_model.py <model_id>")
    print("  3. Build graph:       python scripts/build_graph.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

