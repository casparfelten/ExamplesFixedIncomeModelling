#!/usr/bin/env python3
"""
Build QSIG Macro Graph from Grid Search Results
================================================

This script:
1. Loads event datasets for CPI→HY and Unemp→VIX
2. Runs grid search (or uses cached results)
3. Builds Edge objects for each FN constraint
4. Creates the full Graph structure
5. Saves everything to the registry

Usage:
    python scripts/build_macro_graph.py
    
    # Skip grid search (use cached results)
    python scripts/build_macro_graph.py --skip-search
    
    # Specify output directory
    python scripts/build_macro_graph.py --output registry/
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime

from src.graph import (
    Graph,
    EdgeSpec,
    EdgeBuilder,
    NodeId,
    NodeType,
    save_graph,
    export_graph_summary,
)
from src.graph.edge_builder import build_edge_slot, build_graph_from_specs


def load_cpi_hy_dataset():
    """
    Load CPI → HY OAS event dataset.
    
    Returns:
        train_df, test_df, features list
    """
    from src.data.fred_loader import load_series
    from src.data.merge_panel import build_fed_panel
    from src.data.inflation_announcements_loader import load_inflation_announcements
    
    print("Loading CPI → HY OAS dataset...")
    
    # Load panel
    panel = build_fed_panel()
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel.sort_values('date').reset_index(drop=True)
    
    # Load CPI announcements
    cpi_dates = load_inflation_announcements()
    cpi_dates['release_date'] = pd.to_datetime(cpi_dates['release_date'])
    
    # Load raw CPI for MoM calculation
    cpi_raw = load_series("CPIAUCSL").reset_index()
    cpi_raw.columns = ['date', 'cpi_value']
    cpi_raw['date'] = pd.to_datetime(cpi_raw['date'])
    cpi_raw['year_month'] = cpi_raw['date'].dt.to_period('M')
    cpi_monthly = cpi_raw.groupby('year_month')['cpi_value'].last().reset_index()
    cpi_monthly['data_period'] = cpi_monthly['year_month'].astype(str)
    cpi_dict = dict(zip(cpi_monthly['data_period'], cpi_monthly['cpi_value']))
    
    # Build event-level dataset
    events = []
    
    for idx, ann in cpi_dates.iterrows():
        ann_date = ann['release_date']
        data_period = ann['data_period']
        
        # Get t-1 (before announcement) and t (after)
        ann_rows = panel[panel['date'] == ann_date]
        prev_rows = panel[panel['date'] < ann_date].tail(1)
        
        if ann_rows.empty or prev_rows.empty:
            continue
        
        ann_row = ann_rows.iloc[0]
        prev_row = prev_rows.iloc[0]
        
        # HY OAS change
        hy_after = ann_row.get('hy_oas', np.nan)
        hy_before = prev_row.get('hy_oas', np.nan)
        
        if pd.isna(hy_before) or pd.isna(hy_after):
            continue
        
        hy_change = hy_after - hy_before
        
        # CPI shock (MoM)
        prev_period = _get_prev_month(data_period)
        cpi_current = cpi_dict.get(data_period)
        cpi_prev = cpi_dict.get(prev_period)
        
        cpi_shock = None
        if cpi_current is not None and cpi_prev is not None:
            cpi_shock = cpi_current - cpi_prev
        
        # Compute 20-day yield volatility (10Y)
        recent = panel[panel['date'] < ann_date].tail(20)
        yield_vol_10y = np.nan
        if len(recent) > 5 and 'y_10y' in recent.columns:
            vals = recent['y_10y'].dropna().values
            if len(vals) > 5:
                yield_vol_10y = np.std(np.diff(vals))
        
        # HY volatility
        hy_vol = np.nan
        if len(recent) > 5 and 'hy_oas' in recent.columns:
            vals = recent['hy_oas'].dropna().values
            if len(vals) > 5:
                hy_vol = np.std(np.diff(vals))
        
        event = {
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
        }
        events.append(event)
    
    df = pd.DataFrame(events)
    df = df.dropna(subset=['hy_change', 'cpi_shock_abs'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Chronological split 70/30
    n_test = int(len(df) * 0.30)
    train_df = df.iloc[:-n_test].copy()
    test_df = df.iloc[-n_test:].copy()
    
    features = [
        'cpi_shock_abs',
        'yield_vol_10y',
        'hy_vol',
        'slope_10y_2y',
        'fed_funds',
        'hy_oas_before',
        'stlfsi',
    ]
    
    print(f"  Total events: {len(df)}")
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    
    return train_df, test_df, features


def load_unemp_vix_dataset():
    """
    Load Unemployment → VIX event dataset.
    
    Returns:
        train_df, test_df, features list
    """
    from src.data.fred_loader import load_series
    from src.data.merge_panel import build_fed_panel
    
    print("Loading Unemployment → VIX dataset...")
    
    # Load panel
    panel = build_fed_panel()
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel.sort_values('date').reset_index(drop=True)
    
    # Load unemployment to find announcement dates
    unemp = load_series("UNRATE").reset_index()
    unemp.columns = ['date', 'unemp_value']
    unemp['date'] = pd.to_datetime(unemp['date'])
    
    # Unemployment is released monthly
    # For simplicity, use first business day of each month as proxy
    events = []
    
    for i in range(1, len(unemp)):
        ann_date = unemp.iloc[i]['date']
        
        # Find closest panel date
        ann_rows = panel[(panel['date'] >= ann_date) & (panel['date'] <= ann_date + pd.Timedelta(days=3))]
        prev_rows = panel[panel['date'] < ann_date].tail(1)
        
        if ann_rows.empty or prev_rows.empty:
            continue
        
        ann_row = ann_rows.iloc[0]
        prev_row = prev_rows.iloc[0]
        
        # VIX change
        vix_after = ann_row.get('vix', np.nan)
        vix_before = prev_row.get('vix', np.nan)
        
        if pd.isna(vix_before) or pd.isna(vix_after):
            continue
        
        vix_change = vix_after - vix_before
        
        # Unemployment surprise
        unemp_now = unemp.iloc[i]['unemp_value']
        unemp_prev = unemp.iloc[i-1]['unemp_value']
        unemp_surprise = unemp_now - unemp_prev
        
        # Compute volatilities
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
        
        event = {
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
        }
        events.append(event)
    
    df = pd.DataFrame(events)
    df = df.dropna(subset=['vix_change', 'unemp_surprise_abs'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Chronological split 70/30
    n_test = int(len(df) * 0.30)
    train_df = df.iloc[:-n_test].copy()
    test_df = df.iloc[-n_test:].copy()
    
    features = [
        'unemp_surprise_abs',
        'vix_vol',
        'yield_vol_10y',
        'slope_10y_2y',
        'fed_funds',
        'vix_before',
        'stlfsi',
    ]
    
    print(f"  Total events: {len(df)}")
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    
    return train_df, test_df, features


def _get_prev_month(period: str) -> str:
    """Get previous month's period string."""
    year, month = map(int, period.split('-'))
    if month == 1:
        return f"{year - 1}-12"
    else:
        return f"{year}-{month - 1:02d}"


def run_grid_search_for_spec(
    spec: EdgeSpec,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list,
):
    """Run grid search for a given spec."""
    from src.models.event_grid_search import run_full_grid_search
    
    print(f"\nRunning grid search for {spec.slot_id}...")
    
    results = run_full_grid_search(
        train_df, test_df,
        features=features,
        target_col=spec.target_col,
        large_thresholds=spec.large_threshold_candidates,
        verbose=True,
    )
    
    print(f"  Total configurations: {len(results)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Build QSIG Macro Graph")
    parser.add_argument("--skip-search", action="store_true", help="Skip grid search (use manual configs)")
    parser.add_argument("--output", type=str, default="registry", help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("QSIG MACRO GRAPH BUILDER")
    print("=" * 70)
    
    # Define node IDs
    nodes = {
        "CPI": NodeId(NodeType.EVENT, "CPI", "Consumer Price Index release"),
        "UNEMPLOYMENT": NodeId(NodeType.EVENT, "UNEMPLOYMENT", "BLS Employment Situation"),
        "HY_OAS": NodeId(NodeType.INSTRUMENT, "HY_OAS", "High-Yield OAS spread"),
        "VIX": NodeId(NodeType.INSTRUMENT, "VIX", "CBOE Volatility Index"),
    }
    
    # Define edge specs
    cpi_hy_spec = EdgeSpec(
        slot_id="CPI->HY_OAS",
        from_node=nodes["CPI"],
        to_node=nodes["HY_OAS"],
        active_factor="cpi_shock_abs",
        background_features=["yield_vol_10y", "hy_vol", "slope_10y_2y", "fed_funds", "hy_oas_before", "stlfsi"],
        target_col="hy_change",
        target_unit="bp",
        large_threshold_candidates=[0.05, 0.08, 0.10, 0.12, 0.15],
        fn_constraints=[0.01, 0.05],
    )
    
    unemp_vix_spec = EdgeSpec(
        slot_id="UNEMP->VIX",
        from_node=nodes["UNEMPLOYMENT"],
        to_node=nodes["VIX"],
        active_factor="unemp_surprise_abs",
        background_features=["vix_vol", "yield_vol_10y", "slope_10y_2y", "fed_funds", "vix_before", "stlfsi"],
        target_col="vix_change",
        target_unit="points",
        large_threshold_candidates=[1.0, 1.5, 2.0, 2.5, 3.0],
        fn_constraints=[0.01, 0.05],
    )
    
    specs = [cpi_hy_spec, unemp_vix_spec]
    
    # Create builder
    builder = EdgeBuilder(
        model_output_dir=str(model_dir),
        version_prefix=datetime.now().strftime("%Y-%m-%d"),
        created_by="build_macro_graph.py",
    )
    
    edges_by_slot = {}
    
    # Process each spec
    for spec in specs:
        print(f"\n{'='*70}")
        print(f"PROCESSING: {spec.slot_id}")
        print("=" * 70)
        
        # Load dataset
        if spec.slot_id == "CPI->HY_OAS":
            train_df, test_df, features = load_cpi_hy_dataset()
        elif spec.slot_id == "UNEMP->VIX":
            train_df, test_df, features = load_unemp_vix_dataset()
        else:
            print(f"Unknown spec: {spec.slot_id}")
            continue
        
        if args.skip_search:
            # Use known best configurations from methodology doc
            print("\nUsing known best configurations (skipping grid search)...")
            
            if spec.slot_id == "CPI->HY_OAS":
                # LogReg, w=50, 15bp, @0.80
                edges = [
                    builder.build_from_config(
                        spec=spec,
                        model_type="LogReg",
                        weight=50,
                        large_threshold=0.15,
                        prob_cutoff=0.80,
                        fn_constraint=0.01,
                        train_df=train_df,
                        test_df=test_df,
                        features=features,
                    ),
                    builder.build_from_config(
                        spec=spec,
                        model_type="LogReg",
                        weight=50,
                        large_threshold=0.15,
                        prob_cutoff=0.80,
                        fn_constraint=0.05,
                        train_df=train_df,
                        test_df=test_df,
                        features=features,
                    ),
                ]
            elif spec.slot_id == "UNEMP->VIX":
                # FN≤1%: RF_shallow, w=50, 2.5pt, @0.20
                # FN≤5%: GB, w=20, 2.0pt, @0.05
                edges = [
                    builder.build_from_config(
                        spec=spec,
                        model_type="RF_shallow",
                        weight=50,
                        large_threshold=2.5,
                        prob_cutoff=0.20,
                        fn_constraint=0.01,
                        train_df=train_df,
                        test_df=test_df,
                        features=features,
                    ),
                    builder.build_from_config(
                        spec=spec,
                        model_type="GB",
                        weight=20,
                        large_threshold=2.0,
                        prob_cutoff=0.05,
                        fn_constraint=0.05,
                        train_df=train_df,
                        test_df=test_df,
                        features=features,
                    ),
                ]
            else:
                edges = []
        else:
            # Run grid search
            grid_results = run_grid_search_for_spec(spec, train_df, test_df, features)
            
            if len(grid_results) == 0:
                print(f"No grid search results for {spec.slot_id}")
                continue
            
            # Build edges from grid search
            edges = builder.build_from_grid_search(
                spec=spec,
                grid_results=grid_results,
                train_df=train_df,
                test_df=test_df,
                features=features,
            )
        
        edges_by_slot[spec.slot_id] = edges
        
        # Print edge summaries
        print(f"\nBuilt {len(edges)} edges:")
        for edge in edges:
            print(f"\n{edge.summary()}")
    
    # Build full graph
    print(f"\n{'='*70}")
    print("BUILDING GRAPH")
    print("=" * 70)
    
    graph = build_graph_from_specs(
        specs=specs,
        edges_by_slot=edges_by_slot,
        name="QSIG Macro Graph",
        description="Event → Instrument large move prediction edges",
        version="1.0.0",
    )
    
    print(f"\n{graph.summary()}")
    
    # Save graph
    graph_path = output_dir / "macro_graph.json"
    save_graph(graph, graph_path, exclude_test_predictions=True)
    
    # Save full graph with predictions
    full_graph_path = output_dir / "macro_graph_full.json"
    save_graph(graph, full_graph_path, exclude_test_predictions=False)
    
    # Export summary
    summary_path = output_dir / "macro_graph_summary.md"
    export_graph_summary(graph, summary_path)
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  Graph (compact): {graph_path}")
    print(f"  Graph (full):    {full_graph_path}")
    print(f"  Summary:         {summary_path}")
    print(f"  Models:          {model_dir}/")
    
    return graph


if __name__ == "__main__":
    main()

