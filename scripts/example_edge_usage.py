#!/usr/bin/env python3
"""
Example: Using the QSIG Macro Graph Edge System
================================================

This script demonstrates:
1. Loading a graph from the registry
2. Creating EventContext objects with proper feature names
3. Applying edges with different policies (default, safe, balanced)
4. Forcing specific edge variants
5. Inspecting EdgeResult outputs

Run this after building the graph:
    python scripts/build_macro_graph.py
    python scripts/example_edge_usage.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph import (
    Graph,
    EdgeRunner,
    EventContext,
    NodeId,
    NodeType,
    load_graph,
)


def print_edge_result(result, label=""):
    """Print a compact summary of an EdgeResult."""
    flag_str = "🔴 LARGE MOVE EXPECTED" if result.flag_large_move else "🟢 Normal move expected"
    
    print(f"\n{label}" if label else "")
    print(f"  Edge: {result.edge_id}")
    print(f"  P(large move): {result.prob_large_move:.1%}")
    print(f"  {flag_str}")
    print(f"  Threshold: ≥{result.large_move_threshold} {result.target_unit}")
    print(f"  FN constraint: ≤{result.fn_constraint*100:.0f}%")
    print(f"  Backtest: AUC={result.backtest['auc']:.3f}, "
          f"FN%={result.backtest['fn_rate']*100:.1f}%, "
          f"TN/FP={result.backtest['tn_fp_ratio']:.2f}x" if result.backtest['tn_fp_ratio'] else "∞")


def main():
    print("=" * 70)
    print("QSIG MACRO GRAPH: EDGE USAGE EXAMPLE")
    print("=" * 70)
    
    # Load the graph
    graph_path = project_root / "registry" / "macro_graph.json"
    
    if not graph_path.exists():
        print(f"\nGraph not found at {graph_path}")
        print("Run 'python scripts/build_macro_graph.py' first.")
        return
    
    print(f"\nLoading graph from {graph_path}...")
    graph = load_graph(graph_path)
    
    print(f"\n{graph.summary()}")
    
    # Create runner with model path relative to project root
    runner = EdgeRunner(graph, model_base_path=project_root)
    
    # ========================================================================
    # Example 1: CPI Event - High shock scenario
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 1: CPI Event - High Shock")
    print("=" * 70)
    
    # Create event context using the feature names expected by CPI->HY edges
    # These match the active_factor + background_features in the EdgeSpec
    cpi_high_shock = EventContext(
        node=NodeId(NodeType.EVENT, "CPI"),
        event_date="2024-03-12",
        features={
            # Active factor
            "cpi_shock_abs": 0.4,       # High CPI shock (40 cents MoM)
            # Background features
            "yield_vol_10y": 0.08,      # Elevated 10Y yield volatility
            "hy_vol": 0.05,             # HY spread volatility
            "slope_10y_2y": 0.5,        # Flat curve
            "fed_funds": 5.25,          # High rates environment
            "hy_oas_before": 4.5,       # Elevated HY spreads
            "stlfsi": 0.3,              # Some financial stress
        },
        meta={
            "release_time_utc": "13:30:00",
            "source_calendar": "BLS",
            "scenario": "High CPI shock",
        }
    )
    
    print(f"\nEvent: CPI on {cpi_high_shock.event_date}")
    print(f"Features: cpi_shock_abs={cpi_high_shock.features['cpi_shock_abs']}, "
          f"yield_vol={cpi_high_shock.features['yield_vol_10y']:.2f}, "
          f"hy_before={cpi_high_shock.features['hy_oas_before']}")
    
    # Apply with default edge
    if "CPI->HY_OAS" in graph.edge_slots:
        result_default = runner.apply("CPI->HY_OAS", cpi_high_shock)
        print_edge_result(result_default, "Default Policy:")
        
        # Apply with "safe" policy (lowest FN constraint)
        result_safe = runner.apply("CPI->HY_OAS", cpi_high_shock, policy="safe")
        print_edge_result(result_safe, "Safe Policy (lowest FN constraint):")
        
        # Force specific edge if available
        slot = graph.get_slot("CPI->HY_OAS")
        for edge_id in slot.edges:
            if "FN1pct" in edge_id:
                result_specific = runner.apply("CPI->HY_OAS", cpi_high_shock, edge_id=edge_id)
                print_edge_result(result_specific, f"Forced {edge_id}:")
                break
    else:
        print("CPI->HY_OAS slot not found in graph")
    
    # ========================================================================
    # Example 2: CPI Event - Low shock scenario
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 2: CPI Event - Low Shock")
    print("=" * 70)
    
    cpi_low_shock = EventContext(
        node=NodeId(NodeType.EVENT, "CPI"),
        event_date="2024-04-10",
        features={
            "cpi_shock_abs": 0.1,       # Low CPI shock
            "yield_vol_10y": 0.03,      # Normal volatility
            "hy_vol": 0.02,
            "slope_10y_2y": 1.2,        # Normal curve
            "fed_funds": 5.25,
            "hy_oas_before": 3.5,       # Normal spreads
            "stlfsi": -0.5,             # Low stress
        },
        meta={"scenario": "Low CPI shock"},
    )
    
    print(f"\nEvent: CPI on {cpi_low_shock.event_date}")
    print(f"Features: cpi_shock_abs={cpi_low_shock.features['cpi_shock_abs']}, "
          f"yield_vol={cpi_low_shock.features['yield_vol_10y']:.2f}")
    
    if "CPI->HY_OAS" in graph.edge_slots:
        result = runner.apply("CPI->HY_OAS", cpi_low_shock)
        print_edge_result(result, "Result:")
    
    # ========================================================================
    # Example 3: Unemployment Event
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Unemployment Event")
    print("=" * 70)
    
    # Create event context using the feature names expected by UNEMP->VIX edges
    unemp_event = EventContext(
        node=NodeId(NodeType.EVENT, "UNEMPLOYMENT"),
        event_date="2024-03-08",
        features={
            # Active factor
            "unemp_surprise_abs": 0.3,  # Surprising jobs number
            # Background features
            "vix_vol": 1.5,             # VIX volatility
            "yield_vol_10y": 0.05,
            "slope_10y_2y": 0.8,
            "fed_funds": 5.25,
            "vix_before": 14.0,         # Low VIX starting point
            "stlfsi": 0.0,
        },
        meta={"scenario": "Jobs surprise"},
    )
    
    print(f"\nEvent: Unemployment on {unemp_event.event_date}")
    print(f"Features: unemp_surprise_abs={unemp_event.features['unemp_surprise_abs']}, "
          f"vix_before={unemp_event.features['vix_before']}")
    
    if "UNEMP->VIX" in graph.edge_slots:
        result_default = runner.apply("UNEMP->VIX", unemp_event)
        print_edge_result(result_default, "Default Policy:")
        
        result_balanced = runner.apply("UNEMP->VIX", unemp_event, policy="balanced")
        print_edge_result(result_balanced, "Balanced Policy:")
    else:
        print("UNEMP->VIX slot not found in graph")
    
    # ========================================================================
    # Example 4: Apply All Edges for an Event
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Apply All Edges for CPI Event")
    print("=" * 70)
    
    all_results = runner.apply_all(cpi_high_shock)
    
    print(f"\nFound {len(all_results)} applicable edge slot(s):")
    for result in all_results:
        flag_str = "🔴" if result.flag_large_move else "🟢"
        print(f"  {flag_str} {result.slot_id}: P={result.prob_large_move:.1%}, "
              f"threshold={result.large_move_threshold}{result.target_unit}")
    
    # ========================================================================
    # Example 5: Inspect Graph Structure
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Graph Structure Inspection")
    print("=" * 70)
    
    print(f"\nNodes ({len(graph.nodes)}):")
    for name, node in graph.nodes.items():
        print(f"  {node.type.value}: {name} - {node.description}")
    
    print(f"\nEdge Slots ({len(graph.edge_slots)}):")
    for slot_id, slot in graph.edge_slots.items():
        print(f"\n  {slot_id}:")
        print(f"    Description: {slot.description}")
        print(f"    Features: {slot.feature_spec.get('active_factor', '')} + "
              f"{slot.feature_spec.get('background_features', [])}")
        print(f"    Default edge: {slot.default_edge_id}")
        print(f"    Edges:")
        
        for edge_id, edge in slot.edges.items():
            is_default = " (default)" if edge_id == slot.default_edge_id else ""
            tn_fp = f"{edge.stats.tn_fp_ratio:.2f}x" if edge.stats.tn_fp_ratio else "∞"
            print(f"      • {edge.model_type} FN≤{edge.fn_constraint*100:.0f}%{is_default}")
            print(f"        Threshold: {edge.large_move_threshold.value} {edge.target_unit} @ p≥{edge.prob_cutoff:.2f}")
            print(f"        Test: AUC={edge.stats.auc:.3f}, FN%={edge.stats.fn_rate*100:.1f}%, TN/FP={tn_fp}")
            print(f"        Confusion: TP={edge.stats.tp}, FP={edge.stats.fp}, FN={edge.stats.fn}, TN={edge.stats.tn}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
