#!/usr/bin/env python3
"""
Example: Using the QSIG Macro Graph Edge System
================================================

This script demonstrates:
1. Loading a graph from the registry
2. Creating EventContext objects
3. Applying edges to get predictions
4. Using different policies (safe vs balanced)

Run this after building the graph:
    python scripts/build_macro_graph.py --skip-search
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


def main():
    print("=" * 70)
    print("QSIG MACRO GRAPH: EDGE USAGE EXAMPLE")
    print("=" * 70)
    
    # Load the graph
    graph_path = project_root / "registry" / "macro_graph.json"
    
    if not graph_path.exists():
        print(f"\nGraph not found at {graph_path}")
        print("Run 'python scripts/build_macro_graph.py --skip-search' first.")
        return
    
    print(f"\nLoading graph from {graph_path}...")
    graph = load_graph(graph_path)
    
    print(f"\n{graph.summary()}")
    
    # Create runner
    runner = EdgeRunner(graph, model_base_path=project_root)
    
    # Example 1: CPI event with high shock
    print("\n" + "=" * 70)
    print("EXAMPLE 1: CPI Event with High Shock")
    print("=" * 70)
    
    cpi_context = EventContext(
        node=NodeId(NodeType.EVENT, "CPI"),
        event_date="2024-03-12",
        features={
            "cpi_shock_abs": 0.4,       # High CPI shock
            "yield_vol_10y": 0.08,      # Elevated volatility
            "hy_vol": 0.05,
            "slope_10y_2y": 0.5,        # Flat curve
            "fed_funds": 5.25,          # High rates
            "hy_oas_before": 4.5,       # Elevated spreads
            "stlfsi": 0.3,              # Some stress
        },
        meta={
            "release_time_utc": "13:30:00",
            "source_calendar": "BLS",
        }
    )
    
    # Apply with default edge
    if "CPI->HY_OAS" in graph.edge_slots:
        result = runner.apply("CPI->HY_OAS", cpi_context)
        print(f"\nDefault Edge Result:")
        print(result.summary())
        
        # Apply with "safe" policy (lowest FN constraint)
        result_safe = runner.apply("CPI->HY_OAS", cpi_context, policy="safe")
        print(f"\nSafe Policy (FN≤1%) Result:")
        print(result_safe.summary())
    else:
        print("CPI->HY_OAS slot not found in graph")
    
    # Example 2: CPI event with low shock
    print("\n" + "=" * 70)
    print("EXAMPLE 2: CPI Event with Low Shock")
    print("=" * 70)
    
    cpi_context_low = EventContext(
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
    )
    
    if "CPI->HY_OAS" in graph.edge_slots:
        result = runner.apply("CPI->HY_OAS", cpi_context_low)
        print(f"\nResult:")
        print(result.summary())
    
    # Example 3: Unemployment event
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Unemployment Event with Surprise")
    print("=" * 70)
    
    unemp_context = EventContext(
        node=NodeId(NodeType.EVENT, "UNEMPLOYMENT"),
        event_date="2024-03-08",
        features={
            "unemp_surprise_abs": 0.3,  # Surprising jobs number
            "vix_vol": 1.5,             # VIX volatility
            "yield_vol_10y": 0.05,
            "slope_10y_2y": 0.8,
            "fed_funds": 5.25,
            "vix_before": 14.0,         # Low VIX starting point
            "stlfsi": 0.0,
        },
    )
    
    if "UNEMP->VIX" in graph.edge_slots:
        result = runner.apply("UNEMP->VIX", unemp_context)
        print(f"\nDefault Edge Result:")
        print(result.summary())
        
        # Try balanced policy
        result_balanced = runner.apply("UNEMP->VIX", unemp_context, policy="balanced")
        print(f"\nBalanced Policy (FN≤5%) Result:")
        print(result_balanced.summary())
    else:
        print("UNEMP->VIX slot not found in graph")
    
    # Example 4: Apply all edges for an event
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Apply All Edges for CPI Event")
    print("=" * 70)
    
    all_results = runner.apply_all(cpi_context)
    
    print(f"\nFound {len(all_results)} applicable edges:")
    for result in all_results:
        flag_str = "🔴" if result.flag_large_move else "🟢"
        print(f"  {flag_str} {result.slot_id}: p={result.prob_large_move:.1%}, "
              f"threshold={result.large_move_threshold}{result.target_unit}")
    
    # Example 5: Inspecting edge details
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Edge Inspection")
    print("=" * 70)
    
    if "CPI->HY_OAS" in graph.edge_slots:
        slot = graph.get_slot("CPI->HY_OAS")
        
        print(f"\nSlot: {slot.slot_id}")
        print(f"Description: {slot.description}")
        print(f"Default edge: {slot.default_edge_id}")
        print(f"\nAvailable edges:")
        
        for edge_id, edge in slot.edges.items():
            print(f"\n  {edge_id}:")
            print(f"    Model: {edge.model_type}")
            print(f"    FN constraint: ≤{edge.fn_constraint*100:.0f}%")
            print(f"    Prob cutoff: {edge.prob_cutoff:.2f}")
            print(f"    Large move threshold: {edge.large_move_threshold.value} {edge.target_unit}")
            print(f"    Backtest AUC: {edge.stats.auc:.3f}")
            print(f"    Backtest TN/FP: {edge.stats.tn_fp_ratio:.2f}x" if edge.stats.tn_fp_ratio else "    Backtest TN/FP: ∞")
            print(f"    Test period: {edge.stats.test_period[0]} to {edge.stats.test_period[1]}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

