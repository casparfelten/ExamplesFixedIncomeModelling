"""
QSIG Macro Graph: Edge-Based Event → Instrument Prediction System
==================================================================

This module implements a graph of macro-event → market-move edges, where:
- Each **edge** is a specific, trained model configuration
- Multiple edges can exist for the same mapping (e.g. CPI→HY with FN≤1% and FN≤5%)
- Edges are **swappable** at runtime based on policy (safety vs noise)
- Each edge carries backtest metadata for confidence intervals and Bayesian priors

Core Concepts:
- **Node**: event (CPI, UNEMPLOYMENT) or instrument (HY_OAS, VIX)
- **EdgeSlot**: logical mapping (e.g. "CPI→HY_OAS")
- **Edge**: specific model configuration implementing an EdgeSlot
- **EventContext**: features for a single macro event instance
- **EdgeResult**: output from applying an Edge to an EventContext

Example Usage:
    from src.graph import Graph, EdgeRunner, EventContext
    
    # Load graph
    graph = Graph.load("registry/macro_graph.json")
    
    # Create event context
    ctx = EventContext(
        node=graph.nodes["CPI"],
        event_date="2024-03-12",
        features={
            "cpi_shock_abs": 0.3,
            "yield_vol_10y": 0.05,
            "slope_10y_2y": 1.2,
            ...
        }
    )
    
    # Run edge
    runner = EdgeRunner(graph)
    result = runner.apply("CPI->HY_OAS", ctx)
    print(result.prob_large_move, result.flag_large_move)

See docs/ for design specification and methodology.
"""

from .types import (
    NodeType,
    NodeId,
    LargeMoveThreshold,
    CVFoldMetric,
    TestPrediction,
    EdgeStats,
    Edge,
    EdgeSlot,
    Graph,
    EventContext,
    EdgeResult,
    EdgeSpec,
)

from .registry import (
    save_graph,
    load_graph,
    save_edge,
    load_edge,
    export_graph_summary,
)


def __getattr__(name):
    """Lazy loading for EdgeRunner and EdgeBuilder to defer sklearn import."""
    if name == "EdgeRunner":
        from .edge_runner import EdgeRunner
        return EdgeRunner
    elif name == "EdgeBuilder":
        from .edge_builder import EdgeBuilder
        return EdgeBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Types
    "NodeType",
    "NodeId",
    "LargeMoveThreshold",
    "CVFoldMetric",
    "TestPrediction",
    "EdgeStats",
    "Edge",
    "EdgeSlot",
    "Graph",
    "EventContext",
    "EdgeResult",
    "EdgeSpec",
    # Runner (lazy loaded)
    "EdgeRunner",
    # Builder (lazy loaded)
    "EdgeBuilder",
    # Registry
    "save_graph",
    "load_graph",
    "save_edge",
    "load_edge",
    "export_graph_summary",
]

