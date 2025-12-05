"""
QSIG Macro Graph: Registry (Serialization)
============================================

Load and save graphs and edges to/from JSON files.

The registry provides:
- JSON serialization of Graph, EdgeSlot, and Edge objects
- Version-controlled storage of edge configurations
- Human-readable format for inspection and debugging

Usage:
    from src.graph import Graph, save_graph, load_graph
    
    # Save graph
    save_graph(graph, "registry/macro_graph.json")
    
    # Load graph
    graph = load_graph("registry/macro_graph.json")
    
    # Save individual edge
    save_edge(edge, "registry/edges/cpi_hy_fn1pct.json")
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .types import Edge, EdgeSlot, Graph, NodeId


def save_graph(
    graph: Graph,
    path: Union[str, Path],
    indent: int = 2,
    exclude_test_predictions: bool = False,
) -> None:
    """
    Save a Graph to a JSON file.
    
    Args:
        graph: The Graph to save
        path: Output file path
        indent: JSON indentation level
        exclude_test_predictions: If True, omit test_predictions from edges
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = graph.to_dict()
    
    # Optionally strip test predictions (can be large)
    if exclude_test_predictions:
        for slot_id, slot in data.get("edge_slots", {}).items():
            for edge_id, edge in slot.get("edges", {}).items():
                if "stats" in edge and "test_predictions" in edge["stats"]:
                    edge["stats"]["test_predictions"] = None
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    
    print(f"Graph saved to {path}")


def load_graph(path: Union[str, Path]) -> Graph:
    """
    Load a Graph from a JSON file.
    
    Args:
        path: Input file path
    
    Returns:
        Graph object
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return Graph.from_dict(data)


def save_edge(
    edge: Edge,
    path: Union[str, Path],
    indent: int = 2,
) -> None:
    """
    Save a single Edge to a JSON file.
    
    Args:
        edge: The Edge to save
        path: Output file path
        indent: JSON indentation level
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(edge.to_dict(), f, indent=indent, default=str)
    
    print(f"Edge saved to {path}")


def load_edge(path: Union[str, Path]) -> Edge:
    """
    Load a single Edge from a JSON file.
    
    Args:
        path: Input file path
    
    Returns:
        Edge object
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Edge file not found: {path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return Edge.from_dict(data)


def save_edge_slot(
    slot: EdgeSlot,
    path: Union[str, Path],
    indent: int = 2,
) -> None:
    """
    Save an EdgeSlot to a JSON file.
    
    Args:
        slot: The EdgeSlot to save
        path: Output file path
        indent: JSON indentation level
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(slot.to_dict(), f, indent=indent, default=str)
    
    print(f"EdgeSlot saved to {path}")


def load_edge_slot(path: Union[str, Path]) -> EdgeSlot:
    """
    Load an EdgeSlot from a JSON file.
    
    Args:
        path: Input file path
    
    Returns:
        EdgeSlot object
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"EdgeSlot file not found: {path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return EdgeSlot.from_dict(data)


class GraphRegistry:
    """
    Registry for managing multiple graphs and their versions.
    
    Provides:
    - Version tracking
    - Listing available graphs
    - Loading specific versions
    
    Example:
        registry = GraphRegistry("registry/")
        
        # Save with version
        registry.save(graph, "macro_graph", version="1.0.0")
        
        # List versions
        versions = registry.list_versions("macro_graph")
        
        # Load specific version
        graph = registry.load("macro_graph", version="1.0.0")
        
        # Load latest
        graph = registry.load("macro_graph")
    """
    
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize the registry.
        
        Args:
            base_dir: Base directory for graph storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        graph: Graph,
        name: str,
        version: Optional[str] = None,
        exclude_test_predictions: bool = False,
    ) -> Path:
        """
        Save a graph with version tracking.
        
        Args:
            graph: Graph to save
            name: Graph name
            version: Version string (defaults to graph.version)
            exclude_test_predictions: Omit test predictions
        
        Returns:
            Path to saved file
        """
        version = version or graph.version
        
        # Create versioned filename
        filename = f"{name}_v{version}.json"
        path = self.base_dir / name / filename
        
        save_graph(graph, path, exclude_test_predictions=exclude_test_predictions)
        
        # Also save as "latest"
        latest_path = self.base_dir / name / f"{name}_latest.json"
        save_graph(graph, latest_path, exclude_test_predictions=exclude_test_predictions)
        
        return path
    
    def load(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Graph:
        """
        Load a graph by name and version.
        
        Args:
            name: Graph name
            version: Version string, or None for latest
        
        Returns:
            Graph object
        """
        if version:
            filename = f"{name}_v{version}.json"
        else:
            filename = f"{name}_latest.json"
        
        path = self.base_dir / name / filename
        return load_graph(path)
    
    def list_versions(self, name: str) -> List[str]:
        """
        List available versions of a graph.
        
        Args:
            name: Graph name
        
        Returns:
            List of version strings
        """
        graph_dir = self.base_dir / name
        
        if not graph_dir.exists():
            return []
        
        versions = []
        for f in graph_dir.glob(f"{name}_v*.json"):
            # Extract version from filename
            version = f.stem.replace(f"{name}_v", "")
            versions.append(version)
        
        return sorted(versions)
    
    def list_graphs(self) -> List[str]:
        """
        List all available graph names.
        
        Returns:
            List of graph names
        """
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
    
    def delete(self, name: str, version: Optional[str] = None) -> None:
        """
        Delete a graph version or all versions.
        
        Args:
            name: Graph name
            version: Version to delete, or None for all
        """
        graph_dir = self.base_dir / name
        
        if not graph_dir.exists():
            return
        
        if version:
            path = graph_dir / f"{name}_v{version}.json"
            if path.exists():
                path.unlink()
        else:
            # Delete all versions
            import shutil
            shutil.rmtree(graph_dir)


def export_graph_summary(
    graph: Graph,
    path: Union[str, Path],
) -> None:
    """
    Export a human-readable summary of a graph.
    
    Args:
        graph: Graph to summarize
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        f"# {graph.name}",
        f"",
        f"**Version**: {graph.version}",
        f"**Created**: {graph.created_at}",
        f"",
        f"## Nodes",
        f"",
    ]
    
    # Group nodes by type
    events = [n for n in graph.nodes.values() if n.type.value == "event"]
    instruments = [n for n in graph.nodes.values() if n.type.value == "instrument"]
    
    lines.append("### Events")
    for n in events:
        lines.append(f"- **{n.name}**: {n.description}")
    
    lines.append("")
    lines.append("### Instruments")
    for n in instruments:
        lines.append(f"- **{n.name}**: {n.description}")
    
    lines.append("")
    lines.append("## Edge Slots")
    lines.append("")
    
    for slot_id, slot in graph.edge_slots.items():
        lines.append(f"### {slot_id}")
        lines.append(f"")
        lines.append(f"{slot.description}")
        lines.append(f"")
        lines.append(f"**Features**: {slot.feature_spec.get('active_factor', '')} + {slot.feature_spec.get('background_features', [])}")
        lines.append(f"")
        lines.append(f"**Edges**:")
        lines.append(f"")
        lines.append("| Edge ID | Model | FN Constraint | Prob Cutoff | AUC | TN/FP |")
        lines.append("|---------|-------|---------------|-------------|-----|-------|")
        
        for edge_id, edge in slot.edges.items():
            tn_fp = f"{edge.stats.tn_fp_ratio:.2f}x" if edge.stats.tn_fp_ratio else "∞"
            default_marker = " ⭐" if edge_id == slot.default_edge_id else ""
            lines.append(
                f"| {edge_id}{default_marker} | {edge.model_type} | "
                f"≤{edge.fn_constraint*100:.0f}% | {edge.prob_cutoff:.2f} | "
                f"{edge.stats.auc:.3f} | {tn_fp} |"
            )
        
        lines.append("")
    
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Summary exported to {path}")

