#!/usr/bin/env python3
"""
Build QSIG Macro Graph from Configuration
==========================================

This script assembles the graph from:
1. Edge configuration (registry/edge_config.yaml)
2. Model registry (registry/models/)

It does NOT run grid search or train models.
Models must be promoted to the registry separately.

Usage:
    python scripts/build_graph.py
    
    # Specify config file
    python scripts/build_graph.py --config registry/edge_config.yaml

Workflow:
    1. Run grid search:     python scripts/run_grid_search.py
    2. Review candidates:   python scripts/review_candidates.py
    3. Promote models:      python scripts/promote_model.py <model_id>
    4. Build graph:         python scripts/build_graph.py
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml

from src.graph import (
    Graph,
    Edge,
    EdgeSlot,
    EdgeStats,
    NodeId,
    NodeType,
    LargeMoveThreshold,
    save_graph,
    export_graph_summary,
)
from src.models.model_registry import ModelRegistry


def load_config(config_path: Path) -> dict:
    """Load edge configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_graph_from_config(
    config: dict,
    registry: ModelRegistry,
    registry_dir: Path,
) -> Graph:
    """
    Build a Graph from configuration and model registry.
    
    Args:
        config: Parsed YAML config
        registry: Model registry
        registry_dir: Path to registry directory
    
    Returns:
        Graph object
    """
    graph_config = config.get('graph', {})
    
    graph = Graph(
        name=graph_config.get('name', 'QSIG Macro Graph'),
        description=graph_config.get('description', ''),
        version=graph_config.get('version', '1.0.0'),
    )
    
    # Add nodes
    nodes_config = config.get('nodes', {})
    
    for event in nodes_config.get('events', []):
        node = NodeId(
            type=NodeType.EVENT,
            name=event['name'],
            description=event.get('description', ''),
        )
        graph.add_node(node)
    
    for instrument in nodes_config.get('instruments', []):
        node = NodeId(
            type=NodeType.INSTRUMENT,
            name=instrument['name'],
            description=instrument.get('description', ''),
        )
        graph.add_node(node)
    
    # Build edge slots
    slots_config = config.get('edge_slots', {})
    
    for slot_id, slot_config in slots_config.items():
        from_node = graph.nodes.get(slot_config['from_node'])
        to_node = graph.nodes.get(slot_config['to_node'])
        
        if from_node is None:
            print(f"Warning: from_node '{slot_config['from_node']}' not found for slot {slot_id}")
            continue
        if to_node is None:
            print(f"Warning: to_node '{slot_config['to_node']}' not found for slot {slot_id}")
            continue
        
        # Create edge slot
        slot = EdgeSlot(
            slot_id=slot_id,
            from_node=from_node,
            to_node=to_node,
            description=slot_config.get('description', ''),
            feature_spec=slot_config.get('feature_spec', {}),
        )
        
        # Add edges from model registry
        edges_config = slot_config.get('edges', {})
        target_unit = slot_config.get('target_unit', 'bp')
        
        for policy_label, model_id in edges_config.items():
            if not registry.has_model(model_id):
                print(f"Warning: Model '{model_id}' not found in registry for {slot_id}/{policy_label}")
                continue
            
            # Load model metadata
            meta = registry.get_metadata(model_id)
            if meta is None:
                print(f"Warning: Could not load metadata for {model_id}")
                continue
            
            # Determine FN constraint from policy label
            fn_constraint = 0.05  # default
            if '1pct' in policy_label or '1%' in policy_label:
                fn_constraint = 0.01
            elif '5pct' in policy_label or '5%' in policy_label:
                fn_constraint = 0.05
            
            # Build edge ID
            edge_id = f"{slot_id}__{policy_label}__{model_id}"
            
            # Model location relative to registry
            model_path = registry.models_dir / model_id / "model.pkl"
            
            # Build EdgeStats from metadata
            stats = EdgeStats(
                train_period=meta.train_period,
                test_period=meta.test_period,
                n_test_events=meta.n_test,
                n_test_pos=meta.n_test_pos,
                n_test_neg=meta.n_test_neg,
                tp=meta.tp,
                fp=meta.fp,
                fn=meta.fn,
                tn=meta.tn,
                auc=meta.auc,
                base_rate=meta.n_test_pos / meta.n_test if meta.n_test > 0 else 0,
                fn_rate=meta.fn_rate,
                fp_rate=meta.fp_rate,
                tn_fp_ratio=meta.tn_fp_ratio,
            )
            
            # Build Edge
            edge = Edge(
                edge_id=edge_id,
                slot_id=slot_id,
                from_node=from_node,
                to_node=to_node,
                model_type=meta.model_type,
                model_location=str(model_path),
                active_factor=slot_config.get('feature_spec', {}).get('active_factor', ''),
                background_features=slot_config.get('feature_spec', {}).get('background_features', []),
                target_series=to_node.name,
                target_unit=target_unit,
                large_move_threshold=LargeMoveThreshold(
                    value=meta.large_move_threshold,
                    definition=meta.large_move_definition,
                ),
                fn_constraint=fn_constraint,
                prob_cutoff=meta.prob_cutoff,
                threshold_selection_method=f"manual promotion from {meta.source}",
                stats=stats,
                version=f"{datetime.now().strftime('%Y-%m-%d')}_{model_id}",
                created_at=datetime.utcnow().isoformat() + "Z",
                created_by="build_graph.py",
                notes=meta.notes,
                model_params={"weight": meta.weight},
                feature_medians=meta.feature_medians,
            )
            
            slot.add_edge(edge)
            print(f"  Added edge: {policy_label} -> {model_id}")
        
        # Set default edge
        default_policy = slot_config.get('default_edge')
        if default_policy and default_policy in edges_config:
            model_id = edges_config[default_policy]
            edge_id = f"{slot_id}__{default_policy}__{model_id}"
            if edge_id in slot.edges:
                slot.default_edge_id = edge_id
        
        if slot.edges:
            graph.add_edge_slot(slot)
            print(f"Added slot: {slot_id} with {len(slot.edges)} edge(s)")
        else:
            print(f"Skipping slot: {slot_id} (no valid edges)")
    
    return graph


def main():
    parser = argparse.ArgumentParser(description="Build QSIG Macro Graph from config")
    parser.add_argument(
        "--config",
        type=str,
        default="registry/edge_config.yaml",
        help="Path to edge configuration file"
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="registry/models",
        help="Path to model registry directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="registry",
        help="Output directory"
    )
    args = parser.parse_args()
    
    config_path = project_root / args.config
    registry_dir = project_root / args.registry
    output_dir = project_root / args.output
    
    print("=" * 70)
    print("QSIG MACRO GRAPH BUILDER")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Registry: {registry_dir}")
    print(f"Output: {output_dir}")
    
    # Check config exists
    if not config_path.exists():
        print(f"\nError: Config file not found: {config_path}")
        print("Create the config file or run grid search first.")
        return 1
    
    # Load config
    print("\nLoading configuration...")
    config = load_config(config_path)
    
    # Load registry
    print("Loading model registry...")
    registry = ModelRegistry(registry_dir)
    available_models = registry.list_models()
    print(f"  {len(available_models)} models in registry")
    
    if not available_models:
        print("\nWarning: No models in registry.")
        print("Run grid search and promote models first:")
        print("  python scripts/run_grid_search.py")
        print("  python scripts/promote_model.py <model_id>")
    
    # Build graph
    print("\nBuilding graph...")
    graph = build_graph_from_config(config, registry, registry_dir)
    
    print(f"\n{graph.summary()}")
    
    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    graph_path = output_dir / "macro_graph.json"
    save_graph(graph, graph_path, exclude_test_predictions=True)
    
    full_graph_path = output_dir / "macro_graph_full.json"
    save_graph(graph, full_graph_path, exclude_test_predictions=False)
    
    summary_path = output_dir / "macro_graph_summary.md"
    export_graph_summary(graph, summary_path)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  Graph (compact): {graph_path}")
    print(f"  Graph (full):    {full_graph_path}")
    print(f"  Summary:         {summary_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

