"""
QSIG Macro Graph: Edge Runner
==============================

Runtime execution of edges against event contexts.

The EdgeRunner:
1. Loads serialized models from disk
2. Applies them to EventContext objects
3. Produces EdgeResult objects with predictions and backtest metadata

Usage:
    from src.graph import Graph, EdgeRunner, EventContext, NodeId, NodeType
    
    # Load graph
    graph = Graph.load("registry/macro_graph.json")
    
    # Create runner
    runner = EdgeRunner(graph)
    
    # Create event context
    ctx = EventContext(
        node=NodeId(NodeType.EVENT, "CPI"),
        event_date="2024-03-12",
        features={"cpi_shock_abs": 0.3, "yield_vol_10y": 0.05, ...}
    )
    
    # Apply edge
    result = runner.apply("CPI->HY_OAS", ctx)
    print(result.prob_large_move, result.flag_large_move)
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .types import (
    Edge,
    EdgeResult,
    EdgeSlot,
    EventContext,
    Graph,
    NodeId,
    NodeType,
)


class EdgeRunner:
    """
    Executes edges against event contexts to produce predictions.
    
    The runner caches loaded models for efficiency, and handles:
    - Feature extraction from EventContext
    - Missing value imputation
    - Model prediction
    - EdgeResult construction
    
    Example:
        runner = EdgeRunner(graph)
        
        # Use default edge for a slot
        result = runner.apply("CPI->HY_OAS", ctx)
        
        # Use specific edge
        result = runner.apply("CPI->HY_OAS", ctx, edge_id="CPI->HY__FN1pct__LogReg")
        
        # Apply all relevant edges for an event
        results = runner.apply_all(ctx)
    """
    
    def __init__(
        self,
        graph: Graph,
        model_base_path: Optional[Path] = None,
        cache_models: bool = True,
    ):
        """
        Initialize the EdgeRunner.
        
        Args:
            graph: The macro graph containing edge definitions
            model_base_path: Base path for model files. If None, uses paths as-is.
            cache_models: Whether to cache loaded models in memory
        """
        self.graph = graph
        self.model_base_path = Path(model_base_path) if model_base_path else None
        self.cache_models = cache_models
        self._model_cache: Dict[str, Any] = {}
        self._scaler_cache: Dict[str, Any] = {}
    
    def apply(
        self,
        slot_id: str,
        context: EventContext,
        edge_id: Optional[str] = None,
        policy: Optional[str] = None,
    ) -> EdgeResult:
        """
        Apply an edge to an event context.
        
        Args:
            slot_id: The edge slot ID (e.g. "CPI->HY_OAS")
            context: The event context with features
            edge_id: Specific edge ID to use, or None for default
            policy: Policy override ("safe" for lowest FN, "balanced" for best TN/FP)
        
        Returns:
            EdgeResult with prediction and metadata
        """
        # Get the edge slot
        slot = self.graph.get_slot(slot_id)
        
        # Select edge based on policy or explicit ID
        if edge_id is not None:
            edge = slot.get_edge(edge_id)
        elif policy == "safe":
            # Find edge with lowest FN constraint
            edge = self._get_safest_edge(slot)
        elif policy == "balanced":
            # Find edge with best TN/FP ratio among those meeting some FN threshold
            edge = self._get_balanced_edge(slot, max_fn_rate=0.05)
        else:
            edge = slot.get_edge()  # default edge
        
        return self._run_edge(edge, context)
    
    def apply_all(
        self,
        context: EventContext,
        policy: Optional[str] = None,
    ) -> List[EdgeResult]:
        """
        Apply all relevant edges for an event type.
        
        Args:
            context: The event context
            policy: Policy for edge selection
        
        Returns:
            List of EdgeResult objects
        """
        results = []
        
        for slot_id, slot in self.graph.edge_slots.items():
            # Check if this slot matches the event type
            if slot.from_node.name == context.node.name:
                try:
                    result = self.apply(slot_id, context, policy=policy)
                    results.append(result)
                except Exception as e:
                    # Log error but continue with other edges
                    print(f"Warning: Failed to apply {slot_id}: {e}")
        
        return results
    
    def _run_edge(self, edge: Edge, context: EventContext) -> EdgeResult:
        """
        Execute a single edge against a context.
        
        Args:
            edge: The edge configuration
            context: The event context
        
        Returns:
            EdgeResult
        """
        # Load model and scaler
        model, scaler = self._load_model(edge)
        
        # Prepare features
        features = self._prepare_features(edge, context)
        
        # Scale features
        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
        
        # Get prediction
        prob = float(model.predict_proba(features_scaled)[0, 1])
        flag = prob >= edge.prob_cutoff
        
        # Build backtest summary
        backtest = {
            "auc": edge.stats.auc,
            "base_rate": edge.stats.base_rate,
            "tp": edge.stats.tp,
            "fp": edge.stats.fp,
            "fn": edge.stats.fn,
            "tn": edge.stats.tn,
            "fn_rate": edge.stats.fn_rate,
            "fp_rate": edge.stats.fp_rate,
            "tn_fp_ratio": edge.stats.tn_fp_ratio,
        }
        
        # Build features used dict
        features_used = {
            name: context.get_feature(name)
            for name in edge.all_features
        }
        
        return EdgeResult(
            edge_id=edge.edge_id,
            slot_id=edge.slot_id,
            from_node=edge.from_node,
            to_node=edge.to_node,
            event_date=context.event_date,
            prob_large_move=prob,
            flag_large_move=flag,
            fn_constraint=edge.fn_constraint,
            large_move_threshold=edge.large_move_threshold.value,
            target_series=edge.target_series,
            target_unit=edge.target_unit,
            backtest=backtest,
            features_used=features_used,
        )
    
    def _load_model(self, edge: Edge):
        """
        Load model and scaler for an edge.
        
        Caches models if cache_models is True.
        """
        if self.cache_models and edge.edge_id in self._model_cache:
            return self._model_cache[edge.edge_id], self._scaler_cache.get(edge.edge_id)
        
        # Determine model path
        model_path = edge.model_location
        if self.model_base_path:
            model_path = self.model_base_path / model_path
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model file
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle different serialization formats
        if isinstance(data, dict):
            model = data.get('model')
            scaler = data.get('scaler')
        else:
            # Assume it's just the model
            model = data
            scaler = None
        
        # Cache if enabled
        if self.cache_models:
            self._model_cache[edge.edge_id] = model
            self._scaler_cache[edge.edge_id] = scaler
        
        return model, scaler
    
    def _prepare_features(self, edge: Edge, context: EventContext) -> np.ndarray:
        """
        Prepare feature vector from context.
        
        Uses feature_medians from edge for missing value imputation.
        """
        feature_names = edge.all_features
        values = []
        
        for name in feature_names:
            val = context.get_feature(name)
            
            # Handle missing values
            if val is None or (isinstance(val, float) and np.isnan(val)):
                if edge.feature_medians and name in edge.feature_medians:
                    val = edge.feature_medians[name]
                else:
                    val = 0.0  # fallback
            
            values.append(val)
        
        return np.array([values])
    
    def _get_safest_edge(self, slot: EdgeSlot) -> Edge:
        """Get edge with lowest FN constraint."""
        if not slot.edges:
            raise ValueError(f"No edges in slot {slot.slot_id}")
        
        return min(slot.edges.values(), key=lambda e: e.fn_constraint)
    
    def _get_balanced_edge(self, slot: EdgeSlot, max_fn_rate: float = 0.05) -> Edge:
        """Get edge with best TN/FP ratio among those meeting FN constraint."""
        candidates = [
            e for e in slot.edges.values()
            if e.stats.fn_rate <= max_fn_rate
        ]
        
        if not candidates:
            # Fall back to safest edge if none meet the constraint
            return self._get_safest_edge(slot)
        
        # Sort by TN/FP ratio (higher is better)
        return max(candidates, key=lambda e: e.stats.tn_fp_ratio or 0)
    
    def clear_cache(self):
        """Clear the model cache."""
        self._model_cache.clear()
        self._scaler_cache.clear()
    
    def preload_models(self, slot_ids: Optional[List[str]] = None):
        """
        Preload all models into cache.
        
        Args:
            slot_ids: List of slot IDs to preload, or None for all
        """
        if slot_ids is None:
            slot_ids = list(self.graph.edge_slots.keys())
        
        for slot_id in slot_ids:
            slot = self.graph.edge_slots[slot_id]
            for edge in slot.edges.values():
                try:
                    self._load_model(edge)
                except Exception as e:
                    print(f"Warning: Failed to preload {edge.edge_id}: {e}")


class BatchEdgeRunner:
    """
    Batch execution of edges for multiple events.
    
    Optimized for processing many events at once.
    
    Example:
        runner = BatchEdgeRunner(graph)
        
        contexts = [
            EventContext(...) for event in events
        ]
        
        results = runner.apply_batch("CPI->HY_OAS", contexts)
    """
    
    def __init__(
        self,
        graph: Graph,
        model_base_path: Optional[Path] = None,
    ):
        self.graph = graph
        self.runner = EdgeRunner(graph, model_base_path, cache_models=True)
    
    def apply_batch(
        self,
        slot_id: str,
        contexts: List[EventContext],
        edge_id: Optional[str] = None,
    ) -> List[EdgeResult]:
        """
        Apply an edge to multiple contexts.
        
        Args:
            slot_id: The edge slot ID
            contexts: List of event contexts
            edge_id: Specific edge ID to use
        
        Returns:
            List of EdgeResult objects
        """
        results = []
        
        for ctx in contexts:
            result = self.runner.apply(slot_id, ctx, edge_id=edge_id)
            results.append(result)
        
        return results
    
    def apply_all_batch(
        self,
        contexts: List[EventContext],
        policy: Optional[str] = None,
    ) -> Dict[str, List[EdgeResult]]:
        """
        Apply all relevant edges for multiple events.
        
        Returns:
            Dict mapping event_date to list of EdgeResults
        """
        results_by_date: Dict[str, List[EdgeResult]] = {}
        
        for ctx in contexts:
            results = self.runner.apply_all(ctx, policy=policy)
            results_by_date[ctx.event_date] = results
        
        return results_by_date

