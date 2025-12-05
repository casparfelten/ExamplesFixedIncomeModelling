"""
QSIG Macro Graph: Type Definitions
===================================

Data structures for the edge-based event → instrument prediction system.

All structures are designed to be:
- JSON-serializable for storage and transport
- Self-contained for independent edge operation
- Rich enough to support CI/Bayesian extensions later

Reference: docs/edge_design_specification.md
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import json


# ============================================================================
# ENUMS AND BASIC TYPES
# ============================================================================

class NodeType(str, Enum):
    """Type of node in the macro graph."""
    EVENT = "event"
    INSTRUMENT = "instrument"


# ============================================================================
# NODE DEFINITIONS
# ============================================================================

@dataclass
class NodeId:
    """
    Represents a node in the macro graph.
    
    Two types:
    - event: CPI, UNEMPLOYMENT, FOMC, GDP, etc.
    - instrument: HY_OAS, VIX, YIELD_2Y, YIELD_10Y, etc.
    
    Example:
        cpi_node = NodeId(type=NodeType.EVENT, name="CPI", description="Consumer Price Index release")
        hy_node = NodeId(type=NodeType.INSTRUMENT, name="HY_OAS", description="High-Yield OAS spread")
    """
    type: NodeType
    name: str
    description: str = ""
    
    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = NodeType(self.type)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "name": self.name,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NodeId":
        return cls(
            type=NodeType(d["type"]),
            name=d["name"],
            description=d.get("description", ""),
        )
    
    def __hash__(self):
        return hash((self.type, self.name))
    
    def __eq__(self, other):
        if not isinstance(other, NodeId):
            return False
        return self.type == other.type and self.name == other.name


# ============================================================================
# LARGE MOVE THRESHOLD
# ============================================================================

@dataclass
class LargeMoveThreshold:
    """
    Defines what constitutes a "large" move for a target instrument.
    
    Example:
        threshold = LargeMoveThreshold(
            value=0.15,  # 15bp
            definition="85th percentile of |ΔHY_OAS| in train 1997–2017"
        )
    """
    value: float
    definition: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "definition": self.definition}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LargeMoveThreshold":
        return cls(value=d["value"], definition=d.get("definition", ""))


# ============================================================================
# CROSS-VALIDATION METRICS
# ============================================================================

@dataclass
class CVFoldMetric:
    """
    Metrics from a single CV fold.
    
    Stored for future CI computation via bootstrap.
    """
    fold_id: int
    train_period: Tuple[str, str]  # (start_date, end_date)
    val_period: Tuple[str, str]
    tp: int
    fp: int
    fn: int
    tn: int
    auc: float
    prob_cutoff: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "train_period": list(self.train_period),
            "val_period": list(self.val_period),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "auc": self.auc,
            "prob_cutoff": self.prob_cutoff,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CVFoldMetric":
        return cls(
            fold_id=d["fold_id"],
            train_period=tuple(d["train_period"]),
            val_period=tuple(d["val_period"]),
            tp=d["tp"],
            fp=d["fp"],
            fn=d["fn"],
            tn=d["tn"],
            auc=d["auc"],
            prob_cutoff=d["prob_cutoff"],
        )


@dataclass
class TestPrediction:
    """
    Single test prediction for bootstrap analysis.
    """
    event_date: str  # YYYY-MM-DD
    y_true: int  # 0 or 1
    y_pred_prob: float
    y_pred_flag: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_date": self.event_date,
            "y_true": self.y_true,
            "y_pred_prob": self.y_pred_prob,
            "y_pred_flag": self.y_pred_flag,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TestPrediction":
        return cls(
            event_date=d["event_date"],
            y_true=d["y_true"],
            y_pred_prob=d["y_pred_prob"],
            y_pred_flag=d["y_pred_flag"],
        )


# ============================================================================
# EDGE STATISTICS
# ============================================================================

@dataclass
class EdgeStats:
    """
    Backtest and CV statistics for an Edge.
    
    Contains enough data to:
    - Compute confidence intervals via bootstrap
    - Build Beta priors for TPR/FPR
    - Compare model performance over time
    
    Example:
        stats = EdgeStats(
            train_period=("1997-01-01", "2017-12-31"),
            test_period=("2018-01-01", "2025-12-01"),
            n_test_events=104,
            n_test_pos=8,
            n_test_neg=96,
            tp=8, fp=19, fn=0, tn=77,
            auc=0.911,
            base_rate=0.077,
            fn_rate=0.0,
            fp_rate=0.198,
            precision=0.296,
            tn_fp_ratio=4.05,
        )
    """
    # Data windows
    train_period: Tuple[str, str]  # (start_date, end_date)
    test_period: Tuple[str, str]
    
    # Counts on test set
    n_test_events: int
    n_test_pos: int  # large moves
    n_test_neg: int  # normal moves
    
    # Confusion matrix on test for THIS threshold
    tp: int
    fp: int
    fn: int
    tn: int
    
    # Scalar metrics
    auc: float
    base_rate: float  # n_test_pos / n_test_events
    fn_rate: float  # fn / (tp+fn)
    fp_rate: float  # fp / (fp+tn)
    precision: Optional[float] = None  # tp / (tp+fp) if tp+fp>0 else null
    tn_fp_ratio: Optional[float] = None  # tn / fp (or null if fp=0)
    
    # Optional: per-fold CV metrics for future CIs
    cv_metrics: Optional[List[CVFoldMetric]] = None
    
    # Optional: raw test predictions for bootstrapping
    test_predictions: Optional[List[TestPrediction]] = None
    
    def __post_init__(self):
        # Compute derived metrics if not provided
        if self.precision is None and (self.tp + self.fp) > 0:
            self.precision = self.tp / (self.tp + self.fp)
        if self.tn_fp_ratio is None and self.fp > 0:
            self.tn_fp_ratio = self.tn / self.fp
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "train_period": list(self.train_period),
            "test_period": list(self.test_period),
            "n_test_events": self.n_test_events,
            "n_test_pos": self.n_test_pos,
            "n_test_neg": self.n_test_neg,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "auc": self.auc,
            "base_rate": self.base_rate,
            "fn_rate": self.fn_rate,
            "fp_rate": self.fp_rate,
            "precision": self.precision,
            "tn_fp_ratio": self.tn_fp_ratio,
        }
        if self.cv_metrics:
            d["cv_metrics"] = [m.to_dict() for m in self.cv_metrics]
        if self.test_predictions:
            d["test_predictions"] = [p.to_dict() for p in self.test_predictions]
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EdgeStats":
        cv_metrics = None
        if "cv_metrics" in d and d["cv_metrics"]:
            cv_metrics = [CVFoldMetric.from_dict(m) for m in d["cv_metrics"]]
        
        test_predictions = None
        if "test_predictions" in d and d["test_predictions"]:
            test_predictions = [TestPrediction.from_dict(p) for p in d["test_predictions"]]
        
        return cls(
            train_period=tuple(d["train_period"]),
            test_period=tuple(d["test_period"]),
            n_test_events=d["n_test_events"],
            n_test_pos=d["n_test_pos"],
            n_test_neg=d["n_test_neg"],
            tp=d["tp"],
            fp=d["fp"],
            fn=d["fn"],
            tn=d["tn"],
            auc=d["auc"],
            base_rate=d["base_rate"],
            fn_rate=d["fn_rate"],
            fp_rate=d["fp_rate"],
            precision=d.get("precision"),
            tn_fp_ratio=d.get("tn_fp_ratio"),
            cv_metrics=cv_metrics,
            test_predictions=test_predictions,
        )
    
    @property
    def recall(self) -> float:
        """Recall = 1 - FN rate = TP / (TP + FN)"""
        return 1.0 - self.fn_rate
    
    @property 
    def specificity(self) -> float:
        """Specificity = 1 - FP rate = TN / (TN + FP)"""
        return 1.0 - self.fp_rate


# ============================================================================
# EDGE (THE MODULAR CONFIGURATION)
# ============================================================================

@dataclass
class Edge:
    """
    One specific model configuration implementing an event → instrument mapping.
    
    This is the core unit of the graph. Each Edge:
    - Is self-contained: can be loaded and run independently
    - Is swap-friendly: can be replaced without touching graph logic
    - Carries full backtest metadata for reliability assessment
    
    Example:
        edge = Edge(
            edge_id="CPI->HY__FN1pct__LogReg_w50_thr15bp",
            slot_id="CPI->HY_OAS",
            from_node=NodeId(NodeType.EVENT, "CPI"),
            to_node=NodeId(NodeType.INSTRUMENT, "HY_OAS"),
            model_type="LogReg",
            model_location="models/cpi_hy_logreg_w50.pkl",
            active_factor="cpi_shock_abs",
            background_features=["yield_vol_10y", "slope_10y_2y", "fed_funds"],
            target_series="HY_OAS",
            target_unit="bp",
            large_move_threshold=LargeMoveThreshold(0.15, "85th pctl"),
            fn_constraint=0.01,
            prob_cutoff=0.80,
            threshold_selection_method="grid_search + FN<=1% constraint on test",
            stats=stats,
            version="2025-12-05_cpi_hy_v1",
            created_at="2025-12-05T10:30:00Z",
            created_by="grid_search_module_0.1",
        )
    """
    # Identity
    edge_id: str  # globally unique, e.g. "CPI->HY__FN1pct__LogReg_w50_thr15bp"
    slot_id: str  # which EdgeSlot it belongs to, e.g. "CPI->HY_OAS"
    from_node: NodeId  # event node
    to_node: NodeId  # instrument node
    
    # Modelling / semantics
    model_type: str  # "LogReg", "RF_shallow", "GB", ...
    model_location: str  # pointer to serialized model (path, artifact id)
    active_factor: str  # primary event variable, e.g. "cpi_shock_abs"
    background_features: List[str]  # additional conditioning factors
    target_series: str  # internal code, e.g. "HY_OAS", "VIX"
    target_unit: str  # "bp", "points"
    large_move_threshold: LargeMoveThreshold
    
    # Decision policy
    fn_constraint: float  # 0.01 for FN≤1%; 0.05 for FN≤5%
    prob_cutoff: float  # p* threshold used in backtest
    threshold_selection_method: str  # how threshold was chosen
    
    # Backtest summary
    stats: EdgeStats
    
    # Versioning / metadata
    version: str
    created_at: str  # ISO format
    created_by: str
    notes: str = ""
    
    # Optional: model hyperparameters
    model_params: Optional[Dict[str, Any]] = None
    
    # Optional: feature medians for imputation
    feature_medians: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if isinstance(self.from_node, dict):
            self.from_node = NodeId.from_dict(self.from_node)
        if isinstance(self.to_node, dict):
            self.to_node = NodeId.from_dict(self.to_node)
        if isinstance(self.large_move_threshold, dict):
            self.large_move_threshold = LargeMoveThreshold.from_dict(self.large_move_threshold)
        if isinstance(self.stats, dict):
            self.stats = EdgeStats.from_dict(self.stats)
    
    @property
    def all_features(self) -> List[str]:
        """Return all features: active_factor + background_features."""
        return [self.active_factor] + self.background_features
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "edge_id": self.edge_id,
            "slot_id": self.slot_id,
            "from_node": self.from_node.to_dict(),
            "to_node": self.to_node.to_dict(),
            "model_type": self.model_type,
            "model_location": self.model_location,
            "active_factor": self.active_factor,
            "background_features": self.background_features,
            "target_series": self.target_series,
            "target_unit": self.target_unit,
            "large_move_threshold": self.large_move_threshold.to_dict(),
            "fn_constraint": self.fn_constraint,
            "prob_cutoff": self.prob_cutoff,
            "threshold_selection_method": self.threshold_selection_method,
            "stats": self.stats.to_dict(),
            "version": self.version,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "notes": self.notes,
        }
        if self.model_params:
            d["model_params"] = self.model_params
        if self.feature_medians:
            d["feature_medians"] = self.feature_medians
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Edge":
        return cls(
            edge_id=d["edge_id"],
            slot_id=d["slot_id"],
            from_node=NodeId.from_dict(d["from_node"]),
            to_node=NodeId.from_dict(d["to_node"]),
            model_type=d["model_type"],
            model_location=d["model_location"],
            active_factor=d["active_factor"],
            background_features=d["background_features"],
            target_series=d["target_series"],
            target_unit=d["target_unit"],
            large_move_threshold=LargeMoveThreshold.from_dict(d["large_move_threshold"]),
            fn_constraint=d["fn_constraint"],
            prob_cutoff=d["prob_cutoff"],
            threshold_selection_method=d["threshold_selection_method"],
            stats=EdgeStats.from_dict(d["stats"]),
            version=d["version"],
            created_at=d["created_at"],
            created_by=d["created_by"],
            notes=d.get("notes", ""),
            model_params=d.get("model_params"),
            feature_medians=d.get("feature_medians"),
        )
    
    def summary(self) -> str:
        """Human-readable summary of this edge."""
        return (
            f"{self.edge_id}\n"
            f"  {self.from_node.name} → {self.to_node.name}\n"
            f"  Model: {self.model_type}, FN≤{self.fn_constraint*100:.0f}%\n"
            f"  Threshold: {self.large_move_threshold.value} {self.target_unit}\n"
            f"  Prob cutoff: {self.prob_cutoff:.2f}\n"
            f"  AUC: {self.stats.auc:.3f}, TN/FP: {self.stats.tn_fp_ratio:.2f}x\n"
            f"  Test: TP={self.stats.tp}, FP={self.stats.fp}, FN={self.stats.fn}, TN={self.stats.tn}"
        )


# ============================================================================
# EDGE SLOT (LOGICAL MAPPING)
# ============================================================================

@dataclass
class EdgeSlot:
    """
    Groups all Edges that implement the same conceptual mapping.
    
    Example:
        slot = EdgeSlot(
            slot_id="CPI->HY_OAS",
            from_node=NodeId(NodeType.EVENT, "CPI"),
            to_node=NodeId(NodeType.INSTRUMENT, "HY_OAS"),
            description="CPI announcement → large HY OAS move probability",
            feature_spec={
                "active_factor": "cpi_shock_abs",
                "background_features": ["yield_vol_10y", "slope_10y_2y", ...]
            },
            edges={
                "CPI->HY__FN1pct__LogReg": edge1,
                "CPI->HY__FN5pct__GB": edge2,
            },
            default_edge_id="CPI->HY__FN5pct__GB",
        )
    """
    slot_id: str
    from_node: NodeId
    to_node: NodeId
    description: str
    feature_spec: Dict[str, Any]  # canonical feature recipe
    edges: Dict[str, Edge] = field(default_factory=dict)
    default_edge_id: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.from_node, dict):
            self.from_node = NodeId.from_dict(self.from_node)
        if isinstance(self.to_node, dict):
            self.to_node = NodeId.from_dict(self.to_node)
        # Convert edge dicts to Edge objects
        for k, v in list(self.edges.items()):
            if isinstance(v, dict):
                self.edges[k] = Edge.from_dict(v)
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to this slot."""
        if edge.slot_id != self.slot_id:
            raise ValueError(f"Edge slot_id '{edge.slot_id}' does not match slot '{self.slot_id}'")
        self.edges[edge.edge_id] = edge
        if self.default_edge_id is None:
            self.default_edge_id = edge.edge_id
    
    def get_edge(self, edge_id: Optional[str] = None) -> Edge:
        """Get an edge by ID, or the default edge."""
        if edge_id is None:
            edge_id = self.default_edge_id
        if edge_id is None:
            raise ValueError(f"No default edge set for slot '{self.slot_id}'")
        if edge_id not in self.edges:
            raise KeyError(f"Edge '{edge_id}' not found in slot '{self.slot_id}'")
        return self.edges[edge_id]
    
    def get_edge_by_fn_constraint(self, fn_constraint: float) -> Optional[Edge]:
        """Find edge matching a specific FN constraint."""
        for edge in self.edges.values():
            if abs(edge.fn_constraint - fn_constraint) < 0.001:
                return edge
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "from_node": self.from_node.to_dict(),
            "to_node": self.to_node.to_dict(),
            "description": self.description,
            "feature_spec": self.feature_spec,
            "edges": {k: v.to_dict() for k, v in self.edges.items()},
            "default_edge_id": self.default_edge_id,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EdgeSlot":
        edges = {k: Edge.from_dict(v) for k, v in d.get("edges", {}).items()}
        return cls(
            slot_id=d["slot_id"],
            from_node=NodeId.from_dict(d["from_node"]),
            to_node=NodeId.from_dict(d["to_node"]),
            description=d["description"],
            feature_spec=d["feature_spec"],
            edges=edges,
            default_edge_id=d.get("default_edge_id"),
        )


# ============================================================================
# GRAPH (TOP-LEVEL CONTAINER)
# ============================================================================

@dataclass
class Graph:
    """
    The full macro graph containing nodes and edge slots.
    
    Example:
        graph = Graph(
            name="QSIG Macro Graph v1",
            nodes={
                "CPI": NodeId(NodeType.EVENT, "CPI"),
                "HY_OAS": NodeId(NodeType.INSTRUMENT, "HY_OAS"),
            },
            edge_slots={
                "CPI->HY_OAS": edge_slot,
            },
        )
    """
    name: str
    description: str = ""
    version: str = "1.0.0"
    nodes: Dict[str, NodeId] = field(default_factory=dict)
    edge_slots: Dict[str, EdgeSlot] = field(default_factory=dict)
    created_at: str = ""
    
    def __post_init__(self):
        # Convert dicts to objects
        for k, v in list(self.nodes.items()):
            if isinstance(v, dict):
                self.nodes[k] = NodeId.from_dict(v)
        for k, v in list(self.edge_slots.items()):
            if isinstance(v, dict):
                self.edge_slots[k] = EdgeSlot.from_dict(v)
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
    
    def add_node(self, node: NodeId) -> None:
        """Add a node to the graph."""
        self.nodes[node.name] = node
    
    def add_edge_slot(self, slot: EdgeSlot) -> None:
        """Add an edge slot to the graph."""
        self.edge_slots[slot.slot_id] = slot
        # Ensure nodes exist
        if slot.from_node.name not in self.nodes:
            self.nodes[slot.from_node.name] = slot.from_node
        if slot.to_node.name not in self.nodes:
            self.nodes[slot.to_node.name] = slot.to_node
    
    def get_slot(self, slot_id: str) -> EdgeSlot:
        """Get an edge slot by ID."""
        if slot_id not in self.edge_slots:
            raise KeyError(f"EdgeSlot '{slot_id}' not found in graph")
        return self.edge_slots[slot_id]
    
    def get_edge(self, slot_id: str, edge_id: Optional[str] = None) -> Edge:
        """Get an edge from a slot."""
        return self.get_slot(slot_id).get_edge(edge_id)
    
    def list_slots(self) -> List[str]:
        """List all edge slot IDs."""
        return list(self.edge_slots.keys())
    
    def list_edges(self, slot_id: Optional[str] = None) -> List[str]:
        """List all edge IDs, optionally filtered by slot."""
        if slot_id:
            return list(self.edge_slots[slot_id].edges.keys())
        return [
            edge_id
            for slot in self.edge_slots.values()
            for edge_id in slot.edges.keys()
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edge_slots": {k: v.to_dict() for k, v in self.edge_slots.items()},
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Graph":
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            version=d.get("version", "1.0.0"),
            nodes={k: NodeId.from_dict(v) for k, v in d.get("nodes", {}).items()},
            edge_slots={k: EdgeSlot.from_dict(v) for k, v in d.get("edge_slots", {}).items()},
            created_at=d.get("created_at", ""),
        )
    
    def summary(self) -> str:
        """Human-readable summary of the graph."""
        lines = [
            f"Graph: {self.name} (v{self.version})",
            f"  Nodes: {len(self.nodes)} ({sum(1 for n in self.nodes.values() if n.type == NodeType.EVENT)} events, "
            f"{sum(1 for n in self.nodes.values() if n.type == NodeType.INSTRUMENT)} instruments)",
            f"  Edge Slots: {len(self.edge_slots)}",
        ]
        for slot_id, slot in self.edge_slots.items():
            lines.append(f"    {slot_id}: {len(slot.edges)} edges")
        return "\n".join(lines)


# ============================================================================
# EVENT CONTEXT (RUNTIME INPUT)
# ============================================================================

@dataclass
class EventContext:
    """
    Represents a single macro event instance with precomputed features.
    
    This is the input to an Edge at runtime.
    
    Example:
        ctx = EventContext(
            node=NodeId(NodeType.EVENT, "CPI"),
            event_date="2024-03-12",
            features={
                "cpi_shock_abs": 0.3,
                "yield_vol_10y": 0.05,
                "slope_10y_2y": 1.2,
                "fed_funds": 5.25,
                "hy_oas_before": 4.0,
                "vix_before": 15.0,
                "stlfsi": 0.2,
            },
            meta={
                "release_time_utc": "13:30:00",
                "source_calendar": "BLS",
            }
        )
    """
    node: NodeId  # must be an event node
    event_date: str  # YYYY-MM-DD
    features: Dict[str, float]  # t-1 features; global feature namespace
    meta: Dict[str, Any] = field(default_factory=dict)  # optional metadata
    
    def __post_init__(self):
        if isinstance(self.node, dict):
            self.node = NodeId.from_dict(self.node)
        if self.node.type != NodeType.EVENT:
            raise ValueError(f"EventContext node must be an event, got {self.node.type}")
    
    def get_feature(self, name: str, default: Optional[float] = None) -> Optional[float]:
        """Get a feature value, with optional default."""
        return self.features.get(name, default)
    
    def has_features(self, names: List[str]) -> bool:
        """Check if all required features are present."""
        return all(name in self.features for name in names)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node.to_dict(),
            "event_date": self.event_date,
            "features": self.features,
            "meta": self.meta,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EventContext":
        return cls(
            node=NodeId.from_dict(d["node"]),
            event_date=d["event_date"],
            features=d["features"],
            meta=d.get("meta", {}),
        )


# ============================================================================
# EDGE RESULT (RUNTIME OUTPUT)
# ============================================================================

@dataclass
class EdgeResult:
    """
    Output from applying an Edge to an EventContext.
    
    This is the atomic unit that the network-level aggregator consumes.
    
    Example:
        result = EdgeResult(
            edge_id="CPI->HY__FN1pct__LogReg_w50_thr15bp",
            slot_id="CPI->HY_OAS",
            from_node=NodeId(NodeType.EVENT, "CPI"),
            to_node=NodeId(NodeType.INSTRUMENT, "HY_OAS"),
            event_date="2024-03-12",
            prob_large_move=0.85,
            flag_large_move=True,
            fn_constraint=0.01,
            large_move_threshold=0.15,
            target_series="HY_OAS",
            target_unit="bp",
            backtest={
                "auc": 0.911,
                "base_rate": 0.077,
                "tp": 8, "fp": 19, "fn": 0, "tn": 77,
                "fn_rate": 0.0, "fp_rate": 0.198,
                "tn_fp_ratio": 4.05,
            }
        )
    """
    # Identity
    edge_id: str
    slot_id: str
    from_node: NodeId
    to_node: NodeId
    event_date: str
    
    # Predictions
    prob_large_move: float  # p̂ = P(|Δtarget| ≥ threshold | features)
    flag_large_move: bool  # p̂ ≥ prob_cutoff
    
    # Policy + config info (for later aggregation)
    fn_constraint: float
    large_move_threshold: float
    target_series: str
    target_unit: str
    
    # Reliability / backtest info (copied from Edge)
    backtest: Dict[str, Any]
    
    # Optional: features used for this prediction
    features_used: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if isinstance(self.from_node, dict):
            self.from_node = NodeId.from_dict(self.from_node)
        if isinstance(self.to_node, dict):
            self.to_node = NodeId.from_dict(self.to_node)
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "edge_id": self.edge_id,
            "slot_id": self.slot_id,
            "from_node": self.from_node.to_dict(),
            "to_node": self.to_node.to_dict(),
            "event_date": self.event_date,
            "prob_large_move": self.prob_large_move,
            "flag_large_move": self.flag_large_move,
            "fn_constraint": self.fn_constraint,
            "large_move_threshold": self.large_move_threshold,
            "target_series": self.target_series,
            "target_unit": self.target_unit,
            "backtest": self.backtest,
        }
        if self.features_used:
            d["features_used"] = self.features_used
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EdgeResult":
        return cls(
            edge_id=d["edge_id"],
            slot_id=d["slot_id"],
            from_node=NodeId.from_dict(d["from_node"]),
            to_node=NodeId.from_dict(d["to_node"]),
            event_date=d["event_date"],
            prob_large_move=d["prob_large_move"],
            flag_large_move=d["flag_large_move"],
            fn_constraint=d["fn_constraint"],
            large_move_threshold=d["large_move_threshold"],
            target_series=d["target_series"],
            target_unit=d["target_unit"],
            backtest=d["backtest"],
            features_used=d.get("features_used"),
        )
    
    def summary(self) -> str:
        """Human-readable summary."""
        flag_str = "🔴 LARGE MOVE EXPECTED" if self.flag_large_move else "🟢 Normal move expected"
        return (
            f"{self.from_node.name} → {self.to_node.name} on {self.event_date}\n"
            f"  P(large move): {self.prob_large_move:.1%}\n"
            f"  {flag_str}\n"
            f"  Edge: {self.edge_id} (FN≤{self.fn_constraint*100:.0f}%)"
        )


# ============================================================================
# EDGE SPEC (FOR GRID SEARCH)
# ============================================================================

@dataclass
class EdgeSpec:
    """
    Specification for grid search to produce Edges.
    
    Example:
        spec = EdgeSpec(
            slot_id="CPI->HY_OAS",
            from_node=NodeId(NodeType.EVENT, "CPI"),
            to_node=NodeId(NodeType.INSTRUMENT, "HY_OAS"),
            active_factor="cpi_shock_abs",
            background_features=["yield_vol_10y", "slope_10y_2y", "fed_funds", "hy_oas_before", "stlfsi"],
            target_col="hy_change",
            large_threshold_candidates=[0.05, 0.08, 0.10, 0.12, 0.15],
            fn_constraints=[0.01, 0.05],
        )
    """
    slot_id: str
    from_node: NodeId
    to_node: NodeId
    active_factor: str
    background_features: List[str]
    target_col: str  # column name in DataFrame
    target_unit: str = "bp"
    large_threshold_candidates: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.15])
    fn_constraints: List[float] = field(default_factory=lambda: [0.01, 0.05])
    
    def __post_init__(self):
        if isinstance(self.from_node, dict):
            self.from_node = NodeId.from_dict(self.from_node)
        if isinstance(self.to_node, dict):
            self.to_node = NodeId.from_dict(self.to_node)
    
    @property
    def all_features(self) -> List[str]:
        """All features for the model."""
        return [self.active_factor] + self.background_features
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "from_node": self.from_node.to_dict(),
            "to_node": self.to_node.to_dict(),
            "active_factor": self.active_factor,
            "background_features": self.background_features,
            "target_col": self.target_col,
            "target_unit": self.target_unit,
            "large_threshold_candidates": self.large_threshold_candidates,
            "fn_constraints": self.fn_constraints,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EdgeSpec":
        return cls(
            slot_id=d["slot_id"],
            from_node=NodeId.from_dict(d["from_node"]),
            to_node=NodeId.from_dict(d["to_node"]),
            active_factor=d["active_factor"],
            background_features=d["background_features"],
            target_col=d["target_col"],
            target_unit=d.get("target_unit", "bp"),
            large_threshold_candidates=d.get("large_threshold_candidates", [0.05, 0.10, 0.15]),
            fn_constraints=d.get("fn_constraints", [0.01, 0.05]),
        )

