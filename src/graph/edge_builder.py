"""
QSIG Macro Graph: Edge Builder
===============================

Converts grid search results into Edge configurations.

The EdgeBuilder:
1. Takes grid search DataFrames from event_grid_search.py
2. Filters by FN constraint
3. Selects best configurations
4. Produces Edge objects with full metadata
5. Optionally serializes trained models

Usage:
    from src.graph import EdgeBuilder, EdgeSpec, NodeId, NodeType
    from src.models.event_grid_search import run_full_grid_search
    
    # Define spec
    spec = EdgeSpec(
        slot_id="CPI->HY_OAS",
        from_node=NodeId(NodeType.EVENT, "CPI"),
        to_node=NodeId(NodeType.INSTRUMENT, "HY_OAS"),
        active_factor="cpi_shock_abs",
        background_features=["yield_vol_10y", "slope_10y_2y", ...],
        target_col="hy_change",
    )
    
    # Run grid search
    results = run_full_grid_search(train_df, test_df, features, ...)
    
    # Build edges
    builder = EdgeBuilder(model_output_dir="models/edges")
    edges = builder.build_from_grid_search(spec, results, train_df, test_df)
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from .types import (
    Edge,
    EdgeSlot,
    EdgeSpec,
    EdgeStats,
    Graph,
    LargeMoveThreshold,
    NodeId,
    NodeType,
    CVFoldMetric,
    TestPrediction,
)


class EdgeBuilder:
    """
    Builds Edge objects from grid search results.
    
    This is the glue between "brute-force search notebook" and "modular edge graph".
    
    Example:
        builder = EdgeBuilder(
            model_output_dir="models/edges",
            version_prefix="2025-12-05",
        )
        
        edges = builder.build_from_grid_search(
            spec=EdgeSpec(...),
            grid_results=results_df,
            train_df=train_df,
            test_df=test_df,
            features=["cpi_shock_abs", "yield_vol_10y", ...],
        )
    """
    
    def __init__(
        self,
        model_output_dir: str = "models/edges",
        version_prefix: str = "",
        created_by: str = "EdgeBuilder",
    ):
        """
        Initialize the EdgeBuilder.
        
        Args:
            model_output_dir: Directory to save trained models
            version_prefix: Prefix for version strings (e.g. "2025-12-05")
            created_by: Creator identifier for metadata
        """
        self.model_output_dir = Path(model_output_dir)
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.version_prefix = version_prefix or datetime.now().strftime("%Y-%m-%d")
        self.created_by = created_by
    
    def build_from_grid_search(
        self,
        spec: EdgeSpec,
        grid_results: pd.DataFrame,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: Optional[List[str]] = None,
        save_models: bool = True,
        include_test_predictions: bool = True,
    ) -> List[Edge]:
        """
        Build Edge objects from grid search results.
        
        For each FN constraint in the spec:
        1. Filter results to those meeting the constraint
        2. Select best config by TN/FP ratio
        3. Retrain the model
        4. Save the model
        5. Build an Edge
        
        Args:
            spec: EdgeSpec defining the mapping and search parameters
            grid_results: DataFrame from run_full_grid_search
            train_df: Training DataFrame with features and target
            test_df: Test DataFrame with features and target
            features: Feature column names (defaults to spec.all_features)
            save_models: Whether to save trained models to disk
            include_test_predictions: Whether to include per-event predictions
        
        Returns:
            List of Edge objects
        """
        features = features or spec.all_features
        edges = []
        
        for fn_constraint in spec.fn_constraints:
            # Find best config for this constraint
            config = self._find_best_config(grid_results, fn_constraint)
            
            if config is None:
                print(f"Warning: No config found for FN≤{fn_constraint*100:.0f}% in {spec.slot_id}")
                continue
            
            # Build the edge
            edge = self._build_edge(
                spec=spec,
                config=config,
                fn_constraint=fn_constraint,
                train_df=train_df,
                test_df=test_df,
                features=features,
                save_models=save_models,
                include_test_predictions=include_test_predictions,
            )
            
            edges.append(edge)
        
        return edges
    
    def build_from_config(
        self,
        spec: EdgeSpec,
        model_type: str,
        weight: Any,
        large_threshold: float,
        prob_cutoff: float,
        fn_constraint: float,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: Optional[List[str]] = None,
        save_model: bool = True,
        include_test_predictions: bool = True,
    ) -> Edge:
        """
        Build a single Edge from explicit configuration.
        
        Args:
            spec: EdgeSpec defining the mapping
            model_type: Model type (e.g. "LogReg", "RF_shallow")
            weight: Class weight
            large_threshold: Large move threshold
            prob_cutoff: Probability threshold
            fn_constraint: FN constraint this config meets
            train_df: Training DataFrame
            test_df: Test DataFrame
            features: Feature names
            save_model: Whether to save the trained model
            include_test_predictions: Include per-event predictions
        
        Returns:
            Edge object
        """
        features = features or spec.all_features
        
        config = {
            'model': model_type,
            'weight': weight,
            'large_threshold': large_threshold,
            'prob_threshold': prob_cutoff,
        }
        
        return self._build_edge(
            spec=spec,
            config=config,
            fn_constraint=fn_constraint,
            train_df=train_df,
            test_df=test_df,
            features=features,
            save_models=save_model,
            include_test_predictions=include_test_predictions,
        )
    
    def _find_best_config(
        self,
        grid_results: pd.DataFrame,
        fn_constraint: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Find best configuration meeting FN constraint.
        
        Selects by highest TN/FP ratio among those with fn_rate <= fn_constraint.
        """
        filtered = grid_results[grid_results['fn_rate'] <= fn_constraint].copy()
        
        if len(filtered) == 0:
            return None
        
        # Find best by TN/FP ratio
        # Handle inf values
        filtered['tn_fp_safe'] = filtered['tn_fp_ratio'].replace([np.inf], 1e9)
        best_idx = filtered['tn_fp_safe'].idxmax()
        best = filtered.loc[best_idx]
        
        return best.to_dict()
    
    def _build_edge(
        self,
        spec: EdgeSpec,
        config: Dict[str, Any],
        fn_constraint: float,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: List[str],
        save_models: bool,
        include_test_predictions: bool,
    ) -> Edge:
        """Build a single Edge from a config dict."""
        from sklearn.preprocessing import StandardScaler
        
        model_type = config['model']
        weight = config['weight']
        large_threshold = config['large_threshold']
        prob_cutoff = config['prob_threshold']
        
        # Create edge ID
        fn_pct = int(fn_constraint * 100)
        weight_str = str(weight).replace('.', 'p')
        thresh_str = f"{large_threshold:.2f}".replace('.', 'p')
        edge_id = f"{spec.slot_id}__FN{fn_pct}pct__{model_type}_w{weight_str}_thr{thresh_str}"
        
        # Define is_large column
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df['is_large'] = train_df[spec.target_col].abs() >= large_threshold
        test_df['is_large'] = test_df[spec.target_col].abs() >= large_threshold
        
        # Prepare features
        X_train = train_df[features].copy()
        X_test = test_df[features].copy()
        
        # Compute medians from train
        feature_medians = X_train.median().to_dict()
        
        # Fill missing
        X_train = X_train.fillna(feature_medians)
        X_test = X_test.fillna(feature_medians)
        
        y_train = train_df['is_large'].values
        y_test = test_df['is_large'].values
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = self._train_model(model_type, weight, X_train_scaled, y_train)
        
        # Evaluate on test
        probs = model.predict_proba(X_test_scaled)[:, 1]
        preds = probs >= prob_cutoff
        
        # Compute metrics
        tp = int(((y_test == True) & (preds == True)).sum())
        fp = int(((y_test == False) & (preds == True)).sum())
        fn = int(((y_test == True) & (preds == False)).sum())
        tn = int(((y_test == False) & (preds == False)).sum())
        
        n_test_pos = int(y_test.sum())
        n_test_neg = int((~y_test).sum())
        n_test = len(y_test)
        
        from sklearn.metrics import roc_auc_score
        try:
            auc = float(roc_auc_score(y_test, probs))
        except:
            auc = 0.5
        
        # Build test predictions if requested
        test_predictions = None
        if include_test_predictions and 'date' in test_df.columns:
            test_predictions = []
            for i, (_, row) in enumerate(test_df.iterrows()):
                test_predictions.append(TestPrediction(
                    event_date=str(row['date'].date()) if hasattr(row['date'], 'date') else str(row['date']),
                    y_true=int(y_test[i]),
                    y_pred_prob=float(probs[i]),
                    y_pred_flag=bool(preds[i]),
                ))
        
        # Build stats
        stats = EdgeStats(
            train_period=(
                str(train_df['date'].min().date()) if 'date' in train_df.columns else "",
                str(train_df['date'].max().date()) if 'date' in train_df.columns else "",
            ),
            test_period=(
                str(test_df['date'].min().date()) if 'date' in test_df.columns else "",
                str(test_df['date'].max().date()) if 'date' in test_df.columns else "",
            ),
            n_test_events=n_test,
            n_test_pos=n_test_pos,
            n_test_neg=n_test_neg,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            auc=auc,
            base_rate=n_test_pos / n_test if n_test > 0 else 0,
            fn_rate=fn / (tp + fn) if (tp + fn) > 0 else 0,
            fp_rate=fp / (fp + tn) if (fp + tn) > 0 else 0,
            test_predictions=test_predictions,
        )
        
        # Save model if requested
        model_location = ""
        if save_models:
            model_filename = f"{edge_id}.pkl"
            model_path = self.model_output_dir / model_filename
            
            model_data = {
                'model': model,
                'scaler': scaler,
                'feature_names': features,
                'feature_medians': feature_medians,
                'prob_cutoff': prob_cutoff,
                'large_threshold': large_threshold,
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            model_location = str(model_path)
        
        # Build large move threshold
        large_move_def = f"{int((1 - n_test_pos/n_test)*100)}th percentile of |Δ{spec.target_col}|"
        
        # Get model params
        model_params = self._get_model_params(model_type, weight)
        
        return Edge(
            edge_id=edge_id,
            slot_id=spec.slot_id,
            from_node=spec.from_node,
            to_node=spec.to_node,
            model_type=model_type,
            model_location=model_location,
            active_factor=spec.active_factor,
            background_features=spec.background_features,
            target_series=spec.to_node.name,
            target_unit=spec.target_unit,
            large_move_threshold=LargeMoveThreshold(
                value=large_threshold,
                definition=large_move_def,
            ),
            fn_constraint=fn_constraint,
            prob_cutoff=prob_cutoff,
            threshold_selection_method=f"grid_search + FN≤{fn_constraint*100:.0f}% constraint on test",
            stats=stats,
            version=f"{self.version_prefix}_{spec.slot_id.lower().replace('->', '_')}_v1",
            created_at=datetime.utcnow().isoformat() + "Z",
            created_by=self.created_by,
            notes=f"n_test_pos={n_test_pos}, treat as fragile if low",
            model_params=model_params,
            feature_medians=feature_medians,
        )
    
    def _train_model(self, model_type: str, weight: Any, X: np.ndarray, y: np.ndarray):
        """Train a model of the specified type."""
        from sklearn.ensemble import (
            RandomForestClassifier,
            GradientBoostingClassifier,
            AdaBoostClassifier,
            ExtraTreesClassifier,
        )
        from sklearn.linear_model import LogisticRegression
        
        # Model configurations (matching event_grid_search.py)
        model_configs = {
            'RF': {
                'class': RandomForestClassifier,
                'params': {'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
                'uses_class_weight': True,
            },
            'RF_deep': {
                'class': RandomForestClassifier,
                'params': {'n_estimators': 200, 'max_depth': 8, 'random_state': 42},
                'uses_class_weight': True,
            },
            'RF_shallow': {
                'class': RandomForestClassifier,
                'params': {'n_estimators': 100, 'max_depth': 3, 'random_state': 42},
                'uses_class_weight': True,
            },
            'ExtraTrees': {
                'class': ExtraTreesClassifier,
                'params': {'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
                'uses_class_weight': True,
            },
            'GB': {
                'class': GradientBoostingClassifier,
                'params': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'random_state': 42},
                'uses_class_weight': False,
            },
            'GB_deep': {
                'class': GradientBoostingClassifier,
                'params': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'random_state': 42},
                'uses_class_weight': False,
            },
            'GB_fast': {
                'class': GradientBoostingClassifier,
                'params': {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.2, 'random_state': 42},
                'uses_class_weight': False,
            },
            'AdaBoost': {
                'class': AdaBoostClassifier,
                'params': {'n_estimators': 100, 'learning_rate': 0.5, 'random_state': 42},
                'uses_class_weight': False,
            },
            'LogReg': {
                'class': LogisticRegression,
                'params': {'max_iter': 1000, 'random_state': 42},
                'uses_class_weight': True,
            },
        }
        
        if model_type not in model_configs:
            raise ValueError(f"Unknown model type: {model_type}")
        
        config = model_configs[model_type]
        params = config['params'].copy()
        
        if config['uses_class_weight']:
            if weight == 'balanced':
                params['class_weight'] = 'balanced'
            else:
                params['class_weight'] = {False: 1, True: weight}
            model = config['class'](**params)
            model.fit(X, y)
        else:
            # Use sample weights for GB, AdaBoost
            model = config['class'](**params)
            if weight == 'balanced':
                n_pos = y.sum()
                n_neg = len(y) - n_pos
                w = n_neg / n_pos if n_pos > 0 else 1
                sample_weights = np.where(y, w, 1)
            else:
                sample_weights = np.where(y, weight, 1)
            model.fit(X, y, sample_weight=sample_weights)
        
        return model
    
    def _get_model_params(self, model_type: str, weight: Any) -> Dict[str, Any]:
        """Get model hyperparameters for metadata."""
        base_params = {
            'RF': {'n_estimators': 100, 'max_depth': 5},
            'RF_deep': {'n_estimators': 200, 'max_depth': 8},
            'RF_shallow': {'n_estimators': 100, 'max_depth': 3},
            'ExtraTrees': {'n_estimators': 100, 'max_depth': 5},
            'GB': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
            'GB_deep': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05},
            'GB_fast': {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.2},
            'AdaBoost': {'n_estimators': 100, 'learning_rate': 0.5},
            'LogReg': {'max_iter': 1000},
        }
        
        params = base_params.get(model_type, {}).copy()
        params['weight'] = weight
        return params


def build_edge_slot(
    spec: EdgeSpec,
    edges: List[Edge],
    default_fn_constraint: float = 0.05,
) -> EdgeSlot:
    """
    Build an EdgeSlot from a spec and list of edges.
    
    Args:
        spec: The EdgeSpec
        edges: List of Edge objects
        default_fn_constraint: FN constraint for default edge selection
    
    Returns:
        EdgeSlot
    """
    edge_dict = {e.edge_id: e for e in edges}
    
    # Select default edge
    default_edge_id = None
    for edge in edges:
        if abs(edge.fn_constraint - default_fn_constraint) < 0.001:
            default_edge_id = edge.edge_id
            break
    
    if default_edge_id is None and edges:
        # Fall back to first edge
        default_edge_id = edges[0].edge_id
    
    return EdgeSlot(
        slot_id=spec.slot_id,
        from_node=spec.from_node,
        to_node=spec.to_node,
        description=f"{spec.from_node.name} → {spec.to_node.name} large move probability",
        feature_spec={
            "active_factor": spec.active_factor,
            "background_features": spec.background_features,
        },
        edges=edge_dict,
        default_edge_id=default_edge_id,
    )


def build_graph_from_specs(
    specs: List[EdgeSpec],
    edges_by_slot: Dict[str, List[Edge]],
    name: str = "QSIG Macro Graph",
    description: str = "",
    version: str = "1.0.0",
) -> Graph:
    """
    Build a Graph from specs and edges.
    
    Args:
        specs: List of EdgeSpecs
        edges_by_slot: Dict mapping slot_id to list of edges
        name: Graph name
        description: Graph description
        version: Version string
    
    Returns:
        Graph
    """
    graph = Graph(
        name=name,
        description=description,
        version=version,
    )
    
    for spec in specs:
        if spec.slot_id in edges_by_slot:
            slot = build_edge_slot(spec, edges_by_slot[spec.slot_id])
            graph.add_edge_slot(slot)
    
    return graph

