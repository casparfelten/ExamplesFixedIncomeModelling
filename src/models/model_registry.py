"""
Model Registry: Storage and Reference System for Trained Models
================================================================

This module provides a registry for storing and referencing trained models
independently from the graph system.

Key concepts:
- Each model has a unique model_id
- Models are stored with full metadata (type, params, metrics, features)
- Models are promoted to the registry manually after human review
- The graph references models by ID, never generates them

Workflow:
1. Run grid search (separate process) → candidates
2. Human reviews → promotes good models to registry
3. Graph build reads config → references models from registry

Usage:
    from src.models.model_registry import ModelRegistry
    
    # Load registry
    registry = ModelRegistry("registry/models")
    
    # List available models
    models = registry.list_models()
    
    # Get a specific model
    model_info = registry.get_model("cpi_hy_logreg_w50_20bp")
    model = registry.load_model("cpi_hy_logreg_w50_20bp")
    
    # Promote a new model (manual process)
    registry.promote_model(
        model_id="cpi_hy_logreg_w50_20bp",
        model=trained_model,
        scaler=fitted_scaler,
        metadata={...}
    )
"""

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ModelMetadata:
    """
    Metadata for a registered model.
    
    This is what we store alongside the model artifact.
    """
    model_id: str
    model_type: str  # "LogReg", "RF_shallow", "GB", etc.
    
    # Training configuration
    weight: Any  # class weight (int or "balanced")
    features: List[str]  # ordered list of feature names
    feature_medians: Dict[str, float]  # for imputation
    
    # Target configuration
    target_event: str  # "CPI", "UNEMPLOYMENT"
    target_instrument: str  # "HY_OAS", "VIX"
    target_col: str  # column name used in training
    large_move_threshold: float
    large_move_definition: str
    prob_cutoff: float
    
    # Performance metrics (from test set)
    train_period: tuple  # (start, end)
    test_period: tuple
    n_train: int
    n_test: int
    n_test_pos: int
    n_test_neg: int
    tp: int
    fp: int
    fn: int
    tn: int
    auc: float
    fn_rate: float
    fp_rate: float
    tn_fp_ratio: Optional[float]
    
    # Provenance
    created_at: str
    created_by: str
    source: str  # e.g. "grid_search_2025-12-05", "manual_tuning"
    notes: str = ""
    
    # Artifact paths (relative to registry)
    model_file: str = ""
    scaler_file: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "weight": self.weight,
            "features": self.features,
            "feature_medians": self.feature_medians,
            "target_event": self.target_event,
            "target_instrument": self.target_instrument,
            "target_col": self.target_col,
            "large_move_threshold": self.large_move_threshold,
            "large_move_definition": self.large_move_definition,
            "prob_cutoff": self.prob_cutoff,
            "train_period": list(self.train_period),
            "test_period": list(self.test_period),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_test_pos": self.n_test_pos,
            "n_test_neg": self.n_test_neg,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "auc": self.auc,
            "fn_rate": self.fn_rate,
            "fp_rate": self.fp_rate,
            "tn_fp_ratio": self.tn_fp_ratio,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "source": self.source,
            "notes": self.notes,
            "model_file": self.model_file,
            "scaler_file": self.scaler_file,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelMetadata":
        return cls(
            model_id=d["model_id"],
            model_type=d["model_type"],
            weight=d["weight"],
            features=d["features"],
            feature_medians=d.get("feature_medians", {}),
            target_event=d["target_event"],
            target_instrument=d["target_instrument"],
            target_col=d["target_col"],
            large_move_threshold=d["large_move_threshold"],
            large_move_definition=d.get("large_move_definition", ""),
            prob_cutoff=d["prob_cutoff"],
            train_period=tuple(d["train_period"]),
            test_period=tuple(d["test_period"]),
            n_train=d["n_train"],
            n_test=d["n_test"],
            n_test_pos=d["n_test_pos"],
            n_test_neg=d["n_test_neg"],
            tp=d["tp"],
            fp=d["fp"],
            fn=d["fn"],
            tn=d["tn"],
            auc=d["auc"],
            fn_rate=d["fn_rate"],
            fp_rate=d["fp_rate"],
            tn_fp_ratio=d.get("tn_fp_ratio"),
            created_at=d["created_at"],
            created_by=d["created_by"],
            source=d.get("source", "unknown"),
            notes=d.get("notes", ""),
            model_file=d.get("model_file", ""),
            scaler_file=d.get("scaler_file", ""),
        )
    
    def summary(self) -> str:
        tn_fp = f"{self.tn_fp_ratio:.2f}x" if self.tn_fp_ratio else "∞"
        return (
            f"{self.model_id}\n"
            f"  Type: {self.model_type}, weight={self.weight}\n"
            f"  Target: {self.target_event} → {self.target_instrument}\n"
            f"  Threshold: {self.large_move_threshold} @ p≥{self.prob_cutoff:.2f}\n"
            f"  Test: TP={self.tp}, FP={self.fp}, FN={self.fn}, TN={self.tn}\n"
            f"  AUC={self.auc:.3f}, FN%={self.fn_rate*100:.1f}%, TN/FP={tn_fp}"
        )


class ModelRegistry:
    """
    Registry for storing and retrieving trained models.
    
    Directory structure:
        registry_dir/
            index.json          # list of all model_ids
            models/
                {model_id}/
                    metadata.json
                    model.pkl
                    scaler.pkl (optional)
    """
    
    def __init__(self, registry_dir: Union[str, Path]):
        self.registry_dir = Path(registry_dir)
        self.models_dir = self.registry_dir / "models"
        self.index_file = self.registry_dir / "index.json"
        
        # Ensure directories exist
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create index
        self._index: Dict[str, str] = {}  # model_id -> path
        self._load_index()
    
    def _load_index(self):
        """Load the model index."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self._index = json.load(f)
        else:
            self._index = {}
    
    def _save_index(self):
        """Save the model index."""
        with open(self.index_file, 'w') as f:
            json.dump(self._index, f, indent=2)
    
    def list_models(self) -> List[str]:
        """List all registered model IDs."""
        return list(self._index.keys())
    
    def list_models_for_mapping(self, event: str, instrument: str) -> List[str]:
        """List models for a specific event→instrument mapping."""
        matching = []
        for model_id in self._index:
            meta = self.get_metadata(model_id)
            if meta and meta.target_event == event and meta.target_instrument == instrument:
                matching.append(model_id)
        return matching
    
    def has_model(self, model_id: str) -> bool:
        """Check if a model exists in the registry."""
        return model_id in self._index
    
    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get metadata for a model."""
        if model_id not in self._index:
            return None
        
        model_dir = self.models_dir / model_id
        meta_file = model_dir / "metadata.json"
        
        if not meta_file.exists():
            return None
        
        with open(meta_file, 'r') as f:
            data = json.load(f)
        
        return ModelMetadata.from_dict(data)
    
    def load_model(self, model_id: str) -> Optional[Any]:
        """Load a trained model object."""
        if model_id not in self._index:
            return None
        
        model_dir = self.models_dir / model_id
        model_file = model_dir / "model.pkl"
        
        if not model_file.exists():
            return None
        
        with open(model_file, 'rb') as f:
            return pickle.load(f)
    
    def load_scaler(self, model_id: str) -> Optional[Any]:
        """Load the scaler for a model."""
        if model_id not in self._index:
            return None
        
        model_dir = self.models_dir / model_id
        scaler_file = model_dir / "scaler.pkl"
        
        if not scaler_file.exists():
            return None
        
        with open(scaler_file, 'rb') as f:
            return pickle.load(f)
    
    def load_model_bundle(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Load everything needed to use a model.
        
        Returns:
            Dict with 'model', 'scaler', 'metadata'
        """
        meta = self.get_metadata(model_id)
        if meta is None:
            return None
        
        model = self.load_model(model_id)
        scaler = self.load_scaler(model_id)
        
        return {
            'model': model,
            'scaler': scaler,
            'metadata': meta,
        }
    
    def promote_model(
        self,
        model_id: str,
        model: Any,
        metadata: ModelMetadata,
        scaler: Optional[Any] = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Promote a model to the registry.
        
        This is a manual operation - the human decides which models to promote.
        
        Args:
            model_id: Unique identifier for the model
            model: The trained sklearn model
            metadata: ModelMetadata with full info
            scaler: Optional fitted scaler
            overwrite: If True, overwrite existing model
        
        Returns:
            Path to the model directory
        """
        if model_id in self._index and not overwrite:
            raise ValueError(f"Model {model_id} already exists. Use overwrite=True to replace.")
        
        # Create model directory
        model_dir = self.models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_dir / "model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Save scaler if provided
        if scaler is not None:
            scaler_file = model_dir / "scaler.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            metadata.scaler_file = "scaler.pkl"
        
        # Update metadata
        metadata.model_file = "model.pkl"
        
        # Save metadata
        meta_file = model_dir / "metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Update index
        self._index[model_id] = str(model_dir.relative_to(self.registry_dir))
        self._save_index()
        
        print(f"Promoted model: {model_id}")
        return model_dir
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the registry."""
        if model_id not in self._index:
            return False
        
        import shutil
        model_dir = self.models_dir / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        del self._index[model_id]
        self._save_index()
        
        return True
    
    def export_catalog(self, output_path: Union[str, Path]) -> None:
        """Export a human-readable catalog of all models."""
        output_path = Path(output_path)
        
        lines = [
            "# Model Registry Catalog",
            "",
            f"Generated: {datetime.utcnow().isoformat()}Z",
            f"Total models: {len(self._index)}",
            "",
        ]
        
        # Group by mapping
        by_mapping: Dict[str, List[ModelMetadata]] = {}
        
        for model_id in self._index:
            meta = self.get_metadata(model_id)
            if meta:
                key = f"{meta.target_event}->{meta.target_instrument}"
                if key not in by_mapping:
                    by_mapping[key] = []
                by_mapping[key].append(meta)
        
        for mapping, models in sorted(by_mapping.items()):
            lines.append(f"## {mapping}")
            lines.append("")
            lines.append("| Model ID | Type | Weight | Threshold | Prob | AUC | FN% | TN/FP |")
            lines.append("|----------|------|--------|-----------|------|-----|-----|-------|")
            
            for meta in models:
                tn_fp = f"{meta.tn_fp_ratio:.2f}x" if meta.tn_fp_ratio else "∞"
                lines.append(
                    f"| {meta.model_id} | {meta.model_type} | {meta.weight} | "
                    f"{meta.large_move_threshold} | {meta.prob_cutoff:.2f} | "
                    f"{meta.auc:.3f} | {meta.fn_rate*100:.1f}% | {tn_fp} |"
                )
            
            lines.append("")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"Catalog exported to {output_path}")

