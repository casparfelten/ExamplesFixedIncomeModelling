#!/usr/bin/env python3
"""
Promote Model to Registry
==========================

Manually promote a model configuration to the model registry.
This is a supervised process - YOU decide which models to promote.

Usage:
    # Promote from grid search results
    python scripts/promote_model.py \\
        --mapping CPI->HY_OAS \\
        --model RF_shallow \\
        --weight 20 \\
        --threshold 0.20 \\
        --prob 0.15 \\
        --id cpi_hy_rf_shallow_w20_20bp

    # List available candidates from grid search
    python scripts/promote_model.py --list --mapping CPI->HY_OAS

    # Promote best config for a given FN constraint
    python scripts/promote_model.py \\
        --mapping CPI->HY_OAS \\
        --fn-max 0.01 \\
        --id cpi_hy_fn1pct_best

Workflow:
    1. Run grid search: python scripts/run_grid_search.py
    2. Review results:  python scripts/promote_model.py --list
    3. Promote models:  python scripts/promote_model.py --mapping ... --id ...
    4. Update config:   Edit registry/edge_config.yaml
    5. Build graph:     python scripts/build_graph.py
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler

from src.models.model_registry import ModelRegistry, ModelMetadata
from src.models.event_grid_search import find_best_configs


# ============================================================================
# DATA LOADERS (same as run_grid_search.py)
# ============================================================================

def load_cpi_hy_dataset():
    """Load CPI → HY OAS event dataset."""
    from src.data.fred_loader import load_series
    from src.data.merge_panel import build_fed_panel
    from src.data.inflation_announcements_loader import load_inflation_announcements
    
    panel = build_fed_panel()
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel.sort_values('date').reset_index(drop=True)
    
    cpi_dates = load_inflation_announcements()
    cpi_dates['release_date'] = pd.to_datetime(cpi_dates['release_date'])
    
    cpi_raw = load_series("CPIAUCSL").reset_index()
    cpi_raw.columns = ['date', 'cpi_value']
    cpi_raw['date'] = pd.to_datetime(cpi_raw['date'])
    cpi_raw['year_month'] = cpi_raw['date'].dt.to_period('M')
    cpi_monthly = cpi_raw.groupby('year_month')['cpi_value'].last().reset_index()
    cpi_monthly['data_period'] = cpi_monthly['year_month'].astype(str)
    cpi_dict = dict(zip(cpi_monthly['data_period'], cpi_monthly['cpi_value']))
    
    def get_prev_month(period):
        year, month = map(int, period.split('-'))
        if month == 1:
            return f"{year - 1}-12"
        return f"{year}-{month - 1:02d}"
    
    features = [
        "cpi_shock_abs", "yield_vol_10y", "hy_vol",
        "slope_10y_2y", "fed_funds", "hy_oas_before", "stlfsi",
    ]
    
    events = []
    for idx, ann in cpi_dates.iterrows():
        ann_date = ann['release_date']
        data_period = ann['data_period']
        
        ann_rows = panel[panel['date'] == ann_date]
        prev_rows = panel[panel['date'] < ann_date].tail(1)
        
        if ann_rows.empty or prev_rows.empty:
            continue
        
        ann_row = ann_rows.iloc[0]
        prev_row = prev_rows.iloc[0]
        
        hy_after = ann_row.get('hy_oas', np.nan)
        hy_before = prev_row.get('hy_oas', np.nan)
        
        if pd.isna(hy_before) or pd.isna(hy_after):
            continue
        
        hy_change = hy_after - hy_before
        
        prev_period = get_prev_month(data_period)
        cpi_current = cpi_dict.get(data_period)
        cpi_prev = cpi_dict.get(prev_period)
        cpi_shock = (cpi_current - cpi_prev) if cpi_current and cpi_prev else None
        
        recent = panel[panel['date'] < ann_date].tail(20)
        yield_vol_10y = np.nan
        if len(recent) > 5 and 'y_10y' in recent.columns:
            vals = recent['y_10y'].dropna().values
            if len(vals) > 5:
                yield_vol_10y = np.std(np.diff(vals))
        
        hy_vol = np.nan
        if len(recent) > 5 and 'hy_oas' in recent.columns:
            vals = recent['hy_oas'].dropna().values
            if len(vals) > 5:
                hy_vol = np.std(np.diff(vals))
        
        events.append({
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
        })
    
    df = pd.DataFrame(events)
    df = df.dropna(subset=['hy_change', 'cpi_shock_abs'])
    df = df.sort_values('date').reset_index(drop=True)
    
    n_test = int(len(df) * 0.30)
    train_df = df.iloc[:-n_test].copy()
    test_df = df.iloc[-n_test:].copy()
    
    return train_df, test_df, features, "hy_change", "CPI", "HY_OAS"


def load_unemp_vix_dataset():
    """Load Unemployment → VIX event dataset."""
    from src.data.fred_loader import load_series
    from src.data.merge_panel import build_fed_panel
    
    panel = build_fed_panel()
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel.sort_values('date').reset_index(drop=True)
    
    unemp = load_series("UNRATE").reset_index()
    unemp.columns = ['date', 'unemp_value']
    unemp['date'] = pd.to_datetime(unemp['date'])
    
    features = [
        "unemp_surprise_abs", "vix_vol", "yield_vol_10y",
        "slope_10y_2y", "fed_funds", "vix_before", "stlfsi",
    ]
    
    events = []
    for i in range(1, len(unemp)):
        ann_date = unemp.iloc[i]['date']
        
        ann_rows = panel[(panel['date'] >= ann_date) & (panel['date'] <= ann_date + pd.Timedelta(days=3))]
        prev_rows = panel[panel['date'] < ann_date].tail(1)
        
        if ann_rows.empty or prev_rows.empty:
            continue
        
        ann_row = ann_rows.iloc[0]
        prev_row = prev_rows.iloc[0]
        
        vix_after = ann_row.get('vix', np.nan)
        vix_before = prev_row.get('vix', np.nan)
        
        if pd.isna(vix_before) or pd.isna(vix_after):
            continue
        
        vix_change = vix_after - vix_before
        unemp_surprise = unemp.iloc[i]['unemp_value'] - unemp.iloc[i-1]['unemp_value']
        
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
        
        events.append({
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
        })
    
    df = pd.DataFrame(events)
    df = df.dropna(subset=['vix_change', 'unemp_surprise_abs'])
    df = df.sort_values('date').reset_index(drop=True)
    
    n_test = int(len(df) * 0.30)
    train_df = df.iloc[:-n_test].copy()
    test_df = df.iloc[-n_test:].copy()
    
    return train_df, test_df, features, "vix_change", "UNEMPLOYMENT", "VIX"


def load_cpi_yield_dataset():
    """Load CPI -> 10Y yield event dataset."""
    from src.data.fred_loader import load_series
    from src.data.merge_panel import build_fed_panel
    from src.data.inflation_announcements_loader import load_inflation_announcements

    panel = build_fed_panel()
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel.sort_values('date').reset_index(drop=True)

    cpi_dates = load_inflation_announcements()
    cpi_dates['release_date'] = pd.to_datetime(cpi_dates['release_date'])

    cpi_raw = load_series("CPIAUCSL").reset_index()
    cpi_raw.columns = ['date', 'cpi_value']
    cpi_raw['date'] = pd.to_datetime(cpi_raw['date'])
    cpi_raw['year_month'] = cpi_raw['date'].dt.to_period('M')
    cpi_monthly = cpi_raw.groupby('year_month')['cpi_value'].last().reset_index()
    cpi_monthly['data_period'] = cpi_monthly['year_month'].astype(str)
    cpi_dict = dict(zip(cpi_monthly['data_period'], cpi_monthly['cpi_value']))

    def get_prev_month(period):
        year, month = map(int, period.split('-'))
        if month == 1:
            return f"{year - 1}-12"
        return f"{year}-{month - 1:02d}"

    features = [
        "cpi_shock_abs",
        "yield_vol_10y",
        "slope_10y_2y",
        "fed_funds",
        "expinf_1y",
        "y_10y_before",
        "stlfsi",
    ]

    events = []
    for idx, ann in cpi_dates.iterrows():
        ann_date = ann['release_date']
        data_period = ann['data_period']

        ann_rows = panel[panel['date'] == ann_date]
        prev_rows = panel[panel['date'] < ann_date].tail(1)

        if ann_rows.empty or prev_rows.empty:
            continue

        ann_row = ann_rows.iloc[0]
        prev_row = prev_rows.iloc[0]

        y_after = ann_row.get('y_10y', np.nan)
        y_before = prev_row.get('y_10y', np.nan)

        if pd.isna(y_before) or pd.isna(y_after):
            continue

        y_change = y_after - y_before

        prev_period = get_prev_month(data_period)
        cpi_current = cpi_dict.get(data_period)
        cpi_prev = cpi_dict.get(prev_period)
        cpi_shock = (cpi_current - cpi_prev) if cpi_current and cpi_prev else None

        recent = panel[panel['date'] < ann_date].tail(20)
        yield_vol_10y = np.nan
        if len(recent) > 5 and 'y_10y' in recent.columns:
            vals = recent['y_10y'].dropna().values
            if len(vals) > 5:
                yield_vol_10y = np.std(np.diff(vals))

        events.append({
            'date': ann_date,
            'y_change': y_change,
            'cpi_shock': cpi_shock,
            'cpi_shock_abs': abs(cpi_shock) if cpi_shock else np.nan,
            'yield_vol_10y': yield_vol_10y,
            'slope_10y_2y': prev_row.get('slope_10y_2y', np.nan),
            'fed_funds': prev_row.get('fed_funds', np.nan),
            'expinf_1y': prev_row.get('expinf_1y', np.nan),
            'y_10y_before': y_before,
            'stlfsi': prev_row.get('stlfsi', np.nan),
        })

    df = pd.DataFrame(events)
    df = df.dropna(subset=['y_change', 'cpi_shock_abs'])
    df = df.sort_values('date').reset_index(drop=True)
    
    n_test = int(len(df) * 0.30)
    train_df = df.iloc[:-n_test].copy()
    test_df = df.iloc[-n_test:].copy()
    
    return train_df, test_df, features, "y_change", "CPI", "YIELD_10Y"


def load_cpi_predicted_yield_dataset():
    """Load CPI_PREDICTED -> 10Y yield event dataset."""
    from src.data.fred_loader import load_series
    from src.data.merge_panel import build_fed_panel
    from src.data.inflation_announcements_loader import load_inflation_announcements

    panel = build_fed_panel()
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel.sort_values('date').reset_index(drop=True)

    cpi_dates = load_inflation_announcements()
    cpi_dates['release_date'] = pd.to_datetime(cpi_dates['release_date'])

    cpi_raw = load_series("CPIAUCSL").reset_index()
    cpi_raw.columns = ['date', 'cpi_value']
    cpi_raw['date'] = pd.to_datetime(cpi_raw['date'])
    cpi_raw['year_month'] = cpi_raw['date'].dt.to_period('M')
    cpi_monthly = cpi_raw.groupby('year_month')['cpi_value'].last().reset_index()
    cpi_monthly['data_period'] = cpi_monthly['year_month'].astype(str)
    cpi_dict = dict(zip(cpi_monthly['data_period'], cpi_monthly['cpi_value']))

    def get_prev_month(period):
        year, month = map(int, period.split('-'))
        if month == 1:
            return f"{year - 1}-12"
        return f"{year}-{month - 1:02d}"

    features = [
        "expinf_change",
        "yield_vol_10y",
        "slope_10y_2y",
        "fed_funds",
        "expinf_1y",
        "y_10y_before",
        "stlfsi",
    ]

    events = []
    for idx, ann in cpi_dates.iterrows():
        ann_date = ann['release_date']
        data_period = ann['data_period']

        ann_rows = panel[panel['date'] == ann_date]
        prev_rows = panel[panel['date'] < ann_date].tail(1)

        if ann_rows.empty or prev_rows.empty:
            continue

        ann_row = ann_rows.iloc[0]
        prev_row = prev_rows.iloc[0]

        expinf_after = ann_row.get('expinf_1y', np.nan)
        expinf_before = prev_row.get('expinf_1y', np.nan)
        y_after = ann_row.get('y_10y', np.nan)
        y_before = prev_row.get('y_10y', np.nan)

        if pd.isna(expinf_before) or pd.isna(expinf_after) or pd.isna(y_before) or pd.isna(y_after):
            continue

        expinf_change = expinf_after - expinf_before
        y_change = y_after - y_before

        prev_period = get_prev_month(data_period)
        cpi_current = cpi_dict.get(data_period)
        cpi_prev = cpi_dict.get(prev_period)
        cpi_shock = (cpi_current - cpi_prev) if cpi_current and cpi_prev else None

        recent = panel[panel['date'] < ann_date].tail(20)
        yield_vol_10y = np.nan
        if len(recent) > 5 and 'y_10y' in recent.columns:
            vals = recent['y_10y'].dropna().values
            if len(vals) > 5:
                yield_vol_10y = np.std(np.diff(vals))

        events.append({
            'date': ann_date,
            'y_change': y_change,
            'expinf_change': expinf_change,
            'cpi_shock': cpi_shock,
            'cpi_shock_abs': abs(cpi_shock) if cpi_shock else np.nan,
            'yield_vol_10y': yield_vol_10y,
            'slope_10y_2y': prev_row.get('slope_10y_2y', np.nan),
            'fed_funds': prev_row.get('fed_funds', np.nan),
            'expinf_1y': expinf_before,
            'y_10y_before': y_before,
            'stlfsi': prev_row.get('stlfsi', np.nan),
        })

    df = pd.DataFrame(events)
    df = df.dropna(subset=['y_change', 'expinf_change', 'cpi_shock_abs'])
    df = df.sort_values('date').reset_index(drop=True)

    n_test = int(len(df) * 0.30)
    train_df = df.iloc[:-n_test].copy()
    test_df = df.iloc[-n_test:].copy()

    return train_df, test_df, features, "y_change", "CPI_PREDICTED", "YIELD_10Y"


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(model_type: str, weight, X: np.ndarray, y: np.ndarray):
    """Train a model of the specified type."""
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        AdaBoostClassifier,
        ExtraTreesClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    
    configs = {
        'RF': (RandomForestClassifier, {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}, True),
        'RF_deep': (RandomForestClassifier, {'n_estimators': 200, 'max_depth': 8, 'random_state': 42}, True),
        'RF_shallow': (RandomForestClassifier, {'n_estimators': 100, 'max_depth': 3, 'random_state': 42}, True),
        'ExtraTrees': (ExtraTreesClassifier, {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}, True),
        'GB': (GradientBoostingClassifier, {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'random_state': 42}, False),
        'GB_deep': (GradientBoostingClassifier, {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'random_state': 42}, False),
        'GB_fast': (GradientBoostingClassifier, {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.2, 'random_state': 42}, False),
        'AdaBoost': (AdaBoostClassifier, {'n_estimators': 100, 'learning_rate': 0.5, 'random_state': 42}, False),
        'LogReg': (LogisticRegression, {'max_iter': 1000, 'random_state': 42}, True),
    }
    
    if model_type not in configs:
        raise ValueError(f"Unknown model type: {model_type}")
    
    cls, params, uses_class_weight = configs[model_type]
    params = params.copy()
    
    if uses_class_weight:
        if weight == 'balanced':
            params['class_weight'] = 'balanced'
        else:
            params['class_weight'] = {False: 1, True: weight}
        model = cls(**params)
        model.fit(X, y)
    else:
        model = cls(**params)
        if weight == 'balanced':
            n_pos = y.sum()
            n_neg = len(y) - n_pos
            w = n_neg / n_pos if n_pos > 0 else 1
            sample_weights = np.where(y, w, 1)
        else:
            sample_weights = np.where(y, weight, 1)
        model.fit(X, y, sample_weight=sample_weights)
    
    return model


# ============================================================================
# MAIN
# ============================================================================

def list_candidates(mapping: str, search_dir: Path, fn_max: float = None):
    """List best candidates from grid search."""
    search_file = search_dir / f"grid_search_{mapping.replace('->', '_')}.csv"
    
    if not search_file.exists():
        print(f"No grid search results found: {search_file}")
        print("Run: python scripts/run_grid_search.py")
        return
    
    results = pd.read_csv(search_file)
    
    fn_constraints = [0.01, 0.05] if fn_max is None else [fn_max]
    
    for fn_max in fn_constraints:
        print(f"\n{'='*70}")
        print(f"FN ≤ {fn_max*100:.0f}% CONFIGURATIONS")
        print("=" * 70)
        
        best = find_best_configs(results, max_fn_rate=fn_max)
        
        if len(best) == 0:
            print("No configurations meet this constraint")
            continue
        
        # Show top 5
        print(f"{'Rank':<5} {'Model':<12} {'Weight':<8} {'Thresh':<8} {'Prob':<6} {'AUC':<6} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'TN/FP':>8}")
        print("-" * 90)
        
        for i, (_, row) in enumerate(best.head(10).iterrows()):
            tn_fp = f"{row['tn_fp_ratio']:.2f}x" if row['tn_fp_ratio'] < 1e6 else "∞"
            print(f"{i+1:<5} {row['model']:<12} {str(row['weight']):<8} {row['large_threshold']:<8.2f} "
                  f"{row['prob_threshold']:<6.2f} {row['auc']:<6.3f} {int(row['TP']):>4} {int(row['FP']):>4} "
                  f"{int(row['FN']):>4} {int(row['TN']):>4} {tn_fp:>8}")


def promote_from_config(
    mapping: str,
    model_type: str,
    weight,
    large_threshold: float,
    prob_cutoff: float,
    model_id: str,
    registry_dir: Path,
    search_dir: Path,
):
    """Promote a specific model configuration."""
    
    print(f"\nPromoting model: {model_id}")
    print(f"  Mapping: {mapping}")
    print(f"  Config: {model_type}, weight={weight}, thresh={large_threshold}, prob={prob_cutoff}")
    
    # Load data
    if mapping == "CPI->HY_OAS":
        train_df, test_df, features, target_col, event, instrument = load_cpi_hy_dataset()
    elif mapping == "UNEMP->VIX":
        train_df, test_df, features, target_col, event, instrument = load_unemp_vix_dataset()
    elif mapping == "CPI->YIELD_10Y":
        train_df, test_df, features, target_col, event, instrument = load_cpi_yield_dataset()
    elif mapping == "CPI_PREDICTED->YIELD_10Y":
        train_df, test_df, features, target_col, event, instrument = load_cpi_predicted_yield_dataset()
    else:
        raise ValueError(f"Unknown mapping: {mapping}")
    
    # Define large move
    train_df['is_large'] = train_df[target_col].abs() >= large_threshold
    test_df['is_large'] = test_df[target_col].abs() >= large_threshold
    
    # Prepare features
    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    
    feature_medians = X_train.median().to_dict()
    X_train = X_train.fillna(feature_medians)
    X_test = X_test.fillna(feature_medians)
    
    y_train = train_df['is_large'].values
    y_test = test_df['is_large'].values
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("  Training model...")
    model = train_model(model_type, weight, X_train_scaled, y_train)
    
    # Evaluate
    probs = model.predict_proba(X_test_scaled)[:, 1]
    preds = probs >= prob_cutoff
    
    tp = int(((y_test == True) & (preds == True)).sum())
    fp = int(((y_test == False) & (preds == True)).sum())
    fn = int(((y_test == True) & (preds == False)).sum())
    tn = int(((y_test == False) & (preds == False)).sum())
    
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, probs)
    
    n_test_pos = int(y_test.sum())
    n_test_neg = int((~y_test).sum())
    
    fn_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    tn_fp_ratio = tn / fp if fp > 0 else None
    
    print(f"  Results: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  AUC={auc:.3f}, FN%={fn_rate*100:.1f}%, TN/FP={tn_fp_ratio:.2f}x" if tn_fp_ratio else f"  AUC={auc:.3f}, FN%={fn_rate*100:.1f}%")
    
    # Create metadata
    metadata = ModelMetadata(
        model_id=model_id,
        model_type=model_type,
        weight=weight,
        features=features,
        feature_medians=feature_medians,
        target_event=event,
        target_instrument=instrument,
        target_col=target_col,
        large_move_threshold=large_threshold,
        large_move_definition=f"abs({target_col}) >= {large_threshold}",
        prob_cutoff=prob_cutoff,
        train_period=(str(train_df['date'].min().date()), str(train_df['date'].max().date())),
        test_period=(str(test_df['date'].min().date()), str(test_df['date'].max().date())),
        n_train=len(train_df),
        n_test=len(test_df),
        n_test_pos=n_test_pos,
        n_test_neg=n_test_neg,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        auc=auc,
        fn_rate=fn_rate,
        fp_rate=fp_rate,
        tn_fp_ratio=tn_fp_ratio,
        created_at=datetime.utcnow().isoformat() + "Z",
        created_by="promote_model.py",
        source=f"grid_search_{datetime.now().strftime('%Y-%m-%d')}",
        notes=f"n_test_pos={n_test_pos}",
    )
    
    # Promote to registry
    registry = ModelRegistry(registry_dir)
    registry.promote_model(
        model_id=model_id,
        model=model,
        metadata=metadata,
        scaler=scaler,
        overwrite=True,
    )
    
    print(f"\n✅ Model promoted: {model_id}")
    print(f"   Add to registry/edge_config.yaml to use in graph")
    
    return model_id


def main():
    parser = argparse.ArgumentParser(description="Promote model to registry")
    parser.add_argument("--list", action="store_true", help="List candidates from grid search")
    parser.add_argument("--mapping", type=str, required=True, help="Mapping (e.g. CPI->HY_OAS)")
    parser.add_argument("--model", type=str, help="Model type (e.g. RF_shallow)")
    parser.add_argument("--weight", type=str, help="Weight (int or 'balanced')")
    parser.add_argument("--threshold", type=float, help="Large move threshold")
    parser.add_argument("--prob", type=float, help="Probability cutoff")
    parser.add_argument("--id", type=str, help="Model ID for registry")
    parser.add_argument("--fn-max", type=float, help="Promote best config for this FN constraint")
    parser.add_argument("--search-dir", type=str, default="registry/search_results")
    parser.add_argument("--registry-dir", type=str, default="registry/models")
    args = parser.parse_args()
    
    search_dir = project_root / args.search_dir
    registry_dir = project_root / args.registry_dir
    
    if args.list:
        list_candidates(args.mapping, search_dir, args.fn_max)
        return 0
    
    # Check required args for promotion
    if args.fn_max:
        # Auto-select best config for FN constraint
        search_file = search_dir / f"grid_search_{args.mapping.replace('->', '_')}.csv"
        if not search_file.exists():
            print(f"Error: No grid search results: {search_file}")
            return 1
        
        results = pd.read_csv(search_file)
        best = find_best_configs(results, max_fn_rate=args.fn_max)
        
        if len(best) == 0:
            print(f"Error: No configs meet FN≤{args.fn_max*100:.0f}%")
            return 1
        
        top = best.iloc[0]
        model_type = top['model']
        weight = top['weight']
        threshold = top['large_threshold']
        prob = top['prob_threshold']
        
        if not args.id:
            fn_pct = int(args.fn_max * 100)
            args.id = f"{args.mapping.lower().replace('->', '_')}_fn{fn_pct}pct_best"
        
    else:
        if not all([args.model, args.weight, args.threshold, args.prob, args.id]):
            print("Error: Need --model, --weight, --threshold, --prob, --id")
            print("Or use --fn-max to auto-select best config")
            return 1
        
        model_type = args.model
        weight = int(args.weight) if args.weight != 'balanced' else 'balanced'
        threshold = args.threshold
        prob = args.prob
    
    promote_from_config(
        mapping=args.mapping,
        model_type=model_type,
        weight=weight,
        large_threshold=threshold,
        prob_cutoff=prob,
        model_id=args.id,
        registry_dir=registry_dir,
        search_dir=search_dir,
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
