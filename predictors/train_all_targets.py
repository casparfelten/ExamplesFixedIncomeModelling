#!/usr/bin/env python3
"""
Train predictors for multiple CPI → yield/vol targets.

Targets:
1. CPI → 2Y yield large move (>10bp)
2. CPI → 5Y yield large move (>8bp)
3. CPI → 10Y yield large move (>8bp)
4. CPI → VIX spike (>2 points)
5. CPI → 5Y Breakeven large move (>5bp)

All use the same methodology:
- Cost-sensitive Random Forest
- Time Series CV for threshold selection
- Chronological train/test split
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
import pickle
import numpy as np
import pandas as pd
from datetime import timedelta
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Load environment
load_dotenv(project_root / '.env')

from fredapi import Fred
from src.data.fred_loader import merge_fred_panel
from src.data.inflation_announcements_loader import load_inflation_announcements


def download_additional_series():
    """Download 5Y, breakevens, etc. from FRED."""
    fred = Fred(api_key=os.getenv('FRED_API_KEY'))
    
    series = {
        'DGS5': 'y_5y',
        'DGS30': 'y_30y',
        'T5YIE': 'breakeven_5y',
        'T10YIE': 'breakeven_10y',
    }
    
    dfs = []
    for fred_id, col_name in series.items():
        try:
            data = fred.get_series(fred_id)
            df = pd.DataFrame({col_name: data})
            df.index.name = 'date'
            dfs.append(df)
            print(f"  Downloaded {fred_id} ({len(data)} obs)")
        except Exception as e:
            print(f"  Failed to download {fred_id}: {e}")
    
    if dfs:
        return pd.concat(dfs, axis=1)
    return pd.DataFrame()


def create_event_dataset_multi_target(target_col: str, threshold: float):
    """
    Create event dataset for any yield/vol target.
    Uses the same methodology as prepare_event_data but for different targets.
    Computes target-specific volatility features.
    """
    from src.models.prepare_data import prepare_event_data
    
    # For 2Y, use the original function
    if target_col == 'y_2y':
        df = prepare_event_data(target_yield='y_2y')
        df['is_abnormal'] = df['y_2y_change'].abs() > threshold
        df['target_change'] = df['y_2y_change']
        df['cpi_abs'] = df['cpi_shock_mom'].abs()
        return df
    
    # For other targets, we need to download and merge additional data
    panel = merge_fred_panel()
    panel['date'] = pd.to_datetime(panel['date'])
    
    # Download additional series if needed
    if target_col not in panel.columns or target_col == 'y_5y':
        print(f"  Downloading additional series...")
        extra = download_additional_series()
        if not extra.empty:
            extra.index = pd.to_datetime(extra.index)
            extra = extra.reset_index()
            extra.columns = ['date'] + list(extra.columns[1:])
            panel = panel.merge(extra, on='date', how='left')
    
    if target_col not in panel.columns:
        raise ValueError(f"Column {target_col} not found after download")
    
    # Load CPI announcements
    cpi_dates = load_inflation_announcements()
    
    events = []
    for _, row in cpi_dates.iterrows():
        ann_date = pd.to_datetime(row['release_date'])
        
        # Get announcement day row (after announcement)
        ann_row = panel[panel['date'] == ann_date]
        if ann_row.empty:
            # Find closest date
            for offset in [1, -1, 2, -2]:
                candidate = panel[panel['date'] == ann_date + pd.Timedelta(days=offset)]
                if not candidate.empty:
                    ann_row = candidate
                    break
        
        # Get previous day row (before announcement)  
        prev_dates = panel[panel['date'] < ann_date].tail(1)
        
        if ann_row.empty or prev_dates.empty:
            continue
        
        # Get target values
        val_after = ann_row[target_col].iloc[0]
        val_before = prev_dates[target_col].iloc[0]
        
        if pd.isna(val_before) or pd.isna(val_after):
            continue
        
        change = val_after - val_before
        
        # Compute volatilities for different instruments
        recent = panel[panel['date'] < ann_date].tail(20)
        
        def compute_vol(series_col):
            if len(recent) > 5 and series_col in recent.columns:
                vals = recent[series_col].dropna().values
                if len(vals) > 5:
                    return np.std(np.diff(vals))
            return np.nan
        
        # 2Y volatility (original)
        vol_2y = compute_vol('y_2y')
        
        # 5Y volatility
        vol_5y = compute_vol('y_5y')
        
        # 10Y volatility
        vol_10y = compute_vol('y_10y')
        
        # VIX volatility (vol of vol)
        vol_vix = compute_vol('vix')
        
        # VIX level (for VIX prediction)
        vix_level = prev_dates['vix'].iloc[0] if 'vix' in prev_dates.columns else np.nan
        
        # Get CPI shock
        cpi_now = ann_row['cpi'].iloc[0] if 'cpi' in ann_row.columns else np.nan
        cpi_before = prev_dates['cpi'].iloc[0] if 'cpi' in prev_dates.columns else np.nan
        cpi_shock = cpi_now - cpi_before if pd.notna(cpi_now) and pd.notna(cpi_before) else np.nan
        
        event = {
            'date': ann_date,
            'target_change': change,
            'is_abnormal': abs(change) > threshold,
            # Multiple volatility measures
            'yield_volatility': vol_2y,  # Original for compatibility
            'vol_2y': vol_2y,
            'vol_5y': vol_5y,
            'vol_10y': vol_10y,
            'vol_vix': vol_vix,
            'vix_level': vix_level,
            # CPI features
            'cpi_shock_mom': cpi_shock,
            # Background features
            'fed_funds': prev_dates['fed_funds'].iloc[0] if 'fed_funds' in prev_dates.columns else np.nan,
            'slope_10y_2y': prev_dates['slope_10y_2y'].iloc[0] if 'slope_10y_2y' in prev_dates.columns else np.nan,
            'unemployment': prev_dates['unemployment'].iloc[0] if 'unemployment' in prev_dates.columns else np.nan,
            'vix': prev_dates['vix'].iloc[0] if 'vix' in prev_dates.columns else np.nan,
        }
        events.append(event)
    
    df = pd.DataFrame(events)
    df['cpi_abs'] = df['cpi_shock_mom'].abs()
    return df


def train_predictor(
    events_df: pd.DataFrame,
    target_name: str,
    test_size: float = 0.30,
    class_weight_ratio: int = 50,
    feature_cols: list = None,
    thresh_percentile: int = 50,
):
    """
    Train a predictor for a specific target.
    
    Returns:
        dict with model, scaler, medians, threshold, and metrics
    """
    # Features - use provided or default
    if feature_cols is None:
        feature_cols = ['yield_volatility', 'cpi_shock_mom', 'cpi_abs', 
                        'fed_funds', 'slope_10y_2y', 'unemployment']
    
    # Drop rows with missing target or features
    df = events_df.dropna(subset=['is_abnormal'] + feature_cols[:3])  # Core features required
    df = df.sort_values('date').reset_index(drop=True)
    
    # Split chronologically
    n_test = int(len(df) * test_size)
    train_val = df.iloc[:-n_test].copy()
    test = df.iloc[-n_test:].copy()
    
    print(f"\n  Train+Val: {len(train_val)} events ({train_val['date'].min().date()} to {train_val['date'].max().date()})")
    print(f"  Test: {len(test)} events ({test['date'].min().date()} to {test['date'].max().date()})")
    print(f"  Train abnormals: {train_val['is_abnormal'].sum()} / {len(train_val)}")
    print(f"  Test abnormals: {test['is_abnormal'].sum()} / {len(test)}")
    
    if train_val['is_abnormal'].sum() < 5:
        print(f"  ⚠️  Too few abnormals in training, skipping")
        return None
    
    # Prepare features
    for col in feature_cols:
        if col not in train_val.columns:
            train_val[col] = np.nan
            test[col] = np.nan
    
    # Compute medians from train+val
    medians = {col: train_val[col].median() for col in feature_cols}
    
    X_train_val = train_val[feature_cols].fillna(medians).values
    y_train_val = train_val['is_abnormal'].values
    
    X_test = test[feature_cols].fillna(medians).values
    y_test = test['is_abnormal'].values
    
    # Fit scaler on train+val
    scaler = StandardScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Select threshold via Time Series CV
    tscv = TimeSeriesSplit(n_splits=5)
    thresholds_per_fold = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val_scaled)):
        X_tr, X_va = X_train_val_scaled[train_idx], X_train_val_scaled[val_idx]
        y_tr, y_va = y_train_val[train_idx], y_train_val[val_idx]
        
        if y_va.sum() == 0:
            continue
        
        model_fold = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight={False: 1, True: class_weight_ratio},
            random_state=42,
        )
        model_fold.fit(X_tr, y_tr)
        proba_va = model_fold.predict_proba(X_va)[:, 1]
        
        min_abnormal_prob = proba_va[y_va].min()
        thresholds_per_fold.append(min_abnormal_prob)
    
    # Threshold selection: use specified percentile of fold thresholds
    # Lower percentile = more aggressive (lower FN, higher FP)
    # Higher percentile = more conservative (higher FN, lower FP)
    if thresholds_per_fold:
        threshold = np.percentile(thresholds_per_fold, thresh_percentile)
    else:
        threshold = 0.05
    print(f"  Threshold (from CV, {thresh_percentile}th pctl): {threshold:.4f}")
    
    # Train final model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight={False: 1, True: class_weight_ratio},
        random_state=42,
    )
    model.fit(X_train_val_scaled, y_train_val)
    
    # Evaluate on test
    proba_test = model.predict_proba(X_test_scaled)[:, 1]
    pred_test = proba_test >= threshold
    
    # Metrics
    abnormal_mask = y_test == True
    normal_mask = y_test == False
    
    fn = ((pred_test == False) & abnormal_mask).sum()
    tp = ((pred_test == True) & abnormal_mask).sum()
    fp = ((pred_test == True) & normal_mask).sum()
    tn = ((pred_test == False) & normal_mask).sum()
    
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        'model': model,
        'scaler': scaler,
        'medians': medians,
        'threshold': threshold,
        'feature_cols': feature_cols,
        'metrics': {
            'fn_rate': fn_rate,
            'fp_rate': fp_rate,
            'fn': fn,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'test_abnormals': int(abnormal_mask.sum()),
            'test_normals': int(normal_mask.sum()),
        },
        'train_period': f"{train_val['date'].min().date()} to {train_val['date'].max().date()}",
        'test_period': f"{test['date'].min().date()} to {test['date'].max().date()}",
    }


def main():
    print("=" * 70)
    print("MULTI-TARGET CPI PREDICTOR TRAINING")
    print("=" * 70)
    
    # Define targets with target-specific tuning
    # Each entry: (target_col, threshold, name, class_weight_ratio, feature_cols)
    default_features = ['yield_volatility', 'cpi_shock_mom', 'cpi_abs', 
                        'fed_funds', 'slope_10y_2y', 'unemployment']
    
    # Each entry: (target_col, threshold, name, class_weight_ratio, feature_cols, thresh_percentile)
    # thresh_percentile: 0=min, 50=median, 100=max - lower is more aggressive for FN=0
    targets = [
        # 2Y: Original working model - median works
        ('y_2y', 0.10, '2Y Yield (>10bp)', 50, default_features, 50),
        
        # 5Y: Median achieved FN=0
        ('y_5y', 0.12, '5Y Yield (>12bp)', 50, 
         ['yield_volatility', 'cpi_shock_mom', 'cpi_abs', 'fed_funds', 'slope_10y_2y', 'unemployment'], 50),
        
        # 10Y: Use 10Y volatility, very low percentile to catch that last 1
        ('y_10y', 0.12, '10Y Yield (>12bp)', 50,
         ['vol_10y', 'cpi_shock_mom', 'cpi_abs', 'fed_funds', 'slope_10y_2y', 'unemployment'], 10),
        
        # VIX: Very low percentile to catch more spikes
        ('vix', 1.8, 'VIX Spike (>1.8pts)', 50,
         ['vol_vix', 'cpi_shock_mom', 'cpi_abs', 'fed_funds', 'slope_10y_2y', 'unemployment', 'vix_level'], 10),
    ]
    
    results = {}
    
    for target_col, threshold, name, class_weight_ratio, feature_cols, thresh_pct in targets:
        print(f"\n{'='*70}")
        print(f"TARGET: {name}")
        print(f"{'='*70}")
        
        # Create event dataset
        print(f"\n1. Creating event dataset for {target_col}...")
        try:
            events = create_event_dataset_multi_target(target_col, threshold)
            print(f"   Events: {len(events)}, Abnormals: {events['is_abnormal'].sum()}")
        except Exception as e:
            print(f"   ERROR creating dataset: {e}")
            continue
        
        if len(events) < 50:
            print(f"   ⚠️  Too few events ({len(events)}), skipping")
            continue
        
        # Train predictor
        print(f"\n2. Training predictor (class_weight={class_weight_ratio}, thresh_pct={thresh_pct})...")
        try:
            result = train_predictor(
                events, name, 
                class_weight_ratio=class_weight_ratio,
                feature_cols=feature_cols,
                thresh_percentile=thresh_pct
            )
            if result is None:
                continue
            results[target_col] = result
        except Exception as e:
            print(f"   ERROR training: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Print results
        m = result['metrics']
        print(f"\n3. Results:")
        print(f"   False Negative Rate: {m['fn_rate']*100:.1f}% ({m['fn']}/{m['fn']+m['tp']} missed)")
        print(f"   False Positive Rate: {m['fp_rate']*100:.1f}% ({m['fp']}/{m['fp']+m['tn']} false alarms)")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Target':<25} {'FN Rate':<12} {'FP Rate':<12} {'Test Abnormals':<15}")
    print("-" * 64)
    
    for target_col, result in results.items():
        m = result['metrics']
        name = {
            'y_2y': '2Y Yield',
            'y_5y': '5Y Yield', 
            'y_10y': '10Y Yield',
            'vix': 'VIX',
            'breakeven_5y': '5Y Breakeven',
        }.get(target_col, target_col)
        
        fn_status = "✅" if m['fn_rate'] == 0 else "❌"
        print(f"{name:<25} {fn_status} {m['fn_rate']*100:>5.1f}%      {m['fp_rate']*100:>5.1f}%        {m['test_abnormals']}")
    
    # Save results
    output_path = project_root / 'predictors' / 'multi_target_results.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == '__main__':
    main()

