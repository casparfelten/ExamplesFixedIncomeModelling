#!/usr/bin/env python3
"""
Train the CPI Large Move Predictor

This script:
1. Loads the event data
2. Creates train/val/test splits
3. Trains the Cost-Sensitive Random Forest
4. Selects threshold via Time Series CV
5. Saves the trained model

Usage:
    python -m predictors.cpi_large_move.train
    
    # Or from project root:
    python predictors/cpi_large_move/train.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from src.models.prepare_data import prepare_event_data, create_train_test_split
from predictors.cpi_large_move.model import CPILargeMovePredictor


def main():
    print("=" * 60)
    print("Training CPI Large Move Predictor")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    df = prepare_event_data()
    print(f"   Total events: {len(df)}")
    
    # Split data (70% train+val, 30% test)
    train_val_df, test_df = create_train_test_split(df, test_size=0.30)
    print(f"   Train+Val: {len(train_val_df)} | Test: {len(test_df)}")
    print(f"   Train+Val period: {train_val_df['date'].min()} to {train_val_df['date'].max()}")
    print(f"   Test period: {test_df['date'].min()} to {test_df['date'].max()}")
    
    # Define target and features
    # y_2y_change is the 2Y yield change around CPI announcement
    def is_abnormal(row):
        return abs(row['y_2y_change']) > 0.10
    
    feature_cols = ['yield_volatility', 'cpi_shock_mom', 'fed_funds', 'slope_10y_2y', 'unemployment']
    
    # Prepare train+val data
    print("\n2. Preparing features...")
    train_val_df = train_val_df.dropna(subset=feature_cols + ['y_2y_change'])
    test_df = test_df.dropna(subset=feature_cols + ['y_2y_change'])
    
    # Create feature matrices
    X_train_val = train_val_df[feature_cols].copy()
    X_train_val['cpi_abs'] = X_train_val['cpi_shock_mom'].abs()
    X_train_val = X_train_val[['yield_volatility', 'cpi_shock_mom', 'cpi_abs', 
                                'fed_funds', 'slope_10y_2y', 'unemployment']].values
    y_train_val = train_val_df.apply(is_abnormal, axis=1).values
    
    X_test = test_df[feature_cols].copy()
    X_test['cpi_abs'] = X_test['cpi_shock_mom'].abs()
    X_test = X_test[['yield_volatility', 'cpi_shock_mom', 'cpi_abs',
                     'fed_funds', 'slope_10y_2y', 'unemployment']].values
    y_test = test_df.apply(is_abnormal, axis=1).values
    
    print(f"   Features: {feature_cols + ['cpi_abs']}")
    print(f"   Train+Val abnormals: {y_train_val.sum()} / {len(y_train_val)}")
    print(f"   Test abnormals: {y_test.sum()} / {len(y_test)}")
    
    # Compute medians from train+val
    medians = {
        'yield_volatility': float(np.nanmedian(X_train_val[:, 0])),
        'cpi_shock_mom': float(np.nanmedian(X_train_val[:, 1])),
        'cpi_abs': float(np.nanmedian(X_train_val[:, 2])),
        'fed_funds': float(np.nanmedian(X_train_val[:, 3])),
        'slope_10y_2y': float(np.nanmedian(X_train_val[:, 4])),
        'unemployment': float(np.nanmedian(X_train_val[:, 5])),
    }
    print(f"   Medians computed from train+val only")
    
    # Fit scaler on train+val
    print("\n3. Fitting scaler on train+val...")
    scaler = StandardScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Select threshold via Time Series CV
    print("\n4. Selecting threshold via Time Series CV...")
    tscv = TimeSeriesSplit(n_splits=5)
    thresholds_per_fold = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val_scaled)):
        X_tr, X_va = X_train_val_scaled[train_idx], X_train_val_scaled[val_idx]
        y_tr, y_va = y_train_val[train_idx], y_train_val[val_idx]
        
        # Train model on fold
        model_fold = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight={False: 1, True: 50},
            random_state=42,
        )
        model_fold.fit(X_tr, y_tr)
        
        # Get probabilities for validation
        proba_va = model_fold.predict_proba(X_va)[:, 1]
        
        # Find threshold for FN=0
        if y_va.sum() > 0:
            min_abnormal_prob = proba_va[y_va].min()
            thresholds_per_fold.append(min_abnormal_prob)
            print(f"   Fold {fold+1}: threshold={min_abnormal_prob:.4f} (abnormals={y_va.sum()})")
        else:
            print(f"   Fold {fold+1}: no abnormals in validation")
    
    # Use maximum threshold (most conservative)
    threshold = max(thresholds_per_fold) if thresholds_per_fold else 0.10
    print(f"   Selected threshold: {threshold:.4f} (max across folds)")
    
    # Train final model on all train+val
    print("\n5. Training final model on all train+val...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight={False: 1, True: 50},
        random_state=42,
    )
    model.fit(X_train_val_scaled, y_train_val)
    
    # Evaluate on test
    print("\n6. Evaluating on held-out test set...")
    proba_test = model.predict_proba(X_test_scaled)[:, 1]
    pred_test = proba_test >= threshold
    
    # Compute metrics
    abnormal_mask = y_test == True
    normal_mask = y_test == False
    
    fn = ((pred_test == False) & abnormal_mask).sum()
    tp = ((pred_test == True) & abnormal_mask).sum()
    fp = ((pred_test == True) & normal_mask).sum()
    tn = ((pred_test == False) & normal_mask).sum()
    
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\n   Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                    Normal    Abnormal")
    print(f"   Actual Normal    {tn:<9} {fp}")
    print(f"   Actual Abnormal  {fn:<9} {tp}")
    print(f"\n   False Negative Rate: {fn_rate*100:.1f}%")
    print(f"   False Positive Rate: {fp_rate*100:.1f}%")
    
    # Create and save predictor
    print("\n7. Saving trained model...")
    predictor = CPILargeMovePredictor(
        model=model,
        scaler=scaler,
        medians=medians,
        threshold=threshold,
    )
    predictor.save()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    # Quick test
    print("\n8. Quick test...")
    result = predictor.predict({
        'yield_volatility': 0.05,
        'cpi_shock_mom': 0.1,
        'fed_funds': 2.5,
        'slope_10y_2y': 1.0,
        'unemployment': 4.0,
    })
    print(f"   Test prediction: {result}")
    
    return predictor


if __name__ == '__main__':
    main()

