"""
Grid Search for Event-Based Macro Shock Detection

This module provides reusable functions for exhaustive grid search
over model types, weights, and thresholds for event-based classification.

Usage:
    from src.models.event_grid_search import run_full_grid_search, find_best_configs
    
    results = run_full_grid_search(
        X_train, y_train, X_test, y_test,
        large_thresholds=[0.10, 0.15, 0.20],
        target_col='hy_change'
    )
    
    best = find_best_configs(results, max_fn_rate=0.05)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier, 
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

MODEL_CONFIGS = {
    'RF': {
        'class': RandomForestClassifier,
        'params': {'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
        'uses_class_weight': True,
        'uses_sample_weight': False
    },
    'RF_deep': {
        'class': RandomForestClassifier,
        'params': {'n_estimators': 200, 'max_depth': 8, 'random_state': 42},
        'uses_class_weight': True,
        'uses_sample_weight': False
    },
    'RF_shallow': {
        'class': RandomForestClassifier,
        'params': {'n_estimators': 100, 'max_depth': 3, 'random_state': 42},
        'uses_class_weight': True,
        'uses_sample_weight': False
    },
    'ExtraTrees': {
        'class': ExtraTreesClassifier,
        'params': {'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
        'uses_class_weight': True,
        'uses_sample_weight': False
    },
    'GB': {
        'class': GradientBoostingClassifier,
        'params': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'random_state': 42},
        'uses_class_weight': False,
        'uses_sample_weight': True
    },
    'GB_deep': {
        'class': GradientBoostingClassifier,
        'params': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'random_state': 42},
        'uses_class_weight': False,
        'uses_sample_weight': True
    },
    'GB_fast': {
        'class': GradientBoostingClassifier,
        'params': {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.2, 'random_state': 42},
        'uses_class_weight': False,
        'uses_sample_weight': True
    },
    'AdaBoost': {
        'class': AdaBoostClassifier,
        'params': {'n_estimators': 100, 'learning_rate': 0.5, 'random_state': 42},
        'uses_class_weight': False,
        'uses_sample_weight': True
    },
    'LogReg': {
        'class': LogisticRegression,
        'params': {'max_iter': 1000, 'random_state': 42},
        'uses_class_weight': True,
        'uses_sample_weight': False
    }
}

# Default weight options
DEFAULT_WEIGHTS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 'balanced']

# Default probability thresholds to evaluate
DEFAULT_PROB_THRESHOLDS = np.arange(0.05, 0.95, 0.05)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_model(X_train, y_train, model_name, weight):
    """
    Train a single model with given weight configuration.
    
    Args:
        X_train: Scaled training features
        y_train: Training labels
        model_name: Key from MODEL_CONFIGS
        weight: Class weight (int or 'balanced')
    
    Returns:
        Trained model or None if failed
    """
    if model_name not in MODEL_CONFIGS:
        return None
    
    config = MODEL_CONFIGS[model_name]
    model_class = config['class']
    params = config['params'].copy()
    
    try:
        if config['uses_class_weight']:
            if weight == 'balanced':
                params['class_weight'] = 'balanced'
            else:
                params['class_weight'] = {False: 1, True: weight}
            model = model_class(**params)
            model.fit(X_train, y_train)
        
        elif config['uses_sample_weight']:
            model = model_class(**params)
            if weight == 'balanced':
                n_pos = y_train.sum()
                n_neg = len(y_train) - n_pos
                w = n_neg / n_pos if n_pos > 0 else 1
                sample_weights = np.where(y_train, w, 1)
            else:
                sample_weights = np.where(y_train, weight, 1)
            model.fit(X_train, y_train, sample_weight=sample_weights)
        
        else:
            model = model_class(**params)
            model.fit(X_train, y_train)
        
        return model
    
    except Exception as e:
        return None


def evaluate_at_threshold(probs, y_test, prob_threshold):
    """
    Evaluate predictions at a given probability threshold.
    
    Returns dict with TP, FP, FN, TN and derived metrics.
    """
    pred = probs >= prob_threshold
    
    n_pos = y_test.sum()
    n_neg = (~y_test).sum()
    
    TP = (y_test & pred).sum()
    FP = (~y_test & pred).sum()
    FN = (y_test & ~pred).sum()
    TN = (~y_test & ~pred).sum()
    
    return {
        'prob_threshold': prob_threshold,
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN),
        'TN': int(TN),
        'n_positive': int(n_pos),
        'n_negative': int(n_neg),
        'recall': TP / n_pos if n_pos > 0 else 0,
        'fn_rate': FN / n_pos if n_pos > 0 else 0,
        'specificity': TN / n_neg if n_neg > 0 else 0,
        'fp_rate': FP / n_neg if n_neg > 0 else 0,
        'precision': TP / (TP + FP) if (TP + FP) > 0 else 0,
        'tn_fp_ratio': TN / FP if FP > 0 else float('inf')
    }


def evaluate_model(model, X_test, y_test, prob_thresholds=None):
    """
    Evaluate a trained model at multiple probability thresholds.
    
    Returns:
        list of dicts with metrics at each threshold, plus AUC
    """
    if prob_thresholds is None:
        prob_thresholds = DEFAULT_PROB_THRESHOLDS
    
    try:
        if hasattr(model, 'classes_') and len(model.classes_) < 2:
            return None, 0.5
        
        probs = model.predict_proba(X_test)[:, 1]
        
        try:
            auc = roc_auc_score(y_test, probs)
        except:
            auc = 0.5
        
        results = []
        for prob_t in prob_thresholds:
            metrics = evaluate_at_threshold(probs, y_test, prob_t)
            results.append(metrics)
        
        return results, auc
    
    except Exception as e:
        return None, 0.5


# ============================================================================
# GRID SEARCH
# ============================================================================

def run_grid_search(
    X_train, y_train, X_test, y_test,
    model_names=None,
    weights=None,
    prob_thresholds=None,
    verbose=True
):
    """
    Run grid search over model types, weights, and probability thresholds.
    
    Args:
        X_train, y_train: Training data (already scaled)
        X_test, y_test: Test data (already scaled)
        model_names: List of model names (keys from MODEL_CONFIGS)
        weights: List of weight values to try
        prob_thresholds: Probability thresholds to evaluate
        verbose: Print progress
    
    Returns:
        DataFrame with all results
    """
    if model_names is None:
        model_names = list(MODEL_CONFIGS.keys())
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if prob_thresholds is None:
        prob_thresholds = DEFAULT_PROB_THRESHOLDS
    
    all_results = []
    total = len(model_names) * len(weights)
    count = 0
    
    for model_name in model_names:
        for weight in weights:
            count += 1
            if verbose and count % 20 == 0:
                print(f"  Progress: {count}/{total}")
            
            model = train_model(X_train, y_train, model_name, weight)
            if model is None:
                continue
            
            eval_results, auc = evaluate_model(model, X_test, y_test, prob_thresholds)
            if eval_results is None:
                continue
            
            for metrics in eval_results:
                all_results.append({
                    'model': model_name,
                    'weight': weight,
                    'auc': auc,
                    **metrics
                })
    
    return pd.DataFrame(all_results)


def run_full_grid_search(
    train_df, test_df, features, target_col,
    large_thresholds,
    model_names=None,
    weights=None,
    prob_thresholds=None,
    verbose=True
):
    """
    Run grid search across multiple large-move thresholds.
    
    Args:
        train_df, test_df: DataFrames with features and target
        features: List of feature column names
        target_col: Column name for the target change (e.g., 'hy_change')
        large_thresholds: List of absolute thresholds to define "large" moves
        model_names, weights, prob_thresholds: Grid search parameters
    
    Returns:
        DataFrame with all results including large_threshold column
    """
    all_results = []
    
    for large_thresh in large_thresholds:
        if verbose:
            print(f"\nLarge threshold: {large_thresh}")
        
        # Define large moves
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df['is_large'] = train_df[target_col].abs() >= large_thresh
        test_df['is_large'] = test_df[target_col].abs() >= large_thresh
        
        n_train_large = train_df['is_large'].sum()
        n_test_large = test_df['is_large'].sum()
        
        if verbose:
            print(f"  Train: {n_train_large} large / {len(train_df)} total")
            print(f"  Test: {n_test_large} large / {len(test_df)} total")
        
        if n_train_large < 5 or n_test_large < 2:
            if verbose:
                print("  Skipping: too few large events")
            continue
        
        # Prepare features
        X_train = train_df[features].fillna(train_df[features].median())
        X_test = test_df[features].fillna(train_df[features].median())
        y_train = train_df['is_large'].values
        y_test = test_df['is_large'].values
        
        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # Grid search
        results = run_grid_search(
            X_train_s, y_train, X_test_s, y_test,
            model_names=model_names,
            weights=weights,
            prob_thresholds=prob_thresholds,
            verbose=verbose
        )
        
        if len(results) > 0:
            results['large_threshold'] = large_thresh
            results['n_train_large'] = n_train_large
            results['n_test_large'] = n_test_large
            all_results.append(results)
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


# ============================================================================
# FINDING BEST CONFIGURATIONS
# ============================================================================

def find_best_configs(results_df, max_fn_rate=0.05, min_recall=None):
    """
    Find best configurations that meet FN rate constraint.
    
    Args:
        results_df: DataFrame from run_full_grid_search
        max_fn_rate: Maximum allowed FN rate (e.g., 0.05 for 5%)
        min_recall: Alternative to max_fn_rate (min_recall = 1 - max_fn_rate)
    
    Returns:
        DataFrame with best config for each large_threshold, sorted by TN/FP ratio
    """
    if min_recall is not None:
        max_fn_rate = 1 - min_recall
    
    # Filter to configs meeting FN constraint
    filtered = results_df[results_df['fn_rate'] <= max_fn_rate].copy()
    
    if len(filtered) == 0:
        print(f"No configs found with FN rate <= {max_fn_rate*100:.1f}%")
        return pd.DataFrame()
    
    # For each large_threshold, find best TN/FP ratio
    best_configs = []
    
    for thresh in filtered['large_threshold'].unique():
        thresh_df = filtered[filtered['large_threshold'] == thresh]
        best_idx = thresh_df['tn_fp_ratio'].idxmax()
        best_configs.append(thresh_df.loc[best_idx])
    
    best_df = pd.DataFrame(best_configs)
    best_df = best_df.sort_values('tn_fp_ratio', ascending=False)
    
    return best_df


def print_best_configs(best_df, title="Best Configurations"):
    """Pretty print best configurations."""
    print(f"\n{'='*100}")
    print(title)
    print('='*100)
    print(f"{'Thresh':<8} │ {'Model':<12} {'Wt':>4} {'ProbT':>6} │ {'AUC':>5} │ {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} │ {'FN%':>5} {'Recall':>6} │ {'TN/FP':>8}")
    print('-'*100)
    
    for _, row in best_df.iterrows():
        tn_fp = f"{row['tn_fp_ratio']:.2f}x" if row['tn_fp_ratio'] < 1000 else "∞"
        print(f"{row['large_threshold']:<8.3f} │ {row['model']:<12} {str(row['weight']):>4} {row['prob_threshold']:>6.2f} │ "
              f"{row['auc']:>5.3f} │ {int(row['TP']):>3} {int(row['FP']):>3} {int(row['FN']):>3} {int(row['TN']):>3} │ "
              f"{row['fn_rate']*100:>4.1f}% {row['recall']*100:>5.1f}% │ {tn_fp:>8}")


def generate_markdown_table(best_df, title="Best Configurations"):
    """Generate markdown table for documentation."""
    print(f"\n### {title}\n")
    print("| Threshold | Model | Weight | Prob | AUC | TP | FP | FN | TN | FN% | Recall | TN/FP |")
    print("|-----------|-------|--------|------|-----|-----|-----|-----|-----|-----|--------|-------|")
    
    for _, row in best_df.iterrows():
        tn_fp = f"{row['tn_fp_ratio']:.2f}x" if row['tn_fp_ratio'] < 1000 else "∞"
        print(f"| {row['large_threshold']:.3f} | {row['model']} | {row['weight']} | {row['prob_threshold']:.2f} | "
              f"{row['auc']:.3f} | {int(row['TP'])} | {int(row['FP'])} | {int(row['FN'])} | {int(row['TN'])} | "
              f"{row['fn_rate']*100:.1f}% | {row['recall']*100:.1f}% | {tn_fp} |")


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def quick_search(train_df, test_df, features, target_col, 
                 large_thresholds, fn_constraints=[0.01, 0.05]):
    """
    Quick grid search with common settings and multiple FN constraints.
    
    Args:
        train_df, test_df: DataFrames
        features: Feature column names
        target_col: Target change column
        large_thresholds: List of large-move thresholds
        fn_constraints: List of max FN rates to report (e.g., [0.01, 0.05])
    
    Returns:
        dict with results for each FN constraint
    """
    print("Running exhaustive grid search...")
    
    results = run_full_grid_search(
        train_df, test_df, features, target_col,
        large_thresholds=large_thresholds,
        verbose=True
    )
    
    print(f"\nTotal configurations tested: {len(results)}")
    
    output = {'all_results': results}
    
    for fn_max in fn_constraints:
        print(f"\n" + "="*80)
        print(f"FN <= {fn_max*100:.0f}% CONSTRAINT")
        print("="*80)
        
        best = find_best_configs(results, max_fn_rate=fn_max)
        if len(best) > 0:
            print_best_configs(best, f"Best configs with FN <= {fn_max*100:.0f}%")
            output[f'fn_max_{fn_max}'] = best
        else:
            print(f"No configs found meeting FN <= {fn_max*100:.0f}%")
    
    return output


if __name__ == "__main__":
    print("Event Grid Search Module")
    print("Import and use: from src.models.event_grid_search import quick_search")

