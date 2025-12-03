"""Quick test of two-stage model implementation"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.models.prepare_data import prepare_event_data, create_train_test_split
from src.models.cpi_yield_model import TwoStageCPIBondYieldModel
from src.utils.logging_utils import setup_logging

setup_logging()

print("Testing Two-Stage Model Implementation...")
print("="*60)

# Prepare data
print("\n1. Preparing event data...")
events_df = prepare_event_data(
    target_yield="y_2y",
    prediction_horizon=0,
    start_date=pd.Timestamp("2000-01-01"),  # Use recent data for faster test
    end_date=None
)

print(f"   Dataset shape: {events_df.shape}")
print(f"   Date range: {events_df['date'].min()} to {events_df['date'].max()}")

# Train/test split
print("\n2. Creating train/test split...")
train_df, test_df = create_train_test_split(events_df, test_size=0.3, min_train_size=50)
print(f"   Train: {len(train_df)}, Test: {len(test_df)}")

# Initialize and fit model
print("\n3. Initializing two-stage model...")
model = TwoStageCPIBondYieldModel(
    regime_model_type="ridge",
    shock_model_type="ridge",
    regime_alpha=1.0,
    shock_alpha=1.0,
    scale_features=True
)

print("\n4. Fitting model...")
try:
    train_metrics = model.fit(train_df, target_yield="y_2y", cv=None)
    print("   ✓ Model fitted successfully!")
    print(f"   Regime R²: {train_metrics.get('regime_train_r2', 'N/A'):.4f}")
    print(f"   Shock R²: {train_metrics.get('shock_train_r2', 'N/A'):.4f}")
except Exception as e:
    print(f"   ✗ Error fitting model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test predictions
print("\n5. Testing predictions...")
try:
    predictions = model.predict(test_df, target_yield="y_2y")
    print(f"   ✓ Predictions generated: {len(predictions)} predictions")
    print(f"   Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
except Exception as e:
    print(f"   ✗ Error making predictions: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Evaluate
print("\n6. Evaluating on test set...")
try:
    test_metrics = model.evaluate(test_df, target_yield="y_2y")
    print("   ✓ Evaluation complete!")
    print(f"   Regime RMSE: {test_metrics.get('regime_rmse', 'N/A'):.4f}")
    print(f"   Shock RMSE: {test_metrics.get('shock_rmse', 'N/A'):.4f}")
    print(f"   Directional Accuracy: {test_metrics.get('directional_accuracy', 'N/A'):.4f}")
except Exception as e:
    print(f"   ✗ Error evaluating: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ All tests passed! Two-stage model is working.")
print("="*60)

