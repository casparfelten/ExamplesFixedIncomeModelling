"""Walk-forward backtest framework for CPI-Bond Yield models"""

from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import confusion_matrix
import seaborn as sns

from src.models.cpi_yield_model import CPIBondYieldModel, TwoStageCPIBondYieldModel
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class WalkForwardBacktest:
    """
    Walk-forward backtest with month-sized blocks to prevent overfitting.
    """
    
    def __init__(
        self,
        block_size_days: int = 90,
        min_train_size: int = 50,
        retrain_frequency: str = "block"  # "block" or "event"
    ):
        """
        Initialize walk-forward backtest.
        
        Args:
            block_size_days: Size of validation blocks in days (default: 90 = ~3 months)
            min_train_size: Minimum number of events in training set
            retrain_frequency: When to retrain model ('block' or 'event')
        """
        self.block_size_days = block_size_days
        self.min_train_size = min_train_size
        self.retrain_frequency = retrain_frequency
        
        logger.info(f"Initialized walk-forward backtest: block_size={block_size_days} days, min_train={min_train_size}")
    
    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model,  # CPIBondYieldModel or TwoStageCPIBondYieldModel
        target_yield: str = "y_2y"
    ) -> Dict:
        """
        Run walk-forward backtest.
        
        Args:
            train_df: Training data (used for initial training and expanding window)
            test_df: Test data (held out, not used for training)
            model: Model instance to backtest
            target_yield: Name of target yield
        
        Returns:
            Dictionary with backtest results and metrics
        """
        logger.info("Starting walk-forward backtest...")
        logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        # Sort by date
        train_df = train_df.sort_values('date').reset_index(drop=True)
        test_df = test_df.sort_values('date').reset_index(drop=True)
        
        # Split test set into blocks
        test_blocks = self._split_into_blocks(test_df)
        logger.info(f"Split test set into {len(test_blocks)} blocks")
        
        # Initial training on train_df
        logger.info("Initial training on training set...")
        model.fit(train_df, target_yield=target_yield)
        
        # Run backtest on each block
        all_predictions = []
        all_actuals = []
        all_dates = []
        block_results = []
        
        current_train = train_df.copy()
        
        for block_idx, block_df in enumerate(test_blocks):
            logger.info(f"Processing block {block_idx + 1}/{len(test_blocks)} ({len(block_df)} events)...")
            
            # Make predictions on this block
            predictions = model.predict(block_df, target_yield=target_yield)
            actuals = block_df[f'{target_yield}_change'].values
            
            # Check if model is in classification mode
            is_classification = hasattr(model, 'use_classification') and model.use_classification
            
            # For classification, convert bins to representative values for metrics
            if is_classification:
                predictions_continuous = model._bins_to_representative_values(predictions)
                actuals_bins = model._create_yield_bins(actuals)
                # Store both bin predictions and continuous for metrics
                all_predictions.extend(predictions)  # Store bins
            else:
                predictions_continuous = predictions
                all_predictions.extend(predictions)
            
            all_actuals.extend(actuals)
            all_dates.extend(block_df['date'].tolist())
            
            # Compute block metrics
            # For classification, use continuous representation for RMSE/MAE
            valid_mask_block = ~(np.isnan(actuals) | np.isnan(predictions_continuous))
            if valid_mask_block.sum() > 0:
                block_rmse = np.sqrt(np.mean((actuals[valid_mask_block] - predictions_continuous[valid_mask_block]) ** 2))
                block_mae = np.mean(np.abs(actuals[valid_mask_block] - predictions_continuous[valid_mask_block]))
                block_r2 = 1 - np.sum((actuals[valid_mask_block] - predictions_continuous[valid_mask_block]) ** 2) / np.sum((actuals[valid_mask_block] - np.mean(actuals[valid_mask_block])) ** 2)
                block_directional = np.mean(np.sign(actuals[valid_mask_block]) == np.sign(predictions_continuous[valid_mask_block]))
                
                # For classification, also compute accuracy
                if is_classification:
                    block_accuracy = (predictions[valid_mask_block] == actuals_bins[valid_mask_block]).mean()
                else:
                    block_accuracy = np.nan
            else:
                block_rmse = block_mae = block_r2 = block_directional = np.nan
                block_accuracy = np.nan
            
            block_result = {
                'block_idx': block_idx,
                'start_date': block_df['date'].min(),
                'end_date': block_df['date'].max(),
                'n_events': len(block_df),
                'rmse': block_rmse,
                'mae': block_mae,
                'r2': block_r2,
                'directional_accuracy': block_directional
            }
            if is_classification:
                block_result['accuracy'] = block_accuracy
            block_results.append(block_result)
            
            # Retrain model: add this block to training set (expanding window)
            if self.retrain_frequency == "block":
                current_train = pd.concat([current_train, block_df], ignore_index=True)
                logger.info(f"Retraining model on expanded training set ({len(current_train)} events)...")
                model.fit(current_train, target_yield=target_yield)
        
        # Convert to arrays
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        
        # Check if model is in classification mode
        is_classification = hasattr(model, 'use_classification') and model.use_classification
        
        # For classification, convert bins to continuous for metrics
        if is_classification:
            all_predictions_continuous = model._bins_to_representative_values(all_predictions)
            all_actuals_bins = model._create_yield_bins(all_actuals)
        else:
            all_predictions_continuous = all_predictions
        
        # Remove NaN values
        valid_mask = ~(np.isnan(all_predictions_continuous) | np.isnan(all_actuals))
        all_predictions_continuous = all_predictions_continuous[valid_mask]
        all_predictions = all_predictions[valid_mask]
        all_actuals = all_actuals[valid_mask]
        all_dates = [d for d, v in zip(all_dates, valid_mask) if v]
        
        if is_classification:
            all_actuals_bins = all_actuals_bins[valid_mask]
        
        # Overall metrics
        overall_rmse = np.sqrt(np.mean((all_actuals - all_predictions_continuous) ** 2))
        overall_mae = np.mean(np.abs(all_actuals - all_predictions_continuous))
        overall_r2 = 1 - np.sum((all_actuals - all_predictions_continuous) ** 2) / np.sum((all_actuals - np.mean(all_actuals)) ** 2)
        overall_directional = np.mean(np.sign(all_actuals) == np.sign(all_predictions_continuous))
        
        # Classification accuracy
        if is_classification:
            overall_accuracy = (all_predictions == all_actuals_bins).mean()
        else:
            overall_accuracy = np.nan
        
        # Binned evaluation (use continuous for binning, not bin labels)
        bin_metrics = self._compute_binned_metrics(all_actuals, all_predictions_continuous)
        
        # Final test score (weighted combination of bin accuracy and inverse RMSE)
        final_test_score = self._compute_final_test_score(bin_metrics, overall_rmse)
        
        results = {
            'predictions': all_predictions,
            'actuals': all_actuals,
            'dates': all_dates,
            'overall_rmse': overall_rmse,
            'overall_mae': overall_mae,
            'overall_r2': overall_r2,
            'overall_directional_accuracy': overall_directional,
            'bin_metrics': bin_metrics,
            'final_test_score': final_test_score,
            'block_results': pd.DataFrame(block_results),
            'n_blocks': len(test_blocks),
            'n_predictions': len(all_predictions)
        }
        
        # Add classification metrics if applicable
        if is_classification:
            results['overall_accuracy'] = overall_accuracy
            results['predictions_bins'] = all_predictions
            results['actuals_bins'] = all_actuals_bins
        
        # If two-stage model, also evaluate regime model
        if hasattr(model, 'regime_fitted') and model.regime_fitted:
            try:
                # Get baseline predictions for test set
                baseline_pred = model.predict_regime(test_df, target_yield=target_yield)
                baseline_actual = test_df[f'{target_yield}_before'].values
                valid_mask = ~np.isnan(baseline_actual)
                
                if valid_mask.sum() > 0:
                    regime_rmse = np.sqrt(np.mean((baseline_actual[valid_mask] - baseline_pred[valid_mask]) ** 2))
                    regime_r2 = 1 - np.sum((baseline_actual[valid_mask] - baseline_pred[valid_mask]) ** 2) / np.sum((baseline_actual[valid_mask] - np.mean(baseline_actual[valid_mask])) ** 2)
                    results['regime_rmse'] = regime_rmse
                    results['regime_r2'] = regime_r2
            except Exception as e:
                logger.warning(f"Could not evaluate regime model: {e}")
        
        logger.info(f"Backtest complete: RMSE={overall_rmse:.4f}, MAE={overall_mae:.4f}, R²={overall_r2:.4f}, Dir Acc={overall_directional:.4f}")
        logger.info(f"Final Test Score: {final_test_score:.4f}")
        
        return results
    
    def _categorize_yield_change(self, change: float) -> str:
        """
        Categorize yield change into bins.
        
        Args:
            change: Yield change value
        
        Returns:
            Bin category: 'small', 'medium', or 'large'
        """
        abs_change = abs(change)
        if abs_change < 0.05:
            return 'small'
        elif abs_change < 0.15:
            return 'medium'
        else:
            return 'large'
    
    def _compute_binned_metrics(self, actuals: np.ndarray, predictions: np.ndarray) -> Dict:
        """
        Compute binned evaluation metrics.
        
        Args:
            actuals: Actual yield changes
            predictions: Predicted yield changes
        
        Returns:
            Dictionary with binned metrics
        """
        # Categorize actuals and predictions
        actual_bins = [self._categorize_yield_change(a) for a in actuals]
        pred_bins = [self._categorize_yield_change(p) for p in predictions]
        
        # Overall bin accuracy
        bin_accuracy = np.mean([a == p for a, p in zip(actual_bins, pred_bins)])
        
        # Per-bin accuracy
        bin_labels = ['small', 'medium', 'large']
        bin_accuracies = {}
        bin_counts = {}
        
        for label in bin_labels:
            mask = np.array([a == label for a in actual_bins])
            if mask.sum() > 0:
                bin_accuracies[label] = np.mean([p == label for p, m in zip(pred_bins, mask) if m])
                bin_counts[label] = mask.sum()
            else:
                bin_accuracies[label] = np.nan
                bin_counts[label] = 0
        
        # Confusion matrix
        cm = confusion_matrix(actual_bins, pred_bins, labels=bin_labels)
        
        return {
            'overall_bin_accuracy': bin_accuracy,
            'bin_accuracies': bin_accuracies,
            'bin_counts': bin_counts,
            'confusion_matrix': cm,
            'bin_labels': bin_labels
        }
    
    def _compute_final_test_score(
        self,
        bin_metrics: Dict,
        rmse: float,
        bin_weight: float = 0.6,
        rmse_weight: float = 0.4
    ) -> float:
        """
        Compute final test score combining bin accuracy and RMSE.
        
        Args:
            bin_metrics: Dictionary with binned metrics
            rmse: Root mean squared error
            bin_weight: Weight for bin accuracy (default: 0.6)
            rmse_weight: Weight for inverse RMSE (default: 0.4)
        
        Returns:
            Final test score (0-1, higher is better)
        """
        # Bin accuracy component (weighted by importance of large moves)
        overall_bin_acc = bin_metrics['overall_bin_accuracy']
        bin_accuracies = bin_metrics['bin_accuracies']
        
        # Weighted bin accuracy (emphasize large moves)
        if not np.isnan(bin_accuracies.get('large', np.nan)):
            weighted_bin_acc = (
                0.2 * bin_accuracies.get('small', 0) +
                0.3 * bin_accuracies.get('medium', 0) +
                0.5 * bin_accuracies.get('large', 0)
            )
        else:
            weighted_bin_acc = overall_bin_acc
        
        # RMSE component (normalize to 0-1, inverse so higher is better)
        # Assume max reasonable RMSE is 0.5 (50 bps)
        max_rmse = 0.5
        normalized_rmse = min(rmse / max_rmse, 1.0)
        rmse_score = 1.0 - normalized_rmse
        
        # Combined score
        final_score = bin_weight * weighted_bin_acc + rmse_weight * rmse_score
        
        return final_score
    
    def _split_into_blocks(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Split DataFrame into time-based blocks.
        
        Args:
            df: DataFrame with 'date' column
        
        Returns:
            List of DataFrames, one per block
        """
        df = df.sort_values('date').reset_index(drop=True)
        
        if len(df) == 0:
            return []
        
        blocks = []
        start_date = df['date'].min()
        end_date = df['date'].max()
        
        current_date = start_date
        
        while current_date <= end_date:
            block_end = current_date + timedelta(days=self.block_size_days)
            
            # Get events in this block
            block_df = df[(df['date'] >= current_date) & (df['date'] < block_end)]
            
            if len(block_df) > 0:
                blocks.append(block_df)
            
            # Move to next block
            current_date = block_end
        
        return blocks
    
    def plot_results(
        self,
        results: Dict,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Plot backtest results.
        
        Args:
            results: Results dictionary from run()
            save_path: Optional path to save figure
            figsize: Figure size
        """
        predictions = results['predictions']
        actuals = results['actuals']
        dates = pd.to_datetime(results['dates'])
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Predictions vs Actuals over time
        ax1 = axes[0, 0]
        ax1.plot(dates, actuals, label='Actual', alpha=0.7, linewidth=1.5)
        ax1.plot(dates, predictions, label='Predicted', alpha=0.7, linewidth=1.5)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Yield Change (%)')
        ax1.set_title('Predictions vs Actuals Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot
        ax2 = axes[0, 1]
        ax2.scatter(actuals, predictions, alpha=0.5, s=20)
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Yield Change (%)')
        ax2.set_ylabel('Predicted Yield Change (%)')
        ax2.set_title('Predictions vs Actuals (Scatter)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Residuals over time
        ax3 = axes[1, 0]
        residuals = actuals - predictions
        ax3.plot(dates, residuals, alpha=0.7, linewidth=1)
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Residual (Actual - Predicted)')
        ax3.set_title('Residuals Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Block-wise performance
        ax4 = axes[1, 1]
        block_results = results['block_results']
        ax4.plot(range(len(block_results)), block_results['rmse'], marker='o', label='RMSE')
        ax4.set_xlabel('Block Index')
        ax4.set_ylabel('RMSE')
        ax4.set_title('Block-wise RMSE')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        plt.show()
    
    def print_summary(self, results: Dict):
        """Print summary of backtest results."""
        print("\n" + "="*60)
        print("BACKTEST SUMMARY")
        print("="*60)
        print(f"Number of blocks: {results['n_blocks']}")
        print(f"Total predictions: {results['n_predictions']}")
        
        # Regime model metrics (if two-stage)
        if 'regime_rmse' in results:
            print(f"\nStage 1 (Regime Model) - Baseline Yield Prediction:")
            print(f"  RMSE: {results['regime_rmse']:.4f}")
            print(f"  R²:   {results['regime_r2']:.4f}")
        
        print(f"\nStage 2 (Shock Model) - Overall Metrics:")
        print(f"  RMSE: {results['overall_rmse']:.4f}")
        print(f"  MAE:  {results['overall_mae']:.4f}")
        print(f"  R²:   {results['overall_r2']:.4f}")
        print(f"  Directional Accuracy: {results['overall_directional_accuracy']:.4f}")
        
        # Binned metrics
        bin_metrics = results.get('bin_metrics', {})
        if bin_metrics:
            print(f"\nBinned Evaluation:")
            print(f"  Overall Bin Accuracy: {bin_metrics['overall_bin_accuracy']:.4f}")
            print(f"  Small Move Accuracy: {bin_metrics['bin_accuracies'].get('small', np.nan):.4f} (n={bin_metrics['bin_counts'].get('small', 0)})")
            print(f"  Medium Move Accuracy: {bin_metrics['bin_accuracies'].get('medium', np.nan):.4f} (n={bin_metrics['bin_counts'].get('medium', 0)})")
            print(f"  Large Move Accuracy: {bin_metrics['bin_accuracies'].get('large', np.nan):.4f} (n={bin_metrics['bin_counts'].get('large', 0)})")
        
        print(f"\nFinal Test Score: {results.get('final_test_score', np.nan):.4f}")
        
        print(f"\nBlock-wise Performance:")
        block_results = results['block_results']
        print(block_results[['block_idx', 'start_date', 'end_date', 'n_events', 'rmse', 'r2', 'directional_accuracy']].to_string(index=False))
        
        print("="*60 + "\n")
    
    def plot_binned_evaluation(
        self,
        results: Dict,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 5)
    ):
        """
        Plot binned evaluation results including confusion matrix.
        
        Args:
            results: Results dictionary from run()
            save_path: Optional path to save figure
            figsize: Figure size
        """
        bin_metrics = results.get('bin_metrics', {})
        if not bin_metrics:
            logger.warning("No binned metrics available for plotting")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Bin accuracy by category
        ax1 = axes[0]
        bin_labels = bin_metrics['bin_labels']
        bin_accuracies = [bin_metrics['bin_accuracies'].get(label, 0) for label in bin_labels]
        bin_counts = [bin_metrics['bin_counts'].get(label, 0) for label in bin_labels]
        
        bars = ax1.bar(bin_labels, bin_accuracies, alpha=0.7, color=['green', 'orange', 'red'])
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_xlabel('Move Size', fontsize=12)
        ax1.set_title('Bin Accuracy by Move Size', fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for bar, count in zip(bars, bin_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({height:.2%})',
                    ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Confusion matrix
        ax2 = axes[1]
        cm = bin_metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                   xticklabels=bin_labels, yticklabels=bin_labels)
        ax2.set_xlabel('Predicted', fontsize=12)
        ax2.set_ylabel('Actual', fontsize=12)
        ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved binned evaluation plot to {save_path}")
        
        plt.show()

