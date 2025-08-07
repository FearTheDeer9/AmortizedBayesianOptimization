"""
Core metrics tracking module for BC training debugging.

This module provides comprehensive metrics tracking for both policy and surrogate
BC models, including classification metrics, embeddings, and confusion matrices.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


class BCMetricsTracker:
    """
    Comprehensive metrics tracker for BC training.
    
    Tracks:
    - Classification metrics (F1, precision, recall, accuracy)
    - Value prediction metrics (MSE, MAE, R²)
    - Node embeddings over time
    - Confusion matrices
    - Per-variable performance
    """
    
    def __init__(self, 
                 save_embeddings_every: int = 10,
                 max_embeddings_per_epoch: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            save_embeddings_every: Save embeddings every N epochs
            max_embeddings_per_epoch: Maximum embeddings to store per epoch
        """
        self.save_embeddings_every = save_embeddings_every
        self.max_embeddings_per_epoch = max_embeddings_per_epoch
        
        # Metrics history
        self.epoch_metrics = []
        self.batch_metrics = []
        self.embeddings_history = {}
        self.confusion_matrices = {}
        self.per_variable_stats = {}
        
        # Current epoch tracking
        self.current_epoch = 0
        self.current_batch_metrics = []
        
    def start_epoch(self, epoch: int):
        """Start tracking a new epoch."""
        self.current_epoch = epoch
        self.current_batch_metrics = []
    
    def track_batch(self,
                   predictions: jnp.ndarray,
                   targets: Dict[str, Any],
                   value_predictions: Optional[jnp.ndarray] = None,
                   value_targets: Optional[jnp.ndarray] = None,
                   embeddings: Optional[jnp.ndarray] = None):
        """
        Track metrics for a single batch.
        
        Args:
            predictions: Predicted variable indices [batch_size]
            targets: Target information including variable indices
            value_predictions: Predicted intervention values (optional)
            value_targets: Target intervention values (optional)
            embeddings: Node embeddings from this batch (optional)
        """
        batch_size = len(predictions)
        
        # Extract target indices
        target_indices = []
        for target_info in targets:
            if isinstance(target_info, dict) and 'variable_idx' in target_info:
                target_indices.append(target_info['variable_idx'])
            elif isinstance(target_info, (int, np.integer)):
                target_indices.append(int(target_info))
            else:
                # Try to extract from label structure
                if 'targets' in target_info and target_info['targets']:
                    # Map variable name to index if needed
                    target_indices.append(0)  # Placeholder
                else:
                    target_indices.append(0)
        
        target_indices = jnp.array(target_indices)
        
        # Classification metrics
        accuracy = jnp.mean(predictions == target_indices)
        
        # Store batch metrics
        batch_metric = {
            'accuracy': float(accuracy),
            'batch_size': batch_size
        }
        
        # Value prediction metrics if available
        if value_predictions is not None and value_targets is not None:
            mse = jnp.mean((value_predictions - value_targets) ** 2)
            mae = jnp.mean(jnp.abs(value_predictions - value_targets))
            batch_metric['value_mse'] = float(mse)
            batch_metric['value_mae'] = float(mae)
        
        self.current_batch_metrics.append(batch_metric)
        
        # Store embeddings if it's the right epoch
        if embeddings is not None and self.current_epoch % self.save_embeddings_every == 0:
            if self.current_epoch not in self.embeddings_history:
                self.embeddings_history[self.current_epoch] = []
            
            # Limit number of embeddings stored
            if len(self.embeddings_history[self.current_epoch]) < self.max_embeddings_per_epoch:
                self.embeddings_history[self.current_epoch].append(np.array(embeddings))
    
    def end_epoch(self, 
                  val_predictions: Optional[jnp.ndarray] = None,
                  val_targets: Optional[jnp.ndarray] = None,
                  variable_names: Optional[List[str]] = None):
        """
        Finalize metrics for the current epoch.
        
        Args:
            val_predictions: Validation set predictions
            val_targets: Validation set targets
            variable_names: Names of variables for confusion matrix
        """
        # Aggregate batch metrics
        if self.current_batch_metrics:
            total_samples = sum(m['batch_size'] for m in self.current_batch_metrics)
            weighted_acc = sum(m['accuracy'] * m['batch_size'] for m in self.current_batch_metrics) / total_samples
            
            epoch_metric = {
                'epoch': self.current_epoch,
                'train_accuracy': weighted_acc,
                'n_batches': len(self.current_batch_metrics)
            }
            
            # Add value metrics if available
            if 'value_mse' in self.current_batch_metrics[0]:
                weighted_mse = sum(m.get('value_mse', 0) * m['batch_size'] 
                                 for m in self.current_batch_metrics) / total_samples
                weighted_mae = sum(m.get('value_mae', 0) * m['batch_size'] 
                                 for m in self.current_batch_metrics) / total_samples
                epoch_metric['train_value_mse'] = weighted_mse
                epoch_metric['train_value_mae'] = weighted_mae
            
            # Validation metrics
            if val_predictions is not None and val_targets is not None:
                val_predictions = np.array(val_predictions)
                val_targets = np.array(val_targets)
                
                # Classification metrics
                accuracy = np.mean(val_predictions == val_targets)
                
                # Only compute F1, precision, recall if we have multiple classes
                if len(np.unique(val_targets)) > 1:
                    f1 = f1_score(val_targets, val_predictions, average='weighted', zero_division=0)
                    precision = precision_score(val_targets, val_predictions, average='weighted', zero_division=0)
                    recall = recall_score(val_targets, val_predictions, average='weighted', zero_division=0)
                else:
                    f1 = precision = recall = accuracy
                
                epoch_metric.update({
                    'val_accuracy': accuracy,
                    'val_f1': f1,
                    'val_precision': precision,
                    'val_recall': recall
                })
                
                # Confusion matrix
                if variable_names and self.current_epoch % 10 == 0:
                    cm = confusion_matrix(val_targets, val_predictions)
                    self.confusion_matrices[self.current_epoch] = {
                        'matrix': cm,
                        'labels': variable_names[:cm.shape[0]]
                    }
                
                # Per-variable accuracy
                self._update_per_variable_stats(val_predictions, val_targets, variable_names)
            
            self.epoch_metrics.append(epoch_metric)
    
    def _update_per_variable_stats(self, 
                                   predictions: np.ndarray,
                                   targets: np.ndarray,
                                   variable_names: Optional[List[str]] = None):
        """Update per-variable performance statistics."""
        unique_vars = np.unique(targets)
        
        for var_idx in unique_vars:
            mask = targets == var_idx
            if mask.sum() > 0:
                var_accuracy = np.mean(predictions[mask] == var_idx)
                
                var_name = variable_names[var_idx] if variable_names and var_idx < len(variable_names) else f"var_{var_idx}"
                
                if var_name not in self.per_variable_stats:
                    self.per_variable_stats[var_name] = {
                        'attempts': 0,
                        'correct': 0,
                        'accuracy_history': []
                    }
                
                self.per_variable_stats[var_name]['attempts'] += mask.sum()
                self.per_variable_stats[var_name]['correct'] += (predictions[mask] == var_idx).sum()
                self.per_variable_stats[var_name]['accuracy_history'].append(
                    (self.current_epoch, var_accuracy)
                )
    
    def get_embedding_diversity(self, epoch: int) -> Optional[float]:
        """
        Compute embedding diversity for a specific epoch.
        
        Returns:
            Average pairwise distance between embeddings
        """
        if epoch not in self.embeddings_history:
            return None
        
        embeddings = self.embeddings_history[epoch]
        if not embeddings:
            return None
        
        # Concatenate all embeddings
        all_embeddings = np.concatenate(embeddings, axis=0)
        
        # Compute pairwise distances
        n_samples = min(100, len(all_embeddings))  # Limit for efficiency
        sample_indices = np.random.choice(len(all_embeddings), n_samples, replace=False)
        sampled = all_embeddings[sample_indices]
        
        # Compute average pairwise L2 distance
        distances = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(sampled[i] - sampled[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all tracked metrics.
        
        Returns:
            Dictionary with metric summaries
        """
        if not self.epoch_metrics:
            return {}
        
        latest = self.epoch_metrics[-1]
        
        # Find best validation metrics
        val_accuracies = [m.get('val_accuracy', 0) for m in self.epoch_metrics]
        best_epoch = np.argmax(val_accuracies) if val_accuracies else 0
        
        summary = {
            'total_epochs': len(self.epoch_metrics),
            'latest_metrics': latest,
            'best_epoch': best_epoch,
            'best_val_accuracy': val_accuracies[best_epoch] if val_accuracies else 0,
            'embedding_epochs': list(self.embeddings_history.keys()),
            'confusion_matrix_epochs': list(self.confusion_matrices.keys())
        }
        
        # Add per-variable summary
        if self.per_variable_stats:
            var_summary = {}
            for var_name, stats in self.per_variable_stats.items():
                if stats['attempts'] > 0:
                    var_summary[var_name] = stats['correct'] / stats['attempts']
            summary['per_variable_accuracy'] = var_summary
        
        return summary
    
    def save(self, filepath: Path):
        """Save all metrics to disk."""
        data = {
            'epoch_metrics': self.epoch_metrics,
            'batch_metrics': self.batch_metrics,
            'embeddings_history': self.embeddings_history,
            'confusion_matrices': self.confusion_matrices,
            'per_variable_stats': self.per_variable_stats,
            'config': {
                'save_embeddings_every': self.save_embeddings_every,
                'max_embeddings_per_epoch': self.max_embeddings_per_epoch
            }
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: Path) -> 'BCMetricsTracker':
        """Load metrics from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        tracker = cls(
            save_embeddings_every=data['config']['save_embeddings_every'],
            max_embeddings_per_epoch=data['config']['max_embeddings_per_epoch']
        )
        
        tracker.epoch_metrics = data['epoch_metrics']
        tracker.batch_metrics = data['batch_metrics']
        tracker.embeddings_history = data['embeddings_history']
        tracker.confusion_matrices = data['confusion_matrices']
        tracker.per_variable_stats = data['per_variable_stats']
        
        return tracker


def extract_embeddings_from_model(net, params, input_tensor, target_idx, layer_name: str = 'encoder'):
    """
    Extract intermediate embeddings from a model.
    
    Args:
        net: Haiku transformed network
        params: Model parameters
        input_tensor: Input data tensor
        target_idx: Target variable index
        layer_name: Name of layer to extract embeddings from
        
    Returns:
        Embeddings array
    """
    # This would need to be customized based on the specific model architecture
    # For now, return a placeholder
    rng = jax.random.PRNGKey(0)
    outputs = net.apply(params, rng, input_tensor, target_idx)
    
    # Try to extract embeddings if they're in the output
    if isinstance(outputs, dict) and 'embeddings' in outputs:
        return outputs['embeddings']
    
    # Otherwise return None
    return None