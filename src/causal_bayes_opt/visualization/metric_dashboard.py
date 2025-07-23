"""
Real-time Training Metrics Dashboard

Provides live plotting and analysis of F1 scores, parent likelihoods, and other
training metrics during model training.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import time
import threading
import queue
from dataclasses import dataclass, field

import numpy as onp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a specific episode."""
    episode: int
    timestamp: float
    mean_reward: float
    f1_score: Optional[float] = None
    true_parent_likelihood: Optional[float] = None
    shd: Optional[int] = None
    marginal_probs: Optional[Dict[str, float]] = None
    scm_type: str = "unknown"


class MetricDashboard:
    """Real-time dashboard for training metrics."""
    
    def __init__(self, 
                 update_interval: float = 1.0,
                 max_history: int = 1000,
                 save_plots: bool = True,
                 output_dir: Optional[Path] = None):
        """
        Initialize metric dashboard.
        
        Args:
            update_interval: How often to update plots (seconds)
            max_history: Maximum number of episodes to keep in history
            save_plots: Whether to save plots to disk
            output_dir: Directory to save plots (if save_plots=True)
        """
        self.update_interval = update_interval
        self.max_history = max_history
        self.save_plots = save_plots
        self.output_dir = output_dir or Path("training_plots")
        
        # Create output directory
        if self.save_plots:
            self.output_dir.mkdir(exist_ok=True)
        
        # Metric storage
        self.metrics_history: List[MetricSnapshot] = []
        self.metrics_queue = queue.Queue()
        
        # Plotting setup
        self.fig: Optional[Figure] = None
        self.axes: Optional[Dict[str, plt.Axes]] = None
        self.lines: Dict[str, plt.Line2D] = {}
        
        # Animation and threading
        self.animation: Optional[animation.FuncAnimation] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        logger.info(f"Initialized metric dashboard with {max_history} history, "
                   f"save_plots={save_plots}, output_dir={self.output_dir}")
    
    def start(self):
        """Start real-time metric dashboard."""
        if self.running:
            logger.warning("Dashboard already running")
            return
        
        self.running = True
        self._setup_plots()
        
        # Start background thread for plot updates
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        
        logger.info("Started metric dashboard")
    
    def stop(self):
        """Stop real-time metric dashboard."""
        if not self.running:
            return
        
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        if self.animation:
            self.animation.event_source.stop()
        
        logger.info("Stopped metric dashboard")
    
    def update_metrics(self, episode: int, metrics: Dict[str, Any]):
        """
        Update metrics with new data.
        
        Args:
            episode: Current episode number
            metrics: Dictionary of metric values
        """
        snapshot = MetricSnapshot(
            episode=episode,
            timestamp=time.time(),
            mean_reward=metrics.get('mean_reward', 0.0),
            f1_score=metrics.get('f1_score'),
            true_parent_likelihood=metrics.get('true_parent_likelihood'),
            shd=metrics.get('shd'),
            marginal_probs=metrics.get('marginal_probs'),
            scm_type=metrics.get('scm_type', 'unknown')
        )
        
        # Add to queue for thread-safe updates
        self.metrics_queue.put(snapshot)
    
    def _setup_plots(self):
        """Set up the plotting interface."""
        # Create figure with subplots
        self.fig, axes_array = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Training Metrics Dashboard', fontsize=16)
        
        # Flatten axes array for easier access
        axes_flat = axes_array.flatten()
        
        self.axes = {
            'reward': axes_flat[0],
            'structure': axes_flat[1],
            'shd': axes_flat[2],
            'marginals': axes_flat[3]
        }
        
        # Setup individual plots
        self._setup_reward_plot()
        self._setup_structure_plot()
        self._setup_shd_plot()
        self._setup_marginals_plot()
        
        plt.tight_layout()
        plt.ion()  # Turn on interactive mode
        plt.show()
    
    def _setup_reward_plot(self):
        """Setup reward progression plot."""
        ax = self.axes['reward']
        ax.set_title('Mean Reward Over Time')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Mean Reward')
        ax.grid(True, alpha=0.3)
        
        # Create line objects
        self.lines['reward'] = ax.plot([], [], 'b-', linewidth=2, label='Mean Reward')[0]
        ax.legend()
    
    def _setup_structure_plot(self):
        """Setup structure learning metrics plot."""
        ax = self.axes['structure']
        ax.set_title('Structure Learning Metrics')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        # Create line objects
        self.lines['f1_score'] = ax.plot([], [], 'g-', linewidth=2, label='F1 Score')[0]
        self.lines['parent_likelihood'] = ax.plot([], [], 'r-', linewidth=2, label='P(Parents|Data)')[0]
        
        # Add threshold lines
        ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% Threshold')
        ax.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='70% Threshold')
        ax.legend()
    
    def _setup_shd_plot(self):
        """Setup Structural Hamming Distance plot."""
        ax = self.axes['shd']
        ax.set_title('Structural Hamming Distance')
        ax.set_xlabel('Episode')
        ax.set_ylabel('SHD (Lower is Better)')
        ax.grid(True, alpha=0.3)
        
        # Create line objects
        self.lines['shd'] = ax.plot([], [], 'orange', linewidth=2, label='SHD')[0]
        ax.axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Perfect Recovery')
        ax.legend()
    
    def _setup_marginals_plot(self):
        """Setup marginal probabilities plot."""
        ax = self.axes['marginals']
        ax.set_title('Marginal Parent Probabilities')
        ax.set_xlabel('Variable')
        ax.set_ylabel('P(is_parent)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        # This will be updated dynamically based on available variables
        self.lines['marginals'] = None
    
    def _update_loop(self):
        """Main update loop running in background thread."""
        while self.running:
            try:
                # Process all queued metrics
                updated = False
                while not self.metrics_queue.empty():
                    snapshot = self.metrics_queue.get_nowait()
                    self._add_metric_snapshot(snapshot)
                    updated = True
                
                # Update plots if new data available
                if updated:
                    self._update_plots()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in metric dashboard update loop: {e}")
                time.sleep(1.0)  # Avoid tight error loop
    
    def _add_metric_snapshot(self, snapshot: MetricSnapshot):
        """Add new metric snapshot to history."""
        self.metrics_history.append(snapshot)
        
        # Trim history if too long
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
    
    def _update_plots(self):
        """Update all plots with current data."""
        if not self.metrics_history:
            return
        
        try:
            # Extract data for plotting
            episodes = [m.episode for m in self.metrics_history]
            rewards = [m.mean_reward for m in self.metrics_history]
            f1_scores = [m.f1_score for m in self.metrics_history if m.f1_score is not None]
            parent_likelihoods = [m.true_parent_likelihood for m in self.metrics_history if m.true_parent_likelihood is not None]
            shd_values = [m.shd for m in self.metrics_history if m.shd is not None]
            
            # Update reward plot
            self.lines['reward'].set_data(episodes, rewards)
            self.axes['reward'].relim()
            self.axes['reward'].autoscale_view()
            
            # Update structure metrics plot
            if f1_scores:
                f1_episodes = [m.episode for m in self.metrics_history if m.f1_score is not None]
                self.lines['f1_score'].set_data(f1_episodes, f1_scores)
            
            if parent_likelihoods:
                parent_episodes = [m.episode for m in self.metrics_history if m.true_parent_likelihood is not None]
                self.lines['parent_likelihood'].set_data(parent_episodes, parent_likelihoods)
            
            self.axes['structure'].relim()
            self.axes['structure'].autoscale_view()
            
            # Update SHD plot
            if shd_values:
                shd_episodes = [m.episode for m in self.metrics_history if m.shd is not None]
                self.lines['shd'].set_data(shd_episodes, shd_values)
                self.axes['shd'].relim()
                self.axes['shd'].autoscale_view()
            
            # Update marginal probabilities plot
            if self.metrics_history:
                latest_snapshot = self.metrics_history[-1]
                if latest_snapshot.marginal_probs:
                    self._update_marginals_plot(latest_snapshot.marginal_probs)
            
            # Refresh the display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            # Save plots if enabled
            if self.save_plots and len(self.metrics_history) % 50 == 0:
                self._save_current_plots()
            
        except Exception as e:
            logger.error(f"Error updating plots: {e}")
    
    def _update_marginals_plot(self, marginal_probs: Dict[str, float]):
        """Update marginal probabilities bar plot."""
        ax = self.axes['marginals']
        ax.clear()
        
        # Setup plot again
        ax.set_title('Marginal Parent Probabilities (Latest)')
        ax.set_xlabel('Variable')
        ax.set_ylabel('P(is_parent)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        # Create bar plot
        variables = list(marginal_probs.keys())
        probabilities = list(marginal_probs.values())
        
        bars = ax.bar(variables, probabilities, alpha=0.7)
        
        # Color bars based on probability
        for bar, prob in zip(bars, probabilities):
            if prob > 0.7:
                bar.set_color('green')
            elif prob > 0.3:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.3f}', ha='center', va='bottom')
        
        ax.set_xticklabels(variables, rotation=45)
    
    def _save_current_plots(self):
        """Save current plots to disk."""
        try:
            timestamp = int(time.time())
            filename = self.output_dir / f"training_metrics_{timestamp}.png"
            
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training metrics plot to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save plots: {e}")
    
    def export_metrics(self, filename: Optional[Path] = None) -> Path:
        """
        Export metrics history to CSV file.
        
        Args:
            filename: Optional filename, defaults to timestamped file
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = int(time.time())
            filename = self.output_dir / f"training_metrics_{timestamp}.csv"
        
        try:
            import pandas as pd
            
            # Convert metrics to DataFrame
            data = []
            for snapshot in self.metrics_history:
                row = {
                    'episode': snapshot.episode,
                    'timestamp': snapshot.timestamp,
                    'mean_reward': snapshot.mean_reward,
                    'f1_score': snapshot.f1_score,
                    'true_parent_likelihood': snapshot.true_parent_likelihood,
                    'shd': snapshot.shd,
                    'scm_type': snapshot.scm_type
                }
                
                # Add marginal probabilities as separate columns
                if snapshot.marginal_probs:
                    for var, prob in snapshot.marginal_probs.items():
                        row[f'marginal_{var}'] = prob
                
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            
            logger.info(f"Exported {len(data)} metric snapshots to {filename}")
            return filename
            
        except ImportError:
            logger.error("pandas not available, cannot export to CSV")
            raise
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            raise
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of training metrics."""
        if not self.metrics_history:
            return {}
        
        rewards = [m.mean_reward for m in self.metrics_history]
        f1_scores = [m.f1_score for m in self.metrics_history if m.f1_score is not None]
        parent_likelihoods = [m.true_parent_likelihood for m in self.metrics_history if m.true_parent_likelihood is not None]
        shd_values = [m.shd for m in self.metrics_history if m.shd is not None]
        
        summary = {
            'total_episodes': len(self.metrics_history),
            'reward_stats': {
                'mean': float(onp.mean(rewards)),
                'std': float(onp.std(rewards)),
                'min': float(onp.min(rewards)),
                'max': float(onp.max(rewards)),
                'final': rewards[-1],
                'improvement': rewards[-1] - rewards[0] if len(rewards) > 1 else 0.0
            }
        }
        
        if f1_scores:
            summary['f1_stats'] = {
                'mean': float(onp.mean(f1_scores)),
                'std': float(onp.std(f1_scores)),
                'min': float(onp.min(f1_scores)),
                'max': float(onp.max(f1_scores)),
                'final': f1_scores[-1],
                'improvement': f1_scores[-1] - f1_scores[0] if len(f1_scores) > 1 else 0.0
            }
        
        if parent_likelihoods:
            summary['parent_likelihood_stats'] = {
                'mean': float(onp.mean(parent_likelihoods)),
                'std': float(onp.std(parent_likelihoods)),
                'min': float(onp.min(parent_likelihoods)),
                'max': float(onp.max(parent_likelihoods)),
                'final': parent_likelihoods[-1],
                'improvement': parent_likelihoods[-1] - parent_likelihoods[0] if len(parent_likelihoods) > 1 else 0.0
            }
        
        if shd_values:
            summary['shd_stats'] = {
                'mean': float(onp.mean(shd_values)),
                'std': float(onp.std(shd_values)),
                'min': int(onp.min(shd_values)),
                'max': int(onp.max(shd_values)),
                'final': shd_values[-1]
            }
        
        return summary


class TrainingMetricsLogger:
    """Integration class for adding metric dashboard to training."""
    
    def __init__(self, dashboard: Optional[MetricDashboard] = None):
        """
        Initialize metrics logger.
        
        Args:
            dashboard: Optional metric dashboard, creates default if None
        """
        self.dashboard = dashboard or MetricDashboard()
        self.enabled = True
    
    def start_logging(self):
        """Start metric logging and dashboard."""
        if self.enabled:
            self.dashboard.start()
    
    def stop_logging(self):
        """Stop metric logging and dashboard."""
        if self.enabled:
            self.dashboard.stop()
    
    def log_episode_metrics(self, episode: int, metrics: Any):
        """
        Log metrics from a training episode.
        
        Args:
            episode: Episode number
            metrics: TrainingMetrics object or dict
        """
        if not self.enabled:
            return
        
        # Convert TrainingMetrics to dict if needed
        if hasattr(metrics, '__dict__'):
            metrics_dict = {
                'mean_reward': metrics.mean_reward,
                'f1_score': metrics.f1_score,
                'true_parent_likelihood': metrics.true_parent_likelihood,
                'shd': metrics.shd,
                'marginal_probs': metrics.marginal_probs,
                'scm_type': metrics.scm_type
            }
        else:
            metrics_dict = dict(metrics)
        
        self.dashboard.update_metrics(episode, metrics_dict)
    
    def export_final_metrics(self) -> Optional[Path]:
        """Export final metrics after training completion."""
        if self.enabled:
            return self.dashboard.export_metrics()
        return None
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics."""
        if self.enabled:
            return self.dashboard.get_summary_statistics()
        return {}


# Export public interface
__all__ = [
    'MetricDashboard',
    'TrainingMetricsLogger',
    'MetricSnapshot'
]