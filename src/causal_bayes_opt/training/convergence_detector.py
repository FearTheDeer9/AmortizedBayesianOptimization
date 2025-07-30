"""
Convergence Detection for GRPO Training

This module provides convergence detection to prevent over-training on solved SCMs
and maintain a balanced exploration/exploitation distribution during training.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np
from collections import deque

from .data_structures import TrainingMetrics


@dataclass
class ConvergenceConfig:
    """Configuration for convergence detection."""
    structure_accuracy_threshold: float = 0.95
    reward_variance_threshold: float = 0.05  # Tightened from 0.1
    patience: int = 5  # Reduced from 10 - faster convergence detection
    min_episodes: int = 5  # Reduced from 20 - allow early convergence
    max_episodes_per_scm: int = 30  # Reduced from 100 - prevent over-training
    use_rolling_window: bool = True
    window_size: int = 5  # Reduced from 10 - more responsive


@dataclass
class SCMTrainingState:
    """Tracks training state for a single SCM."""
    scm_name: str
    episodes_trained: int = 0
    best_structure_accuracy: float = 0.0
    converged: bool = False
    convergence_episode: Optional[int] = None
    
    # Rolling metrics
    recent_accuracies: deque = field(default_factory=lambda: deque(maxlen=5))
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=5))
    recent_f1_scores: deque = field(default_factory=lambda: deque(maxlen=5))
    
    # History for analysis
    accuracy_history: List[float] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)
    f1_history: List[float] = field(default_factory=list)
    
    def update(self, metrics: TrainingMetrics) -> None:
        """Update state with new metrics."""
        self.episodes_trained += 1
        
        # Update rolling metrics
        self.recent_accuracies.append(metrics.structure_accuracy)
        self.recent_rewards.append(metrics.mean_reward)
        if metrics.f1_score is not None:
            self.recent_f1_scores.append(metrics.f1_score)
        
        # Update history
        self.accuracy_history.append(metrics.structure_accuracy)
        self.reward_history.append(metrics.mean_reward)
        if metrics.f1_score is not None:
            self.f1_history.append(metrics.f1_score)
        
        # Track best accuracy
        self.best_structure_accuracy = max(
            self.best_structure_accuracy, 
            metrics.structure_accuracy
        )
    
    def get_recent_stats(self) -> Dict[str, float]:
        """Get statistics from recent episodes."""
        stats = {}
        
        if self.recent_accuracies:
            stats['mean_accuracy'] = np.mean(list(self.recent_accuracies))
            stats['std_accuracy'] = np.std(list(self.recent_accuracies))
        
        if self.recent_rewards:
            stats['mean_reward'] = np.mean(list(self.recent_rewards))
            stats['std_reward'] = np.std(list(self.recent_rewards))
        
        if self.recent_f1_scores:
            stats['mean_f1'] = np.mean(list(self.recent_f1_scores))
            stats['std_f1'] = np.std(list(self.recent_f1_scores))
        
        return stats


class ConvergenceDetector:
    """
    Detects when an SCM has been sufficiently learned to avoid over-training.
    
    This prevents the posterior collapse that occurs when we continue training
    on solved SCMs, leading to over-representation of greedy optimization.
    """
    
    def __init__(self, config: ConvergenceConfig):
        self.config = config
        self.scm_states: Dict[str, SCMTrainingState] = {}
        self.patience_counters: Dict[str, int] = {}
    
    def update(self, scm_name: str, metrics: TrainingMetrics) -> None:
        """Update convergence tracking with new metrics."""
        # Initialize state if needed
        if scm_name not in self.scm_states:
            self.scm_states[scm_name] = SCMTrainingState(scm_name=scm_name)
            self.patience_counters[scm_name] = 0
        
        # Update state
        state = self.scm_states[scm_name]
        state.update(metrics)
    
    def check_convergence(self, scm_name: str) -> Tuple[bool, str]:
        """
        Check if SCM has converged based on multiple criteria.
        
        Returns:
            (converged, reason): Whether converged and explanation
        """
        if scm_name not in self.scm_states:
            return False, "No training data"
        
        state = self.scm_states[scm_name]
        
        # Already converged
        if state.converged:
            return True, f"Already converged at episode {state.convergence_episode}"
        
        # Check max episodes
        if state.episodes_trained >= self.config.max_episodes_per_scm:
            state.converged = True
            state.convergence_episode = state.episodes_trained
            return True, f"Reached max episodes ({self.config.max_episodes_per_scm})"
        
        # Need minimum episodes
        if state.episodes_trained < self.config.min_episodes:
            return False, f"Only {state.episodes_trained} episodes (min: {self.config.min_episodes})"
        
        # Get recent statistics
        stats = state.get_recent_stats()
        
        # Check structure accuracy convergence
        if 'mean_accuracy' in stats:
            if stats['mean_accuracy'] >= self.config.structure_accuracy_threshold:
                # Check if accuracy is stable
                if stats['std_accuracy'] < 0.05:  # Less than 5% variation
                    self.patience_counters[scm_name] += 1
                    
                    if self.patience_counters[scm_name] >= self.config.patience:
                        state.converged = True
                        state.convergence_episode = state.episodes_trained
                        return True, (
                            f"Structure converged: accuracy={stats['mean_accuracy']:.3f} "
                            f"(std={stats['std_accuracy']:.3f})"
                        )
                else:
                    self.patience_counters[scm_name] = 0
            else:
                self.patience_counters[scm_name] = 0
        
        # Check reward stability (indicates optimization convergence)
        if 'std_reward' in stats and 'mean_accuracy' in stats:
            # Only consider reward stability if structure is reasonably learned
            if stats['mean_accuracy'] >= 0.8 and stats['std_reward'] < self.config.reward_variance_threshold:
                # Reward is stable and structure is good
                if state.episodes_trained > self.config.min_episodes:
                    state.converged = True
                    state.convergence_episode = state.episodes_trained
                    return True, (
                        f"Optimization converged: accuracy={stats['mean_accuracy']:.3f}, "
                        f"reward_std={stats['std_reward']:.3f}"
                    )
        
        return False, f"Not converged (patience: {self.patience_counters[scm_name]}/{self.config.patience})"
    
    def should_continue_training(self, scm_name: str) -> bool:
        """Check if we should continue training on this SCM."""
        converged, _ = self.check_convergence(scm_name)
        return not converged
    
    def get_scm_summary(self, scm_name: str) -> Dict[str, any]:
        """Get training summary for an SCM."""
        if scm_name not in self.scm_states:
            return {"error": "SCM not found"}
        
        state = self.scm_states[scm_name]
        stats = state.get_recent_stats()
        
        return {
            "scm_name": scm_name,
            "episodes_trained": state.episodes_trained,
            "converged": state.converged,
            "convergence_episode": state.convergence_episode,
            "best_structure_accuracy": state.best_structure_accuracy,
            "recent_stats": stats
        }
    
    def get_training_summary(self) -> Dict[str, any]:
        """Get summary of all SCM training."""
        total_episodes = sum(s.episodes_trained for s in self.scm_states.values())
        converged_scms = sum(1 for s in self.scm_states.values() if s.converged)
        
        # Calculate phase distribution
        discovery_episodes = 0
        exploitation_episodes = 0
        
        for state in self.scm_states.values():
            if state.accuracy_history:
                # Count episodes in discovery phase (accuracy < 90%)
                discovery_count = sum(1 for acc in state.accuracy_history if acc < 0.9)
                discovery_episodes += discovery_count
                exploitation_episodes += len(state.accuracy_history) - discovery_count
        
        return {
            "total_scms": len(self.scm_states),
            "converged_scms": converged_scms,
            "total_episodes": total_episodes,
            "average_episodes_per_scm": total_episodes / max(1, len(self.scm_states)),
            "discovery_episodes": discovery_episodes,
            "exploitation_episodes": exploitation_episodes,
            "discovery_ratio": discovery_episodes / max(1, total_episodes),
            "scm_summaries": {
                name: self.get_scm_summary(name) 
                for name in self.scm_states
            }
        }