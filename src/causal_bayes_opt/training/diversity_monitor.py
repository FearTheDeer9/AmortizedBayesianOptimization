"""Diversity monitoring system for GRPO training.

This module implements diversity monitoring to prevent mode collapse and reward
gaming, inspired by the verifiers repository's emphasis on reward diversity
within training groups.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import logging

import jax.numpy as jnp
import numpy as onp

from ..acquisition.reward_rubric import RewardResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiversityMetrics:
    """Diversity metrics for monitoring reward distribution health.
    
    Args:
        reward_variance: Variance of total rewards in the batch
        component_entropies: Entropy of each reward component
        mode_collapse_risk: Risk score for mode collapse (0-1)
        below_threshold: Whether diversity is below warning threshold
        batch_size: Size of the analyzed batch
        timestamp: When these metrics were computed
    """
    reward_variance: float
    component_entropies: Dict[str, float]
    mode_collapse_risk: float
    below_threshold: bool
    batch_size: int
    timestamp: float
    
    def get_overall_health_score(self) -> float:
        """Compute overall diversity health score (0-1, higher is better)."""
        # Base score from variance (normalized)
        variance_score = min(self.reward_variance / 1.0, 1.0)  # Assume 1.0 is good variance
        
        # Penalty for mode collapse risk
        collapse_penalty = self.mode_collapse_risk
        
        # Entropy score (average of component entropies)
        entropy_score = 0.0
        if self.component_entropies:
            entropy_score = sum(self.component_entropies.values()) / len(self.component_entropies)
            entropy_score = min(entropy_score / 2.0, 1.0)  # Normalize assuming 2.0 is good entropy
        
        # Combined score
        health_score = 0.4 * variance_score + 0.3 * entropy_score + 0.3 * (1.0 - collapse_penalty)
        return max(0.0, min(1.0, health_score))


@dataclass(frozen=True)
class DiversityAlert:
    """Alert for diversity-related issues.
    
    Args:
        severity: Alert severity (low, medium, high, critical)
        message: Human-readable alert message
        metrics: Associated diversity metrics
        suggested_action: Recommended remediation action
    """
    severity: str
    message: str
    metrics: DiversityMetrics
    suggested_action: str


class DiversityMonitor:
    """Monitor and prevent reward gaming through diversity analysis.
    
    This class tracks reward diversity within GRPO batches and detects
    concerning patterns that indicate mode collapse or reward hacking.
    
    Args:
        diversity_threshold: Minimum reward variance to maintain
        history_length: Number of recent batches to track
        alert_threshold: Health score below which to raise alerts
    """
    
    def __init__(
        self,
        diversity_threshold: float = 0.3,
        history_length: int = 50,
        alert_threshold: float = 0.5
    ):
        self.diversity_threshold = diversity_threshold
        self.history_length = history_length
        self.alert_threshold = alert_threshold
        
        # Internal state
        self._metrics_history: List[DiversityMetrics] = []
        self._alert_history: List[DiversityAlert] = []
        
    def compute_batch_diversity(
        self,
        reward_results: List[RewardResult],
        timestamp: Optional[float] = None
    ) -> DiversityMetrics:
        """Compute diversity metrics for a batch of rewards.
        
        Args:
            reward_results: List of reward results from GRPO batch
            timestamp: Optional timestamp for this batch
            
        Returns:
            DiversityMetrics for this batch
        """
        if not reward_results:
            return DiversityMetrics(
                reward_variance=0.0,
                component_entropies={},
                mode_collapse_risk=1.0,  # Max risk for empty batch
                below_threshold=True,
                batch_size=0,
                timestamp=timestamp or 0.0
            )
        
        # Extract total rewards
        total_rewards = jnp.array([r.total_reward for r in reward_results])
        reward_variance = float(jnp.var(total_rewards))
        
        # Compute per-component entropies
        component_entropies = {}
        all_component_names = set()
        for result in reward_results:
            all_component_names.update(result.component_rewards.keys())
        
        for name in all_component_names:
            values = jnp.array([
                result.component_rewards.get(name, 0.0) 
                for result in reward_results
            ])
            
            # Compute entropy from value distribution
            # Discretize values into bins for entropy calculation
            if jnp.std(values) > 1e-6:  # Only if there's variation
                hist, _ = onp.histogram(values, bins=min(10, len(values)))
                hist = hist + 1e-10  # Avoid log(0)
                probs = hist / hist.sum()
                entropy = float(-jnp.sum(probs * jnp.log(probs)))
            else:
                entropy = 0.0  # No diversity
            
            component_entropies[name] = entropy
        
        # Compute mode collapse risk
        mode_collapse_risk = self._compute_mode_collapse_risk(
            total_rewards, reward_results
        )
        
        # Check if below threshold
        below_threshold = reward_variance < self.diversity_threshold
        
        metrics = DiversityMetrics(
            reward_variance=reward_variance,
            component_entropies=component_entropies,
            mode_collapse_risk=mode_collapse_risk,
            below_threshold=below_threshold,
            batch_size=len(reward_results),
            timestamp=timestamp or 0.0
        )
        
        # Add to history
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > self.history_length:
            self._metrics_history.pop(0)
        
        return metrics
    
    def _compute_mode_collapse_risk(
        self,
        total_rewards: jnp.ndarray,
        reward_results: List[RewardResult]
    ) -> float:
        """Compute risk of mode collapse based on reward patterns."""
        if len(total_rewards) < 2:
            return 1.0
        
        # Check for suspiciously uniform rewards
        reward_std = float(jnp.std(total_rewards))
        if reward_std < 1e-6:
            return 1.0  # All rewards identical
        
        # Check for clustering in high rewards (potential gaming)
        reward_mean = float(jnp.mean(total_rewards))
        high_reward_mask = total_rewards > reward_mean + reward_std
        high_reward_fraction = float(jnp.sum(high_reward_mask)) / len(total_rewards)
        
        # High fraction of high rewards suggests potential gaming
        gaming_risk = min(high_reward_fraction * 2.0, 1.0)
        
        # Check for component dominance (one component driving all rewards)
        component_dominance_risk = 0.0
        if reward_results:
            for result in reward_results:
                if result.component_rewards:
                    max_component = max(result.component_rewards.values())
                    total_reward = result.total_reward
                    if total_reward > 0:
                        dominance = max_component / total_reward
                        component_dominance_risk = max(component_dominance_risk, dominance)
        
        # Combined risk score
        overall_risk = 0.4 * gaming_risk + 0.6 * component_dominance_risk
        return min(overall_risk, 1.0)
    
    def check_for_alerts(self, metrics: DiversityMetrics) -> List[DiversityAlert]:
        """Check if metrics warrant any alerts.
        
        Args:
            metrics: Current diversity metrics
            
        Returns:
            List of alerts to raise
        """
        alerts = []
        
        # Check overall health
        health_score = metrics.get_overall_health_score()
        if health_score < self.alert_threshold:
            severity = "critical" if health_score < 0.3 else "high"
            alerts.append(DiversityAlert(
                severity=severity,
                message=f"Low diversity health score: {health_score:.3f}",
                metrics=metrics,
                suggested_action="increase_task_difficulty"
            ))
        
        # Check variance threshold
        if metrics.below_threshold:
            alerts.append(DiversityAlert(
                severity="medium",
                message=f"Reward variance {metrics.reward_variance:.3f} below threshold {self.diversity_threshold}",
                metrics=metrics,
                suggested_action="increase_environment_diversity"
            ))
        
        # Check mode collapse risk
        if metrics.mode_collapse_risk > 0.7:
            alerts.append(DiversityAlert(
                severity="high",
                message=f"High mode collapse risk: {metrics.mode_collapse_risk:.3f}",
                metrics=metrics,
                suggested_action="reset_training_or_adjust_rewards"
            ))
        
        # Check component entropy
        low_entropy_components = [
            name for name, entropy in metrics.component_entropies.items()
            if entropy < 0.5
        ]
        if len(low_entropy_components) > len(metrics.component_entropies) / 2:
            alerts.append(DiversityAlert(
                severity="medium",
                message=f"Low entropy in components: {low_entropy_components}",
                metrics=metrics,
                suggested_action="adjust_component_weights"
            ))
        
        # Store alerts
        self._alert_history.extend(alerts)
        if len(self._alert_history) > self.history_length:
            self._alert_history = self._alert_history[-self.history_length:]
        
        return alerts
    
    def get_recent_trends(self, window_size: int = 10) -> Dict[str, Any]:
        """Analyze recent diversity trends.
        
        Args:
            window_size: Number of recent batches to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        if len(self._metrics_history) < 2:
            return {"trend": "insufficient_data"}
        
        recent_metrics = self._metrics_history[-window_size:]
        
        # Variance trend
        variances = [m.reward_variance for m in recent_metrics]
        variance_trend = "improving" if variances[-1] > variances[0] else "declining"
        
        # Health score trend
        health_scores = [m.get_overall_health_score() for m in recent_metrics]
        health_trend = "improving" if health_scores[-1] > health_scores[0] else "declining"
        
        # Alert frequency
        recent_alerts = [
            alert for alert in self._alert_history
            if any(alert.metrics.timestamp >= m.timestamp for m in recent_metrics)
        ]
        alert_frequency = len(recent_alerts) / len(recent_metrics)
        
        return {
            "variance_trend": variance_trend,
            "health_trend": health_trend,
            "alert_frequency": alert_frequency,
            "current_health": health_scores[-1] if health_scores else 0.0,
            "current_variance": variances[-1] if variances else 0.0,
        }
    
    def suggest_intervention(self, metrics: DiversityMetrics) -> str:
        """Suggest intervention based on current metrics.
        
        Args:
            metrics: Current diversity metrics
            
        Returns:
            Suggested intervention action
        """
        health_score = metrics.get_overall_health_score()
        
        if health_score < 0.3:
            return "restart_training"
        elif metrics.mode_collapse_risk > 0.8:
            return "increase_exploration_noise"
        elif metrics.below_threshold:
            return "increase_task_difficulty"
        elif len(metrics.component_entropies) > 0:
            min_entropy = min(metrics.component_entropies.values())
            if min_entropy < 0.3:
                return "rebalance_reward_weights"
        
        return "continue_monitoring"
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get current monitoring status summary.
        
        Returns:
            Dictionary with current status
        """
        if not self._metrics_history:
            return {"status": "no_data"}
        
        latest_metrics = self._metrics_history[-1]
        recent_alerts = [a for a in self._alert_history[-5:]]
        trends = self.get_recent_trends()
        
        return {
            "current_health": latest_metrics.get_overall_health_score(),
            "current_variance": latest_metrics.reward_variance,
            "mode_collapse_risk": latest_metrics.mode_collapse_risk,
            "recent_alerts": len(recent_alerts),
            "trend": trends.get("health_trend", "unknown"),
            "suggested_action": self.suggest_intervention(latest_metrics),
            "batches_monitored": len(self._metrics_history),
        }


def create_diversity_monitor(
    config_threshold: float = 0.3,
    history_length: int = 50,
    alert_threshold: float = 0.5,
) -> DiversityMonitor:
    """Create a diversity monitor with standard configuration.
    
    Args:
        config_threshold: Minimum reward variance threshold
        history_length: Number of batches to track in history
        alert_threshold: Health score threshold for alerts
        
    Returns:
        Configured DiversityMonitor instance
    """
    return DiversityMonitor(
        diversity_threshold=config_threshold,
        history_length=history_length,
        alert_threshold=alert_threshold
    )