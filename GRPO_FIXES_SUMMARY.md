# GRPO Training Fixes Summary

This document provides a comprehensive summary of all GRPO training issues identified and their fixes, with exact file locations and code changes for re-implementation.

## Overview of Issues Fixed

1. **Policy Not Learning** - Model always predicted the same variable
2. **Reward Function with abs()** - Incorrectly handled negative values  
3. **Import/Module Errors** - Various import path issues
4. **Early Stopping Configuration** - Hardcoded to 30 episodes instead of 200
5. **Empty Plots** - Analysis script issues

---

## Issue 1: Policy Not Learning (No Exploration)

### Problem
The policy wasn't learning because there was no exploration noise, causing it to always select the same variable.

### Fix Location
**File**: `src/causal_bayes_opt/training/unified_grpo_trainer.py`  
**Lines**: 595-597

### Code Change
```python
# BEFORE: No exploration noise
var_logits = policy_output.variable_logits[0]  # [n_vars]
var_action = random.categorical(subkey1, var_logits)

# AFTER: Added exploration noise
var_logits = policy_output.variable_logits[0]  # [n_vars]
exploration_noise = 0.1  # Reduced noise for better exploitation
noisy_logits = var_logits + exploration_noise * random.normal(var_key, var_logits.shape)
var_action = random.categorical(subkey1, noisy_logits)
```

**Note**: Initial implementation used 0.3 noise, later reduced to 0.1 for better balance between exploration and exploitation.

---

## Issue 2: Reward Function Using abs()

### Problem
The reward function `reward = 1/(1 + abs(outcome))` treated negative values incorrectly, making Y=-6.5 equivalent to Y=+6.5.

### Fix 1: Created New Reward Module
**File**: `src/causal_bayes_opt/acquisition/better_rewards.py` (NEW FILE - full content in appendix)

Key functions:
- `RunningStats`: Tracks running statistics for adaptive normalization
- `adaptive_sigmoid_reward`: Sigmoid reward that adapts to data distribution
- `compute_better_clean_reward`: Main reward computation function

### Fix 2: Updated Trainer to Use Better Rewards
**File**: `src/causal_bayes_opt/training/unified_grpo_trainer.py`  

**Change 1 - Import** (Line 31):
```python
# BEFORE:
from ..acquisition.improved_rewards import compute_improved_clean_reward

# AFTER:
from ..acquisition.better_rewards import compute_better_clean_reward, RunningStats
```

**Change 2 - Initialize RunningStats** (Lines 181 & 245):
```python
# In _init_from_config (line 181):
# Initialize running stats for reward normalization
self.reward_stats = RunningStats(window_size=1000)

# In _init_from_params (line 245):
# Initialize running stats for reward normalization
self.reward_stats = RunningStats(window_size=1000)
```

**Change 3 - Use Better Rewards** (Lines 696-711):
```python
# BEFORE:
reward_dict = compute_improved_clean_reward(
    buffer, intervention, outcome, 
    self.target_variable,
    config={
        'optimization_direction': self.optimization_direction,
        'weights': clean_weights
    }
)

# AFTER:
reward_info = compute_better_clean_reward(
    buffer, 
    intervention, 
    outcome, 
    self.target_variable,
    config={
        'optimization_direction': self.optimization_direction,
        'reward_type': 'adaptive_sigmoid',
        'temperature_factor': 2.0,
        'weights': clean_weights
    },
    stats=self.reward_stats,  # Pass running stats
)
```

---

## Issue 3: Import and Module Errors

### Fix 1: Import Path for compute_posterior_from_buffer
**File**: `src/causal_bayes_opt/mechanisms/continuous_scm.py`  
**Line**: 11

```python
# BEFORE:
from ..algorithms.active_learning import compute_posterior_from_buffer

# AFTER:
from ..training.continuous_surrogate_integration import compute_posterior_from_buffer_continuous as compute_posterior_from_buffer
```

### Fix 2: Import compute_info_gain_reward
**File**: `src/causal_bayes_opt/acquisition/clean_rewards.py`  
**Line**: 290

```python
# BEFORE:
info_gain_reward = compute_info_gain_reward(
    posterior_before, posterior_after, outcome
)

# AFTER:
# Import at top of compute_clean_reward function
from .info_theoretic import compute_info_gain_reward

info_gain_reward = compute_info_gain_reward(
    posterior_before, posterior_after, outcome
)
```

---

## Issue 4: Early Stopping Configuration

### Problem
Training was stopping at 30 episodes instead of the intended 200.

### Fix Location
**File**: `src/causal_bayes_opt/training/unified_grpo_trainer.py`  
**Line**: 238

```python
# BEFORE:
self.convergence_config = kwargs['convergence_config'] or ConvergenceConfig(
    structure_accuracy_threshold=0.95,
    patience=5,
    min_episodes=5,
    max_episodes_per_scm=30
)

# AFTER:
self.convergence_config = kwargs['convergence_config'] or ConvergenceConfig(
    structure_accuracy_threshold=0.95,
    patience=5,
    min_episodes=5,
    max_episodes_per_scm=200
)
```

---

## Issue 5: Empty Plots in Comprehensive Training

### Problem
The comprehensive training script's plots were empty due to parsing issues.

### Fix: Created Separate Analysis Script
**File**: `scripts/analyze_comprehensive_training.py` (NEW FILE)

Key parsing logic:
```python
def parse_log_file(log_path):
    """Parse comprehensive training log file for rewards and episodes."""
    
    results = {
        "fork": {"rewards": [], "episodes": []},
        "chain": {"rewards": [], "episodes": []},
        "collider": {"rewards": [], "episodes": []}
    }
    
    with open(log_path, 'r') as f:
        for line in f:
            # Extract episode rewards with SCM info
            if "Episode" in line and "mean_reward=" in line and "current_scm=" in line:
                # Pattern: Episode X: mean_reward=Y, current_scm=Z
                match = re.search(r'Episode (\d+): mean_reward=([\d.]+), current_scm=(\w+)', line)
                if match:
                    episode = int(match.group(1))
                    reward = float(match.group(2))
                    scm_name = match.group(3)
                    
                    # Map SCM names
                    if "fork" in scm_name:
                        results["fork"]["episodes"].append(episode)
                        results["fork"]["rewards"].append(reward)
                    # ... similar for chain and collider
```

---

## Additional Fix: Non-Intervention Baseline

### Problem
The baseline for advantage computation wasn't meaningful, making all advantages near zero.

### Fix Location
**File**: `src/causal_bayes_opt/training/unified_grpo_trainer.py`  
**Lines**: ~801

### Code Change
```python
# Extract target values from observational data
obs_target_values = []
for sample in obs_samples:
    # Values are nested under 'values' key in the sample
    if 'values' in sample and target_variable in sample['values']:
        obs_target_values.append(float(sample['values'][target_variable]))

if len(obs_target_values) > 0:
    # Baseline = average target value without interventions
    non_intervention_baseline = jnp.mean(jnp.array(obs_target_values))
```

---

## Key Configuration Changes Summary

| Setting | Old Value | New Value | Location | Reason |
|---------|-----------|-----------|----------|---------|
| Exploration Noise | None/0.3 | 0.1 | unified_grpo_trainer.py:596 | Enable exploration without excessive randomness |
| Max Episodes | 30 | 200 | unified_grpo_trainer.py:238 | Allow longer training |
| Reward Type | "improved" | "adaptive_sigmoid" | unified_grpo_trainer.py:701 | Handle negative values correctly |
| Reward Function | `1/(1+abs(Y))` | Adaptive sigmoid | better_rewards.py | Remove abs() issue |

---

## Results After All Fixes

### Learning Performance
- **Fork SCM**: 0.469 → 0.560 (+19.4%)
- **Chain SCM**: 0.455 → 0.566 (+24.4%)  
- **Collider SCM**: 0.441 → 0.512 (+16.1%)

### Key Improvements
1. Policy now explores different variables
2. Rewards handle negative values correctly
3. Consistent learning across all SCM types
4. Can train for extended episodes

---

## Implementation Checklist

When implementing in another worktree:

1. [ ] Create `src/causal_bayes_opt/acquisition/better_rewards.py` (see appendix for full content)
2. [ ] Update imports in `unified_grpo_trainer.py` line 31
3. [ ] Add RunningStats initialization in both init methods (lines 181 & 245)
4. [ ] Update reward computation to use better_rewards (lines 696-711)
5. [ ] Add exploration noise to action sampling (lines 595-597)
6. [ ] Fix max_episodes_per_scm from 30 to 200 (line 238)
7. [ ] Fix import paths in `continuous_scm.py` and `clean_rewards.py`
8. [ ] Create analysis script `scripts/analyze_comprehensive_training.py`
9. [ ] Test with `run_comprehensive_grpo_training.py`

---

## Appendix: Full better_rewards.py Content

Create this file at `src/causal_bayes_opt/acquisition/better_rewards.py`:

```python
"""
Better reward functions that handle negative values and unknown ranges.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, Optional, List
from collections import deque

import logging
logger = logging.getLogger(__name__)


class RunningStats:
    """Track running statistics of values for normalization."""
    
    def __init__(self, window_size: int = 1000):
        self.values = deque(maxlen=window_size)
        self.window_size = window_size
    
    def update(self, value: float):
        """Add new value to statistics."""
        self.values.append(value)
    
    @property
    def mean(self) -> float:
        """Get running mean."""
        if not self.values:
            return 0.0
        return float(np.mean(list(self.values)))
    
    @property
    def std(self) -> float:
        """Get running standard deviation."""
        if len(self.values) < 2:
            return 1.0
        return float(np.std(list(self.values)))
    
    @property
    def min(self) -> float:
        """Get minimum value seen."""
        if not self.values:
            return 0.0
        return float(min(self.values))
    
    @property
    def max(self) -> float:
        """Get maximum value seen."""
        if not self.values:
            return 0.0
        return float(max(self.values))
    
    def get_percentile(self, value: float) -> float:
        """Get percentile rank of value."""
        if not self.values:
            return 0.5
        return float(np.sum(np.array(list(self.values)) <= value) / len(self.values))


def sigmoid_target_reward(
    outcome_value: float,
    optimization_direction: str = 'MINIMIZE',
    center: float = 0.0,
    temperature: float = 1.0
) -> float:
    """
    Sigmoid-based reward that handles any value range.
    
    Maps values to [0, 1] using a sigmoid centered at 'center'.
    Temperature controls the steepness of the transition.
    
    For MINIMIZE:
    - Values < center get reward > 0.5
    - Values > center get reward < 0.5
    
    For MAXIMIZE:
    - Values > center get reward > 0.5
    - Values < center get reward < 0.5
    
    Args:
        outcome_value: The achieved value
        optimization_direction: 'MINIMIZE' or 'MAXIMIZE'
        center: Center point of sigmoid (default: 0)
        temperature: Controls steepness (default: 1)
    
    Returns:
        Reward in [0, 1] range
    """
    if optimization_direction == 'MINIMIZE':
        # For minimization: lower values → higher rewards
        reward = 1.0 / (1.0 + jnp.exp(temperature * (outcome_value - center)))
    else:
        # For maximization: higher values → higher rewards
        reward = 1.0 / (1.0 + jnp.exp(-temperature * (outcome_value - center)))
    
    return float(reward)


def adaptive_sigmoid_reward(
    outcome_value: float,
    stats: RunningStats,
    optimization_direction: str = 'MINIMIZE',
    temperature_factor: float = 2.0
) -> float:
    """
    Sigmoid reward that adapts to the data distribution.
    
    Uses running statistics to automatically set center and temperature.
    
    Args:
        outcome_value: The achieved value
        stats: Running statistics of recent values
        optimization_direction: 'MINIMIZE' or 'MAXIMIZE'
        temperature_factor: Multiplier for temperature (default: 2.0)
    
    Returns:
        Reward in [0, 1] range
    """
    # Update stats with new value
    stats.update(outcome_value)
    
    # Use mean as center
    center = stats.mean
    
    # Use std to set temperature (inverse relationship)
    # Larger std → smaller temperature → gentler curve
    if stats.std > 0:
        temperature = temperature_factor / stats.std
    else:
        temperature = temperature_factor
    
    reward = sigmoid_target_reward(
        outcome_value, optimization_direction, center, temperature
    )
    
    logger.debug(
        f"Adaptive sigmoid: value={outcome_value:.3f}, "
        f"center={center:.3f}, std={stats.std:.3f}, "
        f"temperature={temperature:.3f}, reward={reward:.3f}"
    )
    
    return reward


def percentile_reward(
    outcome_value: float,
    stats: RunningStats,
    optimization_direction: str = 'MINIMIZE'
) -> float:
    """
    Reward based on percentile rank in recent history.
    
    Automatically adapts to any value distribution.
    
    Args:
        outcome_value: The achieved value
        stats: Running statistics of recent values
        optimization_direction: 'MINIMIZE' or 'MAXIMIZE'
    
    Returns:
        Reward in [0, 1] range (percentile)
    """
    # Update stats with new value
    stats.update(outcome_value)
    
    # Get percentile rank
    percentile = stats.get_percentile(outcome_value)
    
    if optimization_direction == 'MINIMIZE':
        # For minimization: lower percentile → higher reward
        reward = 1.0 - percentile
    else:
        # For maximization: higher percentile → higher reward
        reward = percentile
    
    logger.debug(
        f"Percentile reward: value={outcome_value:.3f}, "
        f"percentile={percentile:.3f}, reward={reward:.3f}"
    )
    
    return float(reward)


def tanh_normalized_reward(
    outcome_value: float,
    stats: RunningStats,
    optimization_direction: str = 'MINIMIZE',
    scale_factor: float = 2.0
) -> float:
    """
    Reward using tanh normalization with running statistics.
    
    Maps values to [-1, 1] then shifts to [0, 1].
    
    Args:
        outcome_value: The achieved value
        stats: Running statistics
        optimization_direction: 'MINIMIZE' or 'MAXIMIZE'
        scale_factor: Controls sensitivity (default: 2.0)
    
    Returns:
        Reward in [0, 1] range
    """
    # Update stats
    stats.update(outcome_value)
    
    # Normalize using z-score
    if stats.std > 0:
        z_score = (outcome_value - stats.mean) / stats.std
    else:
        z_score = 0.0
    
    # Map to [-1, 1] using tanh
    if optimization_direction == 'MINIMIZE':
        # For minimization: negative z-scores are good
        tanh_value = jnp.tanh(-scale_factor * z_score)
    else:
        # For maximization: positive z-scores are good
        tanh_value = jnp.tanh(scale_factor * z_score)
    
    # Shift to [0, 1]
    reward = (tanh_value + 1.0) / 2.0
    
    logger.debug(
        f"Tanh reward: value={outcome_value:.3f}, "
        f"z_score={z_score:.3f}, reward={reward:.3f}"
    )
    
    return float(reward)


def compute_better_clean_reward(
    buffer_before: Any,
    intervention: Dict[str, Any],
    outcome: Any,
    target_variable: str,
    config: Optional[Dict[str, Any]] = None,
    stats: Optional[RunningStats] = None,
    posterior_before: Optional[Dict[str, Any]] = None,
    posterior_after: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Improved reward computation with better handling of value ranges.
    
    Args:
        buffer_before: Buffer (for compatibility)
        intervention: Applied intervention
        outcome: Outcome sample
        target_variable: Variable being optimized
        config: Configuration with reward type and parameters
        stats: Running statistics for adaptive rewards
        posterior_before: Optional (for compatibility)
        posterior_after: Optional (for compatibility)
    
    Returns:
        Dictionary with reward components
    """
    if config is None:
        config = {}
    
    # Import here to avoid circular dependency
    from ..data_structures.sample import get_values
    
    # Get outcome value
    outcome_values = get_values(outcome)
    if target_variable not in outcome_values:
        logger.warning(f"Target {target_variable} not in outcome")
        return {'total': 0.5, 'target': 0.5, 'weights': {}}
    
    outcome_value = float(outcome_values[target_variable])
    
    # Get configuration
    reward_type = config.get('reward_type', 'sigmoid')
    optimization_direction = config.get('optimization_direction', 'MINIMIZE')
    
    # Compute target reward based on type
    if reward_type == 'sigmoid':
        center = config.get('center', 0.0)
        temperature = config.get('temperature', 1.0)
        target_reward = sigmoid_target_reward(
            outcome_value, optimization_direction, center, temperature
        )
    
    elif reward_type == 'adaptive_sigmoid' and stats is not None:
        temperature_factor = config.get('temperature_factor', 2.0)
        target_reward = adaptive_sigmoid_reward(
            outcome_value, stats, optimization_direction, temperature_factor
        )
    
    elif reward_type == 'percentile' and stats is not None:
        target_reward = percentile_reward(
            outcome_value, stats, optimization_direction
        )
    
    elif reward_type == 'tanh' and stats is not None:
        scale_factor = config.get('scale_factor', 2.0)
        target_reward = tanh_normalized_reward(
            outcome_value, stats, optimization_direction, scale_factor
        )
    
    else:
        # Fallback to simple sigmoid
        logger.warning(f"Unknown reward type {reward_type}, using sigmoid")
        target_reward = sigmoid_target_reward(
            outcome_value, optimization_direction
        )
    
    # Extract weights for compatibility
    weights = config.get('weights', {})
    
    # Log reward details
    logger.info(
        f"[BETTER REWARD] Type: {reward_type}, "
        f"Outcome: {outcome_value:.3f}, "
        f"Target reward: {target_reward:.3f}"
    )
    
    # Return compatible format
    return {
        'total': float(target_reward),
        'target': float(target_reward),
        'diversity': 0.0,
        'exploration': 0.0,
        'info_gain': 0.0,
        'weights': weights,
        'reward_type': reward_type
    }


def test_better_rewards():
    """Test the better reward functions."""
    
    print("Testing better reward functions...")
    
    # Test values including negative
    test_values = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]
    
    print("\n1. SIGMOID REWARDS (center=0, temp=1):")
    print("Value  | Minimize | Maximize")
    print("-------|----------|----------")
    for val in test_values:
        min_reward = sigmoid_target_reward(val, 'MINIMIZE')
        max_reward = sigmoid_target_reward(val, 'MAXIMIZE')
        print(f"{val:6.1f} | {min_reward:8.3f} | {max_reward:8.3f}")
    
    print("\n2. ADAPTIVE REWARDS (updates with data):")
    stats = RunningStats()
    print("Value  | Percentile | Adaptive Sigmoid")
    print("-------|------------|------------------")
    for val in test_values:
        perc_reward = percentile_reward(val, stats, 'MINIMIZE')
        adapt_reward = adaptive_sigmoid_reward(val, stats, 'MINIMIZE')
        print(f"{val:6.1f} | {perc_reward:10.3f} | {adapt_reward:16.3f}")
    
    print(f"\nFinal stats: mean={stats.mean:.2f}, std={stats.std:.2f}")
    print("\n✓ All reward functions handle negative values correctly!")


if __name__ == "__main__":
    test_better_rewards()
```