#!/usr/bin/env python3
"""
Quick test to verify GRPO rewards are being computed correctly.
"""

import sys
sys.path.append('.')

import numpy as np
from src.causal_bayes_opt.experiments.benchmark_scms import create_chain_scm
from src.causal_bayes_opt.acquisition.grpo_rewards import compute_grpo_reward, compute_group_advantages
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm

# Create chain SCM
scm = create_chain_scm()
print(f"Chain SCM: X0 -> X1 -> X2 (target)")

# Create buffer with observational data
buffer = ExperienceBuffer()
obs_samples = sample_from_linear_scm(scm, 10, seed=42)
for sample in obs_samples:
    buffer.add_observation(sample)

# Test interventions on different variables
test_cases = [
    ("X0", -1.0),  # Non-parent
    ("X1", -2.0),  # Parent (good)
    ("X0", 2.0),   # Non-parent
    ("X1", 1.0),   # Parent (bad value for minimization)
]

rewards = []
for var, value in test_cases:
    # Create intervention
    intervention = create_perfect_intervention(
        targets=frozenset([var]),
        values={var: value}
    )
    
    # Sample outcome
    outcomes = sample_with_intervention(scm, intervention, n_samples=1, seed=42)
    outcome = outcomes[0]
    
    # Compute reward
    reward_comp = compute_grpo_reward(
        scm=scm,
        intervention=intervention,
        outcome=outcome,
        target_variable='X2',
        buffer_before=buffer,
        config={
            'optimization_direction': 'MINIMIZE',
            'improvement_threshold': 0.1,
            'reward_weights': {
                'variable_selection': 0.5,
                'value_selection': 0.5,
                'parent_bonus': 0.3,
                'improvement_bonus': 0.2
            }
        }
    )
    
    rewards.append(reward_comp.total_reward)
    
    print(f"\nIntervention: {var} = {value}")
    print(f"  Parent: {reward_comp.correct_parent}")
    print(f"  Improved: {reward_comp.improved_beyond_threshold}")
    print(f"  Total reward: {reward_comp.total_reward:.3f}")
    print(f"  Components: parent={reward_comp.parent_intervention:.3f}, "
          f"improvement={reward_comp.target_improvement:.3f}, "
          f"value={reward_comp.value_optimization:.3f}")

# Test group advantages
print(f"\nGroup rewards: {[f'{r:.3f}' for r in rewards]}")
advantages = compute_group_advantages(rewards, method='zscore')
print(f"Z-score advantages: {[f'{a:.3f}' for a in advantages]}")

# Check if advantages sum to ~0
print(f"Sum of advantages: {sum(advantages):.6f} (should be ~0)")
print(f"Mean advantage: {np.mean(advantages):.6f}")
print(f"Std advantage: {np.std(advantages):.6f}")