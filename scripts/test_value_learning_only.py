#!/usr/bin/env python3
"""
Test value learning in isolation by fixing variable selection to X1.
This helps diagnose if the value optimization component works.
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from src.causal_bayes_opt.experiments.benchmark_scms import create_chain_scm
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.acquisition.grpo_rewards import compute_grpo_reward
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm

# Create chain SCM
scm = create_chain_scm()
print("Testing value learning on Chain SCM: X0 -> X1 -> X2 (target)")

# Create buffer
buffer = ExperienceBuffer()
obs_samples = sample_from_linear_scm(scm, 20, seed=42)
for sample in obs_samples:
    buffer.add_observation(sample)

# Test different intervention values on X1 (the parent)
test_values = np.linspace(-3, 3, 25)
rewards = []
target_outcomes = []
reward_components = []

for value in test_values:
    # Create intervention on X1 with this value
    intervention = create_perfect_intervention(
        targets=frozenset(['X1']),
        values={'X1': float(value)}
    )
    
    # Sample outcome
    outcomes = sample_with_intervention(scm, intervention, n_samples=1, seed=42)
    outcome = outcomes[0]
    
    # Get target value
    from src.causal_bayes_opt.data_structures.sample import get_values
    outcome_values = get_values(outcome)
    target_value = float(outcome_values['X2'])
    target_outcomes.append(target_value)
    
    # Compute reward
    reward_comp = compute_grpo_reward(
        scm=scm,
        intervention=intervention,
        outcome=outcome,
        target_variable='X2',
        buffer_before=buffer,
        config={
            'optimization_direction': 'MINIMIZE',
            'improvement_threshold': 0.05,
            'reward_weights': {
                'variable_selection': 0.3,
                'value_selection': 0.7,
                'parent_bonus': 0.3,
                'improvement_bonus': 0.3
            }
        }
    )
    
    rewards.append(reward_comp.total_reward)
    reward_components.append(reward_comp)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Target outcome vs intervention value
ax = axes[0, 0]
ax.plot(test_values, target_outcomes, 'b-', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Intervention Value on X1')
ax.set_ylabel('Target Outcome (X2)')
ax.set_title('Effect of X1 Intervention on Target')
ax.grid(True, alpha=0.3)

# Plot 2: Total reward vs intervention value
ax = axes[0, 1]
ax.plot(test_values, rewards, 'g-', linewidth=2)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Intervention Value on X1')
ax.set_ylabel('Total Reward')
ax.set_title('Reward Function Shape')
ax.grid(True, alpha=0.3)

# Find best intervention value
best_idx = np.argmax(rewards)
best_value = test_values[best_idx]
best_reward = rewards[best_idx]
ax.plot(best_value, best_reward, 'ro', markersize=10, label=f'Best: {best_value:.2f}')
ax.legend()

# Plot 3: Value optimization component
ax = axes[1, 0]
value_opt_rewards = [rc.value_optimization for rc in reward_components]
ax.plot(test_values, value_opt_rewards, 'm-', linewidth=2)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Intervention Value on X1')
ax.set_ylabel('Value Optimization Reward')
ax.set_title('Value Optimization Component')
ax.grid(True, alpha=0.3)

# Plot 4: Summary statistics
ax = axes[1, 1]
ax.axis('off')
summary_text = f"""Value Learning Analysis Summary:

Best intervention value: {best_value:.3f}
Best reward: {best_reward:.3f}
Target at best: {target_outcomes[best_idx]:.3f}

Reward range: [{min(rewards):.3f}, {max(rewards):.3f}]
Target range: [{min(target_outcomes):.3f}, {max(target_outcomes):.3f}]

Negative values better: {sum(1 for i, v in enumerate(test_values) if v < 0 and rewards[i] > rewards[len(test_values)//2])} / {len([v for v in test_values if v < 0])}
Positive values better: {sum(1 for i, v in enumerate(test_values) if v > 0 and rewards[i] > rewards[len(test_values)//2])} / {len([v for v in test_values if v > 0])}

Value optimization correlation with reward: {np.corrcoef(value_opt_rewards, rewards)[0,1]:.3f}
"""
ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
        fontsize=10, verticalalignment='top', fontfamily='monospace')

plt.suptitle('Value Learning Analysis (Fixed Variable = X1)', fontsize=14)
plt.tight_layout()
plt.savefig('value_learning_analysis.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to value_learning_analysis.png")

# Print detailed analysis
print("\nDetailed Analysis:")
print(f"Best intervention value: {best_value:.3f} (reward={best_reward:.3f})")
print(f"This achieves target value: {target_outcomes[best_idx]:.3f}")
print(f"\nFor MINIMIZE objective, negative X1 values should be preferred.")
print(f"Actual preference: {'CORRECT' if best_value < 0 else 'INCORRECT'}")

# Check reward gradient
gradient_at_zero = (rewards[len(test_values)//2 + 1] - rewards[len(test_values)//2 - 1]) / (test_values[1] - test_values[0]) / 2
print(f"\nReward gradient at zero: {gradient_at_zero:.3f}")
print(f"{'Good' if abs(gradient_at_zero) > 0.1 else 'Poor'} differentiation around zero")