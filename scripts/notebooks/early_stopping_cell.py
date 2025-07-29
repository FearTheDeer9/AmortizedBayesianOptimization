"""
Notebook Cell: Configure Early Stopping

This cell should be added to grpo_training_modular.ipynb after configuration creation
to enable early stopping and prevent over-training on solved SCMs.
"""

# Cell content for the notebook:
EARLY_STOPPING_CELL = """
# Configure Early Stopping (NEW - Prevents Over-training)
print("\\n🛑 Configuring Early Stopping...")

# Enable early stopping to prevent over-training on solved SCMs
early_stopping_config = {
    'early_stopping_enabled': True,
    'convergence_accuracy_threshold': 0.95,  # Consider converged at 95% accuracy
    'convergence_patience': 10,               # Wait 10 episodes before declaring convergence
    'min_episodes_per_scm': 10,              # Train at least 10 episodes per SCM
    'max_episodes_per_scm': 50,              # Stop after 50 episodes even if not converged
    'reward_variance_threshold': 0.1         # Also check reward stability
}

# Update the configuration
config.training.update(early_stopping_config)

print("✓ Early stopping enabled")
print(f"  - Convergence threshold: {early_stopping_config['convergence_accuracy_threshold']}")
print(f"  - Patience: {early_stopping_config['convergence_patience']} episodes")
print(f"  - Min episodes per SCM: {early_stopping_config['min_episodes_per_scm']}")
print(f"  - Max episodes per SCM: {early_stopping_config['max_episodes_per_scm']}")

# Also update the configuration to use the fixed entropy coefficient
if hasattr(config, 'grpo_config'):
    original_entropy = config.grpo_config.get('entropy_coeff', 0.01)
    config.grpo_config['entropy_coeff'] = 0.1  # Increased to prevent collapse
    print(f"\\n✓ Updated entropy coefficient: {original_entropy} → {config.grpo_config['entropy_coeff']}")

# Ensure global standardization is enabled
if hasattr(config.training, 'state_config'):
    config.training.state_config['standardize_values'] = True
    config.training.state_config['use_global_standardization'] = True
    print("✓ Global standardization enabled")
else:
    # Add state config if missing
    config.training['state_config'] = {
        'standardize_values': True,
        'use_global_standardization': True
    }
    print("✓ Added state config with global standardization")

print("\\n📊 Expected behavior with early stopping:")
print("  - Training will progress through SCMs dynamically")
print("  - SCMs that converge quickly will be skipped early") 
print("  - More time will be spent on difficult SCMs")
print("  - Overall training distribution will be more balanced")
"""

# Alternative: Using the quick config function
QUICK_CONFIG_CELL = """
# Alternative: Use Quick Training Config with Early Stopping
from src.causal_bayes_opt.training.grpo_fixed_config import (
    create_quick_training_config_with_early_stopping
)

# Create configuration with early stopping enabled
quick_config = create_quick_training_config_with_early_stopping(
    n_episodes=200,
    n_scms=10
)

# Merge with existing config
for key, value in quick_config.items():
    if hasattr(config, key):
        getattr(config, key).update(value)
    else:
        setattr(config, key, value)

print("✓ Applied quick training configuration with early stopping")
"""

def print_notebook_instructions():
    """Print instructions for updating the notebook."""
    print("=== Instructions for Updating grpo_training_modular.ipynb ===")
    print()
    print("Add the following cell after the configuration creation cell:")
    print()
    print("```python")
    print(EARLY_STOPPING_CELL)
    print("```")
    print()
    print("This will enable early stopping and prevent over-training on solved SCMs.")
    print()
    print("Benefits:")
    print("- Prevents posterior collapse from over-representation of greedy optimization")
    print("- Reduces total training time by skipping converged SCMs")
    print("- Maintains balanced exploration/exploitation distribution")
    print("- Provides better generalization to test SCMs")


if __name__ == "__main__":
    print_notebook_instructions()