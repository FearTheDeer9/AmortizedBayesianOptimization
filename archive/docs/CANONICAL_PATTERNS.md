# Canonical Patterns for ACBO Development

This document defines the canonical patterns and best practices for developing within the ACBO codebase. These patterns have been extracted from the working implementation in `scripts/train_acbo_methods.py` and `scripts/evaluate_acbo_methods.py`.

## Overview

The ACBO pipeline follows a clear, modular architecture with well-defined interfaces between components. This document serves as the authoritative guide for:

1. Training models
2. Saving/loading checkpoints
3. Evaluating models
4. Creating new components

## Core Principles

1. **Simplicity over complexity** - Use simple pickle-based serialization
2. **Consistency** - All components follow the same patterns
3. **Modularity** - Clear separation between training and inference
4. **Minimal dependencies** - Components should be self-contained

## Canonical Training Pattern

### 1. Training Script Structure

```python
# scripts/train_your_method.py

import argparse
from pathlib import Path
from src.causal_bayes_opt.utils.canonical_utils import (
    save_checkpoint,
    create_training_config,
    format_training_results
)

def main():
    # 1. Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    
    # 2. Create config using canonical pattern
    config = create_training_config(
        method=args.method,
        episodes=args.episodes,
        # ... other args
    )
    
    # 3. Initialize trainer
    if args.method == 'grpo':
        trainer = create_unified_grpo_trainer(config)
    elif args.method == 'bc':
        trainer = PolicyBCTrainer(**config)
    
    # 4. Train
    results = trainer.train(...)
    
    # 5. Save checkpoint using canonical pattern
    save_checkpoint(
        checkpoint_path=Path(args.checkpoint_dir),
        params=results['params'],
        config=config,
        metadata=results['metadata']
    )
```

### 2. Trainer Implementation Pattern

Every trainer should follow this pattern:

```python
class YourTrainer:
    def __init__(self, **config):
        """Initialize with config dict."""
        self.config = config
        # Initialize model, optimizer, etc.
    
    def train(self, data) -> Dict[str, Any]:
        """
        Train model and return results.
        
        Returns:
            Dictionary with keys:
            - params: trained parameters
            - config: training configuration
            - metrics: training metrics
            - metadata: additional info
        """
        # Training logic...
        
        return format_training_results(
            params=self.model_params,
            config=self.config,
            metrics={
                'final_loss': final_loss,
                'training_time': time.time() - start,
                # ... other metrics
            },
            trainer_type=self.__class__.__name__,
            model_type='your_model_type'
        )
    
    def save_checkpoint(self, path: Path, results: Dict[str, Any]):
        """Save checkpoint using canonical pattern."""
        save_checkpoint(
            checkpoint_path=path,
            params=results['params'],
            config=results['config'],
            metadata=results['metadata']
        )
```

## Canonical Checkpoint Format

All checkpoints are saved as pickle files with this structure:

```python
checkpoint = {
    'params': model_params,          # JAX pytree of parameters
    'config': training_config,       # Dict of training configuration
    'metadata': {                    # Additional metadata
        'trainer_type': 'PolicyBCTrainer',
        'model_type': 'policy',      # or 'surrogate', 'acquisition', etc.
        'training_time': 123.45,
        'final_metrics': {...}
    }
}

# For GRPO models, additional fields:
checkpoint['policy_params'] = params     # Alias for compatibility
checkpoint['has_surrogate'] = True/False
checkpoint['surrogate_params'] = ...     # If has_surrogate
```

## Canonical Evaluation Pattern

### 1. Model Loading for Evaluation

```python
from src.causal_bayes_opt.utils.canonical_utils import load_checkpoint

# Load checkpoint
checkpoint = load_checkpoint(checkpoint_path)
params = checkpoint['params']
config = checkpoint['config']

# Create model function
model_fn = create_your_model(**config)

# Create acquisition function for evaluation
def acquisition_fn(tensor, posterior, target):
    key = jax.random.PRNGKey(seed)
    outputs = model_fn.apply(params, key, tensor, target)
    # ... extract intervention from outputs
    return intervention
```

### 2. Evaluation Script Pattern

```python
# scripts/evaluate_your_method.py

from src.causal_bayes_opt.evaluation.universal_evaluator import create_universal_evaluator
from src.causal_bayes_opt.utils.canonical_utils import (
    load_checkpoint,
    create_evaluation_config
)

def evaluate_method(checkpoint_path, scms, config):
    # 1. Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    
    # 2. Create acquisition function
    acquisition_fn = create_acquisition_from_checkpoint(checkpoint)
    
    # 3. Evaluate
    evaluator = create_universal_evaluator()
    results = evaluator.evaluate(
        acquisition_fn=acquisition_fn,
        scm=scm,
        config=config
    )
    
    return results
```

## Data Preprocessing Pattern

### Demonstration Loading

```python
from src.causal_bayes_opt.training.data_preprocessing import (
    load_demonstrations_from_path,
    preprocess_demonstration_batch
)

# Load raw demonstrations
raw_demos = load_demonstrations_from_path(demo_path, max_files=max_demos)

# Preprocess for specific use case
preprocessed = preprocess_demonstration_batch(raw_demos)
policy_data = preprocessed['policy_data']      # For policy training
surrogate_data = preprocessed['surrogate_data'] # For surrogate training
```

## Directory Structure

Follow this standard directory structure:

```
checkpoints/
├── grpo/
│   └── checkpoint.pkl
├── bc_policy/
│   └── checkpoint.pkl
└── bc_surrogate/
    └── checkpoint.pkl

evaluation_results/
├── evaluation_results.json
├── improvement_comparison.png
├── target_trajectories.png
└── structure_trajectories.png
```

## Common Pitfalls to Avoid

1. **Don't create new serialization formats** - Use the canonical checkpoint format
2. **Don't create new model interfaces** - Use the (tensor, posterior, target) → intervention pattern
3. **Don't duplicate data loading** - Use existing preprocessing functions
4. **Don't create complex configuration systems** - Use simple dictionaries

## Adding New Methods

When adding a new training method:

1. Create trainer following the pattern above
2. Add method to `scripts/train_acbo_methods.py`
3. Add model interface to `src/causal_bayes_opt/evaluation/model_interfaces.py`
4. Use canonical utilities for checkpoint management

### Important: Training Method Arguments

The `train_acbo_methods.py` script uses `--method` to determine what to train:
- `--method grpo`: Trains GRPO policy (with optional surrogate if `--use_surrogate`)
- `--method bc`: Trains BC policy (always policy, not surrogate)
- `--method surrogate`: Trains BC surrogate for structure learning
- `--method both`: Trains both GRPO and BC policy (not surrogate)

Note: There is no `--model_type` argument. The method determines what type of model is trained.

## Example: Adding a New Policy Method

```python
# 1. Create trainer: src/causal_bayes_opt/training/your_policy_trainer.py
class YourPolicyTrainer:
    def __init__(self, hidden_dim=256, learning_rate=1e-3, **kwargs):
        self.config = {
            'hidden_dim': hidden_dim,
            'learning_rate': learning_rate,
            **kwargs
        }
        # Initialize model...
    
    def train(self, data):
        # Train...
        return format_training_results(...)

# 2. Add to train_acbo_methods.py
elif args.method == 'your_method':
    trainer = YourPolicyTrainer(**config)
    results = trainer.train(data)
    save_checkpoint(...)

# 3. Add to model_interfaces.py
def create_your_method_acquisition(checkpoint_path, seed=42):
    checkpoint = load_checkpoint(checkpoint_path)
    # Create acquisition function...
    return acquisition_fn
```

## Testing Your Implementation

Always test the full pipeline:

```bash
# 1. Train
poetry run python scripts/train_acbo_methods.py \
    --method your_method \
    --episodes 10 \
    --checkpoint_dir test/your_method

# 2. Evaluate  
poetry run python scripts/evaluate_acbo_methods.py \
    --your_method test/your_method/checkpoint.pkl \
    --n_scms 2 \
    --n_interventions 5

# 3. Verify results exist
ls test/your_method/  # Should see checkpoint.pkl
```

## Posterior Extraction Pattern

### Working with Structure Learning Posteriors

The ACBO system uses `ParentSetPosterior` objects to represent structure learning predictions. Here's how to properly extract and use marginal probabilities:

```python
from src.causal_bayes_opt.avici_integration.parent_set import (
    get_marginal_parent_probabilities,
    get_parent_set_probability,
    get_most_likely_parents
)

# 1. Extract marginal probabilities from ParentSetPosterior
def extract_marginals(posterior, variable_order):
    """
    Extract marginal parent probabilities for 5-channel tensor integration.
    
    Args:
        posterior: ParentSetPosterior object
        variable_order: List of variable names
        
    Returns:
        Dict mapping variable names to probabilities
    """
    # Use canonical function
    marginals = get_marginal_parent_probabilities(posterior, variable_order)
    
    # Ensure target has zero probability
    if posterior.target_variable in marginals:
        marginals[posterior.target_variable] = 0.0
        
    return marginals

# 2. Handle different posterior formats
def extract_from_any_posterior(posterior, variable_order, target):
    """Handle various posterior formats consistently."""
    
    if isinstance(posterior, ParentSetPosterior):
        # Use canonical extraction
        return get_marginal_parent_probabilities(posterior, variable_order)
        
    elif isinstance(posterior, dict):
        # Direct dictionary format
        if 'marginal_parent_probs' in posterior:
            return posterior['marginal_parent_probs']
        elif 'metadata' in posterior and 'marginal_parent_probs' in posterior['metadata']:
            return posterior['metadata']['marginal_parent_probs']
            
    elif hasattr(posterior, 'metadata') and isinstance(posterior.metadata, dict):
        # Object with metadata attribute
        if 'marginal_parent_probs' in posterior.metadata:
            return posterior.metadata['marginal_parent_probs']
            
    # Fallback
    logger.warning(f"Unknown posterior format: {type(posterior)}")
    return None
```

### 5-Channel Tensor Integration

When integrating surrogate predictions into policy input:

```python
from src.causal_bayes_opt.training.five_channel_converter import (
    buffer_to_five_channel_tensor,
    convert_three_to_five_channel
)
from src.causal_bayes_opt.utils.posterior_validator import PosteriorValidator

# Create 5-channel tensor with surrogate integration
tensor_5ch, var_order, diagnostics = buffer_to_five_channel_tensor(
    buffer=experience_buffer,
    target_variable="Y",
    surrogate_fn=surrogate_predict_fn,  # Returns ParentSetPosterior
    validate_signals=True
)

# Log diagnostics
if diagnostics.get('surrogate_success'):
    stats = diagnostics['surrogate_stats']
    logger.info(f"Surrogate signal: mean={stats['mean_prob']:.3f}, "
                f"max={stats['max_prob']:.3f}, "
                f"nonzero={stats['num_nonzero']}")
```

### Key Posterior Formats

1. **ParentSetPosterior** (canonical format from AVICI integration):
   - Has `parent_set_probs`, `target_variable`, `metadata`
   - Use `get_marginal_parent_probabilities()` to extract marginals

2. **Dictionary with marginals**:
   ```python
   {
       'marginal_parent_probs': {'X': 0.8, 'Z': 0.3},
       'target_variable': 'Y',
       'entropy': 0.5
   }
   ```

3. **Nested metadata format**:
   ```python
   {
       'metadata': {
           'marginal_parent_probs': {'X': 0.8, 'Z': 0.3}
       }
   }
   ```

### Validation and Logging

Always validate and log posterior signals:

```python
# Validate posterior format and extract marginals
is_valid, issues, marginals = PosteriorValidator.validate_posterior(
    posterior, variable_order, target_variable
)

if not is_valid:
    logger.warning(f"Invalid posterior: {', '.join(issues)}")

# Log posterior summary for debugging
PosteriorValidator.log_posterior_summary(
    posterior, variable_order, target_variable, 
    prefix="PolicyName"
)
```

## Summary

By following these canonical patterns, you ensure:
1. **Compatibility** - Your code works with existing infrastructure
2. **Maintainability** - Others can understand and modify your code
3. **Reliability** - Proven patterns that work end-to-end

When in doubt, look at how `PolicyBCTrainer` or `UnifiedGRPOTrainer` are implemented - they exemplify these patterns.