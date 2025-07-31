# ACBO Quick Reference

Quick commands and patterns for common ACBO tasks.

## Training Commands

### Train GRPO (Policy Only)
```bash
poetry run python scripts/train_acbo_methods.py \
    --method grpo \
    --episodes 1000 \
    --checkpoint_dir checkpoints/grpo
```

### Train BC Policy
```bash
poetry run python scripts/train_acbo_methods.py \
    --method bc \
    --episodes 500 \
    --max_demos 10 \
    --checkpoint_dir checkpoints/bc_policy
```

### Train BC Surrogate
```bash
poetry run python scripts/train_acbo_methods.py \
    --method surrogate \
    --episodes 500 \
    --surrogate_lr 1e-3 \
    --surrogate_layers 4 \
    --surrogate_hidden_dim 128 \
    --checkpoint_dir checkpoints/bc_surrogate
```

## Evaluation Commands

### Standard Evaluation
```bash
poetry run python scripts/evaluate_acbo_methods.py \
    --grpo checkpoints/grpo/checkpoint.pkl \
    --bc checkpoints/bc_policy/checkpoint.pkl \
    --n_scms 10 \
    --n_interventions 20 \
    --plot
```

### With Active Learning
```bash
poetry run python scripts/evaluate_acbo_methods.py \
    --grpo checkpoints/grpo/checkpoint.pkl \
    --bc checkpoints/bc_policy/checkpoint.pkl \
    --use_active_learning \
    --n_scms 10 \
    --plot
```

### Flexible Model Registry (New!)
```bash
# Register models and evaluate custom pairs
poetry run python scripts/evaluate_acbo_methods.py \
    --register_policy grpo_trained checkpoints/focused_20250130/grpo/checkpoint_final.pkl \
    --register_policy bc_policy checkpoints/focused_20250130/bc_policy/bc_final \
    --register_surrogate bc_surrogate checkpoints/focused_20250130/bc_surrogate/bc_surrogate_final \
    --register_surrogate active_learning active_learning \
    --evaluate_pairs grpo_trained bc_surrogate \
    --evaluate_pairs grpo_trained active_learning \
    --n_scms 10 \
    --n_interventions 20 \
    --plot
```

## Code Patterns

### Load a Checkpoint
```python
from src.causal_bayes_opt.utils.canonical_utils import load_checkpoint

checkpoint = load_checkpoint('checkpoints/grpo/checkpoint.pkl')
params = checkpoint['params']
config = checkpoint['config']
```

### Save a Checkpoint
```python
from src.causal_bayes_opt.utils.canonical_utils import save_checkpoint

save_checkpoint(
    checkpoint_path=Path('checkpoints/my_model'),
    params=model_params,
    config=training_config,
    metadata={'trainer_type': 'MyTrainer', 'final_loss': 0.123}
)
```

### Create Acquisition Function
```python
from src.causal_bayes_opt.evaluation.model_interfaces import (
    create_grpo_acquisition,
    create_bc_acquisition
)

# From GRPO checkpoint
grpo_fn = create_grpo_acquisition('checkpoints/grpo/checkpoint.pkl')

# From BC checkpoint  
bc_fn = create_bc_acquisition('checkpoints/bc/checkpoint.pkl')
```

### Evaluate a Model
```python
from src.causal_bayes_opt.evaluation.universal_evaluator import create_universal_evaluator

evaluator = create_universal_evaluator()
results = evaluator.evaluate(
    acquisition_fn=your_acquisition_fn,
    scm=test_scm,
    config={'n_initial_obs': 100, 'max_interventions': 20}
)
```

## Common Issues

### "No module named master_trainer"
This is a harmless warning. The master_trainer module is referenced but not needed for basic functionality.

### Checkpoint not found
Make sure to use the full path to checkpoint.pkl, or just the directory containing it:
```bash
# Both work:
--grpo checkpoints/grpo/checkpoint.pkl
--grpo checkpoints/grpo
```

### F1 scores are 0
This happens when not using active learning surrogates. Add `--use_active_learning` to evaluation.

## Directory Structure

```
your_experiment/
├── checkpoints/
│   ├── grpo/
│   │   └── checkpoint.pkl
│   └── bc/
│       └── checkpoint.pkl
├── evaluation_results/
│   ├── evaluation_results.json
│   └── *.png (plots)
└── logs/
```