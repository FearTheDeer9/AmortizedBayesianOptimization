# Joint GRPO Target Training

Production-ready training pipeline for GRPO-based Active Causal Bayesian Optimization.

## Overview

This directory contains a complete training and evaluation pipeline for training ACBO policies using Group Relative Policy Optimization (GRPO). The system has been optimized based on extensive experiments to achieve:

- **60%+ range utilization** on intervention values
- **Average target values < -5.0** on chain SCMs
- **Consistent within-episode improvement**
- **Robust learning across diverse SCM structures**

## Quick Start

### 1. Quick Test (5 minutes)
```bash
# Sanity check that everything works
python train_production.py --config configs/quick_test.yaml
```

### 2. Production Training
```bash
# Full training with curriculum learning
python train_production.py --config configs/production.yaml

# Resume from checkpoint
python train_production.py --resume checkpoints/production/latest.pkl
```

### 3. Evaluation
```bash
# Evaluate trained model
python evaluate_production.py --checkpoint checkpoints/production/latest.pkl

# Quick evaluation
python evaluate_production.py --checkpoint checkpoints/production/latest.pkl --quick
```

## Key Findings from Experiments

Based on extensive testing, the optimal configuration includes:

1. **Architecture**: `simple_permutation_invariant` (no feature extraction)
2. **Exploration**: Fixed std = 1.5 (not learned)
3. **Reward Weights**: 90% target, 10% parent, 0% info gain
4. **GRPO Settings**: Group size 10, PPO epochs 4, entropy 0.01

## Directory Structure

```
experiments/joint-grpo-target-training/
├── configs/
│   ├── production.yaml      # Full training config (1000 episodes)
│   ├── quick_test.yaml      # Quick test config (5 episodes)
│   └── experiment.yaml      # For trying new ideas
├── scripts/
│   ├── train_production.py  # Main training script
│   ├── evaluate_production.py # Evaluation script
│   └── analyze_results.py   # Analysis tools
├── archive/
│   └── debug_scripts/       # Archived debugging scripts
├── checkpoints/             # Saved model checkpoints
├── results/                 # Training and evaluation results
└── README.md               # This file
```

## Configuration Options

### Key Parameters

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `policy_architecture` | `simple_permutation_invariant` | No feature extraction |
| `fixed_std` | 1.5 | Exploration noise level |
| `group_size` | 10 | GRPO candidates per step |
| `entropy_coefficient` | 0.01 | Exploration pressure |
| `learning_rate` | 5e-4 | Policy learning rate |

### Curriculum Learning

The training uses a 3-stage curriculum:

1. **Simple** (episodes 0-200): 3-4 node chains, forks
2. **Medium** (episodes 200-500): 5-7 node structures
3. **Complex** (episodes 500+): 8-10+ node Erdős-Rényi graphs

Progression is automatic based on performance thresholds.

## Performance Benchmarks

Expected performance after training:

| Metric | Target | Typical |
|--------|--------|---------|
| Range Utilization | >60% | 50-70% |
| Avg Target (Chain-3) | < -5.0 | -4.5 to -5.5 |
| Parent Accuracy | >90% | 85-95% |
| Within-Episode Improvement | >50% | 40-60% |

## Training Monitoring

During training, the following metrics are tracked:

- **Target Values**: Average outcome per episode
- **Range Utilization**: How much of [-5, 5] range is used
- **Within-Episode Learning**: First 3 vs last 3 interventions
- **Gradient Health**: Norm and parameter changes

## Troubleshooting

### Issue: Low Range Utilization (<30%)
- Increase `fixed_std` to 2.0
- Increase `entropy_coefficient` to 0.02
- Check if policy is converging too early

### Issue: No Within-Episode Improvement
- Increase `ppo_epochs` to 6
- Reduce `learning_rate` to 3e-4
- Check reward signal computation

### Issue: Training Crashes
- Reduce `group_size` to 5
- Enable gradient clipping
- Check for NaN in observations

## Advanced Usage

### Custom SCM Pool
```python
# Add custom SCMs to training
from train_production import ProductionTrainer

trainer = ProductionTrainer(config)
trainer.curriculum.scm_pool['custom'] = [your_scms]
```

### Adaptive Exploration
```python
# Adjust exploration based on performance
config['fixed_std'] = 2.0 if low_performance else 1.0
```

## Citation

If you use this code, please cite:
```
[Your citation here]
```

## Archive

Previous experimental scripts have been archived in `archive/debug_scripts/`. These include:
- Various architecture tests
- Reward shaping experiments
- Debugging utilities
- Performance analysis tools

These are preserved for reference but are not needed for production use.