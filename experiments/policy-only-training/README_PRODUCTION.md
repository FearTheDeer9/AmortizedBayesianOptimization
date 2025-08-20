# Production GRPO Training System

## Overview
This is the production-ready training and evaluation system for GRPO-based causal Bayesian optimization. It uses curriculum learning to progressively train on harder SCMs.

## Quick Start

### Training
```bash
# Quick test (50 episodes, levels 1-3)
python train_production.py --config quick_test

# Debug mode (10 episodes, verbose)
python train_production.py --config debug

# Full production training (1000 episodes, levels 1-10)
python train_production.py --config production
```

### Evaluation
```bash
# Evaluate latest model from quick test
python evaluate_production_v2.py --checkpoint checkpoints/quick_test/latest.pkl

# Evaluate specific checkpoint
python evaluate_production_v2.py --checkpoint checkpoints/production/best_model.pkl --output results/eval_best
```

## Architecture

### Core Components
1. **SCMCurriculumFactory**: Manages 15 levels of progressive difficulty
   - Level 1-3: Simple structures (chain, fork, collider)
   - Level 4-6: Medium structures (diamond, butterfly)
   - Level 7-9: Random graphs (Erdős-Rényi)
   - Level 10-12: Small-world networks
   - Level 13-15: Scale-free networks

2. **AdaptiveSCMGenerator**: Handles SCM rotation based on performance
   - Tracks F1 scores and rewards per SCM type
   - Advances levels when performance exceeds threshold
   - Minimum episodes per SCM before rotation

3. **ProductionTrainer**: Enhanced JointACBOTrainer with:
   - Checkpoint saving every N episodes
   - Performance tracking per curriculum level
   - Intervention range monitoring

### Configuration
Based on working settings from debug scripts:
- Architecture: `simple_permutation_invariant`
- Fixed std: 0.5 (not learned)
- Learning rate: 5e-4
- GRPO group size: 10
- Reward weights: target=0.9, parent=0.1, info_gain=0.0

### Directory Structure
```
experiments/joint-grpo-target-training/
├── train_production.py         # Main training script
├── evaluate_production_v2.py   # Evaluation script
├── checkpoints/
│   ├── production/             # Full training checkpoints
│   ├── quick_test/             # Quick test checkpoints
│   └── debug/                  # Debug run checkpoints
├── results/                    # Evaluation results
└── archive/                    # Previous debug scripts (preserved)
```

## Training Configurations

### Quick Test (Default)
- 50 episodes
- Curriculum levels 1-3
- Checkpoint every 10 episodes
- Good for testing setup

### Debug
- 10 episodes
- Curriculum levels 1-2
- Verbose logging
- For debugging issues

### Production
- 1000 episodes
- Curriculum levels 1-10
- Checkpoint every 50 episodes
- Full training run

## Evaluation Metrics

The evaluation script tests:
1. **Curriculum Levels**: Performance on training levels 1, 2, 3, 5, 7, 10
2. **Held-out SCMs**: Generalization to unseen structures
3. **Random Baseline**: Comparison to random policy

Key metrics:
- Average target value (lower is better)
- Parent selection rate (% of interventions on true parents)
- Range utilization (% of intervention range used)
- Improvement over random baseline

## What Was Preserved

All working configurations from debug scripts:
- JointACBOTrainer base class (unchanged)
- Exact hyperparameters that work
- Reward computation logic
- GRPO update mechanism
- Buffer management

## Differences from Debug Scripts

Added production features:
- Curriculum learning instead of single SCM
- Adaptive SCM rotation
- Checkpoint saving and recovery
- Comprehensive evaluation suite
- Performance tracking per level

## Troubleshooting

If training seems stuck:
1. Check debug mode for verbose output
2. Verify curriculum is advancing (check logs)
3. Ensure checkpoint directory is writable
4. Try quick_test config first

If evaluation fails:
1. Verify checkpoint exists
2. Check policy architecture matches training
3. Ensure all dependencies are installed