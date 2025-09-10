# Causal Bayesian Optimization - MSc Thesis Codebase

This repository contains the implementation and experimental results for my MSc thesis on causal Bayesian optimization with joint training of surrogate models and intervention policies.

## Quick Start

The main training and evaluation scripts are in the `experiments/` directory. Each script includes comprehensive help documentation accessible via `--help`.

### Training Models

#### 1. Joint Training (Alternating Surrogate and Policy)
```bash
cd experiments/joint-training
python train_joint_orchestrator.py --total-minutes 120 --phase-minutes 10
```

#### 2. Surrogate Model Training (AVICI-style with BC warmup)
```bash
cd experiments/surrogate-only-training/scripts
python train_avici_style.py --num-steps 1000 --batch-size 32
```

#### 3. Policy Training (GRPO)
```bash
cd experiments/policy-only-training
python train_grpo_ground_truth_rotation.py --episodes 100 --structure-types chain
```

### Evaluating Models

```bash
cd experiments/evaluation/stepwise
python full_evaluation.py \
    --policy-path path/to/policy.pkl \
    --surrogate-path path/to/surrogate.pkl \
    --num-episodes 30
```

## Project Structure

### Core Components

- **`src/`** - Core library implementing causal Bayesian optimization primitives. Largely a mess as befits a true research repo but has critical dependenciesfor my later research.

- **`experiments/`** - Main training and evaluation scripts:
  - `joint-training/` - Alternating surrogate-policy training
  - `surrogate-only-training/` - AVICI-style surrogate model training with BC warmup phase
  - `policy-only-training/` - GRPO policy optimization
  - `evaluation/` - Model evaluation and analysis tools

- **`thesis_results/`** - All results, figures, and analyses reported in the thesis:
  - `data_scaling/` - Data efficiency experiments
  - `policy/` - Policy convergence studies
  - `surrogate/` - Surrogate model scaling analysis
  - `capacity_analysis/` - Model capacity investigations
  - `eval_scripts/` - Scripts for reproducing analyses and plots

- **`thesis_model_checkpoints/`** - Model checkpoints used to generate thesis results (specifically from `thesis_results/surrogate/different_size_performance/` experiments)

- **`imperial-vm-checkpoints/`** - Additional checkpoints from compute cluster training runs

## Dependencies

Key requirements (see `poetry.lock` for full list):
- JAX and Haiku for neural networks
- PyRsistent for immutable data structures
- NumPy, Pandas for data processing
- Matplotlib for visualization

## Reproducing Results

The `thesis_results/` directory contains self-contained scripts and data for reproducing all thesis figures and analyses. See `thesis_results/eval_scripts/` for specific reproduction scripts.

## Notes

- The `src/` directory contains the full research codebase including experimental features. For practical use, refer to the validated implementations in `experiments/`.
- All main scripts include detailed help documentation accessible via `--help`.
- Model checkpoints in `thesis_model_checkpoints/` are the exact models used for thesis results.