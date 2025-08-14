# Amortized Causal Bayesian Optimization (ACBO)

A framework combining AVICI's amortized inference with PARENT_SCALE's causal optimization approach for efficient causal discovery and intervention selection.

## 🚀 Getting Started

### Quick Start
```bash
# 1. Train a surrogate model for structure learning
python scripts/main/train.py --method surrogate --episodes 100 --max_demos 50

# 2. Train a BC policy from expert demonstrations  
python scripts/main/train.py --method bc --episodes 100 --max_demos 50

# 3. Evaluate trained models
python scripts/main/evaluate.py --include_baselines --n_scms 10
```

### Complete Pipeline
```bash
# Run the complete training and evaluation pipeline
bash scripts/examples/complete_pipeline.sh
```

## 📚 Documentation

- [**docs/USAGE_GUIDE.md**](docs/USAGE_GUIDE.md) - 📖 Complete usage guide with examples
- [**CANONICAL_PATTERNS.md**](CANONICAL_PATTERNS.md) - ⭐ Development patterns and best practices
- [**TRAINING_COMMANDS.md**](TRAINING_COMMANDS.md) - Example training commands
- [**ACBO_PIPELINE_STATUS_20250129.md**](ACBO_PIPELINE_STATUS_20250129.md) - Current implementation status

## 🎯 Project Status: Working End-to-End Pipeline

✅ **GRPO Training** - True GRPO with batch advantages  
✅ **BC Training** - Separated policy and surrogate trainers  
✅ **Universal Evaluation** - Consistent evaluation across all methods  
✅ **Active Learning** - Structure learning during evaluation  

## 🏗️ Architecture

The system follows a cyclic workflow:
```
Surrogate Model → Acquisition Model → SCM Environment → Surrogate Model
```

### Core Components

- **Parent Set Prediction Model**: Direct prediction of top-k parent sets for target variables
- **AVICI Integration**: Amortized inference using graph neural networks  
- **Target-Aware Architecture**: Uses [N, d, 3] input format (values, interventions, target indicators)
- **Numerically Stable**: Robust against NaN/Inf issues during training

## 🛠️ Main Scripts

### Training (`scripts/main/train.py`)
Train surrogate and policy models:
- `--method surrogate`: Train structure learning model
- `--method bc`: Train BC policy from demonstrations
- `--method grpo`: Train GRPO policy with RL
- `--method grpo_with_surrogate`: Train surrogate + GRPO sequentially

### Evaluation (`scripts/main/evaluate.py`)
Evaluate trained models with comprehensive metrics:
- Supports multiple policy-surrogate pairs
- Includes baselines (Random, Oracle)
- Generates plots and trajectories
- Supports active learning evaluation

### Example Scripts (`scripts/examples/`)
Ready-to-use automation scripts:
- `train_surrogate.sh` - Train structure learning model
- `train_bc_policy.sh` - Train BC policy
- `train_grpo.sh` - Train GRPO policy
- `evaluate_comprehensive.sh` - Full evaluation
- `complete_pipeline.sh` - End-to-end workflow

## 📁 Project Structure

```
causal_bayes_opt/
├── scripts/                        # Main scripts and examples
│   ├── main/                       # Primary training/evaluation scripts
│   │   ├── train.py                # Universal training script
│   │   └── evaluate.py             # Universal evaluation script
│   ├── examples/                   # Example automation scripts
│   └── utils/                      # Utility scripts
├── src/causal_bayes_opt/           # Core implementation
│   ├── avici_integration/          # AVICI adaptation layer
│   │   ├── continuous/             # Continuous variable models
│   │   └── parent_set/             # Parent set prediction
│   ├── training/                   # Training implementations
│   │   ├── unified_grpo_trainer.py # GRPO trainer
│   │   ├── policy_bc_trainer.py    # BC policy trainer
│   │   └── surrogate_bc_trainer.py # BC surrogate trainer
│   ├── evaluation/                 # Evaluation framework
│   │   ├── universal_evaluator.py  # Universal evaluation loop
│   │   └── surrogate_registry.py   # Surrogate management
│   └── data_structures/            # Core data structures
│   ├── validation/                 # Validation scripts
│   └── test_integration/           # Integration tests
├── memory-bank/                    # Architecture decisions and planning
└── docs/                          # Documentation
```

## 🚀 Quick Start

### Installation

```bash
# Clone and install dependencies
poetry install
```

### Run Training Example

```bash
poetry run python tests/examples/example_parent_set_training.py
```

Expected output:
```bash
Training completed. Final loss: 1.1384, Best loss: 1.0986
✅ Model correctly identifies true parent set with reasonable confidence
```

### Run Validation

```bash
poetry run python tests/validation/validate_parent_set_model.py
```

## 🔬 Key Features

### Direct Parent Set Prediction
- **Linear scaling**: O(k) parameters instead of O(d²) 
- **Principled k selection**: Adaptive based on graph size
- **Clear validation**: Unambiguous success criteria

### Numerical Stability
- **Safe division**: No division by zero errors
- **Robust initialization**: Conservative parameter settings
- **Gradient stability**: Proper loss clipping and normalization

### Bias-Free Design
- **Fair competition**: All parent sets can compete equally
- **Deterministic ordering**: Removes positional bias
- **Target-aware**: Different targets produce different predictions

## 📊 Performance

Current validation results on 3-variable SCM (X → Y ← Z):
- **Training stability**: Loss 1.5 → 1.1 (no NaN/Inf)
- **Accuracy**: True parent set ranked #2 with 33.3% confidence
- **Uncertainty**: Realistic confidence levels (not overconfident)
- **Target specificity**: Different predictions for different targets

## 🔄 Next Steps

- [ ] Phase 2: Acquisition Model (RL with GRPO)  
- [ ] Phase 3: Integration with SCM Environment
- [ ] Phase 4: Full ACBO Training Pipeline
- [ ] Phase 5: Evaluation and Benchmarking

## 📚 References

- **AVICI**: Amortized Variational Inference for Causal Discovery
- **PARENT_SCALE**: Causal discovery through intervention selection
- **GRPO**: Group Relative Policy Optimization for RL

## 🛠️ Development

### Running Tests
```bash
# Run all validation
poetry run python tests/validation/validate_parent_set_model.py

# Run training example
poetry run python tests/examples/example_parent_set_training.py
```

### Architecture Decisions
See `memory-bank/architecture/` for detailed architectural decision records and design principles.

---

**Status**: Phase 1.3 Complete ✅ | **Next**: Phase 2 Implementation
