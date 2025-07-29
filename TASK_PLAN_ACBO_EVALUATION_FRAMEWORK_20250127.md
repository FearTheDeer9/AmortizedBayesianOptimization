# Task Plan: Fix ACBO Evaluation Framework
## Date: 2025-01-27

## Objective
Create a clean, principled evaluation framework for the Amortized Causal Bayesian Optimization (ACBO) system that properly trains models, loads checkpoints, and demonstrates the system's capabilities with comprehensive metrics (F1, SHD, target value trajectories).

## Current State Analysis

### What Works:
- Core abstractions are well-designed (SCM, Sample, surrogate models, acquisition functions)
- Unified evaluation infrastructure exists (`src/causal_bayes_opt/evaluation/run_evaluation.py`)
- Metrics collection framework is comprehensive (`metrics_collector.py`)
- ACBO comparison framework has clean architecture (`scripts/core/acbo_comparison/`)

### What's Broken:
1. **GRPO notebooks**: Show training progress but use mock models in evaluation
2. **BC notebook**: Uses mock wrappers instead of loading actual trained models
3. **Model loading**: Evaluators don't properly load and use trained model parameters
4. **Fragmented implementation**: Logic scattered across notebooks making debugging difficult
5. **Inconsistent metrics**: Different notebooks calculate metrics differently

### Root Cause:
The implementation has good individual components but lacks proper integration. The notebooks were developed incrementally and use placeholder/mock implementations instead of loading actual trained models.

## Implementation Plan

### Phase 1: Core Infrastructure (Foundation)

#### 1.1 Checkpoint Utilities
- **File**: `scripts/core/utils/checkpoint_utils.py`
- Unified checkpoint saving/loading
- Model parameter extraction
- Version compatibility handling

#### 1.2 Metric Utilities  
- **File**: `scripts/core/utils/metric_utils.py`
- Standardized F1, SHD calculation
- Trajectory aggregation
- Statistical comparison utilities

### Phase 2: Training Scripts

#### 2.1 GRPO Training
- **File**: `scripts/core/train_grpo.py`
- Clean training with early stopping
- Proper checkpoint saving
- Bootstrap surrogate integration
- Global standardization for stability

#### 2.2 BC Training
- **File**: `scripts/core/train_bc.py`
- Train surrogate and acquisition models
- Load expert demonstrations
- Curriculum learning support
- Proper checkpoint management

### Phase 3: Evaluation Framework

#### 3.1 Unified Evaluation
- **File**: `scripts/core/evaluate_methods.py`
- Load models from checkpoints
- Run standardized evaluation
- Calculate all metrics properly
- Generate visualizations

#### 3.2 Fix Model Loading
- Update `GRPOEvaluator` to load actual policy
- Update `BCEvaluator` to load actual models
- Remove all mock implementations

### Phase 4: Demonstration Script

#### 4.1 Full Demo
- **File**: `scripts/run_full_acbo_demo.py`
- Complete end-to-end demonstration
- Train all models
- Run comprehensive evaluation
- Generate publication-ready plots

### Phase 5: Documentation

#### 5.1 Evaluation Guide
- **File**: `docs/evaluation_guide.md`
- How to run demonstrations
- Understanding metrics
- Interpreting results
- Troubleshooting

## Design Decisions Log

### 2025-01-27: Initial Design
- Decided to create separate scripts instead of fixing notebooks for clarity
- Will use existing unified evaluation framework as foundation
- Keep modular design so components can be run independently
- Prioritize actual model loading over mock implementations

### 2025-01-27: Implementation Choices
- Created unified checkpoint utilities for all model types
- Standardized metric calculations (F1, SHD, trajectories)
- GRPO training includes early stopping and bootstrap surrogate
- BC training uses curriculum learning with JAX compilation
- Evaluation script properly loads checkpoints and runs unified framework

## Problems & Solutions

### Problem 1: Mock models in notebooks
**Solution**: Created proper checkpoint loading in evaluation script that actually uses trained parameters

### Problem 2: Fragmented implementation
**Solution**: Created modular scripts that can be run independently or together

### Problem 3: Inconsistent metrics
**Solution**: Created standardized metric utilities used by all components

## Progress Updates

### 2025-01-27 14:30
- ✅ Created checkpoint utilities (scripts/core/utils/checkpoint_utils.py)
- ✅ Created metric utilities (scripts/core/utils/metric_utils.py)
- ✅ Created GRPO training script with early stopping and fixes
- ✅ Created BC training script with curriculum learning
- ✅ Created unified evaluation script that loads real models
- ✅ Created full demonstration script that runs everything

### 2025-01-27 15:00
- ✅ Created comprehensive evaluation guide (docs/evaluation_guide.md)
- ✅ All core components implemented and documented
- ✅ Task completed successfully

## Summary of Deliverables

### Scripts Created:
1. `scripts/core/utils/checkpoint_utils.py` - Unified checkpoint management
2. `scripts/core/utils/metric_utils.py` - Standardized metric calculations
3. `scripts/core/train_grpo.py` - Clean GRPO training with fixes
4. `scripts/core/train_bc.py` - BC training with curriculum learning
5. `scripts/core/evaluate_methods.py` - Unified evaluation with real model loading
6. `scripts/run_full_acbo_demo.py` - Complete end-to-end demonstration

### Documentation:
- `docs/evaluation_guide.md` - Comprehensive user guide
- `TASK_PLAN_ACBO_EVALUATION_FRAMEWORK_20250127.md` - This planning document

## Key Improvements Achieved

1. **Proper Model Loading**: Evaluation now loads and uses actual trained model parameters instead of mocks
2. **Unified Framework**: Single entry point for complete demonstration
3. **Modular Design**: Each component can be run independently
4. **Comprehensive Metrics**: F1, SHD, and target trajectories properly calculated
5. **Clear Documentation**: Easy to understand and extend

## Usage

Quick demo:
```bash
python scripts/run_full_acbo_demo.py --quick
```

Full demo:
```bash
python scripts/run_full_acbo_demo.py
```

The system is now ready for principled evaluation of the ACBO framework!