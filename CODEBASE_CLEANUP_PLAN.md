# Codebase Cleanup Plan - ACBO Pipeline

## Active Components (DO NOT DELETE)

### Training Components
1. **Main Scripts**
   - `scripts/train_acbo_methods.py` - Main training entry point
   - `scripts/evaluate_acbo_methods.py` - Main evaluation entry point

2. **Trainers (Currently Used)**
   - `src/causal_bayes_opt/training/unified_grpo_trainer.py` - GRPO training
   - `src/causal_bayes_opt/training/policy_bc_trainer.py` - BC policy training
   - `src/causal_bayes_opt/training/surrogate_bc_trainer.py` - BC surrogate training
   - `src/causal_bayes_opt/training/data_preprocessing.py` - Data preprocessing
   - `src/causal_bayes_opt/training/active_learning.py` - AL surrogates

3. **Evaluation Components**
   - `src/causal_bayes_opt/evaluation/universal_evaluator.py` - Main evaluator
   - `src/causal_bayes_opt/evaluation/model_interfaces.py` - Model loaders

4. **Core Infrastructure**
   - `src/causal_bayes_opt/data_structures/scm.py` - SCM operations
   - `src/causal_bayes_opt/experiments/benchmark_scms.py` - SCM creation
   - `src/causal_bayes_opt/policies/clean_bc_policy_factory.py` - BC policy creation
   - `src/causal_bayes_opt/policies/clean_policy_factory.py` - GRPO policy creation

## Verified Duplicates in Training Directory

### Potentially Unused Files (Need to verify imports)
1. **Alternative trainers**:
   - `batch_bc_trainer.py` - Only used by train_acbo_methods_updated.py
   - `base_trainer.py` - Check if used by active trainers
   - `modular_components.py` - Likely unused

2. **Model loading duplicates**:
   - `bc_model_loader.py`
   - `grpo_policy_loader.py`
   - `model_registry.py`
   - `utils/model_loading.py`

3. **Data processing duplicates**:
   - `pure_data_loader.py`
   - `demonstration_converter.py`
   - `trajectory_extractor.py`
   - `trajectory_processor.py`
   - `data_format_adapter.py`
   - `three_channel_converter.py`

4. **BC-related duplicates**:
   - `bc_curriculum_manager.py`
   - `bc_data_pipeline.py`
   - `bc_loss_debug.py`
   - `behavioral_cloning_adapter.py`

5. **Checkpoint management duplicates**:
   - `checkpoint_manager.py`
   - `checkpoint_dataclasses.py`
   - Each trainer has its own methods

6. **Configuration duplicates**:
   - `grpo_config.py`
   - `grpo_fixed_config.py`
   - `acquisition_config.py`

7. **Unused infrastructure**:
   - `async_training.py`
   - `distributed.py`
   - `diversity_monitor.py`
   - `experience_management.py`
   - `curriculum.py`
   - `master_trainer.py` (doesn't exist but referenced)

## Testing Plan

After each deletion, run:
```bash
# Quick test - training
poetry run python scripts/train_acbo_methods.py --method grpo --episodes 10
poetry run python scripts/train_acbo_methods.py --method bc --episodes 10 --max_demos 1

# Quick test - evaluation  
poetry run python scripts/evaluate_acbo_methods.py --n_scms 2 --n_interventions 5
```

## Cleanup Order (Safe to Risky)

### Phase 1: Obvious Duplicates (Low Risk)
- [ ] Scripts in notebooks/ directory
- [ ] Test scripts in root (test_*.py)
- [ ] Old documentation files

### Phase 2: Unused Trainers (Medium Risk)
- [ ] batch_bc_trainer.py (after confirming not imported)
- [ ] bc_trainer.py (old unified version)
- [ ] base_trainer.py (if not used by active trainers)

### Phase 3: Duplicate Utilities (High Risk - Test Carefully)
- [ ] Consolidate checkpoint management
- [ ] Consolidate model loading utilities
- [ ] Consolidate data preprocessing functions

## Current Dependencies to Verify

Before deleting anything, verify these imports don't exist:
```bash
grep -r "from.*batch_bc_trainer" src/ scripts/
grep -r "from.*bc_trainer import" src/ scripts/
grep -r "from.*base_trainer" src/ scripts/
```