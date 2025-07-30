# ACBO Codebase Cleanup Summary

## What We've Cleaned So Far

### Deleted Files (Verified Safe)
1. **Alternative training scripts**:
   - `scripts/train_acbo_clean.py` - Not referenced anywhere
   - `scripts/train_acbo_methods_updated.py` - Not referenced anywhere
   - These used `batch_bc_trainer.py` which is also deleted

2. **Unused trainers**:
   - `src/causal_bayes_opt/training/batch_bc_trainer.py` - Only used by deleted scripts

### Active Components (DO NOT DELETE)

#### Main Entry Points
- `scripts/train_acbo_methods.py` - Primary training script
- `scripts/evaluate_acbo_methods.py` - Primary evaluation script

#### Core Trainers
- `src/causal_bayes_opt/training/unified_grpo_trainer.py` - GRPO training
- `src/causal_bayes_opt/training/policy_bc_trainer.py` - BC policy training  
- `src/causal_bayes_opt/training/surrogate_bc_trainer.py` - BC surrogate training
- `src/causal_bayes_opt/training/data_preprocessing.py` - Shared data preprocessing
- `src/causal_bayes_opt/training/active_learning.py` - Active learning surrogates

#### Evaluation
- `src/causal_bayes_opt/evaluation/universal_evaluator.py` - Main evaluator
- `src/causal_bayes_opt/evaluation/model_interfaces.py` - Model loading for evaluation

## Identified Duplicates (Need Careful Review)

### Model Loading Pattern
The codebase has multiple ways to load models:
1. **For Training**: Each trainer has its own `save_checkpoint`/`load_checkpoint` methods
2. **For Evaluation**: `model_interfaces.py` has its own loading logic
3. **Unused Loaders**: 
   - `bc_model_loader.py` - Used by scripts/core/ experiments
   - `grpo_policy_loader.py` - Used by scripts/core/two_phase_training.py
   - `model_registry.py` - Unclear usage
   - `utils/model_loading.py` - Unclear usage

### Data Processing Pattern
Multiple ways to process demonstrations:
1. **Active**: `data_preprocessing.py` - Used by BC trainers
2. **Active**: `three_channel_converter.py` - Used by GRPO trainer
3. **Possibly unused**:
   - `pure_data_loader.py`
   - `demonstration_converter.py`
   - `trajectory_extractor.py`
   - `trajectory_processor.py`
   - `data_format_adapter.py`

### Configuration Pattern
Multiple configuration systems:
1. **Active**: Direct config dicts in train_acbo_methods.py
2. **Possibly unused**:
   - `config.py`
   - `grpo_config.py`
   - `grpo_fixed_config.py`
   - `acquisition_config.py`

## Safe Cleanup Candidates

### Low Risk (Not imported by active components)
1. Test scripts in root:
   - `test_bc_evaluation_fix.py`
   - `test_bc_evaluation_simple.py`
   - `test_bc_minimal.py`
   - `test_full_pipeline.py`

2. Debug/analysis scripts:
   - `check_bc_checkpoint.py`
   - `find_gradient_issue.py`
   - `fix_bc_notebook_cells.py`
   - `debug_structure_learning.json`

3. Old documentation:
   - Various .md files in root (except active ones)

### Medium Risk (Used by scripts/core/ which may be experimental)
1. `bc_model_loader.py`
2. `grpo_policy_loader.py`
3. The entire `scripts/core/` directory (if confirmed experimental)

### High Risk (Need to verify no hidden dependencies)
1. Base classes that might be inherited:
   - `base_trainer.py`
   - `modular_components.py`

2. Utilities that might be imported indirectly:
   - `checkpoint_manager.py`
   - `model_registry.py`

## Recommendation

1. **Immediate**: Delete test scripts and old docs in root directory
2. **After verification**: Delete scripts/core/ if confirmed experimental
3. **Careful review**: Check inheritance chains before deleting base classes
4. **Keep for now**: Configuration and data processing files until we verify they're truly unused

## Testing After Each Cleanup

Always run these tests after deletions:
```bash
# Training
poetry run python scripts/train_acbo_methods.py --method grpo --episodes 5
poetry run python scripts/train_acbo_methods.py --method bc --episodes 5 --max_demos 1

# Evaluation
poetry run python scripts/evaluate_acbo_methods.py --n_scms 2 --n_interventions 5
```