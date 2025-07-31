# Cleanup Inventory

## Date: 2025-01-31

## Root Directory Files Analysis

### KEEP (Documentation and Core Files)
- `README.md` - Main project documentation
- `CLAUDE.md` - AI instructions 
- `pyproject.toml` - Poetry configuration
- `poetry.lock` - Poetry dependencies
- `pytest.ini` - Test configuration
- `conftest.py` - Test fixtures

### ARCHIVE (Planning and Status Documents)
- `ACBO_PIPELINE_STATUS_20250129.md` - Status document
- `ACBO_VALIDATION_ANALYSIS.md` - Analysis document
- `CANONICAL_PATTERNS.md` - Pattern documentation
- `CLEANUP_SUMMARY.md` - Previous cleanup notes
- `CODEBASE_CLEANUP_PLAN.md` - Previous cleanup plan
- `QUICK_REFERENCE.md` - Quick reference guide
- `SURROGATE_INTEGRATION_STATUS.md` - Integration status
- `TASK_PLAN_20250131.md` - Today's plan (KEEP for now)
- `TRAINING_COMMANDS.md` - Training reference

### DELETE (Temporary Test Scripts)
- `analyze_12_pairs.py` - Temporary analysis
- `analyze_prediction_patterns.py` - Temporary analysis
- `debug_5channel_usage.py` - Debug script
- `debug_bc_active_updates.py` - Debug script
- `debug_surrogate_pipeline.py` - Debug script
- `debug_surrogate_predictions.py` - Debug script
- `extract_marginals.py` - Temporary extraction
- `test_*.py` (all root level test files) - Should be in tests/
- `run_full_evaluation*.sh` - Temporary scripts
- `test_*.sh` - Temporary test runners

### Log Files (DELETE)
- `*.log` - All log files
- `*_log.txt` - All log text files
- `*.png` - Temporary images

## Directory Analysis

### KEEP (Core Directories)
- `src/` - Source code
- `tests/` - Test suite
- `config/` - Configuration files
- `checkpoints/` - Model checkpoints (clean selectively)
- `docs/` - Documentation
- `cluster/` - Cluster scripts

### ARCHIVE (Examples and Experiments)
- `examples/` - Keep some, archive others based on relevance
- `experiments/` - Archive deprecated notebooks

### CLEAN (Results and Temporary)
- `evaluation_results/` - Move to archive or delete old runs
- `experiment_results/` - Move to archive
- `results/` - Clean old results
- `expert_demonstrations/` - Check if empty

## Scripts Directory Analysis

### Core Scripts (KEEP)
- `train_acbo_methods.py` - Main training entry point
- `evaluate_acbo_methods.py` - Main evaluation entry point

### Dependencies of Main Scripts (KEEP)
From imports analysis:
- `src.causal_bayes_opt.training.unified_grpo_trainer`
- `src.causal_bayes_opt.training.policy_bc_trainer`
- `src.causal_bayes_opt.training.surrogate_bc_trainer`
- `src.causal_bayes_opt.experiments.benchmark_scms`
- `src.causal_bayes_opt.evaluation.universal_evaluator`
- `src.causal_bayes_opt.evaluation.model_interfaces`
- `src.causal_bayes_opt.data_structures.scm`

### Other Scripts (CHECK USAGE)
Need to check if these are used by main scripts:
- `analyze_*.py` - Analysis scripts
- `debug_*.py` - Debug scripts
- `test_*.py` - Test scripts
- `validate_*.py` - Validation scripts

## Next Steps

1. Create `archive/` directory structure:
   - `archive/docs/` - Old documentation
   - `archive/scripts/` - Old scripts
   - `archive/results/` - Old results

2. Move files according to categories above

3. Clean checkpoint directory (keep only final models)

4. Update .gitignore to exclude archived files