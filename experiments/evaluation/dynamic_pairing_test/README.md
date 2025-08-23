# Dynamic Pairing Test Experiment

This experiment validates the three core functionalities of the enhanced evaluation framework:

1. **Dynamic Pairing Creation**: Test arbitrary combinations of policy and surrogate models
2. **Untrained Model Usage**: Validate creation and usage of untrained policies and surrogates
3. **Trained Model Loading**: Verify loading from all checkpoint directories and formats

## Purpose

This is a validation experiment that ensures the enhanced infrastructure works correctly before running the full experimental suite. It tests all possible model combinations and validates that they can be loaded, initialized, and executed without errors.

## Test Coverage

### Model Combinations Tested
- Random policy + no surrogate
- Oracle policy + no surrogate  
- Untrained policy + no surrogate
- Untrained policy + untrained surrogate
- Trained policy + no surrogate
- Trained policy + trained surrogate (joint training)
- Trained policy + untrained surrogate
- Untrained policy + trained surrogate

### Checkpoint Loading Validation
- Joint training checkpoints (`joint-training/checkpoints/`)
- Policy-only checkpoints (`policy-only-training/checkpoints/`)
- Surrogate-only checkpoints (`surrogate-only-training/scripts/checkpoints/`)

### Architecture Compatibility
- Verify trained and untrained models can be paired
- Test architecture detection and matching
- Validate parameter initialization compatibility

## Usage

### Quick Validation
```bash
cd experiments/evaluation/dynamic_pairing_test
python scripts/run_validation.py --quick
```

### Full Testing
```bash
python scripts/run_validation.py --comprehensive
```

### Specific Pairing Test
```bash
python scripts/run_validation.py --pairing "Trained Policy + Untrained Surrogate"
```

## Output

Results are saved to `results/validation_[timestamp]/`:
- `pairing_validation.json`: Success/failure status for each pairing
- `performance_summary.csv`: Basic performance metrics
- `error_log.txt`: Detailed error information for failures
- `validation_report.txt`: Human-readable summary

## Expected Results

All pairings should successfully:
1. Load models without errors
2. Execute at least one intervention
3. Produce reasonable outputs
4. Complete within expected time/memory bounds

Any failures indicate infrastructure issues that need to be addressed before running full experiments.