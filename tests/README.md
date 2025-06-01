# Tests Directory

This directory contains validation scripts and examples for the ACBO framework.

## Structure

```
tests/
├── examples/                     # Working examples and demonstrations
│   ├── example_parent_set_training.py   # Complete training example
│   └── example_target_aware_model.py    # Target-aware model demo
├── validation/                   # Validation and testing scripts  
│   ├── validate_parent_set_model.py     # Parent set model validation
│   └── validate_phase_1_3.py           # Phase 1.3 validation
└── test_integration/             # Integration tests
```

## Usage

### Running Examples

```bash
# Run the main training example
python tests/examples/example_parent_set_training.py

# Run target-aware model demo  
python tests/examples/example_target_aware_model.py
```

### Running Validation

```bash
# Validate parent set model
python tests/validation/validate_parent_set_model.py

# Validate Phase 1.3 implementation
python tests/validation/validate_phase_1_3.py
```

## Expected Results

All validation scripts should pass and show:
- ✅ No NaN or Inf values
- ✅ Stable training progression  
- ✅ Reasonable accuracy (~70-80%)
- ✅ Proper parent set identification

The training example should complete successfully with a final loss around 1.0-1.5 and identify the correct parent sets with reasonable confidence levels.
