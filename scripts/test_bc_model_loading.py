#!/usr/bin/env python3
"""
Test script to verify BC model loading and reconstruction works correctly.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import jax.random as random
import pickle

from src.causal_bayes_opt.training.bc_model_loader import (
    load_bc_surrogate_model,
    load_bc_acquisition_model,
    validate_checkpoint
)
from src.causal_bayes_opt.training.model_registry import list_registered_models

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_surrogate_checkpoint(path: Path):
    """Create a test surrogate checkpoint with model configuration."""
    checkpoint_data = {
        'model_type': 'continuous_surrogate',
        'model_config': {
            'variables': ['X1', 'X2', 'X3'],
            'target_variable': 'X3',
            'model_complexity': 'medium',
            'use_attention': True,
            'temperature': 1.0,
            'parameters': {
                'hidden_dim': 64,
                'num_layers': 3,
                'num_heads': 4,
                'key_size': 32
            }
        },
        'model_params': {
            'test_param': jnp.array([1.0, 2.0, 3.0])
        },
        'epoch': 10,
        'config': {}
    }
    
    with open(path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    logger.info(f"Created test surrogate checkpoint at {path}")


def create_test_acquisition_checkpoint(path: Path):
    """Create a test acquisition checkpoint with model configuration."""
    from src.causal_bayes_opt.training.model_registry import create_model_from_config
    
    # Model configuration
    model_config = {
        'hidden_dim': 128,
        'num_layers': 3,
        'num_heads': 4,
        'key_size': 32,
        'dropout': 0.1,
        'use_enhanced_policy': True,
        'num_variables': 3
    }
    
    # Create model and initialize parameters
    haiku_model, _ = create_model_from_config('enhanced_acquisition', model_config)
    
    # Initialize with dummy state
    key = random.PRNGKey(42)
    dummy_state = {
        'state_tensor': jnp.zeros((3, 10)),
        'target_variable_idx': 0,
        'history_tensor': None
    }
    
    # Initialize parameters
    policy_params = haiku_model.init(key, dummy_state, True)
    
    checkpoint_data = {
        'model_type': 'enhanced_acquisition',
        'model_config': model_config,
        'policy_params': policy_params,
        'epoch': 15,
        'config': {}
    }
    
    with open(path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    logger.info(f"Created test acquisition checkpoint at {path}")


def test_model_registry():
    """Test model registry functionality."""
    logger.info("\n=== Testing Model Registry ===")
    
    registered_models = list_registered_models()
    logger.info(f"Registered models: {registered_models}")
    
    expected_models = [
        'continuous_surrogate',
        'jax_unified_surrogate',
        'enhanced_acquisition',
        'standard_acquisition'
    ]
    
    for model in expected_models:
        if model in registered_models:
            logger.info(f"✅ {model} is registered")
        else:
            logger.error(f"❌ {model} is NOT registered")
            
    return all(model in registered_models for model in expected_models)


def test_checkpoint_validation(checkpoint_path: Path):
    """Test checkpoint validation."""
    logger.info(f"\n=== Testing Checkpoint Validation: {checkpoint_path.name} ===")
    
    metadata = validate_checkpoint(str(checkpoint_path))
    logger.info(f"Validation result: {metadata}")
    
    if metadata['valid']:
        logger.info(f"✅ Checkpoint is valid")
        logger.info(f"  Model type: {metadata['model_type']}")
        logger.info(f"  Has params: {metadata['has_params']}")
        logger.info(f"  Epoch: {metadata['epoch']}")
    else:
        logger.error(f"❌ Checkpoint validation failed: {metadata.get('error', 'Unknown error')}")
        
    return metadata['valid']


def test_surrogate_loading(checkpoint_path: Path):
    """Test surrogate model loading."""
    logger.info("\n=== Testing Surrogate Model Loading ===")
    
    try:
        # Load the model
        init_fn, apply_fn, encoder_init, encoder_apply, params = load_bc_surrogate_model(
            str(checkpoint_path)
        )
        
        logger.info("✅ Successfully loaded surrogate model")
        logger.info(f"  Got init_fn: {init_fn is not None}")
        logger.info(f"  Got apply_fn: {apply_fn is not None}")
        logger.info(f"  Got encoder_init: {encoder_init is not None}")
        logger.info(f"  Got encoder_apply: {encoder_apply is not None}")
        logger.info(f"  Got params: {params is not None}")
        
        # Test that we can initialize the model
        key = random.PRNGKey(42)
        dummy_data = jnp.ones((10, 3, 3))  # [N, d, 3] format
        target_idx = 0
        
        # Initialize model
        init_params = init_fn(key, dummy_data, target_idx, True)
        logger.info("✅ Successfully initialized model")
        
        # Test apply
        output = apply_fn(init_params, key, dummy_data, target_idx, False)
        logger.info(f"✅ Successfully applied model, output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load surrogate model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_acquisition_loading(checkpoint_path: Path):
    """Test acquisition model loading."""
    logger.info("\n=== Testing Acquisition Model Loading ===")
    
    try:
        # Load the model
        acquisition_fn = load_bc_acquisition_model(str(checkpoint_path))
        
        logger.info("✅ Successfully loaded acquisition model")
        logger.info(f"  Got callable: {callable(acquisition_fn)}")
        
        # Test that we can call the acquisition function
        key = random.PRNGKey(42)
        
        # Create dummy state
        dummy_state = {
            'state_tensor': jnp.zeros((3, 10)),
            'target_variable_idx': 0,
            'history_tensor': None
        }
        
        # Call acquisition function
        result = acquisition_fn(dummy_state, key)
        logger.info(f"✅ Successfully called acquisition function")
        logger.info(f"  Result keys: {list(result.keys())}")
        logger.info(f"  Intervention variables: {result.get('intervention_variables')}")
        logger.info(f"  Intervention values: {result.get('intervention_values')}")
        logger.info(f"  Confidence: {result.get('confidence')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load acquisition model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("BC MODEL LOADING TEST")
    logger.info("=" * 80)
    
    # Create test directory
    test_dir = Path("test_checkpoints")
    test_dir.mkdir(exist_ok=True)
    
    # Test model registry
    registry_ok = test_model_registry()
    
    # Create test checkpoints
    surrogate_checkpoint = test_dir / "test_surrogate.pkl"
    acquisition_checkpoint = test_dir / "test_acquisition.pkl"
    
    create_test_surrogate_checkpoint(surrogate_checkpoint)
    create_test_acquisition_checkpoint(acquisition_checkpoint)
    
    # Test checkpoint validation
    surrogate_valid = test_checkpoint_validation(surrogate_checkpoint)
    acquisition_valid = test_checkpoint_validation(acquisition_checkpoint)
    
    # Test model loading
    surrogate_ok = test_surrogate_loading(surrogate_checkpoint)
    acquisition_ok = test_acquisition_loading(acquisition_checkpoint)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    all_tests = [
        ("Model Registry", registry_ok),
        ("Surrogate Validation", surrogate_valid),
        ("Acquisition Validation", acquisition_valid),
        ("Surrogate Loading", surrogate_ok),
        ("Acquisition Loading", acquisition_ok)
    ]
    
    for test_name, passed in all_tests:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in all_tests)
    
    # Cleanup
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info("✅ ALL TESTS PASSED")
        logger.info("BC model loading infrastructure is working correctly!")
    else:
        logger.info("❌ SOME TESTS FAILED")
        logger.info("Please check the errors above")
    logger.info("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())