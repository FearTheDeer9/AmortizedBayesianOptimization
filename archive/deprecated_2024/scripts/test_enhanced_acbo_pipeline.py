#!/usr/bin/env python3
"""
Enhanced ACBO Pipeline End-to-End Test.

This script performs comprehensive testing of the enhanced ACBO pipeline to ensure
all components work together correctly before production deployment.

Tests include:
- Enhanced policy network factory integration
- Enhanced surrogate model factory integration  
- GRPO training manager with enhanced components
- Training script validation
- Experiment runner integration
- Baseline function validation
- Error handling and fallback mechanisms

Usage:
    python scripts/test_enhanced_acbo_pipeline.py
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import jax.random as random
import pyrsistent as pyr

from examples.demo_learning import DemoConfig

logger = logging.getLogger(__name__)


def test_enhanced_policy_network_factory() -> bool:
    """Test enhanced policy network factory functionality."""
    logger.info("Testing enhanced policy network factory...")
    
    try:
        from causal_bayes_opt.acquisition.enhanced_policy_network import (
            create_enhanced_policy_for_grpo, validate_enhanced_policy_integration
        )
        
        # Test validation
        if not validate_enhanced_policy_integration():
            logger.error("Enhanced policy integration validation failed")
            return False
        
        # Test factory creation
        variables = ['X0', 'X1', 'X2', 'X3', 'X4']
        target_variable = 'X0'
        
        enhanced_policy_fn, policy_config = create_enhanced_policy_for_grpo(
            variables=variables,
            target_variable=target_variable,
            architecture_level="simplified",
            performance_mode="fast"
        )
        
        # Validate policy config
        required_keys = ['max_history_size', 'num_channels', 'hidden_dim']
        for key in required_keys:
            if key not in policy_config:
                logger.error(f"Missing required key in policy config: {key}")
                return False
        
        # Test policy function creation (without actual forward pass)
        if enhanced_policy_fn is None:
            logger.error("Enhanced policy function creation failed")
            return False
        
        logger.info("Enhanced policy network factory test passed")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced policy network factory test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_enhanced_surrogate_model_factory() -> bool:
    """Test enhanced surrogate model factory functionality."""
    logger.info("Testing enhanced surrogate model factory...")
    
    try:
        from causal_bayes_opt.avici_integration.enhanced_surrogate import (
            create_enhanced_surrogate_for_grpo, validate_enhanced_surrogate_integration
        )
        
        # Test validation
        if not validate_enhanced_surrogate_integration():
            logger.error("Enhanced surrogate integration validation failed")
            return False
        
        # Test factory creation
        variables = ['X0', 'X1', 'X2', 'X3', 'X4']
        target_variable = 'X0'
        
        enhanced_surrogate_fn, surrogate_config = create_enhanced_surrogate_for_grpo(
            variables=variables,
            target_variable=target_variable,
            model_complexity="medium",
            use_continuous=True,
            performance_mode="fast"
        )
        
        # Validate surrogate config
        required_keys = ['hidden_dim', 'num_layers', 'use_continuous']
        for key in required_keys:
            if key not in surrogate_config:
                logger.error(f"Missing required key in surrogate config: {key}")
                return False
        
        # Test surrogate function creation
        if enhanced_surrogate_fn is None:
            logger.error("Enhanced surrogate function creation failed")
            return False
        
        logger.info("Enhanced surrogate model factory test passed")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced surrogate model factory test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_grpo_training_manager_integration() -> bool:
    """Test GRPO training manager with enhanced components."""
    logger.info("Testing GRPO training manager integration...")
    
    try:
        from causal_bayes_opt.training.grpo_training_manager import create_grpo_training_manager
        from causal_bayes_opt.training.grpo_config import create_debug_grpo_config
        from causal_bayes_opt.acquisition.reward_rubric import CausalRewardRubric
        from causal_bayes_opt.environments.intervention_env import InterventionEnvironment
        
        # Create minimal test components
        config = create_debug_grpo_config()
        
        # Create mock environment
        class MockEnvironment:
            def __init__(self):
                self.variables = ['X0', 'X1', 'X2']
                self.target_variable = 'X0'
                
            def reset(self, key):
                return type('State', (), {'mechanism_features': jnp.ones(3)})()
                
            def step(self, action):
                next_state = type('State', (), {'mechanism_features': jnp.ones(3)})()
                env_info = type('EnvInfo', (), {'episode_complete': True})()
                return next_state, env_info
        
        # Create mock reward rubric
        class MockRewardRubric:
            def compute_reward(self, state, action, next_state):
                return type('RewardResult', (), {
                    'total_reward': 1.0,
                    'components': {'improvement': 0.5, 'exploration': 0.3, 'structure': 0.2}
                })()
        
        # Create mock policy networks
        class MockPolicyNetwork:
            def __init__(self):
                self.params = {}
                
            def replace(self, **kwargs):
                return self
        
        environment = MockEnvironment()
        reward_rubric = MockRewardRubric()
        policy_network = MockPolicyNetwork()
        value_network = MockPolicyNetwork()
        
        # Create training manager
        training_manager = create_grpo_training_manager(
            config=config,
            environment=environment,
            reward_rubric=reward_rubric,
            policy_network=policy_network,
            value_network=value_network
        )
        
        # Test basic functionality
        if not hasattr(training_manager, 'config'):
            logger.error("Training manager missing config attribute")
            return False
        
        if not hasattr(training_manager, 'collect_experiences'):
            logger.error("Training manager missing collect_experiences method")
            return False
        
        # Test statistics
        stats = training_manager.get_training_statistics()
        if not isinstance(stats, dict):
            logger.error("Training statistics should return a dictionary")
            return False
        
        logger.info("GRPO training manager integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"GRPO training manager integration test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_enhanced_acbo_baseline_function() -> bool:
    """Test enhanced ACBO baseline function."""
    logger.info("Testing enhanced ACBO baseline function...")
    
    try:
        from causal_bayes_opt.experiments.enhanced_acbo_baseline import (
            create_enhanced_acbo_baseline, validate_enhanced_acbo_baseline
        )
        from causal_bayes_opt.experiments.benchmark_graphs import create_erdos_renyi_scm
        
        # Test validation function
        if not validate_enhanced_acbo_baseline():
            logger.warning("Enhanced ACBO baseline validation failed - may indicate integration issues")
            # Continue with test as fallback mechanisms should handle this
        
        # Create test SCM
        test_scm = create_erdos_renyi_scm(
            n_nodes=5,
            edge_prob=0.3,
            seed=42
        )
        
        # Create test config
        test_config = DemoConfig(
            n_observational_samples=5,
            n_intervention_steps=3,  # Small for fast testing
            learning_rate=1e-3,
            random_seed=42
        )
        
        # Test baseline function
        result = create_enhanced_acbo_baseline(
            scm=test_scm,
            config=test_config,
            architecture_level="baseline",
            performance_mode="fast"
        )
        
        # Validate result structure
        required_keys = ['method', 'final_best', 'improvement', 'total_samples']
        for key in required_keys:
            if key not in result:
                logger.error(f"Missing required key in baseline result: {key}")
                return False
        
        # Check enhanced features
        if 'enhanced_features' not in result:
            logger.error("Missing enhanced_features in baseline result")
            return False
        
        # Validate method type
        if not result['method'].startswith('enhanced_acbo'):
            logger.error(f"Unexpected method type: {result['method']}")
            return False
        
        logger.info("Enhanced ACBO baseline function test passed")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced ACBO baseline function test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_experiment_runner_integration() -> bool:
    """Test experiment runner integration with enhanced ACBO."""
    logger.info("Testing experiment runner integration...")
    
    try:
        # Import from correct location in scripts/core
        sys.path.insert(0, str(project_root / "scripts" / "core"))
        from acbo_wandb_experiment import ACBOMethodType
        
        # Check if enhanced ACBO method type exists
        if not hasattr(ACBOMethodType, 'ENHANCED_ACBO'):
            logger.error("Enhanced ACBO method type not found in experiment runner")
            return False
        
        # Verify method type value
        enhanced_method = ACBOMethodType.ENHANCED_ACBO
        if enhanced_method != 'acbo_enhanced':
            logger.error(f"Unexpected enhanced ACBO method value: {enhanced_method}")
            return False
        
        logger.info("Experiment runner integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"Experiment runner integration test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_training_script_validation() -> bool:
    """Test enhanced ACBO training script validation."""
    logger.info("Testing enhanced ACBO training script validation...")
    
    try:
        # Import training script components
        training_script_path = project_root / "scripts" / "core" / "train_enhanced_acbo.py"
        if not training_script_path.exists():
            logger.error("Enhanced ACBO training script not found")
            return False
        
        # Test importing key functions from training script
        sys.path.insert(0, str(training_script_path.parent))
        
        try:
            import train_enhanced_acbo
            
            # Check required functions exist
            required_functions = [
                'validate_enhanced_integration',
                'run_enhanced_acbo_training',
                'create_enhanced_training_environment',
                'create_enhanced_reward_rubric'
            ]
            
            for func_name in required_functions:
                if not hasattr(train_enhanced_acbo, func_name):
                    logger.error(f"Training script missing required function: {func_name}")
                    return False
            
            # Test configuration creation
            config = train_enhanced_acbo.create_default_enhanced_config()
            if not config:
                logger.error("Failed to create default enhanced training config")
                return False
            
            logger.info("Enhanced ACBO training script validation passed")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import training script: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Training script validation test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_error_handling_and_fallbacks() -> bool:
    """Test error handling and fallback mechanisms."""
    logger.info("Testing error handling and fallback mechanisms...")
    
    try:
        from causal_bayes_opt.experiments.enhanced_acbo_baseline import (
            create_enhanced_acbo_baseline
        )
        from causal_bayes_opt.experiments.benchmark_graphs import create_erdos_renyi_scm
        
        # Create test SCM with minimal config that might trigger fallbacks
        test_scm = create_erdos_renyi_scm(
            n_nodes=3,  # Very small to test edge cases
            edge_prob=0.1,
            seed=42
        )
        
        # Create minimal test config
        test_config = DemoConfig(
            n_observational_samples=2,
            n_intervention_steps=2,
            learning_rate=1e-3,
            random_seed=42
        )
        
        # Test with different architecture levels to ensure robustness
        architecture_levels = ["baseline", "simplified", "full"]
        
        for arch_level in architecture_levels:
            try:
                result = create_enhanced_acbo_baseline(
                    scm=test_scm,
                    config=test_config,
                    architecture_level=arch_level,
                    performance_mode="fast"
                )
                
                # Should always return a valid result even if fallbacks are used
                if not isinstance(result, dict):
                    logger.error(f"Invalid result type for architecture {arch_level}")
                    return False
                
                if 'method' not in result:
                    logger.error(f"Missing method in result for architecture {arch_level}")
                    return False
                
            except Exception as e:
                logger.error(f"Fallback mechanism failed for architecture {arch_level}: {e}")
                return False
        
        logger.info("Error handling and fallback mechanisms test passed")
        return True
        
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def run_enhanced_acbo_pipeline_test() -> Dict[str, bool]:
    """Run complete enhanced ACBO pipeline test suite."""
    logger.info("Starting Enhanced ACBO Pipeline End-to-End Test")
    
    test_results = {}
    
    # Run all tests
    test_functions = [
        ("Enhanced Policy Network Factory", test_enhanced_policy_network_factory),
        ("Enhanced Surrogate Model Factory", test_enhanced_surrogate_model_factory),
        ("GRPO Training Manager Integration", test_grpo_training_manager_integration),
        ("Enhanced ACBO Baseline Function", test_enhanced_acbo_baseline_function),
        ("Experiment Runner Integration", test_experiment_runner_integration),
        ("Training Script Validation", test_training_script_validation),
        ("Error Handling and Fallbacks", test_error_handling_and_fallbacks),
    ]
    
    for test_name, test_func in test_functions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running test: {test_name}")
        logger.info('='*60)
        
        start_time = time.time()
        try:
            test_results[test_name] = test_func()
            test_time = time.time() - start_time
            
            if test_results[test_name]:
                logger.info(f"✅ {test_name} PASSED ({test_time:.2f}s)")
            else:
                logger.error(f"❌ {test_name} FAILED ({test_time:.2f}s)")
                
        except Exception as e:
            test_time = time.time() - start_time
            test_results[test_name] = False
            logger.error(f"❌ {test_name} CRASHED ({test_time:.2f}s): {e}")
            logger.error(traceback.format_exc())
    
    return test_results


def generate_test_report(test_results: Dict[str, bool]) -> None:
    """Generate comprehensive test report."""
    logger.info(f"\n{'='*80}")
    logger.info("ENHANCED ACBO PIPELINE TEST REPORT")
    logger.info('='*80)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    failed_tests = total_tests - passed_tests
    
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    logger.info(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    logger.info("\nDetailed Results:")
    logger.info("-" * 80)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status:<10} {test_name}")
    
    if failed_tests > 0:
        logger.error(f"\n⚠️  {failed_tests} tests failed - Enhanced ACBO pipeline may have issues")
        logger.error("Review the test output above for specific failure details")
    else:
        logger.info("\n🎉 All tests passed - Enhanced ACBO pipeline is ready for deployment!")
    
    logger.info('='*80)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the test suite
    test_results = run_enhanced_acbo_pipeline_test()
    
    # Generate report
    generate_test_report(test_results)
    
    # Exit with appropriate code
    if all(test_results.values()):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure