#!/usr/bin/env python3
"""
Minimal evaluation script using existing proven functions.

Simple approach:
1. Use existing model loading functions
2. Use existing SCM creation from training scripts  
3. Use existing evaluation loop from universal_evaluator
4. Use existing metric computation functions
"""

import sys
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import existing proven functions
from src.causal_bayes_opt.evaluation.model_interfaces import (
    create_grpo_acquisition,
    create_random_acquisition, 
    create_optimal_oracle_acquisition
)
from src.causal_bayes_opt.evaluation.universal_evaluator import create_universal_evaluator

# Use functional SCM factory (the correct one!)
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.data_structures.scm import get_variables, get_target, get_parents

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_scm() -> Any:
    """Create test SCM using functional VariableSCMFactory."""
    print("🔧 DEBUG SCM: Using VariableSCMFactory (functional pmap-based)...")
    
    # Use the correct functional factory like in the working experiments
    factory = VariableSCMFactory(seed=42)
    scm = factory.create_variable_scm(
        num_variables=5,
        structure_type='fork',  # Creates clear parent-child relationships
        edge_density=0.7  # Ensure meaningful causal structure
    )
    
    target = get_target(scm)
    parents = get_parents(scm, target)
    variables = get_variables(scm)
    
    print(f"🔧 DEBUG SCM: VariableSCMFactory SCM created")
    print(f"🔧 DEBUG SCM: Variables={list(variables)}")
    print(f"🔧 DEBUG SCM: Target={target}")
    print(f"🔧 DEBUG SCM: True parents of {target}={parents}")
    print(f"🔧 DEBUG SCM: SCM type={type(scm)}")
    print(f"🔧 DEBUG SCM: SCM ID={id(scm)}")
    
    return scm


def load_models(policy_path: Optional[Path] = None, 
               surrogate_path: Optional[Path] = None,
               scm: Any = None):
    """Load policy and surrogate models using existing functions."""
    
    # Load policy
    if policy_path and policy_path.exists():
        logger.info(f"Loading trained policy: {policy_path}")
        policy_fn = create_grpo_acquisition(policy_path, seed=42)
    else:
        logger.info("Using random policy baseline")
        policy_fn = create_random_acquisition(seed=42)
    
    # Load surrogate (placeholder - use existing loading from training scripts)
    surrogate_fn = None
    if surrogate_path and surrogate_path.exists():
        logger.info(f"Loading trained surrogate: {surrogate_path}")
        # TODO: Use existing surrogate loading pattern
        pass
    
    # Load oracle baseline
    oracle_fn = create_optimal_oracle_acquisition(scm) if scm else None
    
    return policy_fn, surrogate_fn, oracle_fn


def run_simple_evaluation():
    """Run minimal evaluation using existing proven components."""
    
    # Your specific checkpoint paths
    policy_path = Path("experiments/policy-only-training/checkpoints/full_training_3_vars_to_100_no_var_clipping/joint_ep500/policy.pkl")
    surrogate_path = Path("experiments/surrogate-only-training/scripts/checkpoints/avici_runs/avici_style_20250822_213115/checkpoint_step_200.pkl")
    
    # Create test SCM using proven factory
    print("🔧 DEBUG MAIN: Creating test SCM...")
    scm = create_test_scm()
    
    target = get_target(scm)
    parents = get_parents(scm, target)
    variables = get_variables(scm)
    
    print(f"🔧 DEBUG MAIN: SCM created - ID={id(scm)}")
    logger.info(f"Created test SCM: {len(variables)} variables, target={target}")
    logger.info(f"True parents: {parents}")
    print(f"🔧 DEBUG MAIN: Target={target}, True parents={parents}")
    
    # Load models
    print("🔧 DEBUG MAIN: Loading models...")
    policy_fn, surrogate_fn, oracle_fn = load_models(policy_path, surrogate_path, scm)
    print(f"🔧 DEBUG MAIN: Models loaded - oracle_fn ID={id(oracle_fn)}")
    
    # Create evaluator
    evaluator = create_universal_evaluator()
    print(f"🔧 DEBUG MAIN: Evaluator created - ID={id(evaluator)}")
    
    # Evaluation config
    config = {
        'n_observational': 50,
        'max_interventions': 10,
        'n_intervention_samples': 1,
        'optimization_direction': 'MINIMIZE'
    }
    print(f"🔧 DEBUG MAIN: Config={config}")
    
    # Test different methods
    methods = [
        ("Random", create_random_acquisition(seed=42), None),
        ("Oracle", oracle_fn, None),
    ]
    
    # Add trained policy if available
    if policy_path.exists():
        methods.append(("Trained Policy", policy_fn, surrogate_fn))
    
    print(f"🔧 DEBUG MAIN: Testing {len(methods)} methods")
    
    # Run evaluations
    results = {}
    for method_name, acquisition_fn, surrogate_fn in methods:
        print(f"\n🔧 DEBUG MAIN: ===== STARTING {method_name} =====")
        print(f"🔧 DEBUG MAIN: acquisition_fn ID={id(acquisition_fn)}")
        print(f"🔧 DEBUG MAIN: Using SCM ID={id(scm)}")
        print(f"🔧 DEBUG MAIN: Target={get_target(scm)}, Parents={get_parents(scm, get_target(scm))}")
        logger.info(f"\nEvaluating {method_name}...")
        
        try:
            # Use different seeds for each method to avoid identical randomness
            method_seed = 42 + hash(method_name) % 1000
            print(f"🔧 DEBUG MAIN: Using seed={method_seed} for {method_name}")
            print(f"🔧 DEBUG MAIN: About to call evaluator.evaluate() for {method_name}")
            print(f"🔧 DEBUG MAIN: acquisition_fn type={type(acquisition_fn)}")
            print(f"🔧 DEBUG MAIN: scm type={type(scm)}")
            print(f"🔧 DEBUG MAIN: surrogate_fn={surrogate_fn}")
            
            result = evaluator.evaluate(
                acquisition_fn=acquisition_fn,
                scm=scm,
                config=config,
                surrogate_fn=surrogate_fn,
                seed=method_seed
            )
            
            print(f"🔧 DEBUG MAIN: evaluator.evaluate() completed for {method_name}")
            
            if result.success:
                final_metrics = result.final_metrics
                print(f"🔧 DEBUG MAIN: {method_name} RESULT: final={final_metrics.get('final_value', 0):.6f}, best={final_metrics.get('best_value', 0):.6f}")
                logger.info(f"  ✓ {method_name}: "
                           f"Final={final_metrics.get('final_value', 0):.3f}, "
                           f"Best={final_metrics.get('best_value', 0):.3f}, "
                           f"F1={final_metrics.get('final_f1', 0):.3f}")
                results[method_name] = final_metrics
            else:
                logger.error(f"  ✗ {method_name} failed: {result.error_message}")
                
        except Exception as e:
            print(f"🔧 DEBUG MAIN: *** ERROR in {method_name} ***")
            print(f"🔧 DEBUG MAIN: Error type: {type(e)}")
            print(f"🔧 DEBUG MAIN: Error message: {e}")
            import traceback
            print(f"🔧 DEBUG MAIN: Full traceback:")
            traceback.print_exc()
            logger.error(f"  ✗ {method_name} error: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SIMPLE EVALUATION RESULTS")
    print("=" * 60)
    
    for method_name, metrics in results.items():
        print(f"\n{method_name}:")
        print(f"  🎯 Optimization: Final={metrics.get('final_value', 0):.3f}, "
              f"Best={metrics.get('best_value', 0):.3f}, "
              f"Improvement={metrics.get('improvement', 0):.3f}")
        print(f"  🔍 Structure: F1={metrics.get('final_f1', 0):.3f}, "
              f"SHD={metrics.get('final_shd', 0):.1f}")
    
    print("=" * 60)
    
    # Check if dual metrics working
    has_optimization = any(m.get('improvement', 0) > 0 for m in results.values())
    has_structure = any(m.get('final_f1', 0) > 0 for m in results.values())
    
    if has_optimization and has_structure:
        logger.info("✅ SUCCESS: Both optimization and structure metrics working!")
        return True
    elif has_optimization:
        logger.info("✅ PARTIAL: Optimization metrics working, structure learning pending")
        return True
    else:
        logger.error("✗ FAILED: No meaningful metrics")
        return False


if __name__ == "__main__":
    success = run_simple_evaluation()
    exit(0 if success else 1)