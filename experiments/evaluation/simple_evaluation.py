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

# Additional imports for surrogate integration
from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint
from src.causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
from src.causal_bayes_opt.utils.variable_mapping import VariableMapper
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger(__name__)


def create_high_contrast_scm() -> Any:
    """Create high-contrast SCM for clear Oracle vs Random performance difference."""
    print("🔧 DEBUG SCM: Creating high-contrast SCM: X→Y(coeff=10), Z(isolated)")
    
    from src.causal_bayes_opt.data_structures.scm import create_scm
    from src.causal_bayes_opt.mechanisms.linear import create_linear_mechanism
    import pyrsistent as pyr
    
    # Structure: X → Y, Z isolated
    variables = frozenset(['X', 'Y', 'Z'])
    edges = frozenset([('X', 'Y')])  # Only X causes Y
    target = 'Y'
    
    # Create mechanisms with specific coefficients
    mechanisms = {}
    
    # X: Root variable (no parents, small noise)
    mechanisms['X'] = create_linear_mechanism(
        parents=[],
        coefficients={},
        intercept=0.0,
        noise_scale=0.01  # Very small noise
    )
    
    # Y: Caused by X with LARGE coefficient (target variable, small noise)
    mechanisms['Y'] = create_linear_mechanism(
        parents=['X'],
        coefficients={'X': 10.0},  # LARGE coefficient for dramatic effect
        intercept=0.0,
        noise_scale=0.01  # Very small noise
    )
    
    # Z: Isolated variable (no parents, small noise)
    mechanisms['Z'] = create_linear_mechanism(
        parents=[],
        coefficients={},
        intercept=0.0,
        noise_scale=0.01  # Very small noise
    )
    
    # Create SCM
    scm = create_scm(
        variables=variables,
        edges=edges,
        mechanisms=pyr.pmap(mechanisms),
        target=target,
        metadata=pyr.pmap({
            'structure': 'simple_chain',
            'coefficients': {('X', 'Y'): 10.0},
            'high_contrast': True,
            'expected_oracle_strategy': 'Intervene on X with extreme values',
            'variable_ranges': {
                'X': (-10.0, 10.0),  # X can vary from -10 to 10
                'Y': (-100.0, 100.0),  # Y can vary more due to 10*X  
                'Z': (-5.0, 5.0)  # Z has a smaller range
            }
        })
    )
    
    target = get_target(scm)
    parents = get_parents(scm, target)
    variables = get_variables(scm)
    
    print(f"🔧 DEBUG SCM: High-contrast SCM created")
    print(f"🔧 DEBUG SCM: Variables={list(variables)}")
    print(f"🔧 DEBUG SCM: Target={target}")
    print(f"🔧 DEBUG SCM: True parents of {target}={parents}")
    print(f"🔧 DEBUG SCM: Coefficient X→Y = 10.0 (large effect)")
    print(f"🔧 DEBUG SCM: Z is isolated (no effect on Y)")
    print(f"🔧 DEBUG SCM: Expected: Oracle should always pick X, Random picks X/Y/Z equally")
    print(f"🔧 DEBUG SCM: SCM type={type(scm)}, ID={id(scm)}")
    
    return scm


def load_surrogate_model(surrogate_path: Path):
    """Load trained surrogate following train_avici_style.py pattern."""
    print(f"🔧 DEBUG SURROGATE: Checking surrogate path: {surrogate_path}")
    print(f"🔧 DEBUG SURROGATE: Path exists: {surrogate_path.exists()}")
    
    if not surrogate_path.exists():
        print(f"🔧 DEBUG SURROGATE: Surrogate path does not exist!")
        return None
        
    logger.info(f"Loading trained surrogate: {surrogate_path}")
    
    # Load checkpoint
    checkpoint = load_checkpoint(surrogate_path)
    surrogate_params = checkpoint['params']
    surrogate_config = checkpoint.get('architecture', {})
    
    logger.info(f"Surrogate config: {surrogate_config}")
    
    # Create surrogate function (same as train_avici_style.py)
    def surrogate_fn(tensor, target_idx, is_training):
        model = ContinuousParentSetPredictionModel(
            hidden_dim=surrogate_config.get('hidden_dim', 128),
            num_layers=surrogate_config.get('num_layers', 8),
            num_heads=surrogate_config.get('num_heads', 8),
            key_size=surrogate_config.get('key_size', 32),
            dropout=surrogate_config.get('dropout', 0.1)
        )
        return model(tensor, target_idx, is_training)
    
    # Transform function
    surrogate_net = hk.transform(surrogate_fn)
    
    # Create inference function that matches universal_evaluator expectations
    def surrogate_inference(tensor: jnp.ndarray, target_var: str, variables: List[str]) -> Dict[str, Any]:
        """Surrogate inference following train_avici_style.py pattern."""
        print(f"🔧 DEBUG SURROGATE: surrogate_inference CALLED!")
        print(f"🔧 DEBUG SURROGATE: tensor.shape={tensor.shape}")
        print(f"🔧 DEBUG SURROGATE: target_var={target_var}")
        print(f"🔧 DEBUG SURROGATE: variables={variables}")
        
        try:
            # Get target index
            mapper = VariableMapper(variables, target_variable=target_var)
            target_idx = mapper.target_idx
            print(f"🔧 DEBUG SURROGATE: target_idx={target_idx}")
            
            # Run surrogate inference (same pattern as training)
            rng_key = random.PRNGKey(42)  # Fixed seed for deterministic evaluation
            print(f"🔧 DEBUG SURROGATE: About to call surrogate_net.apply...")
            predictions = surrogate_net.apply(
                surrogate_params, rng_key, tensor, target_idx, False  # is_training=False
            )
            print(f"🔧 DEBUG SURROGATE: surrogate_net.apply completed")
            print(f"🔧 DEBUG SURROGATE: predictions keys: {list(predictions.keys())}")
            
            # Extract predictions (same as train_avici_style.py lines 261-265)
            if 'parent_probabilities' in predictions:
                pred_probs = predictions['parent_probabilities']
            else:
                raw_logits = predictions.get('attention_logits', jnp.zeros(len(variables)))
                pred_probs = jax.nn.sigmoid(raw_logits)
            
            # Convert to flat marginal_parent_probs format expected by universal_evaluator
            # Format should be: {var: prob} not {target: {var: prob}}
            marginal_probs = {}
            
            for i, var in enumerate(variables):
                if var != target_var:
                    prob = float(pred_probs[i]) if i < len(pred_probs) else 0.0
                    marginal_probs[var] = prob
            
            print(f"🔧 DEBUG SURROGATE: Created flat marginal_probs: {marginal_probs}")
            
            result = {
                'marginal_parent_probs': marginal_probs,  # Flat format: {var: prob}
                'parent_probabilities': pred_probs,
                'raw_predictions': predictions
            }
            
            print(f"🔧 DEBUG SURROGATE: Returning result keys: {list(result.keys())}")
            print(f"🔧 DEBUG SURROGATE: marginal_parent_probs type: {type(result['marginal_parent_probs'])}")
            
            return result
            
        except Exception as e:
            print(f"🔧 DEBUG SURROGATE: *** SURROGATE ERROR ***")
            print(f"🔧 DEBUG SURROGATE: Error: {e}")
            import traceback
            traceback.print_exc()
            logger.warning(f"Surrogate inference failed: {e}")
            return {}
    
    return surrogate_inference


def load_models(policy_path: Optional[Path] = None, 
               surrogate_path: Optional[Path] = None,
               scm: Any = None):
    """Load policy and surrogate models using existing functions."""
    
    # Load policy with SCM metadata for quantile policies
    if policy_path and policy_path.exists():
        logger.info(f"Loading trained policy: {policy_path}")
        
        # Extract SCM metadata if available
        scm_metadata = None
        if scm and hasattr(scm, 'get'):
            scm_metadata = scm.get('metadata', {})
            logger.info(f"SCM metadata available: {list(scm_metadata.keys())}")
        
        # Check if this is a quantile policy
        from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint
        checkpoint = load_checkpoint(policy_path)
        architecture = checkpoint.get('architecture', {})
        
        if architecture.get('architecture_type') == 'quantile' or architecture.get('policy_architecture') == 'quantile':
            # Use ModelLoader for quantile policies to pass SCM metadata
            from experiments.evaluation.core.model_loader import ModelLoader
            logger.info("Detected quantile policy - using ModelLoader with SCM metadata")
            policy_fn = ModelLoader.load_policy(policy_path, seed=42, scm_metadata=scm_metadata)
        else:
            # Use regular loading for non-quantile policies
            policy_fn = create_grpo_acquisition(policy_path, seed=42)
    else:
        logger.info("Using random policy baseline")
        policy_fn = create_random_acquisition(seed=42)
    
    # Load surrogate using train_avici_style.py pattern
    surrogate_fn = None
    if surrogate_path and surrogate_path.exists():
        surrogate_fn = load_surrogate_model(surrogate_path)
    
    # Load oracle baseline
    oracle_fn = create_optimal_oracle_acquisition(scm) if scm else None
    
    return policy_fn, surrogate_fn, oracle_fn


def run_simple_evaluation():
    """Run minimal evaluation using existing proven components."""
    
    # Your specific checkpoint paths (absolute paths from project root)
    project_root = Path(__file__).parent.parent.parent
    # policy_path = project_root / "experiments/policy-only-training/checkpoints/full_training_3_vars_to_100_no_var_clipping/joint_ep500/policy.pkl"
    policy_path = project_root / "checkpoints/grpo_runs/grpo_multi_scm_20250825_140559/final_policy.pkl"
    surrogate_path = project_root / "experiments/surrogate-only-training/scripts/checkpoints/avici_runs/avici_style_20250822_213115/checkpoint_step_200.pkl"
    
    # Create high-contrast SCM for clear performance difference
    print("🔧 DEBUG MAIN: Creating high-contrast SCM...")
    scm = create_high_contrast_scm()
    
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
    print(f"🔧 DEBUG MAIN: Surrogate loaded: {surrogate_fn is not None}")
    
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
    
    # Test different methods with surrogate combinations
    methods = [
        ("Random Baseline", create_random_acquisition(seed=42), None),
        ("Oracle Baseline", oracle_fn, None),
    ]
    
    # Add trained models if available
    if policy_path.exists() and surrogate_path.exists():
        # Test different combinations
        methods.append(("Random + Trained Surrogate", create_random_acquisition(seed=42), surrogate_fn))
        methods.append(("Trained Policy + Trained Surrogate", policy_fn, surrogate_fn))
    elif policy_path.exists():
        methods.append(("Trained Policy Only", policy_fn, None))
    
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
            print(f"🔧 DEBUG MAIN: surrogate_fn is None: {surrogate_fn is None}")
            print(f"🔧 DEBUG MAIN: surrogate_fn type: {type(surrogate_fn)}")
            
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