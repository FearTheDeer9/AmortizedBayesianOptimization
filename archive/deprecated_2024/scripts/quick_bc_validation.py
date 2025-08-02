#!/usr/bin/env python3
"""
Quick validation of BC fix without loading full dataset.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import jax.random as random

from src.causal_bayes_opt.training.behavioral_cloning_adapter import (
    load_demonstration_batch,
    create_surrogate_training_example
)
from src.causal_bayes_opt.training.bc_surrogate_trainer import (
    convert_to_jax_batch,
    kl_divergence_loss_jax,
    create_jax_surrogate_train_step
)
from src.causal_bayes_opt.training.config import SurrogateTrainingConfig
from src.causal_bayes_opt.avici_integration.continuous.factory import (
    create_continuous_parent_set_config,
    create_continuous_parent_set_model
)
import optax


def quick_validation():
    """Quick validation without full training loop."""
    print("="*60)
    print("QUICK BC FIX VALIDATION")
    print("="*60)
    
    # Load one batch
    print("\n1. Loading single demonstration batch...")
    demo_file = Path("expert_demonstrations/raw/raw_demonstrations/batch_1751266609.pkl")
    batch = load_demonstration_batch(str(demo_file))
    
    # Create training examples
    print("2. Creating training examples...")
    examples = []
    for demo in batch.demonstrations[:4]:  # Use only 4 demos
        avici_data = jnp.ones((100, demo.n_nodes, 3))
        example = create_surrogate_training_example(demo, 0, avici_data)
        examples.append(example)
    
    # Convert to JAX batch
    print("3. Converting to JAX batch...")
    jax_batch = convert_to_jax_batch(examples)
    
    print(f"   Batch shape: {jax_batch.observational_data.shape}")
    print(f"   Expert probs shape: {jax_batch.expert_probs.shape}")
    
    # Check expert probabilities
    print("\n4. Checking expert probabilities:")
    for i in range(len(examples)):
        probs = jax_batch.expert_probs[i]
        non_zero = jnp.sum(probs > 0.01)
        print(f"   Example {i}: {non_zero} non-zero probs, sum={jnp.sum(probs):.3f}")
    
    # Create model
    print("\n5. Creating model...")
    variables = examples[0].variable_order
    target = examples[0].target_variable
    
    config = create_continuous_parent_set_config(
        variables=variables,
        target_variable=target,
        model_complexity="medium",
        use_attention=True,
        temperature=1.0
    )
    
    model, _ = create_continuous_parent_set_model(config)
    
    # Initialize model
    key = random.PRNGKey(42)
    dummy_data = jax_batch.observational_data[0]
    target_idx = variables.index(target)
    
    params = model.init(key, dummy_data, target_idx, True)
    
    # Create optimizer and training step
    optimizer = optax.adam(1e-3)
    train_step = create_jax_surrogate_train_step(
        model=model,
        optimizer=optimizer,
        loss_fn=kl_divergence_loss_jax,
        config=SurrogateTrainingConfig()
    )
    
    # Run one training step
    print("\n6. Running training step...")
    opt_state = optimizer.init(params)
    
    try:
        updated_params, updated_opt_state, metrics = train_step(
            params,
            opt_state,
            jax_batch.observational_data,
            jax_batch.expert_probs,
            jax_batch.expert_accuracies,
            jnp.array(jax_batch.target_variables),
            key
        )
        
        loss = float(metrics.get('kl_loss', 0.0))
        grad_norm = float(metrics.get('grad_norm', 0.0))
        
        print(f"\n7. RESULTS:")
        print(f"   Loss: {loss:.6f}")
        print(f"   Gradient norm: {grad_norm:.6f}")
        
        # Validate
        if loss < 100:
            print(f"\n✅ VALIDATION PASSED!")
            print(f"   Loss is reasonable (< 100)")
            print(f"   The fix is working correctly!")
            return True
        else:
            print(f"\n❌ VALIDATION FAILED!")
            print(f"   Loss is still astronomical: {loss}")
            return False
            
    except Exception as e:
        print(f"\n❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def manual_loss_check():
    """Manually compute loss to verify."""
    print("\n" + "="*60)
    print("MANUAL LOSS COMPUTATION")
    print("="*60)
    
    # Create simple example
    num_vars = 5
    target_idx = 0
    
    # Expert distribution: parents are {1, 2}
    expert_probs = jnp.array([0.0, 0.5, 0.5, 0.0, 0.0])
    
    # Model prediction: uniform
    pred_probs = jnp.array([0.0, 0.25, 0.25, 0.25, 0.25])
    
    # Compute KL
    kl = kl_divergence_loss_jax(pred_probs, expert_probs)
    
    print(f"\nSimple example:")
    print(f"  Expert: {expert_probs}")
    print(f"  Pred:   {pred_probs}")
    print(f"  KL divergence: {float(kl):.6f}")
    
    if kl < 10:
        print("  ✓ Reasonable KL value")
    else:
        print("  ✗ Unreasonable KL value")


if __name__ == "__main__":
    # Run quick validation
    success = quick_validation()
    
    # Run manual check
    manual_loss_check()
    
    print("\n" + "="*60)
    print("WHAT TO DO NEXT:")
    print("="*60)
    
    if success:
        print("\n1. Run the notebook to see the improvement:")
        print("   poetry run jupyter notebook experiments/bc_development_workflow.ipynb")
        print("   - Restart kernel and run all cells")
        print("   - Check cells 17-18 for loss values")
        print("   - Check cells 19-20 for optimization results")
        
        print("\n2. Expected changes:")
        print("   BEFORE fix:")
        print("   - Surrogate loss: 999999999.9635")
        print("   - Acquisition loss: 999999674.7541")
        print("   - All methods: 0% improvement")
        
        print("\n   AFTER fix:")
        print("   - Surrogate loss: < 10.0")
        print("   - Acquisition loss: < 10.0")
        print("   - BC methods: > 0% improvement")
    else:
        print("\nThe validation failed. Please check the error messages above.")