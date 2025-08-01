#!/usr/bin/env python3
"""
Example of how to integrate the new update function interface with evaluation.

This demonstrates how to use the standardized update functions with
the universal evaluator for active learning.
"""

import logging
from pathlib import Path

import jax.numpy as jnp
import pyrsistent as pyr

from src.causal_bayes_opt.utils.update_functions import (
    create_update_function,
    create_surrogate_loss_fn,
    UpdateContext
)
from src.causal_bayes_opt.evaluation.universal_evaluator import create_universal_evaluator
from src.causal_bayes_opt.evaluation.model_interfaces import create_bc_surrogate
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm
from src.causal_bayes_opt.data_structures.scm import get_variables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate active learning with new update interface."""
    
    # Create a simple SCM
    scm = create_fork_scm(n_variables=3)
    variables = list(get_variables(scm))
    
    # Load a pre-trained BC surrogate
    checkpoint_path = Path("checkpoints/bc_surrogate_final")
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Create surrogate with active learning using new interface
    predict_fn, _, params, opt_state = create_bc_surrogate(scm, checkpoint_path)
    
    # Get the network from checkpoint for loss function
    import pickle
    with open(checkpoint_path / "checkpoint.pkl", "rb") as f:
        checkpoint = pickle.load(f)
    
    # Assume we can extract the network (in practice, this would be provided)
    # For demo purposes, we'll create a loss function
    net = None  # Would be extracted from checkpoint
    
    # Create update function using the new interface
    if net is not None:
        loss_fn = create_surrogate_loss_fn(net, scoring_method="bic")
        update_fn_new = create_update_function(
            strategy="gradient",
            loss_fn=loss_fn,
            learning_rate=1e-3,
            min_samples=5
        )
    else:
        # Fallback to no-op for demo
        update_fn_new = create_update_function(strategy="no_op")
    
    # Wrapper to adapt new interface to evaluator expectations
    def update_fn_wrapper(current_params, current_opt_state, posterior, samples, vars, target):
        """Adapt new update interface to evaluator signature."""
        # Create UpdateContext
        from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
        
        # Create temporary buffer from samples
        buffer = ExperienceBuffer()
        for sample in samples:
            if hasattr(sample, 'get') and sample.get('intervention_type') is None:
                buffer.add_observation(sample)
            else:
                # Would need proper intervention handling
                pass
        
        context = UpdateContext(
            buffer=buffer,
            target_variable=target,
            variables=vars,
            step=len(samples),  # Use sample count as step
            metadata={"posterior": posterior}
        )
        
        # Call new update function
        new_params, new_opt_state, metrics = update_fn_new(
            current_params, current_opt_state, context
        )
        
        # Update closure for predict_fn if needed
        nonlocal params
        params = new_params
        
        return new_params, new_opt_state, metrics
    
    # Create evaluator
    evaluator = create_universal_evaluator()
    
    # Run evaluation with active learning
    config = {
        'n_steps': 20,
        'n_samples_obs': 100,
        'n_samples_per_intervention': 10
    }
    
    # Dummy acquisition function for demo
    def dummy_acquisition(tensor, posterior, target, variables):
        # Random intervention for demo
        import jax.random as random
        key = random.PRNGKey(42)
        var_idx = random.choice(key, len(variables))
        return {
            'targets': frozenset([variables[var_idx]]),
            'values': {variables[var_idx]: 0.0}
        }
    
    results = evaluator.evaluate(
        acquisition_fn=dummy_acquisition,
        scm=scm,
        config=config,
        surrogate_fn=predict_fn,
        surrogate_update_fn=update_fn_wrapper,  # Using our new update function
        surrogate_params=params,
        surrogate_opt_state=opt_state
    )
    
    logger.info(f"Evaluation complete with active learning")
    logger.info(f"Final metrics: {results['final_metrics']}")
    
    # The key advantage is that we can now easily swap update strategies:
    # - create_update_function(strategy="no_op") for fixed models
    # - create_update_function(strategy="gradient", ...) for standard updates
    # - create_update_function(strategy="adaptive", ...) for adaptive LR
    # - Custom strategies by implementing the UpdateFunction protocol


if __name__ == "__main__":
    main()