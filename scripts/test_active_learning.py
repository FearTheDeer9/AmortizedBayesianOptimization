#!/usr/bin/env python3
"""
Test active learning functionality with a simple example.
"""

import sys
from pathlib import Path
import logging
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.evaluation.surrogate_registry import SurrogateRegistry
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.interventions.scm_interface import perform_intervention

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_active_learning():
    """Test active learning with BIC updates."""
    logger.info("=" * 80)
    logger.info("TESTING ACTIVE LEARNING")
    logger.info("=" * 80)
    
    # Load BC surrogate
    registry = SurrogateRegistry()
    bc_checkpoint = "checkpoints/comprehensive_20250804_190724/bc_surrogate_final/checkpoint.pkl"
    
    try:
        registry.register('bc', bc_checkpoint)
        logger.info("✓ Loaded BC surrogate")
    except Exception as e:
        logger.error(f"Failed to load surrogate: {e}")
        return
    
    # Get surrogate
    surrogate = registry.get('bc')
    
    # Create test SCM
    scm = create_fork_scm()
    logger.info(f"Created test SCM with variables: {scm['variable_names']}")
    
    # Create buffer and add some observational data
    buffer = ExperienceBuffer(max_size=1000)
    
    # Add initial observations
    logger.info("\nAdding initial observations...")
    for _ in range(50):
        obs = scm['sample'](1)
        buffer.add_observation(obs)
    
    logger.info(f"Buffer size: {buffer.size()}")
    
    # Test surrogate predictions before any interventions
    logger.info("\nTesting surrogate predictions on initial data...")
    
    # Get prediction for target variable
    target_var = scm['Y']['name']
    
    # Test if surrogate can be updated
    if hasattr(surrogate, 'update_from_buffer'):
        logger.info("\n✓ Surrogate has update_from_buffer method")
        
        # Perform some interventions and update
        logger.info("\nPerforming interventions and updating surrogate...")
        
        for i in range(5):
            # Choose random intervention
            intervention_var = np.random.choice(['X', 'Z'])
            intervention_value = np.random.randn()
            
            logger.info(f"\nIntervention {i+1}: {intervention_var} = {intervention_value:.3f}")
            
            # Perform intervention
            result = perform_intervention(
                scm, 
                [(intervention_var, intervention_value)],
                num_samples_obs=0,
                num_samples_int=10
            )
            
            # Add to buffer
            for sample in result.interventional_samples:
                buffer.add_intervention({intervention_var: intervention_value}, sample)
            
            # Update surrogate with BIC
            try:
                logger.info(f"  Buffer size before update: {buffer.size()}")
                update_info = surrogate.update_from_buffer(buffer, strategy='bic')
                logger.info(f"  ✓ Surrogate updated: {update_info}")
            except Exception as e:
                logger.error(f"  ✗ Update failed: {e}")
    else:
        logger.warning("\n✗ Surrogate does not support updates")
        logger.info("  This explains why active learning shows no improvement!")
    
    logger.info("\n" + "=" * 80)
    logger.info("ACTIVE LEARNING TEST COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_active_learning()