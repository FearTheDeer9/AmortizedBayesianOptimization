#!/usr/bin/env python3
"""
Test that the active learning fix works correctly.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.evaluation.surrogate_registry import SurrogateRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_active_learning_update():
    """Test that active learning updates work with the fix."""
    logger.info("=" * 80)
    logger.info("TESTING ACTIVE LEARNING FIX")
    logger.info("=" * 80)
    
    # Create registry and load BC surrogate with active learning
    registry = SurrogateRegistry()
    bc_checkpoint = "checkpoints/comprehensive_20250804_190724/bc_surrogate_final/checkpoint.pkl"
    
    try:
        # Register with active learning enabled
        registry.register('bc_active', bc_checkpoint)
        surrogate = registry.get('bc_active')
        
        logger.info(f"\n✓ Loaded surrogate: {surrogate.__class__.__name__}")
        logger.info(f"  - Name: {surrogate.name}")
        logger.info(f"  - Is active: {surrogate.is_active}")
        logger.info(f"  - Has update method: {hasattr(surrogate, 'update')}")
        logger.info(f"  - Has net: {hasattr(surrogate, '_net') and surrogate._net is not None}")
        logger.info(f"  - Has params: {hasattr(surrogate, '_params') and surrogate._params is not None}")
        logger.info(f"  - Has opt_state: {hasattr(surrogate, '_opt_state') and surrogate._opt_state is not None}")
        
        # Test update method
        logger.info("\nTesting update method...")
        
        # Create dummy data for update
        dummy_samples = [{'X': 0.5, 'Y': 1.2, 'Z': -0.3}]
        dummy_posterior = {'X': 0.3, 'Y': 0.0, 'Z': 0.7}
        
        try:
            params, metrics = surrogate.update(
                dummy_samples, 
                dummy_posterior,
                target='Y',
                variables=['X', 'Y', 'Z']
            )
            
            logger.info("✓ Update method called successfully!")
            logger.info(f"  Metrics: {metrics}")
            
            if 'error' in metrics:
                logger.warning(f"  Update returned error: {metrics['error']}")
            elif 'skipped' in metrics and metrics['skipped']:
                logger.warning("  Update was skipped (insufficient data?)")
            else:
                logger.info("  ✓ Update appears to have processed data")
                
        except Exception as e:
            logger.error(f"✗ Update failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        logger.error(f"Failed to load surrogate: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_active_learning_update()