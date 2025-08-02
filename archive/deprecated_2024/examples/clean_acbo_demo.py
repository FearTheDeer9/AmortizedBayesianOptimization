#!/usr/bin/env python3
"""
Clean ACBO demonstration using direct tensor conversion.

This demo shows:
1. Training GRPO with 3-channel tensor format
2. Evaluating on variable-sized SCMs
3. No AcquisitionState complexity
4. Direct buffer-to-tensor pipeline
"""

import logging
from pathlib import Path
from typing import Dict, Any
import jax.random as random
from omegaconf import DictConfig

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.clean_grpo_trainer import create_clean_grpo_trainer
from src.causal_bayes_opt.evaluation.clean_grpo_evaluator import create_clean_grpo_evaluator
from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_fork_scm, create_chain_scm, create_collider_scm
)
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_training_config() -> DictConfig:
    """Create clean training configuration."""
    return DictConfig({
        'seed': 42,
        'max_episodes': 500,  # Reduced for demo
        'n_variables_range': [3, 8],
        'obs_per_episode': 100,
        'max_interventions': 10,
        'batch_size': 32,
        'learning_rate': 3e-4,
        'checkpoint_dir': 'checkpoints/clean_grpo',
        'architecture': {
            'num_layers': 4,
            'num_heads': 8,
            'hidden_dim': 256,
            'key_size': 32,
            'dropout': 0.1
        }
    })


def create_scm_generator(seed: int = 42):
    """Create SCM generator for variable-sized structures."""
    factory = VariableSCMFactory(seed=seed)
    rng = random.PRNGKey(seed)
    
    def generator():
        nonlocal rng
        rng, key = random.split(rng)
        
        # Sample number of variables
        n_vars = random.randint(key, (), 3, 9)  # 3-8 variables
        
        # Sample structure type
        structure_types = ['chain', 'fork', 'collider', 'random']
        type_idx = random.randint(random.split(key)[1], (), 0, len(structure_types))
        structure_type = structure_types[type_idx]
        
        # Generate SCM
        return factory.create_variable_scm(
            num_variables=int(n_vars),
            structure_type=structure_type
        )
    
    return generator


def demonstrate_training():
    """Demonstrate clean GRPO training."""
    logger.info("=== Clean ACBO Training Demo ===")
    
    # Create configuration
    config = create_training_config()
    
    # Create trainer
    trainer = create_clean_grpo_trainer(config)
    
    # Create SCM generator
    scm_generator = create_scm_generator(seed=config.seed)
    
    # Train
    logger.info("Starting training with variable-sized SCMs...")
    results = trainer.train(scm_generator)
    
    logger.info(f"Training completed in {results['training_time']:.2f} seconds")
    logger.info(f"Final reward: {results['final_metrics']['mean_reward']:.3f}")
    
    return results


def demonstrate_evaluation():
    """Demonstrate clean GRPO evaluation."""
    logger.info("\n=== Clean ACBO Evaluation Demo ===")
    
    # Check for checkpoint
    checkpoint_path = Path('checkpoints/clean_grpo/clean_grpo_final')
    if not checkpoint_path.exists():
        logger.warning("No checkpoint found. Run training first.")
        return None
    
    # Create evaluator
    evaluator = create_clean_grpo_evaluator(checkpoint_path)
    evaluator.initialize()
    
    # Test on different SCM sizes
    test_scms = [
        ("3-var fork", create_fork_scm(noise_scale=1.0, target="Y")),
        ("4-var chain", create_chain_scm(chain_length=4, noise_scale=1.0)),
        ("5-var collider", create_collider_scm(noise_scale=1.0)),
    ]
    
    # Also test on larger variable-sized SCMs
    factory = VariableSCMFactory(seed=123)
    test_scms.extend([
        (f"6-var random", factory.create_variable_scm(6, 'random')),
        (f"8-var chain", factory.create_variable_scm(8, 'chain')),
    ])
    
    eval_config = {
        'max_interventions': 10,
        'n_observational_samples': 100
    }
    
    results = {}
    for name, scm in test_scms:
        logger.info(f"\nEvaluating on {name}...")
        result = evaluator.evaluate_single_run(scm, eval_config, seed=42)
        
        logger.info(f"  Initial value: {result.final_metrics['initial_value']:.3f}")
        logger.info(f"  Final value: {result.final_metrics['final_value']:.3f}")
        logger.info(f"  Improvement: {result.final_metrics['improvement']:.3f}")
        logger.info(f"  Time: {result.total_time:.2f}s")
        
        results[name] = result
    
    return results


def demonstrate_tensor_conversion():
    """Demonstrate the 3-channel tensor conversion."""
    logger.info("\n=== 3-Channel Tensor Format Demo ===")
    
    from src.causal_bayes_opt.training.three_channel_converter import (
        buffer_to_three_channel_tensor, validate_three_channel_tensor
    )
    from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
    from src.causal_bayes_opt.data_structures.sample import create_sample
    
    # Create sample buffer
    buffer = ExperienceBuffer()
    
    # Add observational samples
    for i in range(10):
        values = {'X': float(i), 'Y': float(2*i + 1), 'Z': float(i**2)}
        sample = create_sample(values=values)
        buffer.add_observation(sample)
    
    # Add intervention samples
    from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
    
    intervention_spec = create_perfect_intervention(
        targets=frozenset(['X']),
        values={'X': 5.0}
    )
    
    for i in range(5):
        values = {'X': 5.0, 'Y': float(11 + i), 'Z': float(25 + i)}
        sample = create_sample(
            values=values,
            intervention_type='perfect',
            intervention_targets=frozenset(['X'])
        )
        buffer.add_intervention(intervention_spec, sample)
    
    # Convert to tensor
    tensor, var_order = buffer_to_three_channel_tensor(
        buffer, target_variable='Y', max_history_size=20
    )
    
    logger.info(f"Tensor shape: {tensor.shape}")
    logger.info(f"Variable order: {var_order}")
    
    # Validate tensor
    is_valid = validate_three_channel_tensor(tensor, var_order)
    logger.info(f"Tensor validation: {'PASSED' if is_valid else 'FAILED'}")
    
    # Show channel contents
    logger.info("\nChannel 0 (values) - last 3 timesteps:")
    logger.info(tensor[-3:, :, 0])
    
    logger.info("\nChannel 1 (target) - last timestep:")
    logger.info(tensor[-1, :, 1])
    
    logger.info("\nChannel 2 (interventions) - last 3 timesteps:")
    logger.info(tensor[-3:, :, 2])


def main():
    """Run full clean ACBO demonstration."""
    # Demonstrate tensor conversion
    demonstrate_tensor_conversion()
    
    # Train
    training_results = demonstrate_training()
    
    # Evaluate
    if training_results:
        eval_results = demonstrate_evaluation()
    
    logger.info("\n=== Demo Complete ===")
    logger.info("The clean ACBO implementation:")
    logger.info("✓ Uses direct 3-channel tensor conversion")
    logger.info("✓ No AcquisitionState complexity")
    logger.info("✓ Supports variable-sized SCMs (3-8+ variables)")
    logger.info("✓ Clean, simple interfaces")
    logger.info("\nNext steps:")
    logger.info("- Add surrogate model integration for structure learning")
    logger.info("- Implement proper posterior updates")
    logger.info("- Add F1/SHD metrics to evaluation")


if __name__ == "__main__":
    main()