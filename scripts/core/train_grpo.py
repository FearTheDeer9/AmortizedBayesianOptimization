#!/usr/bin/env python3
"""
Clean GRPO Training Script using Simplified Trainer

Trains GRPO policy with the new simplified trainer that has minimal
configuration requirements and flexible SCM support.
"""

import argparse
import logging
import time
from pathlib import Path
import sys
import json
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.training.simplified_grpo_trainer import SimplifiedGRPOTrainer
from src.causal_bayes_opt.utils.scm_providers import (
    create_toy_scm_rotation, 
    create_benchmark_scms,
    load_scm_dataset
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_grpo(
    output_dir: Path,
    # Training parameters
    n_episodes: int = 100,
    learning_rate: float = 3e-4,
    architecture_level: str = "simplified",
    optimization_direction: str = "MINIMIZE",
    # SCM parameters
    scm_source: str = "toy",  # "toy", "benchmark", "path/to/dataset"
    variable_range: tuple = (3, 5),
    structure_types: list = None,
    # Other options
    use_early_stopping: bool = True,
    checkpoint_interval: int = 50,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Train GRPO policy using simplified trainer.
    
    Args:
        output_dir: Directory for outputs
        n_episodes: Number of training episodes
        learning_rate: Learning rate
        architecture_level: Model size ("baseline", "simplified", "full")
        optimization_direction: "MINIMIZE" or "MAXIMIZE"
        scm_source: Source of SCMs - "toy", "benchmark", or path
        variable_range: Range for toy SCM generation
        structure_types: Types of structures for toy SCMs
        use_early_stopping: Whether to use convergence detection
        checkpoint_interval: Save checkpoint every N episodes
        seed: Random seed
        
    Returns:
        Training results dictionary
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare SCMs based on source
    logger.info(f"Preparing SCMs from source: {scm_source}")
    
    if scm_source == "toy":
        # Generate toy SCMs
        structure_types = structure_types or ["fork", "chain", "collider"]
        scm_list = create_toy_scm_rotation(
            variable_range=variable_range,
            structure_types=structure_types,
            samples_per_config=2,
            seed=seed
        )
        # Convert list of tuples to dict
        scms = dict(scm_list)
    elif scm_source == "benchmark":
        # Use standard benchmarks
        scm_list = create_benchmark_scms("standard", include_variants=True)
        # Convert list of tuples to dict
        scms = dict(scm_list)
    elif Path(scm_source).exists():
        # Load from file/directory - already returns dict
        scms = load_scm_dataset(scm_source)
    else:
        raise ValueError(f"Invalid SCM source: {scm_source}")
        
    logger.info(f"Loaded {len(scms)} SCMs for training")
    
    # Create trainer
    trainer = SimplifiedGRPOTrainer(
        learning_rate=learning_rate,
        n_episodes=n_episodes,
        episode_length=20,
        architecture_level=architecture_level,
        use_early_stopping=use_early_stopping,
        optimization_direction=optimization_direction,
        seed=seed
    )
    
    # Train
    start_time = time.time()
    results = trainer.train(scms=scms)
    training_time = time.time() - start_time
    
    # Save checkpoint
    checkpoint_path = output_dir / "checkpoints" / "final_checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path, results)
    
    # Save training summary
    summary = {
        "training_time": training_time,
        "n_episodes_trained": results["metrics"]["episodes_per_scm"],
        "final_reward": results["metrics"]["final_reward"],
        "checkpoint_path": str(checkpoint_path),
        "optimization_direction": optimization_direction,
        "architecture_level": architecture_level,
        "num_scms": len(scms),
        "converged": results["metadata"].get("converged", False)
    }
    
    # Add convergence summary if available
    if "convergence_summary" in results:
        summary["convergence_summary"] = results["convergence_summary"]
    
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Training completed in {training_time/60:.1f} minutes")
    logger.info(f"Final reward: {results['metrics']['final_reward']:.4f}")
    logger.info(f"Checkpoint saved to: {checkpoint_path}")
    
    return summary


def main():
    """Main entry point for GRPO training."""
    parser = argparse.ArgumentParser(description="Train GRPO policy with simplified trainer")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/grpo_training",
        help="Output directory for checkpoints and results"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of training episodes"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["baseline", "simplified", "full"],
        default="simplified",
        help="Model architecture level"
    )
    parser.add_argument(
        "--optimization",
        type=str,
        choices=["MINIMIZE", "MAXIMIZE"],
        default="MINIMIZE",
        help="Optimization direction"
    )
    parser.add_argument(
        "--scm-source",
        type=str,
        default="toy",
        help="SCM source: 'toy', 'benchmark', or path to dataset"
    )
    parser.add_argument(
        "--variable-range",
        type=int,
        nargs=2,
        default=[3, 5],
        help="Variable range for toy SCMs (min max)"
    )
    parser.add_argument(
        "--structure-types",
        type=str,
        nargs="+",
        default=None,
        help="Structure types for toy SCMs"
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Run training
    results = train_grpo(
        output_dir=Path(args.output_dir),
        n_episodes=args.episodes,
        learning_rate=args.learning_rate,
        architecture_level=args.architecture,
        optimization_direction=args.optimization,
        scm_source=args.scm_source,
        variable_range=tuple(args.variable_range),
        structure_types=args.structure_types,
        use_early_stopping=not args.no_early_stopping,
        seed=args.seed
    )
    
    # Print summary
    print("\n" + "="*60)
    print("GRPO Training Complete!")
    print("="*60)
    print(f"Episodes trained: {sum(results['n_episodes_trained'].values())}")
    print(f"Training time: {results['training_time']/60:.1f} minutes")
    print(f"Final reward: {results['final_reward']:.4f}")
    print(f"Checkpoint: {results['checkpoint_path']}")
    print(f"Converged: {results['converged']}")
    
    return results


if __name__ == "__main__":
    main()