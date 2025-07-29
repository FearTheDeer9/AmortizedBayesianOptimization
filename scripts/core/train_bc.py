#!/usr/bin/env python3
"""
Clean BC Training Script using Simplified Trainer

Trains Behavioral Cloning models (surrogate and acquisition) using the new
simplified trainer with minimal configuration and flexible data loading.
"""

import argparse
import logging
import time
from pathlib import Path
import sys
import json
import pickle
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.training.simplified_bc_trainer import SimplifiedBCTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_expert_demonstrations(demo_path: Optional[str] = None) -> Dict[str, list]:
    """Load expert demonstrations from default location."""
    
    # Use default demo directory if not provided
    if demo_path is None:
        demo_path = project_root / "expert_demonstrations" / "raw" / "raw_demonstrations"
    else:
        demo_path = Path(demo_path)
        
    logger.info(f"Loading demonstrations from {demo_path}")
    
    # Simple loading - adapt based on actual demo format
    all_demos = {"surrogate": [], "acquisition": []}
    
    if demo_path.is_dir():
        # Load all pkl files
        for file_path in sorted(demo_path.glob("*.pkl")):
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    
                # Determine type based on content or filename
                if "surrogate" in str(file_path) or "structure" in str(file_path):
                    all_demos["surrogate"].extend(data if isinstance(data, list) else [data])
                else:
                    all_demos["acquisition"].extend(data if isinstance(data, list) else [data])
                    
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
    else:
        # Single file
        with open(demo_path, "rb") as f:
            data = pickle.load(f)
            # Assume it contains both types
            if isinstance(data, dict):
                all_demos = data
            else:
                # Default to acquisition demos
                all_demos["acquisition"] = data if isinstance(data, list) else [data]
                
    logger.info(f"Loaded {len(all_demos['surrogate'])} surrogate demos, "
               f"{len(all_demos['acquisition'])} acquisition demos")
    
    return all_demos


def train_bc(
    output_dir: Path,
    model_type: str = "both",  # "surrogate", "acquisition", or "both"
    # Training parameters
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 50,
    # Model parameters
    hidden_dim: int = 128,
    num_layers: int = 4,
    # Data parameters
    demo_path: Optional[str] = None,
    validation_split: float = 0.2,
    # Other options
    early_stopping_patience: int = 5,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Train BC models using simplified trainer.
    
    Args:
        output_dir: Directory for outputs
        model_type: "surrogate", "acquisition", or "both"
        learning_rate: Learning rate
        batch_size: Batch size
        max_epochs: Maximum epochs
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        demo_path: Path to demonstrations
        validation_split: Validation split fraction
        early_stopping_patience: Early stopping patience
        seed: Random seed
        
    Returns:
        Dictionary with results for each model type
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load demonstrations
    all_demos = load_expert_demonstrations(demo_path)
    
    results = {}
    
    # Train surrogate if requested
    if model_type in ["surrogate", "both"]:
        if not all_demos["surrogate"]:
            logger.warning("No surrogate demonstrations found, skipping surrogate training")
        else:
            logger.info("Training surrogate model...")
            
            trainer = SimplifiedBCTrainer(
                model_type="surrogate",
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                learning_rate=learning_rate,
                batch_size=batch_size,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
                validation_split=validation_split,
                seed=seed
            )
            
            surrogate_results = trainer.train(demonstrations=all_demos["surrogate"])
            
            # Save checkpoint
            checkpoint_path = output_dir / "checkpoints" / "surrogate_final.pkl"
            trainer.save_checkpoint(checkpoint_path, surrogate_results)
            
            results["surrogate"] = {
                "checkpoint_path": str(checkpoint_path),
                "best_val_loss": surrogate_results["metrics"]["best_val_loss"],
                "epochs_trained": surrogate_results["metrics"]["epochs_trained"],
                "training_time": surrogate_results["metrics"]["training_time"]
            }
            
    # Train acquisition if requested
    if model_type in ["acquisition", "both"]:
        if not all_demos["acquisition"]:
            logger.warning("No acquisition demonstrations found, skipping acquisition training")
        else:
            logger.info("Training acquisition model...")
            
            # Acquisition models typically need larger capacity
            trainer = SimplifiedBCTrainer(
                model_type="acquisition",
                hidden_dim=hidden_dim * 2,  # Larger for policy
                num_layers=num_layers,
                learning_rate=learning_rate,
                batch_size=batch_size,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
                validation_split=validation_split,
                seed=seed + 1  # Different seed
            )
            
            acquisition_results = trainer.train(demonstrations=all_demos["acquisition"])
            
            # Save checkpoint
            checkpoint_path = output_dir / "checkpoints" / "acquisition_final.pkl"
            trainer.save_checkpoint(checkpoint_path, acquisition_results)
            
            results["acquisition"] = {
                "checkpoint_path": str(checkpoint_path),
                "best_val_loss": acquisition_results["metrics"]["best_val_loss"],
                "epochs_trained": acquisition_results["metrics"]["epochs_trained"],
                "training_time": acquisition_results["metrics"]["training_time"]
            }
            
    # Save summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Training completed! Results saved to {output_dir}")


def main():
    """Main entry point for BC training."""
    parser = argparse.ArgumentParser(description="Train BC models with simplified trainer")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/bc_training",
        help="Output directory for checkpoints and results"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["surrogate", "acquisition", "both"],
        default="both",
        help="Which model(s) to train"
    )
    parser.add_argument(
        "--demo-path",
        type=str,
        default=None,
        help="Path to demonstrations file or directory"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of layers"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Run training
    results = train_bc(
        output_dir=Path(args.output_dir),
        model_type=args.model_type,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        demo_path=args.demo_path,
        early_stopping_patience=args.patience,
        seed=args.seed
    )
    
    # Print summary
    print("\n" + "="*60)
    print("BC Training Complete!")
    print("="*60)
    
    if results:
        for model_name, model_results in results.items():
            print(f"\n{model_name.upper()} Model:")
            print(f"  Best validation loss: {model_results['best_val_loss']:.4f}")
            print(f"  Epochs trained: {model_results['epochs_trained']}")
            print(f"  Training time: {model_results['training_time']/60:.1f} minutes")
            print(f"  Checkpoint: {model_results['checkpoint_path']}")
    
    return results


if __name__ == "__main__":
    main()