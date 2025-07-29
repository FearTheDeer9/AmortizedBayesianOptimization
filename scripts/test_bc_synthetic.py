#!/usr/bin/env python3
"""
Quick test of BC training with synthetic data.
"""

import sys
from pathlib import Path
import jax.numpy as jnp
import jax.random as random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.training.simplified_bc_trainer import SimplifiedBCTrainer


def create_synthetic_demos(n_demos=20, n_vars=5):
    """Create synthetic demonstrations for testing."""
    demos = []
    key = random.PRNGKey(42)
    
    for i in range(n_demos):
        key, demo_key = random.split(key)
        
        # Create synthetic state
        state = random.normal(demo_key, (n_vars, 32))
        
        # Create synthetic action
        action = {
            "variable": int(random.randint(demo_key, (), 0, n_vars)),
            "value": float(random.normal(demo_key, ()))
        }
        
        demos.append({
            "state": state,
            "action": action
        })
    
    return demos


def main():
    """Test BC training with synthetic data."""
    print("Creating synthetic demonstrations...")
    demos = create_synthetic_demos(n_demos=50)
    
    print(f"Created {len(demos)} demonstrations")
    print(f"State shape: {demos[0]['state'].shape}")
    print(f"Action: {demos[0]['action']}")
    
    # Create trainer
    print("\nInitializing BC trainer...")
    trainer = SimplifiedBCTrainer(
        model_type="acquisition",
        hidden_dim=64,
        num_layers=2,
        learning_rate=1e-3,
        batch_size=16,
        max_epochs=5,
        seed=42
    )
    
    # Train
    print("\nStarting training...")
    results = trainer.train(demonstrations=demos)
    
    print("\nTraining complete!")
    print(f"Final train loss: {results['metrics']['final_train_loss']:.4f}")
    print(f"Best val loss: {results['metrics']['best_val_loss']:.4f}")
    print(f"Epochs trained: {results['metrics']['epochs_trained']}")
    print(f"Training time: {results['metrics']['training_time']:.1f} seconds")
    
    # Save checkpoint
    output_path = Path("./results/bc_synthetic_test/checkpoint.pkl")
    trainer.save_checkpoint(output_path, results)
    print(f"\nCheckpoint saved to: {output_path}")


if __name__ == "__main__":
    main()