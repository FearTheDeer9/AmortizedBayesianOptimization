#!/usr/bin/env python3
"""
Quick test to verify surrogate loading functionality.
"""

import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.causal_bayes_opt.training.unified_grpo_trainer import create_unified_grpo_trainer
from src.causal_bayes_opt.experiments.test_scms import create_fork_test_scm

# Test loading the surrogate checkpoint
checkpoint_path = Path("checkpoints/validation/bc_surrogate_final")

print(f"Testing surrogate loading from: {checkpoint_path}")

# Check if checkpoint exists
if not checkpoint_path.exists():
    print(f"❌ Checkpoint not found at {checkpoint_path}")
    print("   Please run scripts/train_focused.sh first")
    sys.exit(1)

# Try to load the checkpoint
try:
    if checkpoint_path.is_dir():
        checkpoint_file = checkpoint_path / 'checkpoint.pkl'
    else:
        checkpoint_file = checkpoint_path
    
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
    
    print(f"✓ Loaded checkpoint")
    print(f"  Keys: {list(checkpoint.keys())}")
    
    # Check for surrogate components
    has_net = 'net' in checkpoint
    has_params = 'params' in checkpoint
    has_surrogate_params = 'surrogate_params' in checkpoint
    
    print(f"  Has 'net': {has_net}")
    print(f"  Has 'params': {has_params}")
    print(f"  Has 'surrogate_params': {has_surrogate_params}")
    
    # Check configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"\n  Surrogate Configuration:")
        print(f"    hidden_dim: {config.get('hidden_dim', 'N/A')}")
        print(f"    num_layers: {config.get('num_layers', 'N/A')}")
        print(f"    num_heads: {config.get('num_heads', 'N/A')}")
        print(f"    key_size: {config.get('key_size', 'N/A')}")
    
    # Try creating trainer with surrogate
    print("\nTesting GRPO trainer creation with surrogate...")
    
    # Prepare surrogate components based on what's in checkpoint
    if has_net and has_params:
        pretrained_surrogate = {
            'net': checkpoint['net'],
            'params': checkpoint['params']
        }
    elif has_params:
        # BC trainer saves params but not the network function
        # We need to reconstruct the network
        print("  Note: BC checkpoint format detected, reconstructing network...")
        
        # Import necessary components
        from src.causal_bayes_opt.training.surrogate_bc_trainer import SurrogateBCTrainer
        import haiku as hk
        import jax.numpy as jnp
        from src.causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel
        
        # Get config from checkpoint
        config = checkpoint.get('config', {})
        hidden_dim = config.get('hidden_dim', 128)
        num_layers = config.get('num_layers', 4)
        num_heads = config.get('num_heads', 8)
        key_size = config.get('key_size', 32)
        dropout = config.get('dropout', 0.1)
        
        # Recreate the network function
        def surrogate_fn(data: jnp.ndarray, target_variable: int, is_training: bool = False):
            model = ContinuousParentSetPredictionModel(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                key_size=key_size,
                dropout=dropout
            )
            return model(data, target_variable, is_training)
        
        # Transform with Haiku
        net = hk.transform(surrogate_fn)
        
        pretrained_surrogate = {
            'net': net,
            'params': checkpoint['params']
        }
        print("  ✓ Reconstructed network from BC checkpoint")
    else:
        print("  ⚠️  No surrogate components found in checkpoint")
        pretrained_surrogate = None
    
    # Create trainer
    trainer = create_unified_grpo_trainer(
        learning_rate=3e-4,
        n_episodes=5,
        batch_size=4,
        use_surrogate=True,
        seed=42,
        pretrained_surrogate=pretrained_surrogate
    )
    
    print("✓ Created GRPO trainer")
    
    # Check if surrogate was loaded
    if trainer.surrogate_params is not None:
        print("✓ Surrogate parameters loaded")
        print(f"  Surrogate predict function: {trainer.surrogate_predict_fn is not None}")
    else:
        print("❌ Surrogate parameters not loaded")
    
    # Test a quick training step
    print("\nTesting quick training step...")
    scm = create_fork_test_scm()
    
    # Just train for 1 episode to test
    trainer.max_episodes = 1
    results = trainer.train([scm])
    
    print("✓ Training completed")
    print(f"  Final reward: {results['final_metrics']['mean_reward']:.4f}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()