"""
Diagnostic script to check GRPO evaluation
"""

import logging
from pathlib import Path
import pickle
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_grpo_checkpoint(checkpoint_path: Path):
    """Diagnose issues with GRPO checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    
    print(f"\n=== Diagnosing GRPO Checkpoint: {checkpoint_path} ===\n")
    
    # Check files exist
    print("1. Checking checkpoint files:")
    metadata_path = checkpoint_path / "metadata.json"
    checkpoint_file = checkpoint_path / "checkpoint.pkl"
    
    if metadata_path.exists():
        print(f"   ✓ metadata.json exists")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"   - Optimization direction: {metadata.get('optimization_config', {}).get('direction', 'UNKNOWN')}")
        print(f"   - Training mode: {metadata.get('training_config', {}).get('mode', 'UNKNOWN')}")
    else:
        print(f"   ✗ metadata.json missing")
    
    if checkpoint_file.exists():
        print(f"   ✓ checkpoint.pkl exists ({checkpoint_file.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"   ✗ checkpoint.pkl missing")
        return
    
    # Load and inspect checkpoint
    print("\n2. Loading checkpoint data:")
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        print(f"   ✓ Checkpoint loaded successfully")
        print(f"   - Keys: {list(checkpoint_data.keys())}")
        
        # Check policy params
        if 'policy_params' in checkpoint_data:
            policy_params = checkpoint_data['policy_params']
            print(f"   ✓ policy_params found")
            
            # Inspect structure
            def inspect_params(params, prefix="", max_depth=3, current_depth=0):
                if current_depth >= max_depth:
                    return
                
                if isinstance(params, dict):
                    for key, value in params.items():
                        if isinstance(value, dict):
                            print(f"     {prefix}{key}/")
                            inspect_params(value, prefix + "  ", max_depth, current_depth + 1)
                        else:
                            if hasattr(value, 'shape'):
                                print(f"     {prefix}{key}: shape={value.shape}")
                            else:
                                print(f"     {prefix}{key}: type={type(value).__name__}")
            
            print("   - Policy parameter structure:")
            inspect_params(policy_params)
        else:
            print(f"   ✗ policy_params missing")
        
        # Check policy config
        if 'policy_config' in checkpoint_data:
            print(f"\n   ✓ policy_config found:")
            config = checkpoint_data['policy_config']
            print(f"     - Architecture: {config.get('architecture', {})}")
            print(f"     - State config: {config.get('state_config', {})}")
            print(f"     - GRPO config: {config.get('grpo_config', {})}")
        
    except Exception as e:
        print(f"   ✗ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n3. Testing policy initialization:")
    try:
        from ..training.enriched_trainer import EnrichedGRPOTrainer
        from ..training.modular_trainer import PolicyFactory
        from omegaconf import DictConfig
        
        # Create minimal config
        dummy_config = DictConfig({
            'training': {
                'architecture': checkpoint_data.get('policy_config', {}).get('architecture', {}),
                'state_config': checkpoint_data.get('policy_config', {}).get('state_config', {}),
                'n_variables': 10
            }
        })
        
        policy_factory = PolicyFactory(dummy_config)
        policy_fn, _ = policy_factory.create_policy()
        
        print("   ✓ Policy function created successfully")
        
        # Test policy application
        import jax
        import jax.numpy as jnp
        import jax.random as random
        
        # Create dummy input
        dummy_input = jnp.zeros((100, 4, 5))  # [history, vars, channels]
        target_idx = 1
        key = random.PRNGKey(0)
        
        try:
            output = policy_fn.apply(
                checkpoint_data['policy_params'],
                key,
                dummy_input,
                target_idx,
                False  # Not training
            )
            print("   ✓ Policy applied successfully")
            print(f"     - Output keys: {list(output.keys())}")
            
            if 'variable_logits' in output:
                logits = output['variable_logits']
                print(f"     - Variable logits shape: {logits.shape}")
                print(f"     - Logits: {logits}")
                print(f"     - Logit variance: {float(jnp.var(logits)):.6f}")
                
                if float(jnp.var(logits)) < 1e-6:
                    print("     ⚠️ WARNING: Very low logit variance - indicates collapse!")
            
        except Exception as e:
            print(f"   ✗ Policy application failed: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"   ✗ Failed to initialize policy: {e}")


if __name__ == "__main__":
    # Example usage
    checkpoint_path = Path("checkpoints/grpo_training/grpo_quick_minimize_20250726_161059")
    
    if checkpoint_path.exists():
        diagnose_grpo_checkpoint(checkpoint_path)
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = Path("checkpoints/grpo_training")
        if checkpoint_dir.exists():
            for ckpt in checkpoint_dir.iterdir():
                if ckpt.is_dir():
                    print(f"  - {ckpt.name}")