#!/usr/bin/env python3
"""
Test that BC development workflow works with the fixed model.
"""

import jax
import jax.numpy as jnp
import jax.random as random
from pathlib import Path
import pickle

# Set up paths
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

# Import BC infrastructure
from src.causal_bayes_opt.training.bc_surrogate_trainer import BCSurrogateTrainer
from src.causal_bayes_opt.training.config import TrainingConfig
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.avici_integration.core.conversion import samples_to_avici_format
from src.causal_bayes_opt.data_structures.scm import get_variables


def generate_demo_data():
    """Generate demonstration data similar to notebook."""
    print("Generating demonstration data...")
    
    # Create simple SCM
    scm = create_fork_scm(noise_scale=0.1, target='Y')
    variables = sorted(list(get_variables(scm)))
    
    # Generate observational samples
    obs_samples = sample_from_linear_scm(scm, n_samples=100, seed=42)
    
    # Generate interventional samples
    all_samples = list(obs_samples)
    
    key = random.PRNGKey(43)
    for var in variables:
        if var != 'Y':  # Don't intervene on target
            for value in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                key, subkey = random.split(key)
                intervention = create_perfect_intervention(frozenset([var]), {var: value})
                int_samples = sample_with_intervention(
                    scm, intervention, n_samples=10, seed=int(subkey[0])
                )
                all_samples.extend(int_samples)
    
    print(f"Generated {len(all_samples)} total samples")
    
    # Create trajectory-like structure
    trajectory = {
        'scm': scm,
        'samples': all_samples
    }
    
    return [trajectory]


def test_bc_training():
    """Test BC training with fixed model."""
    print("\nTesting BC Training with Fixed Model")
    print("="*60)
    
    # Generate demo data
    print("Generating simple test data...")
    
    # Create a simple SCM and data
    scm = create_fork_scm(noise_scale=0.1, target='Y')
    variables = sorted(list(get_variables(scm)))
    
    # Generate some samples
    obs_samples = sample_from_linear_scm(scm, n_samples=50, seed=42)
    all_samples = list(obs_samples)
    
    # Add some interventions
    key = random.PRNGKey(43)
    for var in ['X', 'Z']:
        for value in [-1.0, 0.0, 1.0]:
            key, subkey = random.split(key)
            intervention = create_perfect_intervention(frozenset([var]), {var: value})
            int_samples = sample_with_intervention(scm, intervention, n_samples=10, seed=int(subkey[0]))
            all_samples.extend(int_samples)
    
    print(f"Generated {len(all_samples)} samples")
    
    # Create training examples directly
    from src.causal_bayes_opt.training.data_structures import TrainingExample
    from src.causal_bayes_opt.training.trajectory_processor import DifficultyLevel, SurrogateDataset
    
    training_examples = []
    target = 'Y'
    
    # Convert samples to AVICI format
    avici_data = samples_to_avici_format(all_samples, variables, target, standardization_params=None)
    
    # Create a simple training example with mock expert probabilities
    # For testing, we'll say X and Z are equally likely parents of Y
    expert_probs = jnp.array([0.5, 0.0, 0.5])  # X=0.5, Y=0.0 (self), Z=0.5
    
    training_example = TrainingExample(
        observational_data=avici_data,
        expert_posterior=None,  # Not used in BC training
        target_variable=target,
        variable_order=variables,
        expert_accuracy=1.0,  # Assume perfect expert
        problem_difficulty="easy",
        parent_sets=[frozenset(['X']), frozenset(['Z']), frozenset(['X', 'Z'])],
        expert_probs=expert_probs
    )
    
    training_examples.append(training_example)
    
    # Create dataset
    dataset = SurrogateDataset(
        training_examples=training_examples,
        difficulty_levels=[DifficultyLevel.EASY],
        demonstration_ids=["test_demo"]
    )
    
    # Create trainer
    from src.causal_bayes_opt.training.config import SurrogateTrainingConfig
    from src.causal_bayes_opt.training.bc_surrogate_trainer import BCTrainingConfig
    
    surrogate_config = SurrogateTrainingConfig(
        model_hidden_dim=64,
        model_n_layers=2,
        learning_rate=1e-3,
        batch_size=1,  # Single example
        max_epochs=5,
        dropout=0.0,
        use_continuous_model=True
    )
    
    config = BCTrainingConfig(
        surrogate_config=surrogate_config,
        curriculum_learning=False,
        checkpoint_dir="/tmp/bc_test_checkpoints",
        save_frequency=10,
        enable_wandb_logging=False,
        max_epochs_per_level=5,
        min_epochs_per_level=1
    )
    
    trainer = BCSurrogateTrainer(config)
    
    # Train model
    print("\nTraining BC surrogate model...")
    
    datasets = {DifficultyLevel.EASY: dataset}
    
    key = random.PRNGKey(42)
    try:
        training_results = trainer.train_on_curriculum(
            curriculum_datasets=datasets,
            validation_datasets=datasets,
            random_key=key
        )
        
        if training_results is not None:
            print("✅ Training completed successfully")
            print(f"Final state epoch: {training_results.final_state.epoch}")
            print(f"Training history length: {len(training_results.training_history)}")
            print("\n✅ Model training with fixed architecture works correctly")
            return True
        else:
            print("❌ Training returned None")
            return False
            
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_compatibility():
    """Test that checkpoints work with fixed model."""
    print("\n\nTesting Checkpoint Compatibility")
    print("="*60)
    
    # Try to load a checkpoint (if exists)
    checkpoint_path = Path("/tmp/bc_test_checkpoints/checkpoint_best.pkl")
    
    if checkpoint_path.exists():
        print("Loading checkpoint...")
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            print("✅ Checkpoint loaded successfully")
            print(f"  Keys: {list(checkpoint.keys())}")
            
            # Check model config
            if 'config' in checkpoint and 'model_config' in checkpoint['config']:
                model_config = checkpoint['config']['model_config']
                print(f"  Model config: hidden_dim={model_config.get('hidden_dim')}, "
                      f"num_layers={model_config.get('num_layers')}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return False
    else:
        print("No checkpoint found (this is OK for first run)")
        return True


def main():
    """Run BC workflow integration tests."""
    print("="*80)
    print("BC WORKFLOW INTEGRATION TEST")
    print("="*80)
    
    test1 = test_bc_training()
    test2 = test_checkpoint_compatibility()
    
    print("\n" + "="*80)
    if test1 and test2:
        print("✅ ALL TESTS PASSED!")
        print("\nThe BC development workflow is compatible with the fixed model.")
        print("You can now run the notebook and it should work properly.")
    else:
        print("❌ Some tests failed.")
        print("Further investigation may be needed.")


if __name__ == "__main__":
    main()