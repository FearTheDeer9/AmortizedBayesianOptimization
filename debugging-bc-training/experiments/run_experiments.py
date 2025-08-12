#!/usr/bin/env python3
"""
Run multiple BC training experiments with different configurations.
Tests hyperparameters, augmentation strategies, and loss weighting.
"""

import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any
import time

# Experiment configurations
EXPERIMENTS = [
    # Baseline experiments with different hyperparameters
    {
        "name": "baseline_original",
        "trainer": "enhanced_bc_trainer_fixed",
        "hidden_dim": 256,
        "learning_rate": 3e-4,
        "epochs": 50,
        "description": "Original configuration"
    },
    {
        "name": "smaller_model",
        "trainer": "enhanced_bc_trainer_fixed",
        "hidden_dim": 128,
        "learning_rate": 3e-4,
        "epochs": 50,
        "description": "Smaller model, same LR"
    },
    {
        "name": "higher_lr",
        "trainer": "enhanced_bc_trainer_fixed",
        "hidden_dim": 256,
        "learning_rate": 1e-3,
        "epochs": 50,
        "description": "Original size, higher LR"
    },
    {
        "name": "small_high_lr",
        "trainer": "enhanced_bc_trainer_fixed",
        "hidden_dim": 128,
        "learning_rate": 1e-3,
        "epochs": 50,
        "description": "Smaller model, higher LR"
    },
    {
        "name": "tiny_highest_lr",
        "trainer": "enhanced_bc_trainer_fixed",
        "hidden_dim": 64,
        "learning_rate": 3e-3,
        "epochs": 50,
        "description": "Tiny model, highest LR"
    },
    
    # Permutation augmentation experiments
    {
        "name": "permutation_baseline",
        "trainer": "permutation_augmented_trainer",
        "hidden_dim": 256,
        "learning_rate": 3e-4,
        "n_permutations": 5,
        "epochs": 50,
        "description": "Baseline with full permutation augmentation"
    },
    {
        "name": "permutation_small_high",
        "trainer": "permutation_augmented_trainer",
        "hidden_dim": 128,
        "learning_rate": 1e-3,
        "n_permutations": 5,
        "epochs": 50,
        "description": "Smaller model, higher LR, with permutations"
    },
    
    # Weighted loss experiments
    {
        "name": "weighted_baseline",
        "trainer": "weighted_loss_trainer",
        "hidden_dim": 256,
        "learning_rate": 3e-4,
        "epochs": 50,
        "description": "Baseline with weighted loss"
    },
    {
        "name": "weighted_small_high",
        "trainer": "weighted_loss_trainer",
        "hidden_dim": 128,
        "learning_rate": 1e-3,
        "epochs": 50,
        "description": "Smaller model, higher LR, with weighted loss"
    },
    
    # Combined approaches
    {
        "name": "combined_best",
        "trainer": "balanced_trainer",
        "hidden_dim": 128,
        "learning_rate": 1e-3,
        "n_permutations": 5,
        "use_weights": True,
        "epochs": 50,
        "description": "Combined: permutation + weights + tuned hyperparams"
    }
]

def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single experiment.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Results dictionary
    """
    name = config["name"]
    trainer = config["trainer"]
    
    print("\n" + "="*60)
    print(f"Running Experiment: {name}")
    print(f"Description: {config['description']}")
    print("="*60)
    
    # Build command based on trainer type
    base_path = Path(__file__).parent
    
    if trainer == "enhanced_bc_trainer_fixed":
        script_path = base_path.parent / "enhanced_bc_trainer_fixed.py"
    elif trainer == "permutation_augmented_trainer":
        script_path = base_path / "permutation_augmented_trainer.py"
    elif trainer == "weighted_loss_trainer":
        script_path = base_path / "weighted_loss_trainer.py"
    elif trainer == "balanced_trainer":
        script_path = base_path / "balanced_trainer.py"
        # Note: We'll create this file next
        if not script_path.exists():
            print(f"Skipping {name} - trainer not implemented yet")
            return {"status": "skipped", "name": name}
    else:
        print(f"Unknown trainer: {trainer}")
        return {"status": "error", "name": name, "error": f"Unknown trainer: {trainer}"}
    
    # Build command
    output_dir = f"debugging-bc-training/results_experiments/{name}"
    cmd = [
        "python", str(script_path),
        "--demo_path", "expert_demonstrations/raw/raw_demonstrations",
        "--epochs", str(config.get("epochs", 50)),
        "--batch_size", str(config.get("batch_size", 32)),
        "--learning_rate", str(config.get("learning_rate", 3e-4)),
        "--hidden_dim", str(config.get("hidden_dim", 256)),
        "--output_dir", output_dir,
        "--seed", str(config.get("seed", 42))
    ]
    
    # Add trainer-specific arguments
    if "n_permutations" in config:
        cmd.extend(["--n_permutations", str(config["n_permutations"])])
    
    if "manual_weights" in config:
        cmd.extend(["--manual_weights", json.dumps(config["manual_weights"])])
    
    # Limit training data for faster experiments
    if config.get("quick_test", False):
        cmd.extend(["--max_demos", "20"])
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run experiment
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ {name} completed in {duration/60:.1f} minutes")
            
            # Try to load metrics
            metrics_path = Path(output_dir) / "metrics_history.pkl"
            if metrics_path.exists():
                import pickle
                with open(metrics_path, 'rb') as f:
                    metrics = pickle.load(f)
                
                # Extract key metrics
                summary = {}
                if 'epoch_metrics' in metrics and metrics['epoch_metrics']:
                    last_epoch = metrics['epoch_metrics'][-1]
                    summary['final_val_accuracy'] = last_epoch.get('val_accuracy', 0)
                    summary['final_val_f1'] = last_epoch.get('val_f1', 0)
                    
                    # Best metrics
                    val_accs = [e.get('val_accuracy', 0) for e in metrics['epoch_metrics']]
                    summary['best_val_accuracy'] = max(val_accs) if val_accs else 0
                
                # Check X4 performance
                if 'per_variable_stats' in metrics and 'X4' in metrics['per_variable_stats']:
                    x4_stats = metrics['per_variable_stats']['X4']
                    if x4_stats['attempts'] > 0:
                        summary['X4_accuracy'] = x4_stats['correct'] / x4_stats['attempts']
                    else:
                        summary['X4_accuracy'] = 0.0
                
                return {
                    "status": "success",
                    "name": name,
                    "duration_minutes": duration / 60,
                    "metrics": summary
                }
            else:
                return {
                    "status": "success",
                    "name": name,
                    "duration_minutes": duration / 60,
                    "metrics": None
                }
        else:
            print(f"✗ {name} failed with return code {result.returncode}")
            print(f"Error output: {result.stderr[:500]}")
            return {
                "status": "failed",
                "name": name,
                "duration_minutes": duration / 60,
                "error": result.stderr[:500]
            }
    
    except subprocess.TimeoutExpired:
        print(f"✗ {name} timed out after 30 minutes")
        return {
            "status": "timeout",
            "name": name,
            "duration_minutes": 30
        }
    except Exception as e:
        print(f"✗ {name} failed with exception: {e}")
        return {
            "status": "error",
            "name": name,
            "error": str(e)
        }


def main():
    """Run all experiments and save results."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run BC training experiments')
    parser.add_argument('--quick', action='store_true', help='Run quick tests with limited data')
    parser.add_argument('--experiments', nargs='+', help='Specific experiments to run')
    
    args = parser.parse_args()
    
    # Filter experiments if specified
    experiments_to_run = EXPERIMENTS
    if args.experiments:
        experiments_to_run = [e for e in EXPERIMENTS if e['name'] in args.experiments]
    
    if args.quick:
        print("Running in quick test mode (limited data)")
        for exp in experiments_to_run:
            exp['quick_test'] = True
            exp['epochs'] = 10  # Fewer epochs for quick test
    
    print(f"Running {len(experiments_to_run)} experiments...")
    
    results = []
    for config in experiments_to_run:
        result = run_experiment(config)
        results.append(result)
        
        # Save intermediate results
        results_file = Path("debugging-bc-training/results_experiments/experiment_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for result in results:
        status = result['status']
        name = result['name']
        
        if status == 'success' and result.get('metrics'):
            metrics = result['metrics']
            print(f"\n{name}:")
            print(f"  Status: ✓ Success")
            print(f"  Duration: {result['duration_minutes']:.1f} min")
            print(f"  Best Val Accuracy: {metrics.get('best_val_accuracy', 0):.3f}")
            print(f"  Final Val Accuracy: {metrics.get('final_val_accuracy', 0):.3f}")
            print(f"  X4 Accuracy: {metrics.get('X4_accuracy', 0):.3f}")
        else:
            print(f"\n{name}:")
            print(f"  Status: ✗ {status}")
    
    # Find best performer
    successful = [r for r in results if r['status'] == 'success' and r.get('metrics')]
    if successful:
        best_overall = max(successful, key=lambda x: x['metrics'].get('best_val_accuracy', 0))
        best_x4 = max(successful, key=lambda x: x['metrics'].get('X4_accuracy', 0))
        
        print("\n" + "="*60)
        print("BEST PERFORMERS")
        print("="*60)
        print(f"Best Overall Accuracy: {best_overall['name']} "
              f"({best_overall['metrics']['best_val_accuracy']:.3f})")
        print(f"Best X4 Accuracy: {best_x4['name']} "
              f"({best_x4['metrics']['X4_accuracy']:.3f})")
    
    print(f"\nResults saved to: debugging-bc-training/results_experiments/experiment_results.json")


if __name__ == "__main__":
    main()