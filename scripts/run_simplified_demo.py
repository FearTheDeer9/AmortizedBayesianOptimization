#!/usr/bin/env python3
"""
Simplified ACBO Demonstration Script

Clean end-to-end demonstration using the new simplified trainers and evaluators.
Shows the full workflow: training → evaluation → visualization.
"""

import argparse
import logging
import time
from pathlib import Path
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.training.simplified_grpo_trainer import SimplifiedGRPOTrainer
from src.causal_bayes_opt.training.simplified_bc_trainer import SimplifiedBCTrainer
from src.causal_bayes_opt.utils.scm_providers import create_toy_scm_rotation, create_benchmark_scms
from src.causal_bayes_opt.evaluation.run_evaluation import run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimplifiedACBODemo:
    """Simplified demonstration runner."""
    
    def __init__(self, output_dir: Path, config: Dict[str, Any]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        
        # Track checkpoints
        self.checkpoints = {}
        
    def run_training_phase(self) -> None:
        """Run all training tasks."""
        logger.info("="*60)
        logger.info("TRAINING PHASE")
        logger.info("="*60)
        
        # Train GRPO
        if self.config.get("train_grpo", True):
            self._train_grpo()
            
        # Train BC
        if self.config.get("train_bc", True):
            self._train_bc()
            
    def _train_grpo(self) -> None:
        """Train GRPO policy."""
        logger.info("\n--- Training GRPO Policy ---")
        
        grpo_config = self.config.get("grpo", {})
        output_dir = self.output_dir / "grpo"
        output_dir.mkdir(exist_ok=True)
        
        # Create trainer
        trainer = SimplifiedGRPOTrainer(
            learning_rate=grpo_config.get("learning_rate", 3e-4),
            n_episodes=grpo_config.get("n_episodes", 100),
            episode_length=grpo_config.get("episode_length", 20),
            architecture_level=grpo_config.get("architecture_level", "simplified"),
            optimization_direction=grpo_config.get("optimization_direction", "MINIMIZE"),
            use_early_stopping=grpo_config.get("early_stopping", True),
            seed=grpo_config.get("seed", 42)
        )
        
        # Create SCMs
        scm_type = grpo_config.get("scm_type", "toy")
        if scm_type == "toy":
            scms = create_toy_scm_rotation(
                variable_range=grpo_config.get("variable_range", (3, 5)),
                structure_types=grpo_config.get("structure_types", ["fork", "chain", "collider"]),
                samples_per_config=grpo_config.get("samples_per_config", 2)
            )
        else:
            scms = create_benchmark_scms(
                benchmark_name=scm_type,
                n_scms=grpo_config.get("n_scms", 10)
            )
            
        # Extract just SCMs if we got tuples
        if isinstance(scms, list) and scms and isinstance(scms[0], tuple):
            scm_list = [scm for name, scm in scms]
        else:
            scm_list = scms
            
        # Train
        logger.info(f"Training on {len(scm_list) if isinstance(scm_list, list) else 'rotating'} SCMs")
        checkpoint = trainer.train(scm_list)
        
        # Save checkpoint
        checkpoint_path = output_dir / "grpo_final.pkl"
        trainer.save_checkpoint(checkpoint_path, checkpoint)
        
        self.checkpoints["grpo"] = checkpoint_path
        logger.info(f"GRPO training complete. Checkpoint: {checkpoint_path}")
        
        # Print metrics
        metrics = checkpoint["metrics"]
        logger.info(f"  Episodes trained: {len(metrics.get('episode_metrics', []))}")
        logger.info(f"  Final reward: {metrics.get('final_reward', 0.0):.4f}")
        logger.info(f"  Training time: {metrics['training_time']/60:.1f} minutes")
        
    def _train_bc(self) -> None:
        """Train BC models."""
        logger.info("\n--- Training BC Models ---")
        
        bc_config = self.config.get("bc", {})
        output_dir = self.output_dir / "bc"
        output_dir.mkdir(exist_ok=True)
        
        # Train surrogate
        if bc_config.get("train_surrogate", True):
            logger.info("Training BC Surrogate...")
            
            trainer = SimplifiedBCTrainer(
                model_type="surrogate",
                hidden_dim=bc_config.get("surrogate_hidden_dim", 128),
                num_layers=bc_config.get("surrogate_layers", 4),
                learning_rate=bc_config.get("learning_rate", 1e-3),
                batch_size=bc_config.get("batch_size", 32),
                max_epochs=bc_config.get("max_epochs", 50),
                early_stopping_patience=bc_config.get("patience", 5),
                seed=bc_config.get("seed", 42)
            )
            
            # Load demonstrations (simplified for demo)
            demo_path = bc_config.get("demo_path")
            if demo_path:
                checkpoint = trainer.train(demonstrations=demo_path)
            else:
                # Create synthetic demos for testing
                logger.warning("No demonstrations provided - using synthetic data")
                synthetic_demos = self._create_synthetic_demos("surrogate", 100)
                checkpoint = trainer.train(demonstrations=synthetic_demos)
                
            # Save
            checkpoint_path = output_dir / "surrogate_final.pkl"
            trainer.save_checkpoint(checkpoint_path, checkpoint)
            self.checkpoints["bc_surrogate"] = checkpoint_path
            
            logger.info(f"  Epochs: {checkpoint['metrics']['epochs_trained']}")
            logger.info(f"  Best loss: {checkpoint['metrics']['best_val_loss']:.4f}")
            
        # Train acquisition
        if bc_config.get("train_acquisition", True):
            logger.info("Training BC Acquisition...")
            
            trainer = SimplifiedBCTrainer(
                model_type="acquisition",
                hidden_dim=bc_config.get("acquisition_hidden_dim", 256),
                num_layers=bc_config.get("acquisition_layers", 4),
                learning_rate=bc_config.get("learning_rate", 1e-3),
                batch_size=bc_config.get("batch_size", 32),
                max_epochs=bc_config.get("max_epochs", 50),
                early_stopping_patience=bc_config.get("patience", 5),
                seed=bc_config.get("seed", 43)
            )
            
            # Load demonstrations
            demo_path = bc_config.get("demo_path")
            if demo_path:
                checkpoint = trainer.train(demonstrations=demo_path)
            else:
                # Create synthetic demos
                logger.warning("No demonstrations provided - using synthetic data")
                synthetic_demos = self._create_synthetic_demos("acquisition", 100)
                checkpoint = trainer.train(demonstrations=synthetic_demos)
                
            # Save
            checkpoint_path = output_dir / "acquisition_final.pkl"
            trainer.save_checkpoint(checkpoint_path, checkpoint)
            self.checkpoints["bc_acquisition"] = checkpoint_path
            
            logger.info(f"  Epochs: {checkpoint['metrics']['epochs_trained']}")
            logger.info(f"  Best loss: {checkpoint['metrics']['best_val_loss']:.4f}")
            
    def _create_synthetic_demos(self, model_type: str, n_demos: int) -> List[Dict]:
        """Create synthetic demonstrations for testing."""
        import jax.numpy as jnp
        import jax.random as random
        
        demos = []
        key = random.PRNGKey(42)
        
        for i in range(n_demos):
            key, subkey = random.split(key)
            
            if model_type == "surrogate":
                # Surrogate demo: data -> parent probabilities
                n_vars = random.randint(subkey, (), 3, 7)
                n_samples = 100
                
                data = random.normal(subkey, (n_samples, n_vars))
                # Simple synthetic parent probs
                parent_probs = jnp.eye(n_vars) * 0.8 + 0.1
                
                demo = {
                    "data": data,
                    "parent_probs": parent_probs,
                    "n_variables": int(n_vars)
                }
            else:
                # Acquisition demo: state -> action
                n_vars = random.randint(subkey, (), 3, 7)
                state = random.normal(subkey, (n_vars, 32))
                
                # Random action
                var_idx = random.randint(subkey, (), 0, n_vars)
                value = random.normal(subkey, ())
                
                demo = {
                    "state": state,
                    "action": {"variable_idx": int(var_idx), "value": float(value)},
                    "n_variables": int(n_vars)
                }
                
            demos.append(demo)
            
        return demos
        
    def run_evaluation_phase(self) -> None:
        """Run evaluation of trained models."""
        logger.info("\n" + "="*60)
        logger.info("EVALUATION PHASE")
        logger.info("="*60)
        
        eval_config = self.config.get("evaluation", {})
        
        # Determine which methods to evaluate
        methods = ["random", "learning", "oracle"]  # Always include baselines
        
        if "grpo" in self.checkpoints:
            methods.append("grpo")
            
        if "bc_surrogate" in self.checkpoints and "bc_acquisition" in self.checkpoints:
            methods.append("bc_both")
        elif "bc_surrogate" in self.checkpoints:
            methods.append("bc_surrogate")
        elif "bc_acquisition" in self.checkpoints:
            methods.append("bc_acquisition")
            
        logger.info(f"Evaluating methods: {methods}")
        
        # Create evaluation config
        eval_output = self.output_dir / "evaluation"
        
        config = {
            "n_scms": eval_config.get("n_test_scms", 5),
            "n_seeds": eval_config.get("n_runs", 3),
            "experiment": {
                "target": {
                    "max_interventions": eval_config.get("max_interventions", 10),
                    "n_observational_samples": eval_config.get("n_observational", 100),
                    "optimization_direction": eval_config.get("optimization_direction", "MINIMIZE")
                },
                "scm_generation": {
                    "use_variable_factory": True,
                    "variable_range": eval_config.get("variable_range", [3, 5]),
                    "structure_types": eval_config.get("structure_types", ["fork", "chain", "collider"])
                }
            },
            "visualization": {
                "enabled": True,
                "plot_types": ["target_trajectory", "method_comparison"]
            }
        }
        
        # Add BC checkpoints to config if available
        if "bc_surrogate" in self.checkpoints:
            config["bc_surrogate_checkpoint"] = str(self.checkpoints["bc_surrogate"])
        if "bc_acquisition" in self.checkpoints:
            config["bc_acquisition_checkpoint"] = str(self.checkpoints["bc_acquisition"])
            
        # Run evaluation
        results = run_evaluation(
            checkpoint_path=self.checkpoints.get("grpo"),
            output_dir=eval_output,
            config=config,
            methods=methods
        )
        
        # Save checkpoint info
        checkpoint_info = {
            "checkpoints_used": {k: str(v) for k, v in self.checkpoints.items()},
            "evaluation_completed": datetime.now().isoformat()
        }
        
        with open(eval_output / "checkpoint_info.json", 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
            
        logger.info("Evaluation complete!")
        self._print_evaluation_summary(results)
        
    def _print_evaluation_summary(self, results) -> None:
        """Print evaluation summary."""
        logger.info("\n--- Evaluation Summary ---")
        
        # Sort methods by performance
        sorted_methods = sorted(
            results.method_metrics.items(),
            key=lambda x: x[1].mean_final_value
        )
        
        for i, (method, metrics) in enumerate(sorted_methods):
            logger.info(f"{i+1}. {method}:")
            logger.info(f"   Final value: {metrics.mean_final_value:.4f} ± {metrics.std_final_value:.4f}")
            logger.info(f"   Improvement: {metrics.mean_improvement:.4f}")
            
    def run_complete_demo(self) -> None:
        """Run complete demonstration."""
        start_time = time.time()
        
        logger.info("="*80)
        logger.info("SIMPLIFIED ACBO DEMONSTRATION")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        
        # Save config
        with open(self.output_dir / "demo_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
            
        # Run phases
        self.run_training_phase()
        self.run_evaluation_phase()
        
        # Create final report
        total_time = time.time() - start_time
        self._create_final_report(total_time)
        
        logger.info(f"\nDemo complete! Total time: {total_time/60:.1f} minutes")
        logger.info(f"Results saved to: {self.output_dir}")
        
    def _create_final_report(self, total_time: float) -> None:
        """Create final report."""
        report_lines = [
            "="*80,
            "SIMPLIFIED ACBO DEMONSTRATION - FINAL REPORT",
            "="*80,
            "",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total runtime: {total_time/60:.1f} minutes",
            "",
            "Checkpoints Created:",
        ]
        
        for name, path in self.checkpoints.items():
            report_lines.append(f"  {name}: {path.name}")
            
        report_lines.extend([
            "",
            "Output Structure:",
            f"  {self.output_dir}/",
            f"    ├── grpo/           # GRPO training outputs",
            f"    ├── bc/             # BC training outputs", 
            f"    ├── evaluation/     # Evaluation results",
            f"    │   ├── comparison_results.json",
            f"    │   ├── plots/      # Visualizations",
            f"    │   └── evaluation_summary.txt",
            f"    └── demo_config.json",
            "",
            "="*80
        ])
        
        report = "\n".join(report_lines)
        
        with open(self.output_dir / "final_report.txt", 'w') as f:
            f.write(report)
            
        print("\n" + report)


def create_demo_config(quick: bool = False) -> Dict[str, Any]:
    """Create demonstration configuration."""
    
    if quick:
        # Quick demo for testing
        return {
            "train_grpo": True,
            "train_bc": True,
            "grpo": {
                "n_episodes": 20,
                "episode_length": 10,
                "architecture_level": "baseline",
                "learning_rate": 3e-4,
                "optimization_direction": "MINIMIZE",
                "early_stopping": True,
                "patience": 3,
                "use_surrogate": False,  # Faster without
                "scm_type": "toy",
                "variable_range": (3, 4),
                "n_rotations": 5
            },
            "bc": {
                "train_surrogate": False,  # Skip for quick demo
                "train_acquisition": True,
                "max_epochs": 10,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "patience": 2
            },
            "evaluation": {
                "n_test_scms": 2,
                "n_runs": 1,
                "max_interventions": 5,
                "variable_range": [3, 4],
                "structure_types": ["fork", "chain"]
            }
        }
    else:
        # Full demo
        return {
            "train_grpo": True,
            "train_bc": True,
            "grpo": {
                "n_episodes": 100,
                "episode_length": 20,
                "architecture_level": "simplified",
                "learning_rate": 3e-4,
                "optimization_direction": "MINIMIZE",
                "early_stopping": True,
                "patience": 5,
                "use_surrogate": True,
                "scm_type": "toy",
                "variable_range": (3, 6),
                "n_rotations": 10
            },
            "bc": {
                "train_surrogate": True,
                "train_acquisition": True,
                "max_epochs": 50,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "patience": 5,
                "surrogate_hidden_dim": 128,
                "surrogate_layers": 4,
                "acquisition_hidden_dim": 256,
                "acquisition_layers": 4
            },
            "evaluation": {
                "n_test_scms": 5,
                "n_runs": 3,
                "max_interventions": 10,
                "n_observational": 100,
                "variable_range": [3, 6],
                "structure_types": ["fork", "chain", "collider", "mixed"]
            }
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run simplified ACBO demonstration"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/simplified_demo",
        help="Output directory"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick demo with minimal parameters"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training phase (requires existing checkpoints)"
    )
    parser.add_argument(
        "--grpo-checkpoint",
        type=str,
        help="Path to existing GRPO checkpoint"
    )
    parser.add_argument(
        "--bc-surrogate-checkpoint",
        type=str,
        help="Path to existing BC surrogate checkpoint"
    )
    parser.add_argument(
        "--bc-acquisition-checkpoint",
        type=str,
        help="Path to existing BC acquisition checkpoint"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = create_demo_config(quick=args.quick)
    
    # Handle skip training
    if args.skip_training:
        config["train_grpo"] = False
        config["train_bc"] = False
        
    # Create demo runner
    demo = SimplifiedACBODemo(args.output_dir, config)
    
    # Load existing checkpoints if provided
    if args.grpo_checkpoint:
        demo.checkpoints["grpo"] = Path(args.grpo_checkpoint)
    if args.bc_surrogate_checkpoint:
        demo.checkpoints["bc_surrogate"] = Path(args.bc_surrogate_checkpoint)
    if args.bc_acquisition_checkpoint:
        demo.checkpoints["bc_acquisition"] = Path(args.bc_acquisition_checkpoint)
        
    # Run demo
    if args.skip_training:
        demo.run_evaluation_phase()
    else:
        demo.run_complete_demo()


if __name__ == "__main__":
    main()