#!/usr/bin/env python3
"""
Full ACBO Demonstration Script

Complete end-to-end demonstration of the Amortized Causal Bayesian Optimization system.
Trains models, evaluates against baselines, and generates comprehensive results.
"""

import argparse
import logging
import time
from pathlib import Path
import sys
import json
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_demo_config(
    demo_name: str = "acbo_demo",
    optimization_direction: str = "MINIMIZE",
    quick_mode: bool = False
) -> Dict[str, Any]:
    """Create configuration for full demonstration."""
    
    if quick_mode:
        # Quick demo for testing
        config = {
            "demo_name": demo_name,
            "optimization_direction": optimization_direction,
            "grpo": {
                "n_episodes": 50,
                "learning_rate": 1e-3,
                "max_variables": 5
            },
            "bc": {
                "max_epochs": 20,
                "learning_rate": 1e-3
            },
            "evaluation": {
                "n_test_scms": 2,
                "n_runs_per_scm": 1,
                "max_interventions": 5
            }
        }
    else:
        # Full demo configuration
        config = {
            "demo_name": demo_name,
            "optimization_direction": optimization_direction,
            "grpo": {
                "n_episodes": 200,
                "learning_rate": 1e-3,
                "max_variables": 6
            },
            "bc": {
                "max_epochs": 50,
                "learning_rate": 1e-3
            },
            "evaluation": {
                "n_test_scms": 10,
                "n_runs_per_scm": 3,
                "max_interventions": 20
            }
        }
    
    return config


def run_command(cmd: list, cwd: Optional[Path] = None) -> Dict[str, Any]:
    """Run a command and capture output."""
    logger.info(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"Command failed with code {result.returncode}")
        logger.error(f"stderr: {result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }


def train_grpo(config: Dict[str, Any], output_dir: Path) -> Path:
    """Train GRPO model."""
    logger.info("="*60)
    logger.info("Training GRPO Policy")
    logger.info("="*60)
    
    grpo_output = output_dir / "grpo_training"
    
    # Use config file for quick demo
    config_path = project_root / "config" / "grpo_quick_demo.yaml"
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "core" / "train_grpo.py"),
        "--output-dir", str(grpo_output),
        "--config", str(config_path),
        "--optimization", config["optimization_direction"],
        "--episodes", str(config["grpo"]["n_episodes"]),
        "--learning-rate", str(config["grpo"]["learning_rate"]),
        "--max-variables", str(config["grpo"]["max_variables"])
    ]
    
    result = run_command(cmd)
    
    # Extract checkpoint path from output
    checkpoint_path = None
    for line in result["stdout"].split("\n"):
        if "Final checkpoint:" in line:
            checkpoint_path = Path(line.split("Final checkpoint:")[-1].strip())
            break
    
    if not checkpoint_path or not checkpoint_path.exists():
        # Try to find latest checkpoint
        checkpoints_dir = grpo_output / "checkpoints"
        if checkpoints_dir.exists():
            checkpoints = list(checkpoints_dir.glob("grpo_*"))
            if checkpoints:
                checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    if not checkpoint_path:
        raise RuntimeError("Failed to find GRPO checkpoint after training")
    
    logger.info(f"GRPO training complete. Checkpoint: {checkpoint_path}")
    return checkpoint_path


def train_bc(config: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
    """Train BC models."""
    logger.info("="*60)
    logger.info("Training BC Models")
    logger.info("="*60)
    
    bc_output = output_dir / "bc_training"
    
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "core" / "train_bc.py"),
        "--output-dir", str(bc_output),
        "--model-type", "both",
        "--max-epochs", str(config["bc"]["max_epochs"]),
        "--learning-rate", str(config["bc"]["learning_rate"])
    ]
    
    result = run_command(cmd)
    
    # Parse training summary to get checkpoint paths
    summary_path = bc_output / "training_summary.json"
    if not summary_path.exists():
        raise RuntimeError("BC training summary not found")
    
    with open(summary_path, 'r') as f:
        bc_results = json.load(f)
    
    checkpoints = {}
    
    # Extract surrogate checkpoint
    if "surrogate" in bc_results.get("model_results", {}):
        surrogate_path = Path(bc_results["model_results"]["surrogate"]["checkpoint_path"])
        if surrogate_path.exists():
            checkpoints["surrogate"] = surrogate_path
        else:
            logger.warning(f"Surrogate checkpoint not found: {surrogate_path}")
    
    # Extract acquisition checkpoint
    if "acquisition" in bc_results.get("model_results", {}):
        acquisition_path = Path(bc_results["model_results"]["acquisition"]["checkpoint_path"])
        if acquisition_path.exists():
            checkpoints["acquisition"] = acquisition_path
        else:
            logger.warning(f"Acquisition checkpoint not found: {acquisition_path}")
    
    logger.info(f"BC training complete. Checkpoints: {list(checkpoints.keys())}")
    return checkpoints


def evaluate_models(
    grpo_checkpoint: Path,
    bc_checkpoints: Dict[str, Path],
    config: Dict[str, Any],
    output_dir: Path
) -> Path:
    """Evaluate all models."""
    logger.info("="*60)
    logger.info("Evaluating Models")
    logger.info("="*60)
    
    eval_output = output_dir / "evaluation"
    
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "core" / "evaluate_methods.py"),
        "--output-dir", str(eval_output),
        "--grpo-checkpoint", str(grpo_checkpoint),
        "--max-interventions", str(config["evaluation"]["max_interventions"]),
        "--n-test-scms", str(config["evaluation"]["n_test_scms"]),
        "--n-runs", str(config["evaluation"]["n_runs_per_scm"]),
        "--optimization", config["optimization_direction"]
    ]
    
    # Add BC checkpoints if available
    if "surrogate" in bc_checkpoints:
        cmd.extend(["--bc-surrogate-checkpoint", str(bc_checkpoints["surrogate"])])
    if "acquisition" in bc_checkpoints:
        cmd.extend(["--bc-acquisition-checkpoint", str(bc_checkpoints["acquisition"])])
    
    result = run_command(cmd)
    
    logger.info("Evaluation complete")
    return eval_output


def create_final_report(
    output_dir: Path,
    grpo_checkpoint: Path,
    bc_checkpoints: Dict[str, Path],
    eval_dir: Path,
    config: Dict[str, Any],
    total_time: float
) -> str:
    """Create final demonstration report."""
    
    lines = []
    lines.append("="*80)
    lines.append("AMORTIZED CAUSAL BAYESIAN OPTIMIZATION - FULL DEMONSTRATION")
    lines.append("="*80)
    lines.append("")
    
    lines.append(f"Demonstration: {config['demo_name']}")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total runtime: {total_time/60:.1f} minutes")
    lines.append("")
    
    # Configuration summary
    lines.append("Configuration:")
    lines.append(f"  Optimization direction: {config['optimization_direction']}")
    lines.append(f"  GRPO episodes: {config['grpo']['n_episodes']}")
    lines.append(f"  BC epochs: {config['bc']['max_epochs']}")
    lines.append(f"  Test SCMs: {config['evaluation']['n_test_scms']}")
    lines.append(f"  Runs per SCM: {config['evaluation']['n_runs_per_scm']}")
    lines.append(f"  Max interventions: {config['evaluation']['max_interventions']}")
    lines.append("")
    
    # Training summary
    lines.append("Training Results:")
    lines.append(f"  GRPO checkpoint: {grpo_checkpoint.name}")
    for model_type, checkpoint in bc_checkpoints.items():
        lines.append(f"  BC {model_type} checkpoint: {checkpoint.name}")
    lines.append("")
    
    # Evaluation summary
    eval_summary_path = eval_dir / "evaluation_summary.txt"
    if eval_summary_path.exists():
        lines.append("Evaluation Results:")
        lines.append("-"*80)
        
        # Extract key results from evaluation summary
        with open(eval_summary_path, 'r') as f:
            eval_content = f.read()
            
        # Find performance summary section
        if "Performance Summary:" in eval_content:
            perf_section = eval_content.split("Performance Summary:")[1].split("\n\n")[0]
            lines.append(perf_section.strip())
        
        lines.append("")
    
    # Output locations
    lines.append("Output Files:")
    lines.append(f"  Base directory: {output_dir}")
    lines.append(f"  GRPO training: {output_dir / 'grpo_training'}")
    lines.append(f"  BC training: {output_dir / 'bc_training'}")
    lines.append(f"  Evaluation: {eval_dir}")
    lines.append("")
    
    # Visualizations
    plots_dir = eval_dir / "plots"
    if plots_dir.exists():
        plot_files = list(plots_dir.glob("*.png"))
        if plot_files:
            lines.append(f"Visualizations ({len(plot_files)} plots):")
            for plot in plot_files:
                lines.append(f"  - {plot.name}")
            lines.append("")
    
    # Key insights
    lines.append("Key Insights:")
    lines.append("-"*80)
    
    # Check if trained methods outperformed baselines
    comparison_results_path = eval_dir / "comparison_results.json"
    if comparison_results_path.exists():
        with open(comparison_results_path, 'r') as f:
            comp_results = json.load(f)
            
        method_results = comp_results.get("method_results", {})
        
        # Find best trained and baseline methods
        trained_methods = {k: v for k, v in method_results.items() 
                         if any(x in k.lower() for x in ['grpo', 'bc'])}
        baseline_methods = {k: v for k, v in method_results.items() 
                          if any(x in k.lower() for x in ['random', 'oracle', 'learning'])}
        
        if trained_methods and baseline_methods:
            # Get best of each category
            best_trained = min(trained_methods.items(), 
                             key=lambda x: x[1]['mean_final_value'])
            best_baseline = min(baseline_methods.items(), 
                              key=lambda x: x[1]['mean_final_value'])
            
            if config['optimization_direction'] == "MINIMIZE":
                improvement = (best_baseline[1]['mean_final_value'] - 
                             best_trained[1]['mean_final_value'])
                improvement_pct = improvement / abs(best_baseline[1]['mean_final_value']) * 100
            else:
                improvement = (best_trained[1]['mean_final_value'] - 
                             best_baseline[1]['mean_final_value'])
                improvement_pct = improvement / abs(best_baseline[1]['mean_final_value']) * 100
            
            lines.append(f"• Best trained method: {best_trained[0]}")
            lines.append(f"  Final value: {best_trained[1]['mean_final_value']:.4f}")
            lines.append(f"• Best baseline: {best_baseline[0]}")
            lines.append(f"  Final value: {best_baseline[1]['mean_final_value']:.4f}")
            lines.append(f"• Improvement: {improvement_pct:.1f}%")
            
            if improvement_pct > 0:
                lines.append("• ✅ Trained methods outperformed baselines!")
            else:
                lines.append("• ⚠️ Baselines performed better - may need more training")
    
    lines.append("")
    lines.append("="*80)
    lines.append("Demonstration complete! See output directory for full results.")
    lines.append("="*80)
    
    report = "\n".join(lines)
    
    # Save report
    report_path = output_dir / "demonstration_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    return report


def main():
    """Main entry point for full demonstration."""
    parser = argparse.ArgumentParser(
        description="Run full ACBO demonstration"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/acbo_demo",
        help="Base output directory"
    )
    parser.add_argument(
        "--demo-name",
        type=str,
        default="acbo_demo",
        help="Name for this demonstration"
    )
    parser.add_argument(
        "--optimization",
        type=str,
        choices=["MINIMIZE", "MAXIMIZE"],
        default="MINIMIZE",
        help="Optimization direction"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick demonstration with reduced parameters"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and use existing checkpoints"
    )
    parser.add_argument(
        "--grpo-checkpoint",
        type=str,
        default=None,
        help="Existing GRPO checkpoint (if skipping training)"
    )
    parser.add_argument(
        "--bc-surrogate-checkpoint",
        type=str,
        default=None,
        help="Existing BC surrogate checkpoint (if skipping training)"
    )
    parser.add_argument(
        "--bc-acquisition-checkpoint",
        type=str,
        default=None,
        help="Existing BC acquisition checkpoint (if skipping training)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_demo_config(
        demo_name=args.demo_name,
        optimization_direction=args.optimization,
        quick_mode=args.quick
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = output_dir / "demo_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Starting ACBO Full Demonstration")
    logger.info(f"Output directory: {output_dir}")
    
    start_time = time.time()
    
    try:
        # Step 1: Train or load models
        if args.skip_training:
            logger.info("Skipping training - using provided checkpoints")
            
            if not args.grpo_checkpoint:
                raise ValueError("Must provide --grpo-checkpoint when skipping training")
                
            grpo_checkpoint = Path(args.grpo_checkpoint)
            if not grpo_checkpoint.exists():
                raise ValueError(f"GRPO checkpoint not found: {grpo_checkpoint}")
            
            bc_checkpoints = {}
            if args.bc_surrogate_checkpoint:
                path = Path(args.bc_surrogate_checkpoint)
                if path.exists():
                    bc_checkpoints["surrogate"] = path
                    
            if args.bc_acquisition_checkpoint:
                path = Path(args.bc_acquisition_checkpoint)
                if path.exists():
                    bc_checkpoints["acquisition"] = path
        else:
            # Train models
            grpo_checkpoint = train_grpo(config, output_dir)
            bc_checkpoints = train_bc(config, output_dir)
        
        # Step 2: Evaluate models
        eval_dir = evaluate_models(
            grpo_checkpoint,
            bc_checkpoints,
            config,
            output_dir
        )
        
        # Step 3: Create final report
        total_time = time.time() - start_time
        report = create_final_report(
            output_dir,
            grpo_checkpoint,
            bc_checkpoints,
            eval_dir,
            config,
            total_time
        )
        
        # Print report
        print("\n" + report)
        
        logger.info(f"Demonstration complete! Total time: {total_time/60:.1f} minutes")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()