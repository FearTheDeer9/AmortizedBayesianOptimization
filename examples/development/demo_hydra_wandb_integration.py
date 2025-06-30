#!/usr/bin/env python3
"""
Demonstration of Hydra + WandB Integration

This script shows how the enhanced erdos_renyi_scaling_experiment.py now supports:
1. Hydra configuration management 
2. Enhanced WandB experiment tracking
3. CLI parameter overrides and multirun sweeps

Usage examples:
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def demo_basic_usage():
    """Demo 1: Basic usage with default config"""
    print("🎯 Demo 1: Basic usage with default configuration")
    print("Command: python scripts/erdos_renyi_scaling_experiment.py")
    print("This uses config/experiment/erdos_renyi_base.yaml")
    print()

def demo_config_variants():
    """Demo 2: Using different config variants"""
    print("🔧 Demo 2: Using different configuration variants")
    print()
    
    print("Quick test (fast for development):")
    print("python scripts/erdos_renyi_scaling_experiment.py experiment=erdos_renyi_quick")
    print()
    
    print("Large-scale experiment:")
    print("python scripts/erdos_renyi_scaling_experiment.py experiment=erdos_renyi_large")
    print()

def demo_cli_overrides():
    """Demo 3: CLI parameter overrides"""
    print("⚙️  Demo 3: CLI parameter overrides")
    print()
    
    print("Override specific parameters:")
    print("python scripts/erdos_renyi_scaling_experiment.py min_nodes=8 max_nodes=15 n_runs_per_config=5")
    print()
    
    print("Disable WandB logging:")
    print("python scripts/erdos_renyi_scaling_experiment.py wandb.enabled=false")
    print()
    
    print("Change WandB project and tags:")
    print("python scripts/erdos_renyi_scaling_experiment.py wandb.project=my-project wandb.tags=[test,dev]")
    print()

def demo_multirun_sweeps():
    """Demo 4: Hyperparameter sweeps with multirun"""
    print("🔄 Demo 4: Hyperparameter sweeps with multirun")
    print()
    
    print("Sweep over different graph sizes:")
    print("python scripts/erdos_renyi_scaling_experiment.py --multirun min_nodes=3,5,8 max_nodes=10,15,20")
    print()
    
    print("Sweep over edge probabilities:")
    print("python scripts/erdos_renyi_scaling_experiment.py --multirun edge_probability=0.2,0.3,0.5")
    print()
    
    print("Complex sweep with multiple parameters:")
    print("python scripts/erdos_renyi_scaling_experiment.py --multirun \\")
    print("  edge_probability=0.2,0.3,0.5 \\")
    print("  n_intervention_steps=10,20,30 \\")
    print("  learning_rate=1e-4,1e-3,1e-2")
    print()

def demo_wandb_features():
    """Demo 5: WandB features showcased"""
    print("📊 Demo 5: Enhanced WandB experiment tracking includes:")
    print()
    
    features = [
        "• Automatic experiment naming and grouping",
        "• Complete configuration logging", 
        "• Real-time metrics tracking (F1 scores, improvements)",
        "• Per-graph-size detailed metrics",
        "• Results summary tables",
        "• Experiment status tracking",
        "• Validation result logging",
        "• Automatic tagging for easy filtering"
    ]
    
    for feature in features:
        print(feature)
    print()

def demo_hydra_features():
    """Demo 6: Hydra features showcased"""
    print("🔧 Demo 6: Hydra configuration management includes:")
    print()
    
    features = [
        "• Compositional configs (base + variants)",
        "• CLI parameter overrides",
        "• Multirun sweeps for hyperparameter optimization", 
        "• Structured output directories",
        "• Configuration validation",
        "• Easy config sharing and reproducibility"
    ]
    
    for feature in features:
        print(feature)
    print()

def run_quick_demo():
    """Actually run a quick demo if requested"""
    print("🚀 Running quick demo with WandB disabled...")
    
    try:
        cmd = [
            sys.executable, 
            "scripts/erdos_renyi_scaling_experiment.py",
            "experiment=erdos_renyi_quick",
            "wandb.enabled=false"
        ]
        subprocess.run(cmd, cwd=project_root, check=True)
        print("✅ Demo completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed: {e}")
    except Exception as e:
        print(f"❌ Error running demo: {e}")

def main():
    """Show all demonstrations"""
    print("🎉 Hydra + WandB Integration Demo")
    print("=" * 50)
    print()
    
    demo_basic_usage()
    demo_config_variants()
    demo_cli_overrides()
    demo_multirun_sweeps()
    demo_wandb_features()
    demo_hydra_features()
    
    print("💡 To see the full power, try running:")
    print("python scripts/demo_hydra_wandb_integration.py --run-demo")
    print()

if __name__ == "__main__":
    if "--run-demo" in sys.argv:
        run_quick_demo()
    else:
        main()
