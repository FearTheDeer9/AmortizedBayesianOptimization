#!/usr/bin/env python3
"""
Main script to run computational efficiency scaling experiments.

This script implements Experiments 1.1, 1.2, and 1.3 from the research plan:
- Inference time scaling analysis
- Memory usage scaling analysis  
- Training time amortization analysis
"""

import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

# Add paths
import sys
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from experiments.evaluation.computational_efficiency.src.scaling_analyzer import ScalingAnalyzer
from experiments.evaluation.core.plotting_utils import PlottingUtils

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_scaling_plots(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Create scaling analysis plots.
    
    Args:
        results: Scaling analysis results
        output_dir: Directory to save plots
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    PlottingUtils.setup_style()
    
    # Plot 1: Inference time scaling
    if 'inference_time_scaling' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        time_results = results['inference_time_scaling']
        colors = plt.cm.Set1(np.linspace(0, 1, len(time_results)))
        
        for (method, result), color in zip(time_results.items(), colors):
            if result.successful_sizes:
                sizes = result.successful_sizes
                times = [result.time_by_size[s] for s in sizes]
                
                ax.semilogy(sizes, times, 'o-', label=f"{method} ({result.complexity_estimate})",
                          color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel('Number of Variables')
        ax.set_ylabel('Inference Time (seconds, log scale)')
        ax.set_title('Inference Time Scaling Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'inference_time_scaling.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Memory usage scaling
    if 'memory_usage_scaling' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        memory_results = results['memory_usage_scaling']
        colors = plt.cm.Set1(np.linspace(0, 1, len(memory_results)))
        
        for (method, memory_by_size), color in zip(memory_results.items(), colors):
            if memory_by_size:
                sizes = sorted(memory_by_size.keys())
                memories = [memory_by_size[s] for s in sizes]
                
                ax.plot(sizes, memories, 'o-', label=method,
                       color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel('Number of Variables')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory Usage Scaling Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'memory_usage_scaling.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Amortization analysis
    if 'training_amortization' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        amort_data = results['training_amortization']
        breakeven_data = amort_data.get('breakeven_analysis', {})
        
        # Plot break-even curves for different training types
        training_types = list(breakeven_data.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(training_types)))
        
        for training_type, color in zip(training_types, colors):
            if training_type in breakeven_data:
                type_data = breakeven_data[training_type]
                sizes = sorted([s for s in type_data.keys() if isinstance(s, int)])
                breakeven_problems = [type_data[s]['breakeven_problems'] 
                                    for s in sizes if type_data[s]['breakeven_problems'] != float('inf')]
                valid_sizes = sizes[:len(breakeven_problems)]
                
                if valid_sizes:
                    ax.semilogy(valid_sizes, breakeven_problems, 'o-', 
                              label=f"{training_type.replace('_', ' ').title()}",
                              color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel('Number of Variables')
        ax.set_ylabel('Break-even Problems (log scale)')
        ax.set_title('Training Amortization Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'amortization_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Summary plot combining all analyses
    if len(results) >= 2:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Replot key results in summary
        # ... (implementation would mirror above plots in subplots)
        
        fig.suptitle('Computational Efficiency Summary', fontsize=16)
        plt.tight_layout()
        plt.savefig(plots_dir / 'efficiency_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Scaling plots saved to {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run computational efficiency experiments')
    parser.add_argument('--config', type=Path,
                       default=Path(__file__).parent.parent / 'configs' / 'scaling_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=Path,
                       help='Output directory (default: results/scaling_[timestamp])')
    parser.add_argument('--compare-exact', action='store_true',
                       help='Include exact CBO-U comparison (slow for large graphs)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with small sizes')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    # Modify config for quick test
    if args.quick:
        config['scaling_test']['sizes'] = [5, 10, 15]
        config['scaling_test']['n_repetitions'] = 2
        logger.info("Running in quick test mode")
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(__file__).parent.parent / 'results' / f'scaling_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Get experiments directory
    experiments_dir = Path(__file__).parent.parent.parent
    
    # Create scaling analyzer
    analyzer = ScalingAnalyzer(config)
    
    # Run complete analysis
    logger.info("Starting computational efficiency analysis...")
    results = analyzer.run_complete_scaling_analysis(experiments_dir)
    
    # Export results
    analyzer.export_scaling_analysis(results, output_dir)
    
    # Create plots
    create_scaling_plots(results, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPUTATIONAL EFFICIENCY SUMMARY")
    print("=" * 60)
    
    if 'comparative_analysis' in results:
        comp_analysis = results['comparative_analysis']
        
        if 'inference_comparison' in comp_analysis:
            inf_comp = comp_analysis['inference_comparison']
            best_method = inf_comp.get('best_scaling_method')
            max_sizes = inf_comp.get('max_feasible_sizes', {})
            
            print(f"Best scaling method: {best_method}")
            print("Maximum feasible sizes:")
            for method, max_size in max_sizes.items():
                print(f"  {method}: {max_size} variables")
        
        if 'amortization_insights' in comp_analysis:
            amort = comp_analysis['amortization_insights']
            breakeven = amort.get('typical_breakeven_problems', 'N/A')
            speedup = amort.get('long_term_speedup_factor', 'N/A')
            print(f"\nAmortization Analysis:")
            print(f"  Break-even point: {breakeven} problems")
            print(f"  Long-term speedup: {speedup}x")
    
    print(f"\nDetailed results saved to: {output_dir}")
    print("=" * 60)
    
    logger.info("Computational efficiency analysis complete!")


if __name__ == "__main__":
    main()