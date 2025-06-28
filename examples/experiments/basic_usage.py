#!/usr/bin/env python3
"""
Example Usage: New ACBO Experiment Infrastructure

This script demonstrates how to use the new experiment infrastructure to:
1. Run experiments with trajectory tracking
2. Compute derived metrics (true parent likelihood, etc.)
3. Generate visualizations
4. Load and analyze saved results

Run this script to see the complete workflow in action.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Demonstrate the complete experiment workflow."""
    
    print("🧪 ACBO Experiment Infrastructure Demo")
    print("=" * 50)
    
    # Import our new modules
    from src.causal_bayes_opt.experiments.runner import (
        create_erdos_renyi_scms, run_experiments, 
        load_and_analyze_results, generate_summary_plots
    )
    from src.causal_bayes_opt.analysis.trajectory_metrics import (
        compute_true_parent_likelihood, extract_metrics_from_experiment_result
    )
    from src.causal_bayes_opt.visualization.plots import (
        plot_convergence, plot_method_comparison
    )
    from src.causal_bayes_opt.storage.results import (
        load_experiment_results, create_results_summary
    )
    
    # Step 1: Create SCMs to test
    print("\\n📊 Step 1: Creating test SCMs...")
    scms = create_erdos_renyi_scms(
        sizes=[5, 8],           # Test on 5 and 8 node graphs
        edge_probs=[0.3],       # 30% edge probability
        n_scms_per_config=2     # 2 SCMs per configuration
    )
    print(f"Created {len(scms)} SCMs for testing")
    
    # Step 2: Run experiments
    print("\\n🔬 Step 2: Running experiments...")
    experiment_results = run_experiments(
        scms=scms,
        methods=['static_surrogate', 'learning_surrogate'],
        n_runs=2,                    # 2 runs per method for statistics
        n_interventions=15,          # 15 intervention steps each
        output_dir="demo_results",
        save_plots=True
    )
    
    print(f"Completed {experiment_results['summary']['total_experiments']} experiments")
    print(f"Success rate: {experiment_results['summary']['successful_experiments']}/{experiment_results['summary']['total_experiments']}")
    
    # Step 3: Demonstrate metrics computation
    print("\\n📈 Step 3: Computing trajectory metrics...")
    
    # Take first successful result as example
    successful_results = [r for r in experiment_results['results'] if r.get('success', True)]
    if successful_results:
        example_result = successful_results[0]
        true_parents = example_result.get('true_parents', [])
        
        # Extract trajectory metrics
        trajectory_metrics = extract_metrics_from_experiment_result(example_result, true_parents)
        
        print(f"Example trajectory has {len(trajectory_metrics['steps'])} steps")
        if trajectory_metrics['true_parent_likelihood']:
            print(f"Final true parent likelihood: {trajectory_metrics['true_parent_likelihood'][-1]:.3f}")
            print(f"Final F1 score: {trajectory_metrics['f1_scores'][-1]:.3f}")
    
    # Step 4: Load and re-analyze results
    print("\\n💾 Step 4: Loading and analyzing saved results...")
    
    reloaded_analysis = load_and_analyze_results("demo_results")
    print(f"Reloaded {len(reloaded_analysis['results'])} results")
    
    if reloaded_analysis['analysis']:
        overall_stats = reloaded_analysis['analysis']['overall_stats']
        print(f"Mean F1 score across all methods: {overall_stats['mean_f1_score']:.3f}")
        print(f"Mean target improvement: {overall_stats['mean_improvement']:.3f}")
    
    # Step 5: Generate additional plots
    print("\\n📊 Step 5: Generating additional visualizations...")
    
    if reloaded_analysis['analysis']:
        additional_plots = generate_summary_plots(
            reloaded_analysis['analysis'], 
            "demo_results/additional_plots"
        )
        print(f"Generated {len(additional_plots)} additional plots")
    
    # Step 6: Show example of manual metrics computation
    print("\\n🔍 Step 6: Manual metrics computation example...")
    
    # Example marginal probabilities (would come from real experiment)
    example_marginals = {
        'X1': 0.1,  # Not a parent
        'X2': 0.9,  # Strong parent
        'X3': 0.2,  # Weak parent
        'X4': 0.8   # Strong parent  
    }
    
    example_true_parents = ['X2', 'X4']
    
    likelihood = compute_true_parent_likelihood(example_marginals, example_true_parents)
    print(f"Example true parent likelihood calculation:")
    print(f"  Marginals: {example_marginals}")
    print(f"  True parents: {example_true_parents}")
    print(f"  Likelihood: {likelihood:.3f}")
    print(f"  (= 0.9 * 0.8 * (1-0.1) * (1-0.2) = {0.9 * 0.8 * 0.9 * 0.8:.3f})")
    
    print("\\n✅ Demo completed successfully!")
    print("\\nCheck the following directories for results:")
    print("  📁 demo_results/ - Experiment results and plots")
    print("  📁 demo_results/plots/ - Generated visualizations")
    print("\\nKey features demonstrated:")
    print("  ✓ Automatic trajectory data collection")
    print("  ✓ True parent likelihood computation")  
    print("  ✓ Convergence analysis")
    print("  ✓ Method comparison plots")
    print("  ✓ Results storage and loading")
    print("  ✓ No modifications to core data structures")


def quick_metrics_demo():
    """Quick demo of just the metrics computation."""
    
    print("\\n🎯 Quick Metrics Demo")
    print("-" * 30)
    
    from src.causal_bayes_opt.analysis.trajectory_metrics import (
        compute_true_parent_likelihood, compute_f1_score_from_marginals
    )
    
    # Example scenarios
    scenarios = [
        {
            'name': 'Perfect Discovery',
            'marginals': {'X1': 1.0, 'X2': 0.0, 'X3': 1.0},
            'true_parents': ['X1', 'X3']
        },
        {
            'name': 'Partial Discovery', 
            'marginals': {'X1': 0.8, 'X2': 0.2, 'X3': 0.7},
            'true_parents': ['X1', 'X3']
        },
        {
            'name': 'Poor Discovery',
            'marginals': {'X1': 0.3, 'X2': 0.6, 'X3': 0.4},
            'true_parents': ['X1', 'X3']
        }
    ]
    
    for scenario in scenarios:
        likelihood = compute_true_parent_likelihood(
            scenario['marginals'], 
            scenario['true_parents']
        )
        f1_score = compute_f1_score_from_marginals(
            scenario['marginals'],
            scenario['true_parents']
        )
        
        print(f"\\n{scenario['name']}:")
        print(f"  True Parent Likelihood: {likelihood:.3f}")
        print(f"  F1 Score: {f1_score:.3f}")


if __name__ == "__main__":
    try:
        main()
        quick_metrics_demo()
    except KeyboardInterrupt:
        print("\\n\\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\\n\\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()