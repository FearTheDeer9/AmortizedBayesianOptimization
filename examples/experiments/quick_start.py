#!/usr/bin/env python3
"""
Quick Start: ACBO Experiment Infrastructure

Minimal example showing how to run fair comparison experiments.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Run a quick experiment comparison."""
    print("🚀 ACBO Experiment Quick Start")
    print("=" * 40)
    
    # Import the experiment infrastructure
    from src.causal_bayes_opt.experiments.runner import run_experiments, create_erdos_renyi_scms
    
    # 1. Create test SCMs
    print("📊 Creating test SCMs...")
    scms = create_erdos_renyi_scms(
        sizes=[5],           # Small graphs for quick test
        edge_probs=[0.3],    # 30% connectivity
        n_scms_per_config=2  # 2 SCMs per configuration
    )
    print(f"Created {len(scms)} SCMs")
    
    # 2. Run fair comparison
    print("\n🔬 Running fair comparison experiments...")
    results = run_experiments(
        scms=scms,
        methods=['static_surrogate', 'learning_surrogate'],
        n_runs=2,            # Quick test with 2 runs
        n_interventions=30,  # Shorter for demo
        output_dir="quick_start_results",
        save_plots=True
    )
    
    # 3. Show results
    print(f"\n✅ Completed {results['summary']['total_experiments']} experiments")
    print(f"Success rate: {results['summary']['successful_experiments']}/{results['summary']['total_experiments']}")
    
    # Show method performance
    successful_results = [r for r in results['results'] if r.get('success', True)]
    
    if successful_results:
        print("\n📈 Results Summary:")
        for method in ['static_surrogate', 'learning_surrogate']:
            method_results = [r for r in successful_results if r.get('method') == method]
            if method_results:
                avg_f1 = sum(r.get('final_f1_score', 0) for r in method_results) / len(method_results)
                print(f"  {method}: avg F1 = {avg_f1:.3f}")
    
    print(f"\n📁 Results saved to: quick_start_results/")
    print(f"📊 Plots saved to: quick_start_results/plots/")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()