#!/usr/bin/env python3
"""
Clean deterministic SCM test using new GRPO implementation.

This tests the same parent selection task as the complex system
but with clean, efficient implementation focused on strong gradients.
"""

import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from typing import List, Dict, Any

# Add parent directories to path  
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from core.grpo_trainer import CleanGRPOTrainer, compare_with_current_implementation
from core.data_pipeline import validate_tensor_quality


def create_simple_deterministic_scm():
    """Create the same deterministic SCM: X → Y, Z isolated."""
    # Reuse the working SCM creation
    from src.causal_bayes_opt.data_structures.scm import create_scm
    from src.causal_bayes_opt.mechanisms.linear import create_linear_mechanism
    
    variables = frozenset(['X', 'Y', 'Z'])
    edges = frozenset([('X', 'Y')])  # Only X causes Y
    
    mechanisms = {
        'X': create_linear_mechanism([], {}, 0.0, 1.0),  # Root with noise
        'Y': create_linear_mechanism(['X'], {'X': 10.0}, 0.0, 0.0),  # Y = 10*X (deterministic)
        'Z': create_linear_mechanism([], {}, 0.0, 1.0)   # Isolated with noise
    }
    
    return create_scm(variables, edges, mechanisms, 'Y', metadata={
        'description': 'Clean deterministic: Y = 10*X, Z isolated'
    })


def sample_observations(scm, n_samples: int = 10) -> List[Dict[str, float]]:
    """Sample initial observations from SCM."""
    from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
    return sample_from_linear_scm(scm, n_samples, seed=42)


def test_parent_selection_learning():
    """Test clean GRPO on parent selection task."""
    
    print("🧪 CLEAN GRPO: Parent Selection Test")
    print("="*50)
    print("Task: Learn that X is parent of Y (X → Y, Z isolated)")
    print("Expected: X probability 50% → 90%+ in <5 interventions")
    print("Current system: Takes 20+ interventions with tiny gradients\n")
    
    # Create clean trainer
    trainer = CleanGRPOTrainer(
        policy_architecture="clean",
        hidden_dim=256,
        learning_rate=2e-2,  # Start with moderate LR
        group_size=8
    )
    
    # Create SCM and run test
    scm = create_simple_deterministic_scm()
    target_variable = 'Y'
    
    try:
        metrics = trainer.train_on_scm(scm, target_variable, max_interventions=10)
        
        # Success criteria
        x_prob_initial = metrics['variable_probabilities'][0][0]  # X is first variable
        x_prob_final = metrics['variable_probabilities'][-1][0]
        x_improvement = x_prob_final - x_prob_initial
        
        avg_grad_norm = np.mean(metrics['gradient_norms'])
        
        print(f"\n🎯 RESULTS:")
        print(f"  X probability change: {x_prob_initial:.3f} → {x_prob_final:.3f} ({x_improvement:+.3f})")
        print(f"  Average gradient norm: {avg_grad_norm:.6f}")
        print(f"  Interventions needed: {len(metrics['variable_probabilities'])}")
        
        # Success assessment
        if x_improvement > 0.3 and avg_grad_norm > 0.01:
            print(f"  🎉 EXCELLENT: Fast learning with strong gradients!")
        elif x_improvement > 0.1:
            print(f"  ✅ GOOD: Clear learning detected")  
        elif avg_grad_norm > 0.01:
            print(f"  ⚠️ MIXED: Strong gradients but slow learning")
        else:
            print(f"  ❌ POOR: Weak gradients and minimal learning")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Clean GRPO test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_different_architectures():
    """Test different policy architectures for gradient efficiency."""
    
    print("\n🔬 ARCHITECTURE COMPARISON:")
    print("="*50)
    
    architectures = ["clean", "ultra_simple"]
    results = {}
    
    scm = create_simple_deterministic_scm()
    
    for arch in architectures:
        print(f"\nTesting {arch} architecture...")
        
        trainer = CleanGRPOTrainer(
            policy_architecture=arch,
            learning_rate=2e-2,
            group_size=8
        )
        
        try:
            metrics = trainer.train_on_scm(scm, 'Y', max_interventions=5)
            results[arch] = metrics
            
            # Quick assessment
            if metrics:
                x_improvement = (metrics['variable_probabilities'][-1][0] - 
                               metrics['variable_probabilities'][0][0])
                avg_grad = np.mean(metrics['gradient_norms'])
                print(f"  {arch}: X improvement = {x_improvement:+.3f}, Avg gradient = {avg_grad:.6f}")
            
        except Exception as e:
            print(f"  ❌ {arch} failed: {e}")
            results[arch] = None
    
    # Find best architecture
    best_arch = None
    best_score = -1
    
    for arch, result in results.items():
        if result is not None:
            x_improvement = (result['variable_probabilities'][-1][0] - 
                           result['variable_probabilities'][0][0])
            avg_grad = np.mean(result['gradient_norms'])
            score = x_improvement * avg_grad  # Combined metric
            
            if score > best_score:
                best_score = score
                best_arch = arch
    
    print(f"\n🏆 Best architecture: {best_arch} (score: {best_score:.6f})")
    return best_arch, results


def main():
    """Run clean GRPO validation tests."""
    
    print("🚀 CLEAN GRPO IMPLEMENTATION VALIDATION")
    print("="*60)
    print("Goal: Achieve strong gradients and fast learning on simple tasks")
    print("Baseline: Current system needs 20+ interventions with tiny gradients\n")
    
    # Test 1: Basic parent selection learning
    clean_metrics = test_parent_selection_learning()
    
    if clean_metrics is None:
        print("❌ Basic test failed - stopping validation")
        return
    
    # Test 2: Architecture comparison
    best_arch, arch_results = test_different_architectures()
    
    # Test 3: Compare with current system (if data available)
    print(f"\n📈 COMPARISON WITH CURRENT SYSTEM:")
    print(f"Current system gradient norm: ~0.0004 (from diagnostic)")
    print(f"Current system learning: X:0.496 → X:0.793 in 20 interventions")
    
    if clean_metrics:
        clean_grad_avg = np.mean(clean_metrics['gradient_norms'])
        clean_x_change = (clean_metrics['variable_probabilities'][-1][0] - 
                         clean_metrics['variable_probabilities'][0][0])
        
        print(f"Clean system gradient norm: {clean_grad_avg:.6f}")
        print(f"Clean system learning: X change = {clean_x_change:+.3f} in {len(clean_metrics['variable_probabilities'])} interventions")
        
        # Improvement metrics
        grad_improvement = clean_grad_avg / 0.0004 if clean_grad_avg > 0 else 0
        efficiency_improvement = (clean_x_change / len(clean_metrics['variable_probabilities'])) / (0.297 / 20)
        
        print(f"\nImprovement factors:")
        print(f"  Gradient strength: {grad_improvement:.1f}x")
        print(f"  Learning efficiency: {efficiency_improvement:.1f}x")
        
        if grad_improvement > 10:
            print(f"\n🎉 CLEAN IMPLEMENTATION SUCCESS!")
            print(f"  → Ready to replace current complex system")
        else:
            print(f"\n⚠️ PARTIAL IMPROVEMENT")
            print(f"  → Further optimization needed")


if __name__ == "__main__":
    main()