#!/usr/bin/env python3
"""
Quick test of just the improved validation metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jnp
import numpy as onp

def test_metrics_only():
    """Test just the validation metrics without full training."""
    print("🧪 Testing Improved Validation Metrics")
    
    try:
        from src.causal_bayes_opt.training.acquisition_validation_metrics import (
            compute_comprehensive_validation_metrics,
            top_k_accuracy,
            mean_reciprocal_rank,
            compute_diversity_bonus
        )
        print("✅ Successfully imported validation metrics")
        
        # Create test data
        batch_size, n_variables = 10, 12
        
        # Policy favors first 5 variables (like our BC should)
        policy_logits = onp.random.randn(batch_size, n_variables).astype(onp.float32)
        policy_logits[:, :5] += 2.0  # Strong preference for first 5 variables
        policy_logits = jnp.array(policy_logits)
        
        # Expert choices are from first 5 variables 
        expert_choices = jnp.array([onp.random.randint(0, 5) for _ in range(batch_size)])
        
        # Test metrics
        top_1 = top_k_accuracy(policy_logits, expert_choices, k=1)
        top_3 = top_k_accuracy(policy_logits, expert_choices, k=3)
        mrr = mean_reciprocal_rank(policy_logits, expert_choices)
        
        print(f"\n📊 Test Results:")
        print(f"  Top-1 accuracy: {top_1:.3f}")
        print(f"  Top-3 accuracy: {top_3:.3f}")  
        print(f"  Mean reciprocal rank: {mrr:.3f}")
        
        # Test comprehensive metrics
        comprehensive = compute_comprehensive_validation_metrics(
            policy_logits=policy_logits,
            expert_choices=expert_choices,
            intervention_history=[0, 1, 2, 0, 3],
            total_variables=n_variables
        )
        
        print(f"\n🔍 Comprehensive Metrics:")
        for name, value in comprehensive.items():
            print(f"  {name}: {value:.4f}")
        
        # Test diversity bonus
        frequent_bonus = compute_diversity_bonus(0, [0, 1, 0, 2, 0])  # 0 appears often
        rare_bonus = compute_diversity_bonus(4, [0, 1, 0, 2, 0])     # 4 is rare
        
        print(f"\n🔄 Diversity Bonus:")
        print(f"  Frequent choice bonus: {frequent_bonus:.3f}")
        print(f"  Rare choice bonus: {rare_bonus:.3f}")
        print(f"  Ratio: {rare_bonus/frequent_bonus:.2f}")
        
        # Verify results are reasonable
        assert top_1 > 0.2, f"Top-1 too low: {top_1}"
        assert top_3 > 0.5, f"Top-3 too low: {top_3}"
        assert mrr > 0.4, f"MRR too low: {mrr}"
        assert rare_bonus > frequent_bonus, "Diversity bonus not working"
        
        print("\n✅ All metrics working correctly!")
        
        # Show the improvement
        print("\n🎯 Improvement Summary:")
        print(f"  OLD: 0% exact match accuracy")
        print(f"  NEW: {top_3:.1%} top-3 accuracy")
        print(f"  OLD: 414M+ cross-entropy loss")  
        print(f"  NEW: Diversity-weighted loss with {rare_bonus:.2f}x bonus for exploration")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_metrics_only()
    if success:
        print("\n🎉 SUCCESS! Improved metrics are working.")
        print("Ready to test in notebook with real training.")
    else:
        print("\n❌ FAILED! Need to debug metric implementation.")
    sys.exit(0 if success else 1)