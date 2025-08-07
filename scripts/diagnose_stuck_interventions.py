#!/usr/bin/env python3
"""
Diagnose why GRPO policy is stuck intervening on the same variable.
"""

import sys
from pathlib import Path


def diagnose_stuck_pattern():
    """Analyze the log snippet to understand why interventions are stuck."""
    
    print("="*80)
    print("DIAGNOSING STUCK INTERVENTION PATTERN")
    print("="*80)
    
    # The log shows:
    # Episode 10, Step 0: use_surrogate=False
    # Intervening on X with value 0.5
    # Target reward: 0.596918523311615, info_gain_reward=0.0
    # Total reward: 0.596918523311615
    
    print("\n🔍 OBSERVATIONS FROM LOG:")
    print("1. use_surrogate=False")
    print("   - No surrogate model is being used")
    print("   - This means NO info gain rewards (all 0.0)")
    print("   - Policy only receives target rewards")
    
    print("\n2. All interventions on variable 'X'")
    print("   - Policy is stuck on one variable")
    print("   - No exploration of other variables")
    
    print("\n3. Reward signal analysis:")
    print("   - Target reward: 0.5969")
    print("   - Info gain reward: 0.0")
    print("   - Total reward = Target reward only")
    
    print("\n⚠️  ROOT CAUSES:")
    print("\n1. MISSING SURROGATE:")
    print("   - Without surrogate, there's no info gain reward")
    print("   - Info gain rewards incentivize exploring different variables")
    print("   - Target rewards alone may not provide enough signal")
    
    print("\n2. INSUFFICIENT EXPLORATION:")
    print("   - Policy may have low entropy coefficient")
    print("   - Initial random exploration may be too limited")
    print("   - Policy converged to local optimum (always pick X)")
    
    print("\n3. REWARD STRUCTURE:")
    print("   - Target rewards might be similar across variables")
    print("   - Without info gain, no incentive to explore graph structure")
    print("   - Policy can't learn which interventions are most informative")
    
    print("\n✅ SOLUTIONS:")
    print("\n1. Enable surrogate integration:")
    print("   - Train with --method grpo_with_surrogate")
    print("   - Ensure surrogate checkpoint exists")
    print("   - Verify use_surrogate=True in logs")
    
    print("\n2. Increase exploration:")
    print("   - Increase entropy coefficient in GRPO config")
    print("   - Add initial random exploration phase")
    print("   - Use epsilon-greedy exploration")
    
    print("\n3. Improve reward signal:")
    print("   - Add diversity bonus for trying different variables")
    print("   - Penalize repetitive interventions")
    print("   - Consider curriculum learning (start simple, increase complexity)")
    
    print("\n4. Check GRPO configuration:")
    print("   - Verify learning rate is appropriate")
    print("   - Check if gradients are flowing properly")
    print("   - Ensure policy network has sufficient capacity")
    
    print("\n📊 EXPECTED BEHAVIOR WITH SURROGATE:")
    print("- Info gain rewards > 0 when exploring informative variables")
    print("- Policy learns to intervene on parents of target")
    print("- Diverse intervention strategy based on graph beliefs")
    print("- Adaptive behavior as surrogate updates (if active learning)")
    
    print("\n🔧 DEBUGGING STEPS:")
    print("1. Check if surrogate was trained properly:")
    print("   python scripts/test_bc_checkpoints.py")
    print("\n2. Test GRPO WITH surrogate:")
    print("   python scripts/debug_grpo_surrogate_training.py")
    print("\n3. Analyze full training log:")
    print("   python scripts/analyze_log_file.py <full_log_path>")
    print("\n4. Visualize policy behavior:")
    print("   python scripts/analyze_intervention_patterns.py <full_log_path>")
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    diagnose_stuck_pattern()