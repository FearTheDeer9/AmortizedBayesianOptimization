#!/usr/bin/env poetry run python
"""
Example comparing standard BC training (3-channel) vs enhanced BC training (5-channel).

This demonstrates how the enhanced BC trainer uses structural knowledge from
expert demonstrations, while the standard trainer ignores it.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_training_difference():
    """
    Visualize the difference between 3-channel and 5-channel BC training.
    """
    logger.info("=== BC Training Comparison ===\n")
    
    # Standard BC Training (3-channel)
    logger.info("1. Standard BC Training (3-channel):")
    logger.info("   Input channels:")
    logger.info("   - Channel 0: Variable values")
    logger.info("   - Channel 1: Target indicator") 
    logger.info("   - Channel 2: Intervention history")
    logger.info("   Missing: Structural knowledge!\n")
    
    logger.info("   What BC learns:")
    logger.info("   - 'When values look like X, intervene on Y'")
    logger.info("   - No understanding of WHY Y was chosen")
    logger.info("   - Has to infer patterns from outcomes alone\n")
    
    # Enhanced BC Training (5-channel)
    logger.info("2. Enhanced BC Training (5-channel):")
    logger.info("   Input channels:")
    logger.info("   - Channel 0: Variable values")
    logger.info("   - Channel 1: Target indicator")
    logger.info("   - Channel 2: Intervention history") 
    logger.info("   - Channel 3: Parent probabilities ← NEW!")
    logger.info("   - Channel 4: Intervention recency ← NEW!\n")
    
    logger.info("   What BC learns:")
    logger.info("   - 'When values look like X AND Z is likely parent, intervene on Z'")
    logger.info("   - Understands structural reasoning behind expert choices")
    logger.info("   - Can generalize better to new SCMs\n")
    
    # Example scenario
    logger.info("=== Example Scenario ===")
    logger.info("SCM: X → Y ← Z (fork structure)")
    logger.info("Target: Y")
    logger.info("Expert knows: P(X is parent) = 1.0, P(Z is parent) = 1.0\n")
    
    logger.info("Standard BC sees:")
    logger.info("  Time  |  X    Y    Z  | Target | Intervened | Expert Action")
    logger.info("  t=0   | 0.5  -1.2  0.8 |   Y    |    none    | intervene(X)")
    logger.info("  t=1   | 1.0  -0.5  0.8 |   Y    |     X      | intervene(Z)")
    logger.info("  → Must guess why expert chose X then Z\n")
    
    logger.info("Enhanced BC sees:")
    logger.info("  Time  |  X    Y    Z  | Target | Intervened | Parent Probs | Expert Action")
    logger.info("  t=0   | 0.5  -1.2  0.8 |   Y    |    none    | X:1.0 Z:1.0  | intervene(X)")
    logger.info("  t=1   | 1.0  -0.5  0.8 |   Y    |     X      | X:1.0 Z:1.0  | intervene(Z)")
    logger.info("  → Sees that both X and Z are parents, understands alternating strategy\n")
    
    # Key benefits
    logger.info("=== Key Benefits of 5-Channel Training ===")
    logger.info("1. Train-test consistency: Same format during training and evaluation")
    logger.info("2. Structure awareness: Policy learns to use causal knowledge")
    logger.info("3. Better generalization: Can adapt to new SCMs with different structures")
    logger.info("4. Matches GRPO: Both methods now use structural information\n")
    
    # Implementation notes
    logger.info("=== Implementation Notes ===")
    logger.info("• Expert demonstrations already contain posterior distributions")
    logger.info("• No need for mock surrogates - use expert's actual beliefs")
    logger.info("• Direct transformation: Demo → 5-channel tensor → Training")
    logger.info("• Simpler than buffer abstractions, more principled design")


def show_code_example():
    """Show code example of using enhanced BC trainer."""
    logger.info("\n=== Code Example ===")
    
    code = '''
# Old approach (3-channel)
from src.causal_bayes_opt.training.policy_bc_trainer import PolicyBCTrainer

trainer = PolicyBCTrainer(hidden_dim=256)
results = trainer.train("expert_demonstrations/")
# → Trains on 3-channel data, ignoring structure

# New approach (5-channel)
from src.causal_bayes_opt.training.enhanced_bc_trainer import EnhancedBCTrainer

trainer = EnhancedBCTrainer(hidden_dim=256)
results = trainer.train("expert_demonstrations/")
# → Trains on 5-channel data, including structure

# The policy learns to make decisions like:
# "If P(X is parent) > 0.8 and X hasn't been intervened recently, choose X"
# Instead of just:
# "If we're at step 3, choose X (because that's what expert did)"
'''
    
    logger.info(code)


def main():
    """Run the comparison demo."""
    visualize_training_difference()
    show_code_example()
    
    logger.info("\n" + "="*60)
    logger.info("This enhanced approach ensures BC policies can leverage the same")
    logger.info("structural information available during evaluation, eliminating")
    logger.info("the train-test mismatch and enabling more intelligent decisions.")
    logger.info("="*60)


if __name__ == "__main__":
    main()