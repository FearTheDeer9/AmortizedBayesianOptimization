#!/usr/bin/env python3
"""
Debug script to understand why X4 is never predicted correctly.
This runs a minimal training session focusing on X4 cases.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from enhanced_bc_trainer_fixed import FixedEnhancedBCTrainer
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

print("="*60)
print("X4 DEBUGGING SESSION")
print("="*60)
print("\nThis will train for just 1 epoch with debugging enabled.")
print("When it hits an X4 target, it will break into pdb.")
print("\nIn pdb, you can inspect:")
print("  - var_logits: The raw model outputs")
print("  - probs: The softmax probabilities")
print("  - example_mapper.variables: The variable ordering")
print("  - input_tensor: The actual input data")
print("\nType 'c' to continue to the next X4 case, or 'q' to quit.")
print("="*60)

# Create trainer with minimal settings
trainer = FixedEnhancedBCTrainer(
    hidden_dim=256,  # Use same as trained model
    learning_rate=3e-4,
    batch_size=4,  # Small batch to find X4 quickly
    max_epochs=1,  # Just 1 epoch
    save_embeddings_every=1,
    seed=42
)

# Train with limited data to quickly find X4 cases
print("\nStarting training... Will break when X4 is encountered.")
results = trainer.train(
    demonstrations_path='expert_demonstrations/raw/raw_demonstrations',
    max_demos=20,  # Use just 20 demos
    output_dir='debugging-bc-training/debug_x4_output/'
)

print("\nDebug session complete!")
print("Check the output above to understand why X4 fails.")