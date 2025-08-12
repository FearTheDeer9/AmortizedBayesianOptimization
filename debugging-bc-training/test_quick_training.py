#!/usr/bin/env python3
"""
Quick test of enhanced BC training with small dataset.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from enhanced_bc_trainer_fixed import FixedEnhancedBCTrainer

# Create trainer with small settings for quick test
trainer = FixedEnhancedBCTrainer(
    hidden_dim=64,  # Smaller model
    learning_rate=1e-3,  # Higher LR for faster convergence
    batch_size=16,  # Smaller batch
    max_epochs=10,  # Just 10 epochs
    save_embeddings_every=2,
    early_stopping_patience=5,  # Shorter patience
    seed=42
)

print("Starting quick training test...")

# Train with very limited data
results = trainer.train(
    demonstrations_path='expert_demonstrations/raw/raw_demonstrations',
    max_demos=20,  # Only 20 demos for speed
    output_dir='debugging-bc-training/results_quick_test/'
)

print("\n" + "="*60)
print("QUICK TEST RESULTS")
print("="*60)

# Check results
metrics = results['metrics']
print(f"Training time: {metrics['training_time']:.1f}s")
print(f"Epochs trained: {metrics['epochs_trained']}")
print(f"Final train loss: {metrics.get('final_train_loss', 'N/A')}")
print(f"Best val loss: {metrics.get('best_val_loss', 'N/A')}")

if 'best_val_accuracy' in metrics:
    print(f"Best val accuracy: {metrics['best_val_accuracy']:.3f}")

if 'per_variable_accuracy' in metrics:
    print("\nPer-variable accuracy:")
    for var, acc in metrics['per_variable_accuracy'].items():
        print(f"  {var}: {acc:.3f}")

print("\nTest complete! Check debugging-bc-training/results_quick_test/ for full metrics.")