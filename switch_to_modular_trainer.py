#!/usr/bin/env python3
"""
Script to switch experiments to use the new modular UnifiedGRPOTrainer.

This creates a backup of the original trainer and switches imports to use
the new modular version.
"""

import os
import shutil
from pathlib import Path

def main():
    # Backup original trainer
    original_path = Path("src/causal_bayes_opt/training/unified_grpo_trainer.py")
    backup_path = Path("src/causal_bayes_opt/training/unified_grpo_trainer_backup.py")
    
    print("Creating backup of original trainer...")
    shutil.copy2(original_path, backup_path)
    print(f"✓ Backed up to {backup_path}")
    
    # Replace original with modular version
    modular_path = Path("src/causal_bayes_opt/training/unified_grpo_trainer_v2.py")
    
    # Read modular version
    with open(modular_path, 'r') as f:
        modular_content = f.read()
    
    # Create new content that imports from v2 but maintains same interface
    new_content = '''"""
Unified GRPO trainer - now uses modular components.

This maintains backward compatibility while using the new modular architecture.
"""

# Import the modular version but expose same interface
from .unified_grpo_trainer_v2 import (
    UnifiedGRPOTrainerV2 as UnifiedGRPOTrainer,
    create_unified_grpo_trainer
)

# Expose all the same components for imports
from .grpo_trainer_core import GRPOTrainerCore, GRPOTrainerConfig
from .grpo_reward_computer import GRPORewardComputer, GRPORewardConfig  
from .grpo_logger import GRPOLogger

__all__ = [
    'UnifiedGRPOTrainer',
    'create_unified_grpo_trainer',
    'GRPOTrainerCore',
    'GRPOTrainerConfig', 
    'GRPORewardComputer',
    'GRPORewardConfig',
    'GRPOLogger'
]
'''
    
    # Write new simplified trainer
    with open(original_path, 'w') as f:
        f.write(new_content)
    
    print(f"✓ Replaced {original_path} with modular version")
    print("\nModular trainer is now active!")
    print(f"Original backed up to: {backup_path}")
    
    # List the new modular files
    print("\nNew modular components:")
    print("  - src/causal_bayes_opt/training/grpo_trainer_core.py")
    print("  - src/causal_bayes_opt/training/grpo_reward_computer.py") 
    print("  - src/causal_bayes_opt/training/grpo_logger.py")
    print("  - src/causal_bayes_opt/training/unified_grpo_trainer_v2.py")

if __name__ == "__main__":
    main()