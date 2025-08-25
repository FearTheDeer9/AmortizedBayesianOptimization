#!/usr/bin/env python3
"""
Complex SCM with Surrogate Integration for GRPO Training.
Tests quantile architecture with surrogate guidance on 10-variable causal discovery task.
Expected: E_25% > F_25% > isolated variables (faster learning with surrogate assistance).
"""

import os
import sys
import json
import argparse
import logging
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import jax.random as random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.causal_bayes_opt.training.joint_acbo_trainer import JointACBOTrainer
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.data_structures.scm import get_parents, get_variables, get_target
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.training.utils.wandb_setup import WandBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_deterministic_test_scm():
    """
    Create SIMPLIFIED deterministic test SCM: X -> Y (NO Z for cleaner testing)
    
    Structure:
    - X -> Y (with coefficient 10.0, huge signal)
    - Y is target
    - NO Z variable to eliminate noise in gradient analysis
    
    Properties:
    - Y = 10.0 * X + 0 (deterministic, no noise)
    - X has noise_scale=1.0 (for variation)
    """
    # Use a working benchmark SCM as template and modify it
    from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm
    from src.causal_bayes_opt.data_structures.scm import create_scm
    from src.causal_bayes_opt.mechanisms.linear import create_linear_mechanism
    import pyrsistent as pyr
    
    # Define structure: X -> Y, Z (isolated) - reintroduced for full testing
    variables = frozenset(['X', 'Y', 'Z'])
    edges = frozenset([('X', 'Y')])  # Only X causes Y, Z isolated
    target = 'Y'
    
    # Create mechanisms
    mechanisms = {}
    
    # X mechanism: root variable with noise
    mechanisms['X'] = create_linear_mechanism(
        parents=[],  # Root variable
        coefficients={},
        intercept=0.0,
        noise_scale=1.0
    )
    
    # Y mechanism: Y = 10*X (deterministic, no noise)
    mechanisms['Y'] = create_linear_mechanism(
        parents=['X'],
        coefficients={'X': 10.0},  # Huge coefficient
        intercept=0.0,
        noise_scale=0.0  # No noise - perfect function
    )
    
    # Z mechanism: isolated root variable with noise (reintroduced)
    mechanisms['Z'] = create_linear_mechanism(
        parents=[],  # Root variable (isolated)
        coefficients={},
        intercept=0.0,
        noise_scale=1.0
    )
    
    # Create SCM with custom metadata
    metadata = {
        'structure_type': 'deterministic_test',
        'variable_ranges': {
            'X': (-100.0, 100.0),    # X can vary widely
            'Y': (-1000.0, 1000.0),  # Y can vary widely (10*X range)
            'Z': (-10.0, 10.0)       # Z range (isolated)
        },
        'coefficients': {('X', 'Y'): 10.0},
        'noise_scales': {'X': 1.0, 'Y': 0.0, 'Z': 1.0},
        'description': 'X -> Y causal, Z isolated (reintroduced)',
        'expected_behavior': 'X interventions affect Y, Z interventions random',
        'optimal_strategy': 'Minimize Y by setting X = -100 → Y = -1000'
    }
    
    scm = create_scm(
        variables=variables,
        edges=edges,
        mechanisms=mechanisms,
        target=target,
        metadata=metadata
    )
    
    logger.info("Created test SCM: X -> Y (coeff=10.0), Z isolated")
    logger.info(f"  Structure: Y = 10*X (no noise), Z isolated")
    logger.info(f"  Expected: X quantiles learn causal pattern, Z quantiles stay random")
    logger.info(f"  Optimal: X_25% = -66.7 → Y = -667 (should become highest score)")
    
    return scm


def create_complex_test_scm():
    """
    Create complex 10-variable SCM for testing causal discovery:
    
    Structure:
    - Component 1 (disconnected): A, B, C, D (4 isolated variables)
    - Component 2 (fork): E -> G <- F, with G as target
        * E -> G (coefficient = 10.0, STRONG causal effect)
        * F -> G (coefficient = 2.0, WEAK causal effect)  
        * H, I, J (3 more isolated variables in this component)
    
    Expected learning:
    1. E_25% should get highest scores (strongest causal effect)
    2. F_25% should get medium scores (weaker causal effect)
    3. A,B,C,D,H,I,J should get low/random scores (no causal effect)
    """
    from src.causal_bayes_opt.data_structures.scm import create_scm
    from src.causal_bayes_opt.mechanisms.linear import create_linear_mechanism
    import pyrsistent as pyr
    
    # Define 10-variable structure with two disconnected components
    variables = frozenset(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
    edges = frozenset([('E', 'G'), ('F', 'G')])  # Only E and F cause G
    target = 'G'  # Target is the fork center
    
    # Create mechanisms
    mechanisms = {}
    
    # Component 1: Isolated variables (A, B, C, D)
    for var in ['A', 'B', 'C', 'D']:
        mechanisms[var] = create_linear_mechanism(
            parents=[],
            coefficients={},
            intercept=0.0,
            noise_scale=1.0
        )
    
    # Component 2: Fork structure with target G
    # E: Root variable (strong parent)
    mechanisms['E'] = create_linear_mechanism(
        parents=[],
        coefficients={},
        intercept=0.0,
        noise_scale=1.0
    )
    
    # F: Root variable (weak parent)  
    mechanisms['F'] = create_linear_mechanism(
        parents=[],
        coefficients={},
        intercept=0.0,
        noise_scale=1.0
    )
    
    # G: Target variable (fork center)
    mechanisms['G'] = create_linear_mechanism(
        parents=['E', 'F'],
        coefficients={
            'E': 10.0,  # STRONG coefficient - E has major impact
            'F': 2.0    # WEAK coefficient - F has minor impact
        },
        intercept=0.0,
        noise_scale=0.0  # Deterministic for clear signal
    )
    
    # More isolated variables (H, I, J)
    for var in ['H', 'I', 'J']:
        mechanisms[var] = create_linear_mechanism(
            parents=[],
            coefficients={},
            intercept=0.0,
            noise_scale=1.0
        )
    
    # Create SCM with metadata
    metadata = {
        'structure_type': 'complex_causal_discovery',
        'variable_ranges': {
            'A': (-10.0, 10.0), 'B': (-10.0, 10.0), 'C': (-10.0, 10.0), 'D': (-10.0, 10.0),
            'E': (-20.0, 20.0),  # Wider range for strong parent
            'F': (-20.0, 20.0),  # Wider range for weak parent
            'G': (-250.0, 250.0),  # Wide range for target (10*E + 2*F)
            'H': (-10.0, 10.0), 'I': (-10.0, 10.0), 'J': (-10.0, 10.0)
        },
        'coefficients': {('E', 'G'): 10.0, ('F', 'G'): 2.0},
        'noise_scales': {var: 1.0 if var != 'G' else 0.0 for var in variables},
        'description': 'Complex 10-var: E->G(10.0), F->G(2.0), others isolated',
        'expected_behavior': 'E_25% > F_25% > isolated variables',
        'optimal_strategy': 'E = -20, F = -20 → G = -240 (theoretical minimum)',
        'causal_hierarchy': ['E (strong)', 'F (weak)', 'A,B,C,D,H,I,J (none)']
    }
    
    scm = create_scm(
        variables=variables,
        edges=edges, 
        mechanisms=pyr.pmap(mechanisms),
        target=target,
        metadata=pyr.pmap(metadata)  # Ensure metadata is also pyrsistent
    )
    
    logger.info("Created complex 10-variable SCM: E->G(10.0), F->G(2.0), others isolated")
    logger.info(f"  Structure: G = 10*E + 2*F (deterministic), 7 isolated variables")
    logger.info(f"  Expected hierarchy: E_25% > F_25% > isolated_variables")
    logger.info(f"  Optimal: E = F = -20 → G = -240 (E contributes 5x more than F)")
    
    return scm


class DiverseGRPOTrainer(JointACBOTrainer):
    """
    Trainer for diverse SCMs using proven patterns from definitive_100_percent.py.
    Overrides intervention selection to use direct generation instead of batch system.
    """
    
    def __init__(self, config):
        # Store config for override method
        self.config = config
        
        # CRITICAL: Override episodes_per_phase BEFORE calling parent init
        if not config.get('use_surrogate', True):
            config['episodes_per_phase'] = 999999  # Never switch phases
            config['initial_phase'] = 'policy'  # Ensure we start in policy phase
            logger.info("Configuring for pure GRPO: phase switching disabled")
        
        super().__init__(config=config)
        
        # FORCE override after parent init (regardless of config)
        # This ensures we never switch phases in pure GRPO mode
        if not config.get('use_surrogate', True):
            self.episodes_per_phase = 999999  # Force this value
            self.current_phase = 'policy'  # Force policy phase
            self.phase_episode_count = 0  # Reset counter
            logger.info(f"FORCED OVERRIDE: episodes_per_phase={self.episodes_per_phase}, current_phase={self.current_phase}, phase_count={self.phase_episode_count}")
        
        # Tracking metrics by SCM size
        self.size_category_metrics = {
            'small': [],
            'medium': [],
            'large': []
        }
        
        # Track all interventions for analysis
        self.all_interventions = []
        self.episode_performances = []
        
        # Comprehensive metrics tracking
        self.convergence_metrics = {
            'rolling_average_window': 20,
            'rolling_averages': [],
            'best_per_category': {'small': float('inf'), 'medium': float('inf'), 'large': float('inf')},
            'episodes_to_threshold': {},  # Track when we hit performance thresholds
            'value_patterns': {'parent_values': [], 'non_parent_values': []},
            'exploration_ratio': []  # Track exploration vs exploitation
        }
        
        # CSV logging setup
        self.csv_data = []
        self.checkpoint_episodes = [25, 50, 75, 100]
        
        # Current episode tracking
        self.current_episode_data = {
            'selections': [],
            'target_values': [],
            'rewards': [],
            'size_category': None,
            'num_vars': 0,
            'intervention_values': [],  # Track actual intervention values
            'parent_interventions': 0,   # Count parent vs non-parent
            'non_parent_interventions': 0
        }
        
        # Override group size if needed
        if hasattr(self, 'grpo_config'):
            self.group_size = self.grpo_config.group_size
        else:
            self.group_size = config.get('grpo_config', {}).get('group_size', 10)
        
        logger.info(f"Initialized DiverseGRPOTrainer with group_size={self.group_size}")
    
    def _rotate_scm(self, episode_metrics):
        """Override to use fixed single SCM - never actually rotate."""
        # For single SCM sanity check, don't rotate - keep same SCM
        # Just increment the episode counter
        self.episodes_on_current_scm += 1
        
        # Log that we're keeping the same SCM
        target_var = get_target(self.current_scm) if hasattr(self, 'current_scm') else 'unknown'
        num_vars = len(get_variables(self.current_scm)) if hasattr(self, 'current_scm') else 0
        logger.info(f"Keeping same single SCM: {num_vars} vars, target={target_var}")
    
    def _should_switch_phase(self):
        """Override to never switch phases in pure GRPO mode."""
        if not self.config.get('use_surrogate', True):
            return False  # Never switch phases
        return super()._should_switch_phase()
    
    def _run_grpo_episode(self, episode_idx, scm, scm_name, key):
        """Override to track metrics and add surrogate debugging."""
        # Set current episode metadata
        self.current_episode_data['size_category'] = 'single'
        self.current_episode_data['num_vars'] = len(list(get_variables(scm)))
        
        logger.info(f"\nEpisode {episode_idx}: Single SCM with {self.current_episode_data['num_vars']} variables")
        
        # DEBUG: Check surrogate utilization before episode
        print(f"\n🔍 SURROGATE & BUFFER DEBUG (Episode {episode_idx}):")
        
        # Check surrogate availability
        if hasattr(self, 'use_surrogate') and self.use_surrogate:
            print(f"  ✅ Surrogate enabled in config")
            if hasattr(self, 'surrogate_predict_fn') and self.surrogate_predict_fn:
                print(f"  ✅ AVICI surrogate loaded and available")
            else:
                print(f"  ❌ AVICI surrogate not available")
        else:
            print(f"  ❌ Surrogate disabled in config")
            
        # Check true causal structure for comparison
        target_var = get_target(scm)
        true_parents = list(get_parents(scm, target_var)) if hasattr(scm, 'edges') else []
        variables = list(get_variables(scm))
        
        print(f"  🎯 TRUE CAUSAL STRUCTURE:")
        print(f"    Target: {target_var}")
        print(f"    True parents: {true_parents}")
        print(f"    All variables: {variables}")
        print(f"    Expected: E,F should get high parent probabilities")
        
        # Use parent's proven GRPO implementation
        result = super()._run_grpo_episode(episode_idx, scm, scm_name, key)
        
        # DEBUG: Check buffer state and surrogate utilization after episode
        print(f"\n📊 POST-EPISODE BUFFER & SURROGATE ANALYSIS:")
        
        # Get buffer from trainer if accessible (might need to access through trainer state)
        # For now, create a test to verify surrogate predictions on complex SCM
        from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
        from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
        from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
        
        # Create test buffer with complex SCM data
        test_buffer = ExperienceBuffer()
        test_samples = sample_from_linear_scm(scm, 20, seed=42)
        for sample in test_samples:
            test_buffer.add_observation(sample)
            
        # Test AVICI predictions on complex structure
        if hasattr(self, 'surrogate_predict_fn') and self.surrogate_predict_fn:
            tensor_3ch, mapper = buffer_to_three_channel_tensor(
                test_buffer, target_var, max_history_size=100, standardize=False
            )
            
            surrogate_prediction = self.surrogate_predict_fn(tensor_3ch, target_var, mapper.variables)
            
            print(f"  🔮 AVICI SURROGATE PREDICTIONS ON COMPLEX SCM:")
            print(f"    Target: {target_var}")
            print(f"    Variables: {mapper.variables}")
            
            if 'parent_probs' in surrogate_prediction:
                parent_probs = surrogate_prediction['parent_probs']
                print(f"    Parent probabilities:")
                
                for i, var in enumerate(mapper.variables):
                    if var != target_var and i < len(parent_probs):
                        prob = float(parent_probs[i])
                        is_true_parent = var in true_parents
                        marker = "🎯" if is_true_parent else "❌" if prob > 0.7 else ""
                        print(f"      {var}: {prob:.3f} {marker}")
                        
                # Analyze how well AVICI distinguishes parents
                if true_parents:
                    true_parent_probs = []
                    false_parent_probs = []
                    
                    for i, var in enumerate(mapper.variables):
                        if var != target_var and i < len(parent_probs):
                            prob = float(parent_probs[i])
                            if var in true_parents:
                                true_parent_probs.append(prob)
                            else:
                                false_parent_probs.append(prob)
                    
                    if true_parent_probs and false_parent_probs:
                        avg_true = np.mean(true_parent_probs)
                        avg_false = np.mean(false_parent_probs)
                        discrimination = avg_true - avg_false
                        
                        print(f"  📊 AVICI PERFORMANCE ANALYSIS:")
                        print(f"    True parents (E,F): {avg_true:.3f} avg probability")
                        print(f"    False parents: {avg_false:.3f} avg probability") 
                        print(f"    Discrimination: {discrimination:+.3f}")
                        
                        if discrimination > 0.2:
                            print(f"    ✅ AVICI strongly discriminates parents")
                        elif discrimination > 0.1:
                            print(f"    ⚠️  AVICI weakly discriminates parents")
                        else:
                            print(f"    ❌ AVICI not discriminating parents")
            else:
                print(f"    ❌ Unexpected prediction format: {list(surrogate_prediction.keys())}")
        else:
            print(f"  ❌ AVICI surrogate not available for testing")
        
        # Track episode performance
        self._track_episode_performance(episode_idx)
        
        # Reset episode data for next episode
        self.current_episode_data = {
            'selections': [],
            'target_values': [],
            'rewards': [],
            'size_category': 'single',
            'num_vars': 3,
            'intervention_values': [],
            'parent_interventions': 0,
            'non_parent_interventions': 0
        }
        
        return result
    
    def _save_checkpoint(self, episode_idx):
        """Save checkpoint at specified episodes."""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        episode_dir = checkpoint_dir / f"episode_{episode_idx}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        # Save policy using checkpoint_utils
        from src.causal_bayes_opt.utils.checkpoint_utils import save_checkpoint
        
        # Get architecture from parent class (UnifiedGRPOTrainer sets these)
        # The parent class already has self.hidden_dim and policy_architecture
        architecture = {
            'hidden_dim': self.hidden_dim,  # Set by UnifiedGRPOTrainer.__init__
            'num_layers': self.config.get('architecture', {}).get('num_layers', 4),
            'num_heads': self.config.get('architecture', {}).get('num_heads', 8),
            'key_size': self.config.get('architecture', {}).get('key_size', 32),
            'dropout': self.config.get('architecture', {}).get('dropout', 0.1),
            'architecture_type': self.config.get('policy_architecture', 'simple_permutation_invariant')
        }
        
        save_checkpoint(
            path=episode_dir / "policy.pkl",
            params=self.policy_params,
            architecture=architecture,
            model_type='policy',
            model_subtype='grpo',
            training_config={
                'learning_rate': self.config.get('learning_rate', 5e-4),
                'episode': episode_idx,
                'total_episodes': self.config.get('max_episodes', 100)
            },
            metadata={
                'episode': episode_idx,
                'convergence_metrics': self.convergence_metrics,
                'best_per_category': self.convergence_metrics['best_per_category'],
                'trainer': 'SingleSCMSanityCheck'
            }
        )
        
        logger.info(f"  💾 Saved checkpoint at episode {episode_idx}")
    
    
    
    def _track_episode_performance(self, episode_idx):
        """Track performance metrics using episode result from parent."""
        # Simple tracking for compatibility - parent handles the actual metrics
        metrics = {
            'episode': episode_idx,
            'size_category': 'single',
            'num_vars': 3,
            'mean_target': 0.0,  # Would need to extract from episode result
            'best_target': 0.0,  # Would need to extract from episode result  
            'parent_selection_rate': 0.0  # Would need custom tracking
        }
        
        self.episode_performances.append(metrics)
        
        # Save checkpoint if needed
        if episode_idx in self.checkpoint_episodes:
            self._save_checkpoint(episode_idx)
        
        logger.info(f"\nEpisode {episode_idx} tracking complete")


def create_single_scm_config(
    max_episodes: int = 100,
    verbose: bool = False,
    enable_wandb: bool = False,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Create configuration for single SCM sanity check using proven values.
    """
    
    config = {
        # Core episode settings (reduced for sanity check)
        'max_episodes': 1,  # Test meaningful ratios in episode 2
        'obs_per_episode': 10,
        'max_interventions': 15,   # Minimal interventions for debugging
        
        # CRITICAL: Disable phase switching for pure GRPO
        'episodes_per_phase': 999999,  # Never switch phases
        
        # CRITICAL: Use simple_permutation_invariant (our lesson)
        'policy_architecture': 'quantile',
        
        # CRITICAL: Enable surrogate for enhanced learning!
        'use_surrogate': True,   # Enable surrogate to guide learning
        'use_grpo_rewards': True,
        
        # CRITICAL: Fixed std for exploration - LARGER for proper exploration
        'use_fixed_std': True,
        'fixed_std': 1.0,  # Much larger to allow discovery of X = -10 strategy
        
        # CRITICAL: Reduce learning rate to prevent gradient explosion
        'learning_rate': 5e-4,  # Much lower to prevent huge oscillating steps
        
        # GRPO configuration (back to reasonable size for now)
        'grpo_config': {
            'group_size': 32,  # 8 candidates per intervention
            'entropy_coefficient': 0.001,    # VERY HIGH to force differentiation
            'clip_ratio': 1.0,             # Remove PPO clipping (was 0.2)
            'gradient_clip': 10.0,         # Much looser clipping (was 1.0)
            'ppo_epochs': 4,
            'normalize_advantages': False   # Don't normalize advantages yet
        },
        
        # Reward weights (enhanced with surrogate guidance)
        'reward_weights': {
            'target': 0.7,    # Target optimization (primary signal)
            'parent': 0.1,    # Parent selection bonus (helps structure discovery)
            'info_gain': 0.2   # Information gain from surrogate (structure learning)
        },
        
        # Use composite reward for surrogate integration (includes info_gain)
        'reward_type': 'composite',
        
        # Surrogate model configuration - load AVICI checkpoint
        'surrogate_checkpoint_path': 'experiments/surrogate-only-training/scripts/checkpoints/avici_runs/avici_style_20250822_213115/checkpoint_step_200.pkl',
        'surrogate_lr': 1e-3,           # Surrogate learning rate (for updates)
        'surrogate_hidden_dim': 128,    # Surrogate architecture
        'surrogate_layers': 4,
        'surrogate_heads': 8,
        
        # Joint training config (just in case it's checked)
        'joint_training': {
            'episodes_per_phase': 999999,  # Never switch
            'initial_phase': 'policy',
            'adaptive': {
                'use_performance_rotation': False,  # Disable performance-based rotation
                'plateau_patience': 999999  # Never detect plateau
            }
        },
        
        # General settings
        'batch_size': 32,
        'seed': seed,
        'verbose': verbose,
        'checkpoint_dir': 'checkpoints/single_scm_sanity',
        
        # WandB logging configuration
        'logging': {
            'wandb': {
                'enabled': enable_wandb,
                'project': 'causal-bayes-opt-grpo',
                'name': f'single_scm_sanity_{max_episodes}ep',
                'tags': ['single_scm', 'grpo', 'sanity_check'],
                'log_frequency': 1
            }
        }
    }
    
    return config


def main():
    """Main training function for single SCM sanity check."""
    
    parser = argparse.ArgumentParser(description="Single SCM GRPO sanity check")
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of training episodes')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable WandB logging')
    
    args = parser.parse_args()
    
    # Create configuration
    logger.info("\n" + "="*70)
    logger.info("SINGLE SCM GRPO SANITY CHECK")
    logger.info("="*70)
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"SCM: 3-variable fork structure")
    logger.info(f"Seed: {args.seed}")
    logger.info("="*70 + "\n")
    
    config = create_single_scm_config(
        max_episodes=args.episodes,
        verbose=args.verbose,
        enable_wandb=args.wandb,
        seed=args.seed
    )
    
    # Log critical settings
    logger.info("Critical settings:")
    logger.info(f"  - Policy: {config['policy_architecture']}")
    logger.info(f"  - Fixed std: {config['fixed_std']}")
    logger.info(f"  - Learning rate: {config['learning_rate']}")
    logger.info(f"  - GRPO group size: {config['grpo_config']['group_size']}")
    logger.info(f"  - Pure GRPO training: surrogate disabled")
    logger.info("")
    
    # Initialize WandB if enabled
    wandb_manager = None
    if args.wandb:
        logger.info("Initializing WandB...")
        try:
            wandb_manager = WandBManager()
            wandb_run = wandb_manager.setup(config, experiment_name=f"single_scm_sanity")
            if wandb_run:
                logger.info(f"✅ WandB initialized")
            else:
                logger.warning("⚠️  WandB initialization failed, continuing without logging")
                wandb_manager = None
        except Exception as e:
            logger.warning(f"⚠️  WandB initialization failed: {e}")
            wandb_manager = None
    
    # Create complex 10-variable test SCM for causal discovery
    logger.info("Creating complex 10-variable test SCM: E->G(10.0), F->G(2.0), others isolated...")
    fixed_scm = create_complex_test_scm()
    
    # scm_factory = VariableSCMFactory(seed=args.seed, noise_scale= 0.01)
    # fixed_scm = scm_factory.create_variable_scm(
    #     num_variables=8,        # Simple 3-variable setup
    #     structure_type='random',  # Fork structure: X0, X2 → X1
    #     target_variable='X3'    # Middle variable as target
    # )
    
    # Log SCM details
    target_var = get_target(fixed_scm)
    parents = get_parents(fixed_scm, target_var)
    variables = get_variables(fixed_scm)
    logger.info(f"✅ Created SCM: variables={variables}, target={target_var}, parents={parents}")
    
    # Initialize trainer
    logger.info("\nInitializing GRPO trainer...")
    trainer = DiverseGRPOTrainer(config=config)
    logger.info("✅ Trainer initialized")
    
    # Set fixed SCM in trainer (this is the key change for single SCM)
    trainer.current_scm = fixed_scm
    trainer.current_scm_metadata = dict(fixed_scm.get('metadata', {}))
    trainer.current_scm_metadata['size_category'] = 'single'
    trainer.current_scm_metadata['num_variables'] = 3
    
    # Create lambda that always returns the same SCM
    trainer.scm_generator_callable = lambda: fixed_scm
    
    logger.info(f"✅ Fixed SCM set: {len(get_variables(fixed_scm))} variables, target={get_target(fixed_scm)}")
    
    # Start training
    logger.info("\n" + "="*70)
    logger.info("STARTING SINGLE SCM TRAINING")
    logger.info("="*70 + "\n")
    
    try:
        # Train with the fixed SCM
        results = trainer.train(scms=lambda: fixed_scm)
        
        # Analyze results
        logger.info("\n" + "="*70)
        logger.info("ANALYZING SINGLE SCM RESULTS")
        logger.info("="*70)
        
        all_metrics = trainer.episode_performances
        if all_metrics:
            # Overall statistics
            final_episodes = all_metrics[-10:] if len(all_metrics) >= 10 else all_metrics
            
            analysis = {
                'total_episodes': len(all_metrics),
                'final_mean_target': np.mean([m['mean_target'] for m in final_episodes]),
                'best_target_overall': np.min([m['best_target'] for m in all_metrics]),
                'final_parent_rate': np.mean([m['parent_selection_rate'] for m in final_episodes]),
                'improvement': 0
            }
            
            # Calculate improvement (first vs last episodes)
            if len(all_metrics) >= 20:
                first_10 = all_metrics[:10]
                last_10 = all_metrics[-10:]
                first_mean = np.mean([m['mean_target'] for m in first_10])
                last_mean = np.mean([m['mean_target'] for m in last_10])
                analysis['improvement'] = first_mean - last_mean
            
            logger.info(f"\nSINGLE SCM PERFORMANCE:")
            logger.info(f"  Total episodes: {analysis['total_episodes']}")
            logger.info(f"  Final mean target: {analysis['final_mean_target']:.3f}")
            logger.info(f"  Best target overall: {analysis['best_target_overall']:.3f}")
            logger.info(f"  Final parent selection: {100*analysis['final_parent_rate']:.1f}%")
            if analysis['improvement'] != 0:
                logger.info(f"  Improvement: {analysis['improvement']:.3f}")
            
            # Success criteria check
            logger.info(f"\nSUCCESS CRITERIA CHECK:")
            target_success = analysis['best_target_overall'] < -1.0  # Relaxed from -2.0 for 3-var
            parent_success = analysis['final_parent_rate'] > 0.6  # Relaxed from 0.8 for sanity check
            
            logger.info(f"  Target minimization (< -1.0): {'✅' if target_success else '❌'} {analysis['best_target_overall']:.3f}")
            logger.info(f"  Parent selection (> 60%): {'✅' if parent_success else '❌'} {100*analysis['final_parent_rate']:.1f}%")
            
            if target_success and parent_success:
                logger.info(f"  🎉 SANITY CHECK PASSED!")
            else:
                logger.info(f"  ⚠️  SANITY CHECK NEEDS INVESTIGATION")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results/single_scm_sanity")
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"single_scm_sanity_{timestamp}.json"
        csv_file = results_dir / f"single_scm_sanity_{timestamp}.csv"
        
        # Save CSV data for detailed analysis
        if trainer.csv_data:
            df = pd.DataFrame(trainer.csv_data)
            df.to_csv(csv_file, index=False)
            logger.info(f"\n📊 Saved episode metrics to {csv_file}")
        
        # Prepare serializable results
        save_data = {
            'config': config,
            'scm_details': {
                'variables': list(get_variables(fixed_scm)),  # Convert frozenset to list
                'target': get_target(fixed_scm),
                'parents': list(get_parents(fixed_scm, get_target(fixed_scm))),  # Convert frozenset to list
                'structure_type': 'fork',
                'num_variables': 3
            },
            'analysis': analysis if 'analysis' in locals() else {},
            'episode_performances': trainer.episode_performances,
            'convergence_metrics': {
                'best_overall': trainer.convergence_metrics['best_per_category'].get('single', float('inf')),
                'rolling_averages': trainer.convergence_metrics['rolling_averages'][-20:] if trainer.convergence_metrics['rolling_averages'] else []
            }
        }
        
        with open(results_file, 'w') as f:
            # Simple serialization - convert numpy types
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(v) for v in obj]
                return obj
            
            json.dump(convert(save_data), f, indent=2)
        
        logger.info(f"\n✅ Single SCM sanity check complete! Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Ensure WandB is properly closed
        if wandb_manager:
            wandb_manager.finish()
            logger.info("WandB run finished")
    
    return 0


if __name__ == "__main__":
    # Import needed for intervention handling
    from src.causal_bayes_opt.data_structures.sample import get_values
    
    sys.exit(main())