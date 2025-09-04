#!/usr/bin/env python3
"""
Single-SCM GRPO Training for Detailed Learning Analysis.

This script trains on a SINGLE SCM with many interventions to prove that
the system can learn optimal policies on a fixed graph. This isolates the
learning problem from the generalization problem.

Key features:
- Single episode on one SCM with many interventions (default: 100)
- Detailed logging of every intervention and policy update
- Convergence tracking showing trajectory to optimal actions
- Direct verbose output (no subprocess needed)
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np
import jax
import jax.numpy as jnp
import pyrsistent as pyr

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.causal_bayes_opt.training.joint_acbo_trainer import JointACBOTrainer
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory, get_scm_info
from src.causal_bayes_opt.data_structures.scm import get_parents, get_variables, get_target
from src.causal_bayes_opt.training.utils.wandb_setup import WandBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class SingleSCMTrainer(JointACBOTrainer):
    """
    Simple wrapper to add convergence tracking to JointACBOTrainer.
    
    Uses the same proven training code, just adds analysis metrics.
    """
    
    def __init__(self, config):
        """Initialize trainer with convergence tracking."""
        super().__init__(config=config)
        
        # Convergence tracking
        self.convergence_data = []  # Track selections and optimality
        self.intervention_history = []  # Full intervention details
        self.target_var = None
        self.true_parents = []
        self.optimal_parent = None
        self.scm_details = {}  # Full SCM specification
        
        # Initialize convergence metrics like parent class
        self.convergence_metrics = {
            'total_selections': 0,
            'total_optimal': 0,
            'consecutive_optimal': 0,
            'selections': []
        }
        
        # Probability-based early stopping settings
        self.use_early_stopping = config.get('use_early_stopping', False)
        convergence_config = config.get('convergence', {})
        self.convergence_probability_threshold = convergence_config.get('probability_threshold', 0.9)
        self.convergence_patience = convergence_config.get('patience', 3)
        
        # Probability-based convergence tracking
        self.last_selected_var = None
        self.consecutive_high_prob_count = 0
        
        # Initialize tracking for batch interception
        self._intervention_count = 0
        self._episode_interventions = []
        
    def _run_policy_episode_with_interventions(self, episode_idx, scm, scm_name, key):
        """
        Override to add convergence tracking and extract SCM details.
        This is the method actually called during training.
        """
        # Store SCM details for convergence tracking
        self.target_var = get_target(scm)
        variables = list(get_variables(scm))
        self.true_parents = list(get_parents(scm, self.target_var))
        
        # Extract comprehensive SCM details
        metadata = dict(scm.get('metadata', {}))
        # Coefficients are stored in metadata for this SCM type
        raw_coefficients = dict(metadata.get('coefficients', {}))
        # Keep coefficients as dict with tuple keys for internal use
        coefficients = raw_coefficients
        
        # Get intervention ranges for each variable from metadata
        variable_ranges = {}
        if 'variable_ranges' in metadata:
            variable_ranges = dict(metadata['variable_ranges'])
        else:
            # Default ranges if not specified
            for var in variables:
                variable_ranges[var] = (-10, 10)
        
        # Extract parent coefficients from the coefficients dict
        parent_coeffs = {}
        for parent in self.true_parents:
            coeff_key = (parent, self.target_var)
            if coeff_key in coefficients:
                parent_coeffs[parent] = coefficients[coeff_key]
            else:
                # Log warning if coefficient not found
                logger.warning(f"Coefficient not found for edge {parent} → {self.target_var}")
        
        # Store comprehensive SCM details (convert coefficient keys to strings)
        self.scm_details = {
            'name': scm_name,
            'variables': variables,
            'target': self.target_var,
            'true_parents': self.true_parents,
            'coefficients': {f"{parent}->{child}": value 
                           for (parent, child), value in coefficients.items()},
            'parent_coefficients': parent_coeffs,
            'variable_ranges': variable_ranges,
            'metadata': {k: v for k, v in metadata.items() if k != 'coefficients'},  # Exclude coefficients from metadata to avoid duplication
            'noise_scale': metadata.get('noise_scale', 0.5),
            'structure_type': metadata.get('structure_type', 'unknown')
        }
        
        # Determine optimal parent (with largest coefficient * range effect)
        optimal_score = 0
        self.optimal_parent = None
        
        for parent in self.true_parents:
            coeff = parent_coeffs.get(parent, 0.0)
            parent_range = variable_ranges.get(parent, (-10, 10))
            range_size = parent_range[1] - parent_range[0]
            
            # Score = coefficient magnitude × range size
            score = abs(coeff) * range_size
            
            if score > optimal_score:
                optimal_score = score
                self.optimal_parent = parent
                self.optimal_coefficient = coeff
                self.optimal_range = parent_range
        
        # Print detailed SCM info
        logger.info(f"\n" + "="*60)
        logger.info(f"📊 SINGLE-SCM TRAINING DETAILS:")
        logger.info(f"   Variables: {variables}")
        logger.info(f"   Target: {self.target_var}")
        logger.info(f"   True Parents: {self.true_parents}")
        logger.info(f"   Optimal Parent: {self.optimal_parent}")
        if parent_coeffs:
            logger.info(f"   Parent Coefficients:")
            for parent, coeff in parent_coeffs.items():
                parent_range = variable_ranges.get(parent, (-10, 10))
                marker = " ← OPTIMAL" if parent == self.optimal_parent else ""
                logger.info(f"      {parent} → {self.target_var}: {coeff:.3f}, range={parent_range}{marker}")
        logger.info(f"   Variable Ranges:")
        for var, range_vals in variable_ranges.items():
            logger.info(f"      {var}: {range_vals}")
        logger.info(f"="*60 + "\n")
        
        # We need to track interventions during the episode since parent _run_grpo_episode doesn't track convergence
        # Clear tracking for this episode
        self._intervention_count = 0
        self._episode_interventions = []
        
        # Clear convergence data for this episode
        self.convergence_data = []
        self.intervention_history = []
        
        # COPIED FROM PARENT CLASS WITH OUR ADDITIONS
        import time
        import numpy as np
        from jax import random
        from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
        from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
        from src.causal_bayes_opt.data_structures.sample import get_values
        from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
        from src.causal_bayes_opt.training.five_channel_converter import create_uniform_posterior
        
        # Initialize buffer with observations
        buffer = ExperienceBuffer()
        key, obs_key = random.split(key)
        obs_samples = sample_from_linear_scm(scm, self.obs_per_episode, seed=int(obs_key[0]))
        
        # Add initial posterior for observations
        if self.use_surrogate and hasattr(self, '_get_surrogate_predictions'):
            # Get initial posterior
            temp_buffer = ExperienceBuffer()
            for sample in obs_samples:
                temp_buffer.add_observation(sample)
            initial_tensor, mapper = buffer_to_three_channel_tensor(
                temp_buffer, self.target_var, max_history_size=100, standardize=True
            )
            initial_posterior = self._get_surrogate_predictions(temp_buffer, self.target_var, list(get_variables(scm)))
        else:
            initial_posterior = create_uniform_posterior(list(get_variables(scm)), self.target_var)
        
        # Add observations with posterior
        for sample in obs_samples:
            buffer.add_observation(sample, posterior=initial_posterior)
        
        initial_buffer_size = buffer.size()
        logger.info(f"\nEpisode {episode_idx} starting: buffer initialized with {initial_buffer_size} observations")
        
        # INTERVENTION LOOP
        intervention_metrics = []
        all_rewards = []
        all_target_values = []  # Track target progression
        
        for intervention_idx in range(self.max_interventions):
            intervention_start = time.time()
            print(f"\n{'='*50}")
            print(f"INTERVENTION {intervention_idx+1}/{self.max_interventions}")
            print(f"{'='*50}")
            
            # Use parent's proven GRPO implementation for single intervention
            key, intervention_key = random.split(key)
            
            # Run single GRPO intervention (generates candidates, updates policy, returns best)
            single_result = super()._run_single_grpo_intervention(
                buffer, scm, self.target_var, list(get_variables(scm)), intervention_key
            )
            
            # OUR ADDITION: CAPTURE INTERVENTION DATA
            if 'best_intervention' in single_result:
                best_int = single_result['best_intervention']
                
                if best_int.get('outcome') is not None:
                    # Extract data from the best intervention
                    outcome = best_int['outcome']
                    debug_info = best_int.get('debug_info', {})
                    
                    # Get selected variable from debug_info
                    if 'selected_var_idx' in debug_info:
                        var_idx = debug_info['selected_var_idx']
                        selected_var = list(get_variables(scm))[var_idx]
                        probability = debug_info.get('selection_probability', 0.0)
                        
                        # Get intervention value from the intervention object
                        intervention = best_int['intervention']
                        # The intervention is a pyr.PMap, extract values using get()
                        intervention_values = intervention.get('values', {})
                        intervention_value = float(intervention_values.get(selected_var, 0.0))
                        
                        # Get target outcome
                        target_outcome = float(get_values(outcome).get(self.target_var, 0.0))
                        all_target_values.append(target_outcome)
                        
                        # Check optimality
                        is_optimal = (selected_var == self.optimal_parent) if self.optimal_parent else False
                        is_parent = selected_var in self.true_parents
                        
                        # Update optimality metrics (for reporting)
                        if is_optimal:
                            self.convergence_metrics['consecutive_optimal'] += 1
                            self.convergence_metrics['total_optimal'] += 1
                        else:
                            self.convergence_metrics['consecutive_optimal'] = 0
                        self.convergence_metrics['total_selections'] += 1
                        
                        # Check probability-based convergence
                        if probability >= self.convergence_probability_threshold:
                            if self.last_selected_var == selected_var:
                                self.consecutive_high_prob_count += 1
                                logger.info(f"    🎯 Same variable with high prob! Count: {self.consecutive_high_prob_count}/{self.convergence_patience}")
                            else:
                                # Different variable, reset counter
                                self.consecutive_high_prob_count = 1
                                self.last_selected_var = selected_var
                                logger.info(f"    🔄 New high-prob variable: {selected_var}")
                        else:
                            # Low probability, reset tracking
                            if probability > 0:
                                logger.info(f"    ⚠️ Low probability ({probability:.3f} < {self.convergence_probability_threshold}), resetting")
                            self.consecutive_high_prob_count = 0
                            self.last_selected_var = None
                        
                        # Create intervention record
                        intervention_record = {
                            'intervention_idx': len(self.convergence_data),
                            'selected_var': selected_var,
                            'selected_value': float(intervention_value),
                            'is_optimal': is_optimal,
                            'is_parent': is_parent,
                            'probability': float(probability),
                            'target_outcome': target_outcome,
                            'optimal_parent': self.optimal_parent,
                            'true_parents': self.true_parents
                        }
                        
                        self.convergence_data.append(intervention_record)
                        self.intervention_history.append(intervention_record)
                        
                        # Log the actual selection
                        logger.info(f"\n📊 CAPTURED INTERVENTION {len(self.convergence_data)}: "
                                   f"Selected={selected_var} (prob={probability:.3f}), "
                                   f"IsOptimal={is_optimal}, IsParent={is_parent}, "
                                   f"Value={intervention_value:.3f}, "
                                   f"Outcome={target_outcome:.3f}")
                    
                    # Add best intervention to buffer (from parent code)
                    buffer.add_intervention(
                        best_int['intervention'],
                        best_int['outcome'],
                        posterior=best_int.get('posterior')
                    )
                    
                    # Log buffer growth
                    current_size = buffer.size()
                    print(f"Buffer progression: {current_size-1} -> {current_size}")
                    print(f"Selected intervention TARGET: {target_outcome:.3f}")
                
                # Track metrics
                intervention_metrics.append(single_result)
                all_rewards.extend(single_result.get('candidate_rewards', []))
                
                # Log intervention timing
                intervention_duration = time.time() - intervention_start
                logger.info(f"⏱️ Intervention {intervention_idx+1} completed in {intervention_duration:.1f} seconds")
                
                # Check for probability-based early stopping
                if self.use_early_stopping and self.consecutive_high_prob_count >= self.convergence_patience:
                    logger.info(f"\n🎯 CONVERGENCE DETECTED: Variable '{self.last_selected_var}' selected ")
                    logger.info(f"   {self.consecutive_high_prob_count} times with >{self.convergence_probability_threshold:.0%} probability")
                    logger.info(f"   Stopping after {intervention_idx+1}/{self.max_interventions} interventions")
                    break
        
        # Episode summary (from parent)
        final_buffer_size = buffer.size()
        total_interventions_added = final_buffer_size - initial_buffer_size
        
        print(f"\n{'='*60}")
        print(f"EPISODE {episode_idx} COMPLETE")
        print(f"{'='*60}")
        print(f"Total interventions captured: {len(self.convergence_data)}")
        print(f"Buffer growth: {initial_buffer_size} -> {final_buffer_size} (+{total_interventions_added})")
        print(f"Mean reward across all interventions: {np.mean(all_rewards) if all_rewards else 0:.3f}")
        
        # TARGET PROGRESSION ANALYSIS within episode
        if all_target_values:
            print(f"\n📈 TARGET PROGRESSION (within episode):")
            print(f"  Values: {[f'{v:.3f}' for v in all_target_values]}")
            print(f"  Best (lowest): {min(all_target_values):.3f}")
            print(f"  Worst (highest): {max(all_target_values):.3f}")
            print(f"  Trend: {all_target_values[0]:.3f} → {all_target_values[-1]:.3f} ({all_target_values[-1] - all_target_values[0]:+.3f}")
            
            # Check if improving within episode (for minimization)
            improvement = all_target_values[0] - all_target_values[-1]
            if improvement > 0.1:
                print(f"  ✅ IMPROVING within episode! ({improvement:+.3f})")
            elif improvement < -0.1:
                print(f"  ⚠️ Getting worse within episode ({improvement:+.3f})")
            else:
                print(f"  ➖ No clear trend within episode ({improvement:+.3f})")
        
        # Return result similar to parent
        result = {
            'episode': episode_idx,
            'mean_reward': float(np.mean(all_rewards)) if all_rewards else 0.0,
            'n_interventions': len(intervention_metrics),
            'buffer_growth': total_interventions_added,
            'intervention_metrics': intervention_metrics,
            'n_variables': len(list(get_variables(scm))),
            'scm_type': scm_name,
            'target_values': all_target_values,
            'best_target': min(all_target_values) if all_target_values else 0.0,
            'target_improvement': (all_target_values[0] - all_target_values[-1]) if len(all_target_values) >= 2 else 0.0,
            'convergence_data': self.convergence_data  # Our addition
        }
        
        # Analyze convergence from collected data
        self._analyze_convergence()
        
        return result
    
    
    def _analyze_convergence(self):
        """Analyze convergence metrics after episode."""
        if not self.convergence_data:
            logger.info("No convergence data collected yet")
            return
            
        logger.info(f"\n" + "="*60)
        logger.info("📈 CONVERGENCE ANALYSIS")
        logger.info("="*60)
        
        # Calculate convergence metrics in windows
        window_size = min(10, max(1, len(self.convergence_data) // 5))
        
        for start in range(0, len(self.convergence_data), window_size):
            end = min(start + window_size, len(self.convergence_data))
            window_data = self.convergence_data[start:end]
            optimal_count = sum(1 for d in window_data if d.get('is_optimal', False))
            pct = 100 * optimal_count / len(window_data) if window_data else 0
            logger.info(f"  Interventions {start+1}-{end}: "
                       f"{optimal_count}/{len(window_data)} optimal ({pct:.1f}%)")
        
        # Final performance
        last_n = min(10, len(self.convergence_data))
        if last_n > 0:
            last_window = self.convergence_data[-last_n:]
            final_optimal = sum(1 for d in last_window if d.get('is_optimal', False))
            final_pct = 100 * final_optimal / len(last_window)
            
            logger.info(f"\n  FINAL PERFORMANCE (last {last_n} interventions):")
            logger.info(f"    Optimal selections: {final_optimal}/{last_n} ({final_pct:.1f}%)")
            
            if final_pct >= 80:
                logger.info(f"    ✅ CONVERGED to optimal policy!")
            elif final_pct >= 50:
                logger.info(f"    ⚠️  Partially converged")
            else:
                logger.info(f"    ❌ Not converged")
        
        logger.info("="*60 + "\n")


def create_single_scm_config(
    max_interventions: int = 100,
    verbose: bool = True,
    enable_wandb: bool = False,
    seed: int = 42,
    target_weight: float = 0.7,
    parent_weight: float = 0.1,
    info_gain_weight: float = 0.2,
    use_early_stopping: bool = True,
    convergence_patience: int = 3,
    convergence_threshold: float = 0.9
) -> Dict[str, Any]:
    """Create configuration for single-SCM training."""
    
    config = {
        # Single episode with many interventions
        'max_episodes': 1,
        'obs_per_episode': 20,
        'max_interventions': max_interventions,
        
        # Disable phase switching
        'episodes_per_phase': 999999,
        'initial_phase': 'policy',
        
        # Architecture
        'policy_architecture': 'quantile',
        
        # Surrogate integration
        'use_surrogate': True,
        'use_grpo_rewards': True,
        
        # Exploration
        'use_fixed_std': True,
        'fixed_std': 1.0,
        
        # Learning
        'learning_rate': 5e-4,
        
        # GRPO configuration
        'grpo_config': {
            'group_size': 32,
            'entropy_coefficient': 0.001,
            'clip_ratio': 1.0,
            'gradient_clip': 10.0,
            'ppo_epochs': 4,
            'normalize_advantages': True
        },
        
        # Reward weights
        'reward_weights': {
            'target': target_weight,
            'parent': parent_weight,
            'info_gain': info_gain_weight
        },
        
        # Binary rewards
        'reward_type': 'binary',
        'info_gain_type': 'probability_change',
        
        # Buffer configuration
        'buffer_config': {
            'max_history_size': 30,
            'adaptive_history': True,
            'min_history_size': 10
        },
        
        # No rotation for single-SCM
        'rotate_after_episode': False,
        
        # Early stopping configuration
        'use_early_stopping': use_early_stopping,
        'convergence': {
            'patience': convergence_patience,
            'probability_threshold': convergence_threshold,
            'min_episodes': 1,  # For single SCM, we only have 1 episode
            'max_episodes_per_scm': 1,
            'use_rolling_window': False  # Not needed for single SCM
        },
        
        # Surrogate model
        'surrogate_checkpoint_path': 'experiments/surrogate-only-training/scripts/checkpoints/avici_runs/avici_style_20250822_213115/checkpoint_step_200.pkl',
        
        # General settings
        'batch_size': 32,
        'seed': seed,
        'verbose': verbose,
        'checkpoint_dir': 'checkpoints/grpo_single_scm',
        
        # WandB logging
        'logging': {
            'wandb': {
                'enabled': enable_wandb,
                'project': 'causal-bayes-opt-single-scm',
                'name': f'single_scm_{max_interventions}int',
                'tags': ['single_scm', 'grpo', 'convergence_analysis'],
                'log_frequency': 1
            }
        }
    }
    
    return config


def create_hardcoded_collider_scm():
    """
    Create a hardcoded collider SCM with specific structure:
    X0 -> X1, X0 -> X2, X2 -> X3, X1 -> X3
    All coefficients = 2.0, all ranges = (-3, 3)
    """
    from src.causal_bayes_opt.experiments.test_scms import create_simple_linear_scm
    
    variables = ['X0', 'X1', 'X2', 'X3']
    
    # Define the exact edges: X0->X1, X0->X2, X2->X3, X1->X3
    edges = [
        ('X0', 'X1'),
        ('X0', 'X2'),
        ('X2', 'X3'),
        ('X1', 'X3')
    ]
    
    # All coefficients are exactly 2.0
    coefficients = {
        ('X0', 'X1'): 2.0,
        ('X0', 'X2'): 2.0,
        ('X2', 'X3'): 2.0,
        ('X1', 'X3'): 2.0
    }
    
    # Noise scales for all variables (reduced for clearer signal)
    noise_scales = {
        'X0': 0.1,  # Root node
        'X1': 0.1,
        'X2': 0.1,
        'X3': 0.1
    }
    
    # All variables have range (-3, 3)
    variable_ranges = {
        'X0': (-3.0, 3.0),
        'X1': (-3.0, 3.0),
        'X2': (-3.0, 3.0),
        'X3': (-12.0, 12.0)
    }
    
    # X3 is the target (collider node with two parents)
    target = 'X3'
    
    # Create the SCM
    scm = create_simple_linear_scm(
        variables=variables,
        edges=edges,
        coefficients=coefficients,
        noise_scales=noise_scales,
        target=target,
        variable_ranges=variable_ranges,
        output_bounds=None
    )
    
    # Add metadata
    metadata = scm.get('metadata', pyr.pmap({}))
    metadata = metadata.update({
        'structure_type': 'hardcoded_collider',
        'num_variables': 4,
        'num_edges': 4,
        'edge_density': 4 / (4 * 3 / 2),  # 4 edges out of 6 possible
        'target_variable': target,
        'coefficients': coefficients,
        'variable_ranges': variable_ranges,
        'noise_scale': 0.1,
        'description': 'Hardcoded collider: X0->X1, X0->X2, X2->X3, X1->X3'
    })
    
    scm = scm.update({'metadata': metadata})
    
    logger.info("Created hardcoded collider SCM:")
    logger.info(f"  Structure: X0 -> X1, X0 -> X2, X2 -> X3, X1 -> X3")
    logger.info(f"  All coefficients: 2.0")
    logger.info(f"  All ranges: (-3, 3)")
    logger.info(f"  Target: X3 (has parents X1 and X2)")
    
    return scm


def main():
    """Main training function for single-SCM GRPO analysis."""
    
    parser = argparse.ArgumentParser(description="Single-SCM GRPO training for convergence analysis")
    parser.add_argument('--interventions', type=int, default=100,
                        help='Number of interventions on single SCM')
    parser.add_argument('--scm-type', type=str, default='fork',
                        choices=['fork', 'true_fork', 'chain', 'collider', 'mixed', 'random', 'hardcoded_collider',
                                 "scale_free", "two_layer"],
                        help='Type of SCM structure')
    parser.add_argument('--num-vars', type=int, default=4,
                        help='Number of variables in SCM')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Enable verbose logging (default: True)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable WandB logging')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--entropy', type=float, default=None,
                        help='Override entropy coefficient')
    parser.add_argument('--target-weight', type=float, default=0.7,
                        help='Weight for target value reward component (default: 0.7)')
    parser.add_argument('--parent-weight', type=float, default=0.1,
                        help='Weight for parent selection bonus (default: 0.1)')
    parser.add_argument('--info-gain-weight', type=float, default=0.2,
                        help='Weight for information gain reward (default: 0.2)')
    parser.add_argument('--early-stopping', action='store_true', default=True,
                        help='Enable early stopping based on probability convergence (default: True)')
    parser.add_argument('--no-early-stopping', dest='early_stopping', action='store_false',
                        help='Disable early stopping')
    parser.add_argument('--convergence-patience', type=int, default=3,
                        help='Number of consecutive high-probability selections for convergence (default: 3)')
    parser.add_argument('--convergence-threshold', type=float, default=0.9,
                        help='Probability threshold for considering a selection high-confidence (default: 0.9)')
    
    args = parser.parse_args()
    
    # Print configuration
    logger.info("\n" + "="*70)
    logger.info("SINGLE-SCM GRPO TRAINING ANALYSIS")
    logger.info("="*70)
    logger.info(f"Configuration:")
    logger.info(f"  Interventions: {args.interventions}")
    logger.info(f"  SCM Type: {args.scm_type}")
    logger.info(f"  Variables: {args.num_vars}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Verbose: {args.verbose}")
    logger.info("="*70 + "\n")
    
    # Create timestamped run directory
    run_name = f"single_scm_{args.scm_type}_{args.num_vars}vars_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = Path("checkpoints/grpo_single_scm") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    
    # Create configuration
    config = create_single_scm_config(
        max_interventions=args.interventions,
        verbose=args.verbose,
        enable_wandb=args.wandb,
        seed=args.seed,
        target_weight=args.target_weight,
        parent_weight=args.parent_weight,
        info_gain_weight=args.info_gain_weight,
        use_early_stopping=args.early_stopping,
        convergence_patience=args.convergence_patience,
        convergence_threshold=args.convergence_threshold
    )
    
    # Override parameters if provided
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
        logger.info(f"Overriding learning rate: {args.learning_rate}")
    if args.entropy:
        config['grpo_config']['entropy_coefficient'] = args.entropy
        logger.info(f"Overriding entropy coefficient: {args.entropy}")
    
    # Update checkpoint directory
    config['checkpoint_dir'] = str(checkpoint_dir)
    
    # Log critical settings
    logger.info("\nTraining Settings:")
    logger.info(f"  Episodes: {config['max_episodes']} (single SCM)")
    logger.info(f"  Interventions per episode: {config['max_interventions']}")
    logger.info(f"  Learning rate: {config['learning_rate']}")
    logger.info(f"  GRPO group size: {config['grpo_config']['group_size']}")
    logger.info(f"  Entropy coefficient: {config['grpo_config']['entropy_coefficient']}")
    logger.info(f"\n🎯 Reward Weights:")
    logger.info(f"  Target: {config['reward_weights']['target']} (minimize target value)")
    logger.info(f"  Parent: {config['reward_weights']['parent']} (bonus for selecting true parents)")
    logger.info(f"  Info gain: {config['reward_weights']['info_gain']} (information gain from intervention)")
    logger.info(f"  Total must sum to 1.0: {sum(config['reward_weights'].values()):.1f}")
    
    if config['use_early_stopping']:
        logger.info(f"\n⏹️  Early Stopping: ENABLED (Probability-based)")
        logger.info(f"  Convergence threshold: {config['convergence']['probability_threshold']:.0%}")
        logger.info(f"  Patience: {config['convergence']['patience']} consecutive selections")
        logger.info(f"  Will stop when same variable selected {config['convergence']['patience']} times with >{config['convergence']['probability_threshold']:.0%} probability")
    else:
        logger.info(f"\n⏹️  Early Stopping: DISABLED")
    logger.info("")
    
    # Initialize WandB if enabled
    wandb_manager = None
    if args.wandb:
        try:
            wandb_manager = WandBManager()
            wandb_run = wandb_manager.setup(config, experiment_name=run_name)
            if wandb_run:
                logger.info("✅ WandB initialized")
        except Exception as e:
            logger.warning(f"⚠️ WandB initialization failed: {e}")
            wandb_manager = None
    
    # Create SCM based on type
    if args.scm_type == 'hardcoded_collider':
        logger.info("Using hardcoded collider SCM...")
        single_scm = create_hardcoded_collider_scm()
        scm_name = 'hardcoded_collider_4var'
    else:
        # Create SCM factory
        logger.info("Setting up SCM factory...")
        scm_factory = VariableSCMFactory(
            seed=args.seed,
            noise_scale=0.5,
            coefficient_range=(-3.0, 3.0),
            vary_intervention_ranges=True,
            use_output_bounds=True
        )
        
        # Create specific SCM for training
        sampling_config = {
            'variable_counts': [args.num_vars],
            'structure_types': [args.scm_type],
            'edge_density_range': (0.3, 0.5),
            'name_prefix': 'single_scm'
        }
        
        logger.info(f"Generating {args.scm_type} SCM with {args.num_vars} variables...")
        scm_name, single_scm = scm_factory.get_random_scm(**sampling_config)
    
    # Log SCM details and extract full information
    target_var = get_target(single_scm)
    variables = list(get_variables(single_scm))
    true_parents = list(get_parents(single_scm, target_var))
    
    # Extract coefficients and ranges from the SCM for saving
    scm_metadata = dict(single_scm.get('metadata', {}))
    raw_coefficients = dict(scm_metadata.get('coefficients', {}))
    # Convert coefficient keys from tuples to strings for JSON serialization
    scm_coefficients = {f"{parent}->{child}": value 
                       for (parent, child), value in raw_coefficients.items()}
    scm_variable_ranges = dict(scm_metadata.get('variable_ranges', {}))
    # Store parent coefficients separately for easy access
    scm_parent_coeffs = {parent: raw_coefficients.get((parent, target_var), 0.0) 
                        for parent in true_parents if (parent, target_var) in raw_coefficients}
    
    logger.info(f"\n📊 GENERATED SCM:")
    logger.info(f"  Name: {scm_name}")
    logger.info(f"  Structure: {args.scm_type}")
    logger.info(f"  Variables: {variables}")
    logger.info(f"  Target: {target_var}")
    logger.info(f"  True Parents: {true_parents}")
    logger.info("")
    
    # Create generator that returns the single SCM
    def single_scm_generator():
        return scm_name, single_scm
    
    # Initialize trainer with convergence tracking
    logger.info("Initializing trainer...")
    trainer = SingleSCMTrainer(config=config)
    logger.info("✅ Trainer initialized")
    
    # Save experiment configuration
    experiment_config = {
        'args': vars(args),
        'config': config,
        'run_name': run_name,
        'scm_info': {
            'name': scm_name,
            'type': args.scm_type,
            'variables': variables,
            'target': target_var,
            'parents': true_parents,
            'coefficients': scm_coefficients,
            'variable_ranges': scm_variable_ranges,
            'metadata': {k: v for k, v in scm_metadata.items() if k != 'coefficients'}  # Exclude coefficients
        }
    }
    
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(experiment_config, f, indent=2, default=str)
    
    logger.info(f"💾 Saved experiment config to {checkpoint_dir / 'config.json'}")
    
    # Start training
    logger.info("\n" + "="*70)
    logger.info("STARTING SINGLE-SCM TRAINING")
    logger.info("="*70 + "\n")
    
    try:
        # Train on single SCM
        results = trainer.train(scms=single_scm_generator)
        
        # Save final checkpoint
        from src.causal_bayes_opt.utils.checkpoint_utils import save_checkpoint
        
        final_checkpoint_path = checkpoint_dir / 'final_policy.pkl'
        save_checkpoint(
            path=final_checkpoint_path,
            params=trainer.policy_params,
            architecture={
                'hidden_dim': trainer.hidden_dim,
                'num_layers': trainer.num_layers,
                'num_heads': trainer.num_heads,
                'key_size': trainer.key_size,
                'dropout': trainer.dropout,
                'policy_architecture': config['policy_architecture'],
                'architecture_type': 'quantile'
            },
            model_type='policy',
            model_subtype='grpo',
            training_config=config,
            metadata={
                'experiment_type': 'single_scm_analysis',
                'scm_type': args.scm_type,
                'num_vars': args.num_vars,
                'run_name': run_name,
                'convergence_data': trainer.convergence_data
            }
        )
        
        logger.info(f"✅ Training complete!")
        logger.info(f"💾 Final checkpoint saved to: {final_checkpoint_path}")
        
        # Save convergence data
        convergence_file = checkpoint_dir / 'convergence_data.json'
        with open(convergence_file, 'w') as f:
            json.dump({
                'convergence_data': trainer.convergence_data,
                'target': trainer.target_var,
                'true_parents': trainer.true_parents,
                'optimal_parent': trainer.optimal_parent
            }, f, indent=2, default=str)
        
        # Save comprehensive SCM and intervention data
        full_data_file = checkpoint_dir / 'full_experiment_data.json'
        
        # Make sure we have SCM details even if episode didn't run
        if hasattr(trainer, 'scm_details') and trainer.scm_details:
            scm_details = trainer.scm_details
        else:
            # Fallback: extract from the original SCM
            scm_details = {
                'name': scm_name,
                'variables': variables,
                'target': target_var,
                'true_parents': true_parents,
                'coefficients': scm_coefficients,
                'parent_coefficients': scm_parent_coeffs,
                'variable_ranges': scm_variable_ranges,
                'metadata': {k: v for k, v in scm_metadata.items() if k != 'coefficients'},  # Exclude coefficients to avoid tuple key issue
                'noise_scale': scm_metadata.get('noise_scale', 0.5),
                'structure_type': args.scm_type
            }
        
        with open(full_data_file, 'w') as f:
            json.dump({
                'scm_details': scm_details,
                'convergence_data': trainer.convergence_data,
                'intervention_history': trainer.intervention_history,
                'training_config': config,
                'experiment_info': {
                    'scm_type': args.scm_type,
                    'num_vars': args.num_vars,
                    'total_interventions': len(trainer.convergence_data),
                    'seed': args.seed,
                    'convergence_metrics_found': hasattr(trainer, 'convergence_metrics')
                }
            }, f, indent=2, default=str)
        
        logger.info(f"📊 Convergence data saved to: {convergence_file}")
        logger.info(f"📊 Full experiment data saved to: {full_data_file}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        if wandb_manager:
            wandb_manager.finish()
            logger.info("WandB run finished")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())