"""
GRPO training logger for diagnostics and progress tracking.

This module handles all diagnostic logging, progress tracking, and analysis
that was previously mixed into the main training logic.
"""

import logging
from typing import Dict, Any, Optional, List
import jax.numpy as jnp
import jax
from collections import defaultdict

from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.sample import get_values
from ..data_structures.scm import get_parents

logger = logging.getLogger(__name__)


class GRPOLogger:
    """Handles all diagnostic logging and progress tracking for GRPO training."""
    
    def __init__(self, optimization_direction: str = "MINIMIZE"):
        self.optimization_direction = optimization_direction
        
        # Training history
        self.target_value_history = []
        self.probability_history = []
        self.best_target_values = {}
        self.initial_target_values = {}
        
    def log_episode_start(self, episode_idx: int, scm_name: str, target_var: str, true_parents: list):
        """Log episode start information."""
        if episode_idx % 10 == 0:
            logger.info(f"\n{'='*70}")
            logger.info(f"EPISODE {episode_idx} - {scm_name}")
            logger.info(f"Target: {target_var}, True parents: {true_parents}")
    
    def log_variable_probabilities(self, var_probs: jnp.ndarray, mapper: Any, target_var: str, 
                                  episode_idx: int, scm_name: str):
        """Log variable selection probabilities and track changes."""
        # Log probabilities for each variable
        prob_info = []
        for i, var_name in enumerate(mapper.variables):
            if i != mapper.target_idx:
                prob_info.append(f"{var_name}:{var_probs[i]:.3f}")
        logger.info(f"  Variable probabilities: {' '.join(prob_info)}")
        
        # Compute differentiation metrics
    
    def log_quantile_probabilities(self, quantile_scores: jnp.ndarray, variables: List[str], 
                                  target_var: str, episode_idx: int, scm_name: str):
        """Log quantile probability matrix for quantile architecture."""
        
        print(f"\n📊 QUANTILE PROBABILITY MATRIX:")
        quantile_names = ['25%', '50%', '75%']
        
        # Create probability distribution over all quantiles
        flat_scores = quantile_scores.flatten()
        flat_probs = jax.nn.softmax(flat_scores)
        
        # Display matrix format
        for i, var in enumerate(variables):
            if var == target_var:
                print(f"    {var}: [MASKED - target variable]")
            else:
                probs = flat_probs[i*3:(i+1)*3]
                prob_str = ", ".join([f"{q}:{p:.3f}" for q, p in zip(quantile_names, probs)])
                print(f"    {var}: [{prob_str}]")
        
        # Find most likely selection
        max_prob_idx = jnp.argmax(flat_probs)
        max_var_idx, max_quantile_idx = divmod(int(max_prob_idx), 3)
        if max_var_idx < len(variables):
            max_var = variables[max_var_idx]
            max_quantile = quantile_names[max_quantile_idx]
            max_prob = float(flat_probs[max_prob_idx])
            print(f"    Most likely: {max_var}_{max_quantile} ({max_prob:.3f})")
        
        # Track differentiation
        prob_std = float(jnp.std(flat_probs))
        prob_entropy = float(-jnp.sum(flat_probs * jnp.log(flat_probs + 1e-8)))
        
        print(f"    Probability std: {prob_std:.4f}, Entropy: {prob_entropy:.3f}")
        
        if prob_std < 0.01:
            print(f"    ⚠️ WARNING: Low quantile differentiation")
        else:
            print(f"    ✅ Good quantile differentiation")
    
    def log_gradient_analysis(self, grads: Dict[str, Any], policy_output: Dict[str, jnp.ndarray]):
        """Log gradient analysis for both traditional and quantile architectures."""
        
        # Compute gradient norms by component
        quantile_head_total = 0.0
        var_head_total = 0.0
        val_head_total = 0.0
        attention_total = 0.0
        other_total = 0.0
        
        # Flatten gradient tree and categorize
        flat_grads = jax.tree_util.tree_flatten_with_path(grads)[0]
        
        for path_info, grad in flat_grads:
            if not hasattr(grad, 'shape'):
                continue
                
            param_path = '/'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path_info)
            norm = float(jnp.linalg.norm(grad))
            
            # Categorize gradients based on parameter path
            if 'quantile_mlp' in param_path or 'quantile_output' in param_path:
                quantile_head_total += norm
            elif 'var_mlp' in param_path or 'variable' in param_path:
                var_head_total += norm
            elif 'val_mlp' in param_path or 'value' in param_path:
                val_head_total += norm
            elif 'attention' in param_path or 'single_attention' in param_path:
                attention_total += norm
            else:
                other_total += norm
        
        total_grad_norm = quantile_head_total + var_head_total + val_head_total + attention_total + other_total
        
        print(f"\n📊 Gradient Flow Distribution:")
        
        if 'quantile_scores' in policy_output:
            # Quantile architecture analysis
            print(f"  Quantile head: {quantile_head_total:.6f} ({quantile_head_total/total_grad_norm:.1%})")
            print(f"  Attention layers: {attention_total:.6f} ({attention_total/total_grad_norm:.1%})")
            print(f"  Other layers: {other_total:.6f} ({other_total/total_grad_norm:.1%})")
            
            if quantile_head_total / total_grad_norm < 0.1:
                print(f"  ❌ CRITICAL: Quantile head gets <10% of gradients!")
            else:
                print(f"  ✅ Quantile head gets reasonable gradient share")
                
        else:
            # Traditional two-head architecture analysis
            print(f"  Variable head: {var_head_total:.6f} ({var_head_total/total_grad_norm:.1%})")
            print(f"  Value head: {val_head_total:.6f} ({val_head_total/total_grad_norm:.1%})")
            print(f"  Attention layers: {attention_total:.6f} ({attention_total/total_grad_norm:.1%})")
            print(f"  Other layers: {other_total:.6f}")
            
            if var_head_total / total_grad_norm < 0.05:
                print(f"  ❌ CRITICAL: Variable head gets <5% gradients!")
            
        print(f"  Total norm: {total_grad_norm:.6f}")
        
        return total_grad_norm
        valid_probs = var_probs[var_probs > 0]
        prob_std = float(jnp.std(valid_probs))
        max_min_diff = float(jnp.max(valid_probs) - jnp.min(valid_probs))
        entropy = -float(jnp.sum(valid_probs * jnp.log(valid_probs + 1e-8)))
        
        logger.info(f"  Prob metrics - Std: {prob_std:.4f}, Max-Min: {max_min_diff:.4f}, Entropy: {entropy:.3f}")
        
        # Store probability history
        prob_record = {
            'episode': episode_idx,
            'scm': scm_name,
            'target': target_var,
            'probs': {var_name: float(var_probs[i]) for i, var_name in enumerate(mapper.variables) if i != mapper.target_idx},
            'std': prob_std,
            'max_min': max_min_diff,
            'entropy': entropy
        }
        self.probability_history.append(prob_record)
        
        # Check for sudden changes
        if len(self.probability_history) > 1:
            prev_record = self.probability_history[-2]
            if prev_record['scm'] == scm_name:
                prev_max_min = prev_record['max_min']
                change = max_min_diff - prev_max_min
                if abs(change) > 0.5:
                    logger.warning(f"  SUDDEN PROBABILITY CHANGE: {prev_max_min:.4f} -> {max_min_diff:.4f} (change: {change:+.4f})")
    
    def log_buffer_state(self, buffer: ExperienceBuffer):
        """Log buffer state analysis."""
        print(f"\n🔍 BUFFER STATE ANALYSIS:")
        print(f"  Buffer total size: {buffer.size()}")
        print(f"  Observations: {len(buffer.get_observations())}")
        print(f"  Interventions: {len(buffer.get_interventions())}")
        
        # Show last few samples
        all_samples = buffer.get_all_samples()
        print(f"\n📋 BUFFER CONTENT (last 3 samples):")
        for i, sample in enumerate(all_samples[-3:]):
            sample_values = get_values(sample)
            print(f"    Sample {len(all_samples)-3+i+1}: {dict(sample_values)}")
    
    def log_tensor_analysis(self, tensor: jnp.ndarray, mapper: Any, target_var: str):
        """Log detailed tensor construction analysis."""
        print(f"\n🧮 TENSOR CONSTRUCTION ANALYSIS:")
        print(f"  Tensor shape: {tensor.shape}")
        print(f"  Mapper variables: {mapper.variables}")
        print(f"  Target variable: {target_var} (index: {mapper.target_idx})")
        
        # Check each channel
        channel_names = ['Values', 'Target', 'Interv', 'Probs', 'Recency']
        for ch_idx in range(min(tensor.shape[2], len(channel_names))):
            ch_name = channel_names[ch_idx]
            channel = tensor[:, :, ch_idx]
            print(f"  Channel {ch_idx} ({ch_name}):")
            print(f"    Shape: {channel.shape}")
            print(f"    Mean: {jnp.mean(channel):.6f}")
            print(f"    Std: {jnp.std(channel):.6f}")
            print(f"    Range: [{jnp.min(channel):.3f}, {jnp.max(channel):.3f}]")
            print(f"    Last timestep: {channel[-1]}")
            
            # Specific checks
            if ch_name == 'Values':
                var_means = jnp.mean(channel, axis=0)
                var_std_across_vars = jnp.std(var_means)
                print(f"    Variable distinguishability: {var_std_across_vars:.6f}")
                if var_std_across_vars < 0.01:
                    print(f"    ❌ CRITICAL: Variables look identical!")
                else:
                    print(f"    ✅ Variables are distinguishable")
            
            elif ch_name == 'Probs':
                unique_vals = jnp.unique(channel)
                print(f"    Unique probability values: {len(unique_vals)}")
                if len(unique_vals) <= 2:
                    print(f"    ❌ CRITICAL: Posteriors not varying! All: {unique_vals}")
                else:
                    print(f"    ✅ Posteriors show variation")
        
        # Overall tensor analysis
        total_variation = jnp.std(tensor)
        print(f"\n📊 OVERALL TENSOR ANALYSIS:")
        print(f"  Total variation (std): {total_variation:.6f}")
        print(f"  Non-zero elements: {jnp.sum(tensor != 0.0)}/{tensor.size}")
        print(f"  Informative ratio: {jnp.sum(tensor != 0.0)/tensor.size:.3f}")
        
        if total_variation < 0.1:
            print(f"  ⚠️ WARNING: Tensor is mostly zeros - check data population")
    
    def log_candidates_with_rewards(self, grpo_batch_data: Dict, mapper: Any, 
                                   target_var: str, scm: Any, reward_config: Any):
        """Log detailed candidate breakdown with reward components."""
        print(f"\n[GRPO CANDIDATES]:")
        
        for i in range(len(grpo_batch_data['actions'])):
            action = grpo_batch_data['actions'][i] 
            reward = grpo_batch_data['rewards'][i]
            var_name = mapper.get_name(int(action['variable']))
            
            # Get target value from outcome
            if i < len(grpo_batch_data['intervention_details']):
                intervention_info = grpo_batch_data['intervention_details'][i]
                if 'samples' in intervention_info and intervention_info['samples']:
                    target_value = get_values(intervention_info['samples'][0]).get(target_var, 0.0)
                    
                    # Compute components for display
                    target_component = -target_value if self.optimization_direction == "MINIMIZE" else target_value
                    parent_component = 1.0 if var_name in set(get_parents(scm, target_var)) else 0.0
                    
                    # Show weighted breakdown
                    target_weighted = reward_config.target_weight * target_component
                    parent_weighted = reward_config.parent_weight * parent_component
                    
                    print(f"  Candidate {i+1}: {var_name} = {action['value']:.3f} → TARGET = {target_value:.3f}")
                    print(f"    Target: {target_component:.3f} × {reward_config.target_weight:.2f} = {target_weighted:.3f}")
                    print(f"    Parent: {parent_component:.1f} × {reward_config.parent_weight:.2f} = {parent_weighted:.3f}")
                    print(f"    TOTAL REWARD: {reward:.3f}")
    
    def log_grpo_advantages(self, rewards: jnp.ndarray):
        """Log GRPO advantage analysis."""
        baseline = jnp.mean(rewards)
        advantages = rewards - baseline
        best_idx = jnp.argmax(advantages)
        
        print(f"\n📊 GRPO Advantages:")
        print(f"  Baseline: {baseline:.3f}")
        print(f"  Advantages: {[f'{float(a):+.3f}' for a in advantages]}")
        print(f"  Best: #{best_idx+1} (advantage: {advantages[best_idx]:+.3f})")
        
        # Advantage analysis
        advantage_std = jnp.std(advantages)
        max_advantage = jnp.max(jnp.abs(advantages))
        meaningful_advantages = jnp.sum(jnp.abs(advantages) > 0.1)
        
        print(f"\n🔍 ADVANTAGE ANALYSIS:")
        print(f"  Advantage std: {advantage_std:.6f}")
        print(f"  Max |advantage|: {max_advantage:.3f}")
        print(f"  Advantage range: [{jnp.min(advantages):.3f}, {jnp.max(advantages):.3f}]")
        print(f"  Meaningful advantages (>0.1): {meaningful_advantages}/{len(advantages)}")
        
        if advantage_std < 0.01:
            print(f"  ❌ CRITICAL: Advantages too uniform - no learning signal!")
        elif max_advantage > 10.0:
            print(f"  ⚠️ WARNING: Very large advantages - may cause instability")
        else:
            print(f"  ✅ Advantage distribution looks reasonable")
    
    def log_gradient_analysis(self, grads: Dict, rewards: jnp.ndarray):
        """Log detailed gradient flow analysis."""
        print(f"\n🔬 DETAILED GRADIENT FLOW ANALYSIS:")
        
        # Mathematical flow
        baseline = jnp.mean(rewards)
        advantages = rewards - baseline
        
        print(f"  📊 Mathematical Flow:")
        print(f"    Rewards: {[f'{float(r):.3f}' for r in rewards]}")
        print(f"    Baseline: {baseline:.3f}")
        print(f"    Advantages: {[f'{float(a):+.3f}' for a in advantages]}")
        print(f"    Advantage signal strength: {jnp.std(advantages):.6f}")
        
        # Gradient distribution analysis
        var_head_total = 0.0
        val_head_total = 0.0
        other_total = 0.0
        
        flat_grads = jax.tree_util.tree_flatten_with_path(grads)[0]
        
        for path_info, grad in flat_grads:
            if not hasattr(grad, 'shape'):
                continue
                
            norm = jnp.linalg.norm(grad)
            param_path = '/'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path_info)
            
            if 'var_mlp' in param_path or 'variable' in param_path:
                var_head_total += norm
            elif 'val_mlp' in param_path or 'value' in param_path:
                val_head_total += norm
            else:
                other_total += norm
        
        total_grad_norm = var_head_total + val_head_total + other_total
        
        print(f"  📊 Gradient Flow Distribution:")
        if total_grad_norm > 0:
            print(f"    Variable head: {var_head_total:.6f} ({var_head_total/total_grad_norm:.1%})")
            print(f"    Value head: {val_head_total:.6f} ({val_head_total/total_grad_norm:.1%})")
            print(f"    Other layers: {other_total:.6f} ({other_total/total_grad_norm:.1%})")
        print(f"    Total norm: {total_grad_norm:.6f}")
        
        # Critical checks
        if total_grad_norm < 1e-6:
            print(f"  ❌ CRITICAL: Total gradients near zero - no learning possible!")
        elif var_head_total / (total_grad_norm + 1e-8) < 0.05:
            print(f"  ❌ CRITICAL: Variable head gets <5% gradients - can't learn selection!")
        elif jnp.std(advantages) < 0.01:
            print(f"  ❌ CRITICAL: Advantage signal too weak - uniform rewards!")
        else:
            print(f"  ✅ Gradient flow looks reasonable")
    
    def log_policy_evolution(self, old_logits: jnp.ndarray, new_logits: jnp.ndarray):
        """Log how policy output is changing."""
        old_probs = jax.nn.softmax(old_logits)
        new_probs = jax.nn.softmax(new_logits)
        
        print(f"\n🔬 POST-UPDATE DIAGNOSTICS:")
        print(f"  New var logits: {new_logits}")
        print(f"  New var probs: {new_probs}")
        
        # Evolution analysis
        logit_change = new_logits - old_logits
        prob_change = new_probs - old_probs
        
        print(f"  Logit changes: {logit_change}")
        print(f"  Prob changes: {prob_change}")
        print(f"  Max logit change: {jnp.max(jnp.abs(logit_change)):.6f}")
        print(f"  Max prob change: {jnp.max(jnp.abs(prob_change)):.6f}")
        
        # Check meaningfulness
        if jnp.max(jnp.abs(logit_change)) < 1e-4:
            print(f"  ❌ CRITICAL: Logits barely changing!")
        elif jnp.max(jnp.abs(prob_change)) < 1e-3:
            print(f"  ❌ CRITICAL: Probabilities barely changing!")
        else:
            print(f"  ✅ Policy output is evolving")
    
    def log_loss_analysis(self, loss_info: Dict):
        """Log loss component analysis."""
        print(f"\n📊 LOSS COMPONENT ANALYSIS:")
        total_loss = float(loss_info.get('total_loss', 0))
        policy_loss = float(loss_info.get('policy_loss', 0))
        entropy_loss = float(loss_info.get('entropy_loss', 0))
        
        print(f"  Total loss: {total_loss:.6f}")
        print(f"  Policy loss: {policy_loss:.6f}")
        print(f"  Entropy loss: {entropy_loss:.6f}")
        
        if abs(total_loss) > 1e-8:
            policy_ratio = abs(policy_loss) / (abs(total_loss) + 1e-8)
            entropy_ratio = abs(entropy_loss) / (abs(total_loss) + 1e-8)
            
            if entropy_ratio > 0.8:
                print(f"  ❌ CRITICAL: Entropy dominates ({entropy_ratio:.1%}) - may prevent learning!")
            elif policy_ratio > 0.8:
                print(f"  ✅ Policy loss dominates ({policy_ratio:.1%}) - good for learning")
    
    def log_reinforce_comparison(self, grpo_loss: float, rewards: jnp.ndarray, 
                               new_log_probs: jnp.ndarray):
        """Compare GRPO vs REINFORCE loss for debugging."""
        print(f"\n🧪 REINFORCE COMPARISON:")
        
        reinforce_loss = -jnp.mean(rewards * new_log_probs)
        
        print(f"  GRPO policy loss: {grpo_loss:.6f}")
        print(f"  REINFORCE loss: {reinforce_loss:.6f}")
        print(f"  Loss difference: {grpo_loss - reinforce_loss:.6f}")
        print(f"  REINFORCE signal strength: {jnp.abs(reinforce_loss):.6f}")
        
        if jnp.abs(reinforce_loss) < 1e-6:
            print(f"  ❌ CRITICAL: Even REINFORCE has no signal!")
        elif jnp.abs(grpo_loss) < jnp.abs(reinforce_loss) * 0.1:
            print(f"  ⚠️ GRPO loss much weaker than REINFORCE - ratio issue?")
        else:
            print(f"  ✅ GRPO and REINFORCE losses similar magnitude")
    
    def track_target_value(self, episode_count: int, target_var: str, target_val: float, 
                          scm_name: str, scm: Any):
        """Track target value progression."""
        # Initialize target values for this SCM
        if scm_name not in self.initial_target_values:
            from ..mechanisms.linear import sample_from_linear_scm
            initial_samples = sample_from_linear_scm(scm, n_samples=100, seed=42)
            initial_values = [s.get(target_var, 0.0) for s in initial_samples]
            self.initial_target_values[scm_name] = float(np.mean(initial_values))
            self.best_target_values[scm_name] = self.initial_target_values[scm_name]
        
        # Update best value
        if self.optimization_direction == "MINIMIZE":
            if target_val < self.best_target_values[scm_name]:
                self.best_target_values[scm_name] = target_val
        else:
            if target_val > self.best_target_values[scm_name]:
                self.best_target_values[scm_name] = target_val
        
        # Store in history
        self.target_value_history.append({
            'episode': episode_count,
            'scm': scm_name,
            'target_value': target_val,
            'best_so_far': self.best_target_values[scm_name],
            'initial': self.initial_target_values[scm_name],
            'improvement': self.best_target_values[scm_name] - self.initial_target_values[scm_name]
        })
    
    def log_training_progress(self, episode: int, episode_metrics: List[Dict], scm_name: str):
        """Log periodic training progress."""
        if episode % 10 == 0:
            recent_rewards = [m['mean_reward'] for m in episode_metrics[-10:]]
            mean_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
            logger.info(f"Episode {episode}: mean_reward={mean_reward:.4f}, current_scm={scm_name}")
            
            # Report target improvement
            if scm_name in self.best_target_values:
                improvement = self.best_target_values[scm_name] - self.initial_target_values[scm_name]
                logger.info(f"  Target - Initial: {self.initial_target_values[scm_name]:.3f}, "
                           f"Best: {self.best_target_values[scm_name]:.3f}, "
                           f"Improvement: {improvement:.3f}")
    
    def log_final_summary(self):
        """Log final training summary."""
        if not self.target_value_history:
            return
            
        logger.info("\n" + "="*70)
        logger.info("TARGET VALUE IMPROVEMENT SUMMARY")
        logger.info("="*70)
        
        total_improvements = []
        for scm_name in self.best_target_values:
            if scm_name in self.initial_target_values:
                improvement = self.best_target_values[scm_name] - self.initial_target_values[scm_name]
                total_improvements.append(improvement)
                logger.info(f"\n{scm_name}:")
                logger.info(f"  Initial: {self.initial_target_values[scm_name]:.4f}")
                logger.info(f"  Best: {self.best_target_values[scm_name]:.4f}")
                logger.info(f"  Improvement: {improvement:.4f}")
        
        if total_improvements:
            avg_improvement = sum(total_improvements) / len(total_improvements)
            improved_count = sum(1 for imp in total_improvements if imp < 0)
            logger.info(f"\nOverall: {improved_count}/{len(total_improvements)} SCMs improved")
            logger.info(f"Average improvement: {avg_improvement:.4f}")