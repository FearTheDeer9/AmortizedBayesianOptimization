"""
Clean GRPO trainer optimized for strong gradients.

Design principles:
1. Start with REINFORCE (guaranteed strong gradients)
2. Add PPO only if REINFORCE works
3. Focus on gradient magnitude over algorithm sophistication
4. Minimal abstraction layers
"""

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import haiku as hk
from typing import Dict, List, Any, Callable, Tuple, Optional
import logging

from .data_pipeline import SimpleBuffer, create_clean_tensor
from .policy_network import create_clean_policy

logger = logging.getLogger(__name__)


class CleanGRPOTrainer:
    """
    Clean GRPO trainer focused on gradient efficiency.
    
    Core loop:
    1. Generate intervention candidates
    2. Compute rewards (reuse existing composite system)
    3. Compute advantages 
    4. Update policy (start with REINFORCE)
    5. Add best intervention to buffer
    """
    
    def __init__(self, 
                 policy_architecture: str = "clean",
                 hidden_dim: int = 256,
                 learning_rate: float = 1e-2,
                 group_size: int = 8,
                 max_history: int = 50):
        """Initialize clean GRPO trainer."""
        
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.group_size = group_size
        self.max_history = max_history
        
        # Create policy
        if policy_architecture == "clean":
            policy_fn = create_clean_policy(hidden_dim)
        elif policy_architecture == "ultra_simple":
            from .policy_network import create_ultra_simple_policy
            policy_fn = create_ultra_simple_policy(hidden_dim)
        else:
            raise ValueError(f"Unknown architecture: {policy_architecture}")
        
        self.policy = hk.transform(policy_fn)
        
        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate)
        
        # Will be set during training
        self.policy_params = None
        self.optimizer_state = None
        
        logger.info(f"Initialized CleanGRPOTrainer:")
        logger.info(f"  Architecture: {policy_architecture}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Group size: {group_size}")
    
    def initialize_policy(self, n_vars: int):
        """Initialize policy parameters with correct dimensions."""
        # Create dummy input for initialization
        dummy_tensor = jnp.zeros((self.max_history, n_vars, 4))
        dummy_key = random.PRNGKey(42)
        
        # Initialize parameters
        self.policy_params = self.policy.init(dummy_key, dummy_tensor, 0)
        self.optimizer_state = self.optimizer.init(self.policy_params)
        
        # Test gradient flow
        from .policy_network import test_policy_gradient_flow
        gradient_ok = test_policy_gradient_flow(self.policy.apply, self.policy_params, dummy_tensor, 0)
        
        if not gradient_ok:
            logger.warning("Policy gradient flow test failed - may have learning issues")
        
        logger.info(f"Policy initialized for {n_vars} variables")
    
    def train_on_scm(self, scm, target_variable: str, max_interventions: int = 10):
        """
        Train policy on single SCM with clean GRPO.
        
        Returns learning metrics for comparison with current system.
        """
        variables = ['X', 'Y', 'Z']  # For deterministic SCM
        target_idx = variables.index(target_variable)
        
        # Initialize policy if needed
        if self.policy_params is None:
            self.initialize_policy(len(variables))
        
        # Initialize buffer with observations
        buffer = SimpleBuffer()
        
        # Add initial observations (reuse current SCM sampling)
        from ..experiments.deterministic_scm_test import sample_observations
        initial_samples = sample_observations(scm, n_samples=10)
        for sample in initial_samples:
            buffer.add_sample(sample, is_intervention=False)
        
        # Track learning metrics
        learning_metrics = {
            'variable_probabilities': [],
            'target_values': [],
            'gradient_norms': [],
            'policy_losses': []
        }
        
        print(f"🚀 CLEAN GRPO TRAINING:")
        print(f"  SCM: {target_variable} ← X (deterministic)")
        print(f"  Max interventions: {max_interventions}")
        print(f"  Group size: {self.group_size}")
        
        # Main training loop
        for intervention_idx in range(max_interventions):
            print(f"\n--- Clean Intervention {intervention_idx+1}/{max_interventions} ---")
            
            # Run single intervention
            metrics = self._run_clean_intervention(
                buffer, scm, variables, target_variable, target_idx
            )
            
            # Track metrics
            learning_metrics['variable_probabilities'].append(metrics['var_probs'])
            learning_metrics['target_values'].append(metrics['target_value'])
            learning_metrics['gradient_norms'].append(metrics['grad_norm'])
            learning_metrics['policy_losses'].append(metrics['policy_loss'])
            
            # Log progress
            var_probs = metrics['var_probs']
            x_prob = var_probs[0] if variables[0] == 'X' else var_probs[variables.index('X')]
            print(f"  X probability: {x_prob:.3f} (should increase for parent learning)")
            print(f"  Gradient norm: {metrics['grad_norm']:.6f}")
            print(f"  Target value: {metrics['target_value']:.3f}")
        
        # Analyze learning
        self._analyze_learning_performance(learning_metrics, variables)
        
        return learning_metrics
    
    def _run_clean_intervention(self, buffer: SimpleBuffer, scm, variables: List[str], 
                               target_variable: str, target_idx: int) -> Dict[str, Any]:
        """Run single intervention with clean GRPO."""
        
        # Create tensor from current buffer
        tensor, var_order = create_clean_tensor(
            buffer, target_variable, max_history=self.max_history, surrogate_fn=None
        )
        
        # Generate candidates
        candidates = []
        for _ in range(self.group_size):
            candidate = self._generate_candidate(tensor, target_idx, scm, variables)
            candidates.append(candidate)
        
        # Compute rewards (reuse existing system)
        rewards = jnp.array([c['reward'] for c in candidates])
        
        # CLEAN GRPO UPDATE
        old_params = self.policy_params
        
        # Compute pure REINFORCE loss (start simple)
        def reinforce_loss_fn(params):
            total_loss = 0.0
            for candidate in candidates:
                # Get policy output for this candidate's state
                policy_output = self.policy.apply(params, random.PRNGKey(42), tensor, target_idx)
                var_probs = jax.nn.softmax(policy_output['variable_logits'])
                
                # Log probability of selected action
                selected_var_idx = candidate['selected_var_idx']
                log_prob = jnp.log(var_probs[selected_var_idx] + 1e-8)
                
                # REINFORCE: loss = -reward * log_prob
                loss_contribution = -candidate['reward'] * log_prob
                total_loss += loss_contribution
            
            return total_loss / len(candidates)  # Average over batch
        
        # Compute gradients
        loss_value, grads = jax.value_and_grad(reinforce_loss_fn)(self.policy_params)
        
        # Analyze gradient magnitude
        grad_norms = jax.tree.map(jnp.linalg.norm, grads)
        total_grad_norm = sum(jax.tree_leaves(grad_norms))
        
        print(f"  📊 Clean GRPO Metrics:")
        print(f"    Policy loss: {loss_value:.6f}")
        print(f"    Gradient norm: {total_grad_norm:.6f}")
        
        # Apply gradients
        updates, self.optimizer_state = self.optimizer.update(grads, self.optimizer_state, self.policy_params)
        self.policy_params = optax.apply_updates(self.policy_params, updates)
        
        # Get updated policy probabilities
        new_output = self.policy.apply(self.policy_params, random.PRNGKey(42), tensor, target_idx)
        new_var_probs = jax.nn.softmax(new_output['variable_logits'])
        
        # Select best candidate and add to buffer
        best_idx = jnp.argmax(rewards)
        best_candidate = candidates[best_idx]
        buffer.add_sample(best_candidate['outcome'], is_intervention=True)
        
        return {
            'var_probs': new_var_probs,
            'target_value': best_candidate['target_value'],
            'grad_norm': total_grad_norm,
            'policy_loss': loss_value,
            'rewards': rewards,
            'advantages': rewards - jnp.mean(rewards)
        }
    
    def _generate_candidate(self, tensor: jnp.ndarray, target_idx: int, 
                           scm, variables: List[str]) -> Dict[str, Any]:
        """Generate single intervention candidate."""
        
        # Get policy output
        policy_output = self.policy.apply(self.policy_params, random.PRNGKey(42), tensor, target_idx)
        var_probs = jax.nn.softmax(policy_output['variable_logits'])
        value_params = policy_output['value_params']
        
        # Sample variable
        var_key = random.PRNGKey(42)  # Fixed for reproducibility in testing
        selected_var_idx = random.categorical(var_key, jnp.log(var_probs + 1e-8))
        selected_var = variables[selected_var_idx]
        
        # Sample value
        val_key = random.PRNGKey(43)
        mean = value_params[selected_var_idx, 0] 
        log_std = value_params[selected_var_idx, 1]
        std = jnp.exp(log_std)
        intervention_value = mean + std * random.normal(val_key)
        
        # Apply intervention to SCM (reuse existing system)
        from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
        from src.causal_bayes_opt.environments.sampling import sample_with_intervention
        
        intervention = create_perfect_intervention(
            targets=frozenset([selected_var]),
            values={selected_var: float(intervention_value)}
        )
        
        outcome_samples = sample_with_intervention(scm, intervention, n_samples=1, seed=42)
        outcome = outcome_samples[0] if outcome_samples else {}
        
        # Compute reward (reuse existing composite system)
        from src.causal_bayes_opt.acquisition.composite_reward import compute_parent_reward
        
        # Simple reward for testing: parent selection
        reward = compute_parent_reward(scm, selected_var, target_variable)
        target_value = outcome.get(target_variable, 0.0)
        
        return {
            'selected_var': selected_var,
            'selected_var_idx': int(selected_var_idx),
            'intervention_value': float(intervention_value),
            'outcome': outcome,
            'reward': float(reward),
            'target_value': float(target_value),
            'log_prob': float(jnp.log(var_probs[selected_var_idx] + 1e-8))
        }
    
    def _analyze_learning_performance(self, metrics: Dict, variables: List[str]):
        """Analyze if learning is happening efficiently."""
        
        var_probs_history = metrics['variable_probabilities']
        target_history = metrics['target_values']
        grad_norms = metrics['gradient_norms']
        
        print(f"\n📊 CLEAN GRPO LEARNING ANALYSIS:")
        
        # Variable selection learning
        if len(var_probs_history) > 1:
            x_idx = variables.index('X')
            initial_x_prob = var_probs_history[0][x_idx]
            final_x_prob = var_probs_history[-1][x_idx]
            x_improvement = final_x_prob - initial_x_prob
            
            print(f"  X probability: {initial_x_prob:.3f} → {final_x_prob:.3f} ({x_improvement:+.3f})")
            
            if x_improvement > 0.2:
                print(f"  ✅ Strong parent learning!")
            elif x_improvement > 0.05:
                print(f"  ⚠️ Moderate parent learning")
            else:
                print(f"  ❌ Minimal parent learning")
        
        # Gradient efficiency
        avg_grad_norm = jnp.mean(jnp.array(grad_norms))
        print(f"  Average gradient norm: {avg_grad_norm:.6f}")
        
        if avg_grad_norm > 0.01:
            print(f"  ✅ Strong gradients - efficient learning")
        elif avg_grad_norm > 0.001:
            print(f"  ⚠️ Moderate gradients - acceptable")
        else:
            print(f"  ❌ Weak gradients - efficiency issue")
        
        # Learning speed assessment
        interventions_to_good_learning = len(var_probs_history)
        for i, probs in enumerate(var_probs_history):
            if probs[x_idx] > 0.8:  # 80% X preference = good learning
                interventions_to_good_learning = i + 1
                break
        
        print(f"  Interventions to good learning: {interventions_to_good_learning}")
        
        if interventions_to_good_learning <= 5:
            print(f"  ✅ Fast learning!")
        elif interventions_to_good_learning <= 10:
            print(f"  ⚠️ Moderate learning speed")
        else:
            print(f"  ❌ Slow learning - needs optimization")


def compare_with_current_implementation(clean_metrics: Dict, current_metrics: Dict):
    """Compare clean implementation with current complex system."""
    
    print(f"\n🏆 IMPLEMENTATION COMPARISON:")
    print(f"{'Metric':<25} {'Clean':<15} {'Current':<15} {'Improvement'}")
    print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*12}")
    
    # Extract comparable metrics
    clean_grad_avg = jnp.mean(jnp.array(clean_metrics['gradient_norms']))
    current_grad_avg = 0.0004  # From current system diagnostic
    
    clean_x_improvement = (clean_metrics['variable_probabilities'][-1][0] - 
                          clean_metrics['variable_probabilities'][0][0])
    current_x_improvement = 0.297  # From current system (X:0.496 → X:0.793)
    
    # Gradient comparison
    grad_improvement = clean_grad_avg / current_grad_avg if current_grad_avg > 0 else float('inf')
    print(f"{'Gradient Norm':<25} {clean_grad_avg:<15.6f} {current_grad_avg:<15.6f} {grad_improvement:<12.1f}x")
    
    # Learning speed comparison
    learning_improvement = clean_x_improvement / current_x_improvement if current_x_improvement > 0 else float('inf')
    print(f"{'X Probability Change':<25} {clean_x_improvement:<15.3f} {current_x_improvement:<15.3f} {learning_improvement:<12.1f}x")
    
    # Overall assessment
    if grad_improvement > 10 and learning_improvement > 2:
        print(f"\n✅ CLEAN IMPLEMENTATION CLEARLY SUPERIOR!")
        print(f"  → Integrate clean version into main system")
    elif grad_improvement > 3:
        print(f"\n⚠️ CLEAN IMPLEMENTATION BETTER (gradients)")
        print(f"  → Consider adopting for gradient efficiency")
    else:
        print(f"\n❌ NO CLEAR IMPROVEMENT")
        print(f"  → Issue may be more fundamental")


def validate_grpo_math(rewards: jnp.ndarray, advantages: jnp.ndarray, 
                      log_probs: jnp.ndarray) -> bool:
    """Validate GRPO mathematical components."""
    
    print(f"\n🔬 GRPO MATH VALIDATION:")
    print(f"  Rewards: {[f'{r:.3f}' for r in rewards[:8]]}...")
    print(f"  Advantages: {[f'{a:+.3f}' for a in advantages[:8]]}...")
    print(f"  Log probs: {[f'{lp:.3f}' for lp in log_probs[:8]]}...")
    
    # Check advantage properties
    advantage_mean = jnp.mean(advantages)
    advantage_std = jnp.std(advantages)
    
    print(f"  Advantage mean: {advantage_mean:.6f} (should be ~0)")
    print(f"  Advantage std: {advantage_std:.6f} (should be >0.1)")
    
    # Check for valid learning signal
    if jnp.abs(advantage_mean) > 0.01:
        print(f"  ⚠️ Advantages not centered - check baseline computation")
    
    if advantage_std < 0.01:
        print(f"  ❌ No advantage signal - uniform rewards")
        return False
    
    if jnp.any(jnp.isnan(log_probs)) or jnp.any(jnp.isinf(log_probs)):
        print(f"  ❌ Invalid log probabilities")
        return False
    
    print(f"  ✅ GRPO math components look valid")
    return True