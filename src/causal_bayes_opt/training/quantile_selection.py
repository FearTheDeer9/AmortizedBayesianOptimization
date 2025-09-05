"""
Quantile-based intervention selection for unified variable and value prediction.

This module handles the conversion from quantile scores to actual intervention
values using buffer statistics, maintaining gradient flow through the policy.
"""

import jax
import jax.numpy as jnp
import jax.random as random
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def get_deterministic_percentiles(scm, variable_name: str, quantile_indices: List[int]) -> List[float]:
    """
    Get deterministic percentile values from SCM variable ranges.
    
    Instead of using historical buffer data, use the SCM's defined variable ranges
    to create deterministic percentile values. This ensures consistent mapping
    from quantile selection to intervention values.
    
    Args:
        scm: SCM with variable_ranges metadata
        variable_name: Variable to get percentiles for  
        quantile_indices: [0, 1, 2] for [25%, 50%, 75%]
        
    Returns:
        List of percentile values corresponding to middle of range segments
        
    Raises:
        ValueError: If variable ranges not defined in SCM metadata
    """
    metadata = scm.get('metadata', {})
    ranges = metadata.get('variable_ranges', {})
    
    if variable_name not in ranges:
        available_vars = list(ranges.keys()) if ranges else "None"
        raise ValueError(
            f"No range defined for variable '{variable_name}' in SCM metadata. "
            f"Available ranges: {available_vars}. "
            f"SCM must define variable_ranges for deterministic quantile mapping."
        )
    
    var_min, var_max = ranges[variable_name]
    range_size = var_max - var_min
    
    if range_size <= 0:
        raise ValueError(f"Invalid range for {variable_name}: [{var_min}, {var_max}]")
    
    # Calculate middle points of 3 equal segments
    percentile_values = []
    for q_idx in quantile_indices:
        if q_idx == 0:    # 25th percentile - middle of first third
            value = var_min + range_size / 6
        elif q_idx == 1:  # 50th percentile - middle of range
            value = var_min + range_size / 2
        elif q_idx == 2:  # 75th percentile - middle of last third
            value = var_min + 5 * range_size / 6
        else:
            raise ValueError(f"Invalid quantile index: {q_idx}. Must be 0, 1, or 2.")
            
        percentile_values.append(float(value))
    
    # Log only the calculated percentiles
    quantile_names = ['25%', '50%', '75%']
    debug_pairs = [f"{quantile_names[quantile_indices[i]]}={percentile_values[i]:.3f}" 
                   for i in range(len(quantile_indices))]
    logger.debug(f"Deterministic percentiles for {variable_name} range [{var_min}, {var_max}]: {', '.join(debug_pairs)}")
    
    return percentile_values


def select_quantile_intervention(
    policy_output: Dict[str, jnp.ndarray],
    buffer,  # ExperienceBuffer (still needed for compatibility)
    scm,     # SCM with variable_ranges metadata
    variables: List[str],
    target_variable: str,
    key: random.PRNGKey,
    fixed_std: float = 1.0
) -> Tuple[str, float, float, Dict[str, Any]]:
    """
    Select intervention based on quantile scores and map to actual values.
    
    Args:
        policy_output: Policy output containing 'quantile_scores' [n_vars, 3]
        buffer: ExperienceBuffer with historical data
        variables: List of variable names in order
        target_variable: Target variable name (for debugging)
        key: Random key for sampling
        fixed_std: Standard deviation for sampling around percentile
        
    Returns:
        Tuple of (selected_var, intervention_value, log_prob, debug_info)
    """
    quantile_scores = policy_output['quantile_scores']  # [n_vars, 3]
    
    # Create probability distribution over all quantile options
    flat_scores = quantile_scores.flatten()  # [n_vars * 3]
    flat_probs = jax.nn.softmax(flat_scores)  # DIFFERENTIABLE probabilities
    
    # For exploration: sample from distribution
    # For exploitation: use argmax but with proper log_prob
    key, select_key = random.split(key)
    
    # Use categorical sampling to get proper log_prob (essential for GRPO)
    winner_idx = int(random.categorical(select_key, flat_scores))
    selected_var_idx, selected_quantile_idx = divmod(winner_idx, 3)
    
    selected_var = variables[selected_var_idx]
    quantile_names = ['25%', '50%', '75%']
    selected_quantile_name = quantile_names[selected_quantile_idx]
    quantile_percentiles = [0.25, 0.5, 0.75]
    target_percentile = quantile_percentiles[selected_quantile_idx]
    
    # Get deterministic percentile from SCM variable ranges
    try:
        percentiles = get_deterministic_percentiles(scm, selected_var, [selected_quantile_idx])
        percentile_value = percentiles[0]
        
        # Get SCM range info for debugging
        metadata = scm.get('metadata', {})
        ranges = metadata.get('variable_ranges', {})
        var_min, var_max = ranges.get(selected_var, (0.0, 0.0))
        
        # Also get buffer statistics for comparison
        try:
            var_stats = buffer.get_variable_statistics(selected_var)
        except:
            var_stats = {'count': 0}
        
    except Exception as e:
        logger.error(f"Error getting deterministic percentiles for {selected_var}: {e}")
        raise  # Don't silently handle - force error as requested
    
    # Sample around percentile with fixed std
    key, val_key = random.split(key)
    intervention_value = percentile_value + fixed_std * random.normal(val_key)
    
    # CRITICAL: Proper log probability computation for GRPO
    log_prob = float(jnp.log(flat_probs[winner_idx] + 1e-8))
    
    # Debug information with deterministic percentile details
    debug_info = {
        'selected_var': selected_var,
        'selected_var_idx': selected_var_idx,
        'selected_quantile': selected_quantile_name,
        'selected_quantile_idx': selected_quantile_idx,
        'winner_idx': winner_idx,  # Store the actual flat index used
        'quantile_score': float(flat_scores[winner_idx]),
        'percentile_value': percentile_value,
        'intervention_value': float(intervention_value),
        'log_prob': log_prob,
        # Deterministic range info
        'scm_range_min': var_min,
        'scm_range_max': var_max,
        'deterministic_mapping': True,
        # Buffer stats for comparison
        'var_history_count': var_stats.get('count', 0),
        'var_p25': var_stats.get('p25', 0.0),
        'var_p50': var_stats.get('p50', 0.0),
        'var_p75': var_stats.get('p75', 0.0)
    }
    
    debug_info.update({
        # Full scores and probabilities for ALL variables
        'full_quantile_scores': quantile_scores.tolist(),  # [n_vars, 3] matrix
        'full_flat_scores': flat_scores.tolist(),  # [n_vars * 3] flattened
        'full_flat_probs': flat_probs.tolist(),  # [n_vars * 3] probabilities
        'variable_max_probs': {},  # Max probability per variable
        'variable_best_quantiles': {},  # Best quantile per variable
    })
    
    # Compute per-variable max probabilities and best quantiles
    probs_matrix = flat_probs.reshape(len(variables), 3)
    scores_matrix = flat_scores.reshape(len(variables), 3)
    quantile_names_list = ['25%', '50%', '75%']
    
    for i, var in enumerate(variables):
        var_probs = probs_matrix[i]
        var_scores = scores_matrix[i]
        max_prob = float(jnp.max(var_probs))
        best_q_idx = int(jnp.argmax(var_scores))
        debug_info['variable_max_probs'][var] = max_prob
        debug_info['variable_best_quantiles'][var] = quantile_names_list[best_q_idx]
    
    return selected_var, float(intervention_value), log_prob, debug_info


def create_compatible_policy_output(
    policy_output: Dict[str, jnp.ndarray],
    selected_var_idx: int,
    intervention_value: float,
    log_prob: float
) -> Dict[str, jnp.ndarray]:
    """
    Create compatible variable_logits and value_params for existing reward system.
    
    This maintains interface compatibility while using quantile-based selection.
    """
    quantile_scores = policy_output['quantile_scores']  # [n_vars, 3]
    n_vars = quantile_scores.shape[0]
    
    # Variable logits: Use max quantile score per variable
    variable_logits = jnp.max(quantile_scores, axis=1)  # [n_vars]
    
    # Value params: Set intervention value for selected variable
    value_means = jnp.zeros(n_vars)
    value_means = value_means.at[selected_var_idx].set(intervention_value)
    
    value_stds = jnp.ones(n_vars) * jnp.log(1.0)  # Fixed std
    value_params = jnp.stack([value_means, value_stds], axis=1)  # [n_vars, 2]
    
    return {
        'variable_logits': variable_logits,
        'value_params': value_params,
        'quantile_scores': quantile_scores  # Keep original for debugging
    }


def log_quantile_details(quantile_scores: jnp.ndarray, debug_info: Dict[str, Any], 
                        variables: List[str], target_variable: str, scm=None):
    """Enhanced logging for quantile architecture."""
    
    print(f"\n🎯 QUANTILE ARCHITECTURE DETAILS:")
    print(f"  🔍 VARIABLE ORDERING DEBUG:")
    print(f"    Variables list: {variables}")
    print(f"    Target variable: {target_variable}")
    print(f"    Target index should be: {variables.index(target_variable) if target_variable in variables else 'NOT FOUND'}")
    
    print(f"  Full quantile scores matrix [variables × percentiles]:")
    
    quantile_names = ['25%', '50%', '75%']
    for i, var in enumerate(variables):
        scores = quantile_scores[i]
        score_str = ", ".join([f"{q}:{s:.3f}" for q, s in zip(quantile_names, scores)])
        marker = "🎯" if var == debug_info['selected_var'] else "  "
        
        # Check if this variable is masked (target)
        is_masked = all(abs(s + 10.0) < 0.01 for s in scores)  # All values ≈ -10.0
        mask_indicator = " [MASKED]" if is_masked else ""
        
        print(f"    {marker} {var}: [{score_str}]{mask_indicator}")
    
    print(f"\n  Selection details:")
    print(f"    Winner: {debug_info['selected_var']} at {debug_info['selected_quantile']} (score: {debug_info['quantile_score']:.3f})")
    print(f"    Selection probability: {jnp.exp(debug_info['log_prob']):.3f}")
    
    # Show deterministic percentile mapping
    if debug_info.get('deterministic_mapping', False):
        print(f"    🎯 DETERMINISTIC PERCENTILE MAPPING:")
        print(f"      SCM range: [{debug_info['scm_range_min']:.1f}, {debug_info['scm_range_max']:.1f}]")
        
        # Calculate all percentiles for this variable
        try:
            all_percentiles = get_deterministic_percentiles(scm, debug_info['selected_var'], [0, 1, 2])
            print(f"      25%: {all_percentiles[0]:.3f}, 50%: {all_percentiles[1]:.3f}, 75%: {all_percentiles[2]:.3f}")
            print(f"      Selected {debug_info['selected_quantile']}: {debug_info['percentile_value']:.3f}")
        except:
            print(f"      Selected {debug_info['selected_quantile']}: {debug_info['percentile_value']:.3f}")
    else:
        print(f"    Historical {debug_info['selected_quantile']} for {debug_info['selected_var']}: {debug_info['percentile_value']:.3f}")
        if debug_info['var_history_count'] > 3:
            print(f"    Variable percentiles: 25%={debug_info['var_p25']:.3f}, 50%={debug_info['var_p50']:.3f}, 75%={debug_info['var_p75']:.3f}")
        else:
            print(f"    Insufficient history ({debug_info['var_history_count']} samples) - using default")
    
    print(f"    Final intervention value: {debug_info['intervention_value']:.3f}")
    
    # Enhanced strategy analysis with deterministic mapping
    if debug_info['selected_var'] == 'X':
        # Analyze X quantile strategy with deterministic percentiles
        percentile_value = debug_info['percentile_value']
        expected_y = percentile_value * 10  # Y = 10*X relationship
        
        print(f"    🔍 X STRATEGY ANALYSIS (Deterministic):")
        print(f"      Selected X {debug_info['selected_quantile']}: value = {percentile_value:.3f}")
        print(f"      Expected Y outcome: {expected_y:.3f}")
        print(f"      Strategy quality: {'EXCELLENT' if expected_y < -50 else 'GOOD' if expected_y < -5 else 'POOR'}")
        
        # Show theoretical optimal strategy
        if scm and debug_info.get('deterministic_mapping'):
            try:
                all_x_percentiles = get_deterministic_percentiles(scm, 'X', [0, 1, 2])
                print(f"      🎯 THEORETICAL ANALYSIS:")
                for i, (q_name, q_val) in enumerate(zip(['25%', '50%', '75%'], all_x_percentiles)):
                    theoretical_y = q_val * 10
                    quality = 'EXCELLENT' if theoretical_y < -50 else 'GOOD' if theoretical_y < -5 else 'POOR'
                    marker = "👑" if i == 0 else "  "  # 25% should be optimal
                    print(f"        {marker} X_{q_name}: {q_val:.1f} → Y={theoretical_y:.1f} ({quality})")
                
                print(f"      💡 OPTIMAL: X_25% = {all_x_percentiles[0]:.1f} → Y = {all_x_percentiles[0]*10:.1f}")
            except:
                pass
        
        if debug_info['selected_quantile'] == '25%' and expected_y < -50.0:
            print(f"    ✅ OPTIMAL STRATEGY: X + 25th percentile (excellent negative values)!")
        elif debug_info['selected_quantile'] == '75%':
            print(f"    ⚠️ SUBOPTIMAL: X + 75th percentile")
            print(f"      Should learn X_25% = {all_x_percentiles[0]:.1f} is much better")
        else:
            print(f"    📊 X exploration: {debug_info['selected_quantile']} strategy")
    else:
        print(f"    📊 Exploring {debug_info['selected_var']} variable")
    
    # Add quantile score analysis for X variable
    if 'X' in [variables[i] for i in range(len(variables))]:
        x_idx = variables.index('X')
        print(f"    📊 X QUANTILE SCORES:")
        print(f"      X_25%: {quantile_scores[x_idx, 0]:.3f} (should be HIGHEST for optimal)")
        print(f"      X_50%: {quantile_scores[x_idx, 1]:.3f}")  
        print(f"      X_75%: {quantile_scores[x_idx, 2]:.3f} (currently highest - suboptimal)")
        
        if quantile_scores[x_idx, 0] > quantile_scores[x_idx, 2]:
            print(f"      ✅ Learning optimal: 25% > 75%")
        else:
            print(f"      ⚠️ Learning suboptimal: 75% > 25% (needs more training)")


def log_quantile_selection(debug_info: Dict[str, Any], target_variable: str):
    """Backward compatibility wrapper."""
    print(f"🎯 QUANTILE SELECTION:")
    print(f"  Selected: {debug_info['selected_var']} at {debug_info['selected_quantile']} percentile")
    print(f"  Final intervention value: {debug_info['intervention_value']:.3f}")


def test_quantile_architecture_gradients():
    """Test quantile architecture for gradient flow."""
    
    import haiku as hk
    from .clean_policy_factory import create_quantile_policy
    
    print("🧪 TESTING QUANTILE ARCHITECTURE GRADIENTS")
    print("="*50)
    
    # Create policy
    policy_fn = create_quantile_policy(hidden_dim=128)
    policy = hk.transform(policy_fn)
    
    # Test input
    dummy_input = jnp.ones((8, 3, 4))
    key = random.PRNGKey(42)
    
    try:
        # Initialize
        params = policy.init(key, dummy_input, 0)
        output = policy.apply(params, key, dummy_input, 0)
        
        print(f"✅ Quantile policy works!")
        print(f"  Quantile scores shape: {output['quantile_scores'].shape}")
        print(f"  Quantile scores: {output['quantile_scores']}")
        
        # Test gradient flow
        def quantile_loss(p):
            out = policy.apply(p, key, dummy_input, 0)
            # Simple loss: prefer first variable, 75th percentile (index 2)
            target_score = out['quantile_scores'][0, 2]  # X variable, 75% quantile
            return -target_score  # Maximize this score
        
        loss, grads = jax.value_and_grad(quantile_loss)(params)
        grad_norms = jax.tree.map(jnp.linalg.norm, grads)
        total_grad_norm = sum(jax.tree.leaves(grad_norms))
        
        print(f"\n📊 GRADIENT RESULTS:")
        print(f"  Loss: {loss:.6f}")
        print(f"  Gradient norm: {total_grad_norm:.6f}")
        print(f"  Expected improvement vs current: {total_grad_norm/0.000356:.0f}x")
        
        if total_grad_norm > 0.01:
            print(f"  🎉 EXCELLENT: Strong gradients with quantile architecture!")
        elif total_grad_norm > 0.001:
            print(f"  ✅ GOOD: Improved gradients")
        else:
            print(f"  ❌ POOR: Still weak gradients")
            
        return total_grad_norm > 0.001
        
    except Exception as e:
        print(f"❌ Quantile architecture test failed: {e}")
        return False