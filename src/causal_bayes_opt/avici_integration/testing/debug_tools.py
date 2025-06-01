"""Debug and testing utilities (development only)."""

import jax
import jax.numpy as jnp
import jax.random as random
import pyrsistent as pyr
from typing import List, FrozenSet, Dict, Any

# Type aliases
SampleList = List[pyr.PMap]
VariableOrder = List[str]


def debug_conversion_step_by_step(
    samples: SampleList,
    variable_order: VariableOrder, 
    target_variable: str,
    standardize: bool = True
) -> Dict[str, Any]:
    """
    Debug conversion process step by step.
    
    Args:
        samples: List of Sample objects
        variable_order: Ordered list of variable names
        target_variable: Name of target variable
        standardize: Whether to standardize values
        
    Returns:
        Dictionary with detailed step-by-step conversion information
    """
    from ..core.data_extraction import (
        extract_values_matrix,
        extract_intervention_indicators,
        create_target_indicators
    )
    from ..core.standardization import compute_standardization_params, apply_standardization
    from ..utils.analysis import analyze_avici_data
    
    print(f"\n🔍 Step-by-Step Conversion Debug")
    print(f"Target: '{target_variable}'")
    print(f"Variables: {variable_order}")
    print(f"Samples: {len(samples)}")
    print(f"Standardize: {standardize}")
    
    debug_info = {
        'input_summary': {
            'n_samples': len(samples),
            'n_variables': len(variable_order),
            'target_variable': target_variable,
            'variable_order': variable_order,
            'standardize': standardize
        }
    }
    
    # Step 1: Extract values matrix
    print(f"\n📊 Step 1: Extract values matrix")
    values_matrix = extract_values_matrix(samples, variable_order)
    print(f"Values shape: {values_matrix.shape}")
    print(f"Values range: [{jnp.min(values_matrix):.3f}, {jnp.max(values_matrix):.3f}]")
    print(f"Values mean: {jnp.mean(values_matrix):.3f}, std: {jnp.std(values_matrix):.3f}")
    
    debug_info['step1_values'] = {
        'shape': values_matrix.shape,
        'range': [float(jnp.min(values_matrix)), float(jnp.max(values_matrix))],
        'mean': float(jnp.mean(values_matrix)),
        'std': float(jnp.std(values_matrix)),
        'sample_values': values_matrix[:min(3, len(samples))].tolist()  # First 3 samples
    }
    
    # Step 2: Standardization (if requested)
    if standardize:
        print(f"\n📏 Step 2: Standardization")
        std_params = compute_standardization_params(values_matrix, "default")
        standardized_values = apply_standardization(values_matrix, std_params)
        
        print(f"Standardized range: [{jnp.min(standardized_values):.3f}, {jnp.max(standardized_values):.3f}]")
        print(f"Standardized mean: {jnp.mean(standardized_values):.3f}, std: {jnp.std(standardized_values):.3f}")
        
        debug_info['step2_standardization'] = {
            'parameters': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in std_params.items()},
            'standardized_range': [float(jnp.min(standardized_values)), float(jnp.max(standardized_values))],
            'standardized_mean': float(jnp.mean(standardized_values)),
            'standardized_std': float(jnp.std(standardized_values))
        }
        
        final_values = standardized_values
    else:
        print(f"\n⏭️  Step 2: Skipping standardization")
        debug_info['step2_standardization'] = None
        final_values = values_matrix
    
    # Step 3: Extract intervention indicators
    print(f"\n🎯 Step 3: Extract intervention indicators")
    intervention_indicators = extract_intervention_indicators(samples, variable_order)
    n_interventions = int(jnp.sum(intervention_indicators))
    intervention_rate = float(jnp.mean(intervention_indicators))
    
    print(f"Intervention indicators shape: {intervention_indicators.shape}")
    print(f"Total interventions: {n_interventions}")
    print(f"Intervention rate: {intervention_rate:.3f}")
    
    # Show which variables were intervened upon
    intervened_vars = []
    for i, var_name in enumerate(variable_order):
        var_interventions = int(jnp.sum(intervention_indicators[:, i]))
        if var_interventions > 0:
            intervened_vars.append(f"{var_name}: {var_interventions}")
    print(f"Intervened variables: {intervened_vars}")
    
    debug_info['step3_interventions'] = {
        'shape': intervention_indicators.shape,
        'total_interventions': n_interventions,
        'intervention_rate': intervention_rate,
        'per_variable_counts': {
            var_name: int(jnp.sum(intervention_indicators[:, i]))
            for i, var_name in enumerate(variable_order)
        }
    }
    
    # Step 4: Create target indicators
    print(f"\n🎯 Step 4: Create target indicators")
    target_indicators = create_target_indicators(target_variable, variable_order, len(samples))
    target_sum = int(jnp.sum(target_indicators))
    
    print(f"Target indicators shape: {target_indicators.shape}")
    print(f"Target sum: {target_sum} (should equal {len(samples)})")
    print(f"Target consistency: {target_sum == len(samples)}")
    
    debug_info['step4_targets'] = {
        'shape': target_indicators.shape,
        'target_sum': target_sum,
        'expected_sum': len(samples),
        'consistent': target_sum == len(samples),
        'target_variable': target_variable,
        'target_index': variable_order.index(target_variable)
    }
    
    # Step 5: Stack into final tensor
    print(f"\n📦 Step 5: Stack into final AVICI tensor")
    avici_data = jnp.stack([final_values, intervention_indicators, target_indicators], axis=2)
    
    print(f"Final tensor shape: {avici_data.shape}")
    print(f"Channel 0 (values) range: [{jnp.min(avici_data[:, :, 0]):.3f}, {jnp.max(avici_data[:, :, 0]):.3f}]")
    print(f"Channel 1 (interventions) range: [{jnp.min(avici_data[:, :, 1]):.3f}, {jnp.max(avici_data[:, :, 1]):.3f}]")
    print(f"Channel 2 (targets) range: [{jnp.min(avici_data[:, :, 2]):.3f}, {jnp.max(avici_data[:, :, 2]):.3f}]")
    
    debug_info['step5_final'] = {
        'shape': avici_data.shape,
        'channel_ranges': {
            'values': [float(jnp.min(avici_data[:, :, 0])), float(jnp.max(avici_data[:, :, 0]))],
            'interventions': [float(jnp.min(avici_data[:, :, 1])), float(jnp.max(avici_data[:, :, 1]))],
            'targets': [float(jnp.min(avici_data[:, :, 2])), float(jnp.max(avici_data[:, :, 2]))]
        }
    }
    
    # Step 6: Final analysis
    print(f"\n📈 Step 6: Final analysis")
    analysis = analyze_avici_data(avici_data, variable_order)
    debug_info['step6_analysis'] = analysis
    
    print(f"✅ Conversion completed successfully!")
    
    return debug_info


def debug_parent_set_enumeration(
    variables: List[str], 
    target_variable: str, 
    max_parent_size: int = 3
) -> List[FrozenSet[str]]:
    """
    Debug parent set enumeration to verify empty sets are included.
    
    Args:
        variables: List of all variable names
        target_variable: Target variable name
        max_parent_size: Maximum parent set size
        
    Returns:
        List of all enumerated parent sets
    """
    try:
        from ..parent_set.enumeration import enumerate_possible_parent_sets
    except ImportError:
        print("⚠️ Parent set enumeration not available")
        return []
    
    parent_sets = enumerate_possible_parent_sets(variables, target_variable, max_parent_size)
    
    print(f"\n🔍 Parent Set Enumeration Debug")
    print(f"Target: '{target_variable}'")
    print(f"Variables: {variables}")
    print(f"Max parent size: {max_parent_size}")
    print(f"Total parent sets: {len(parent_sets)}")
    
    # Count by size
    size_counts = {}
    for ps in parent_sets:
        size = len(ps)
        size_counts[size] = size_counts.get(size, 0) + 1
    
    print(f"Parent sets by size: {size_counts}")
    
    # Check for empty set
    empty_sets = [ps for ps in parent_sets if len(ps) == 0]
    print(f"Empty sets found: {len(empty_sets)}")
    
    # Show all parent sets
    print(f"\nAll parent sets:")
    for i, ps in enumerate(parent_sets):
        ps_str = set(ps) if ps else "{}"
        print(f"  {i:2d}: {ps_str}")
    
    return parent_sets


def debug_training_step(
    net, 
    params, 
    x: jnp.ndarray, 
    variable_order: VariableOrder, 
    target_variable: str, 
    true_parent_set: FrozenSet[str]
) -> Dict[str, Any]:
    """
    Debug a single training step with detailed output.
    
    Args:
        net: Neural network model
        params: Model parameters
        x: Input data tensor [N, d, 3]
        variable_order: Variable order
        target_variable: Target variable name
        true_parent_set: Ground truth parent set
        
    Returns:
        Dictionary with debug information and model output
    """
    print(f"\n🔍 Training Step Debug")
    print(f"Target: '{target_variable}'")
    print(f"True parent set: {set(true_parent_set) if true_parent_set else '{}'}")
    print(f"Input shape: {x.shape}")
    
    # Debug parent set enumeration first
    parent_sets = debug_parent_set_enumeration(variable_order, target_variable)
    
    # Get model output
    print(f"\n🧠 Model Forward Pass")
    output = net.apply(params, random.PRNGKey(0), x, variable_order, target_variable, False)
    
    if 'parent_set_logits' not in output or 'parent_sets' not in output:
        print(f"⚠️ Model output missing expected keys: {output.keys()}")
        return {'error': 'Invalid model output', 'output': output}
    
    logits = output['parent_set_logits']
    model_parent_sets = output['parent_sets']
    
    print(f"Model parent sets: {len(model_parent_sets)}")
    print(f"Logits shape: {logits.shape}")
    print(f"Logit range: [{jnp.min(logits):.3f}, {jnp.max(logits):.3f}]")
    
    # Convert to probabilities
    probabilities = jax.nn.softmax(logits)
    print(f"Probability range: [{jnp.min(probabilities):.6f}, {jnp.max(probabilities):.6f}]")
    
    # Find true parent set in predictions
    true_idx = None
    for i, ps in enumerate(model_parent_sets):
        if ps == true_parent_set:
            true_idx = i
            break
    
    print(f"\n📊 Predictions Analysis")
    print(f"True parent set in predictions: {'✅' if true_idx is not None else '❌'}")
    
    if true_idx is not None:
        print(f"True parent set position: {true_idx}")
        print(f"True parent set logit: {logits[true_idx]:.4f}")
        print(f"True parent set probability: {probabilities[true_idx]:.6f}")
    
    # Show top predictions
    sorted_indices = jnp.argsort(logits)[::-1]  # Descending order
    print(f"\n🏆 Top 5 Predictions:")
    for rank, idx in enumerate(sorted_indices[:5]):
        ps = model_parent_sets[idx]
        ps_str = set(ps) if ps else "{}"
        is_true = "✅" if ps == true_parent_set else "  "
        print(f"{is_true} {rank+1}: {ps_str} -> logit: {logits[idx]:.4f}, prob: {probabilities[idx]:.6f}")
    
    # Special check for empty set
    for i, ps in enumerate(model_parent_sets):
        if len(ps) == 0:
            print(f"\n∅ Empty Set Analysis:")
            print(f"Position: {i}")
            print(f"Logit: {logits[i]:.4f}")
            print(f"Probability: {probabilities[i]:.6f}")
            break
    
    debug_info = {
        'target_variable': target_variable,
        'true_parent_set': list(true_parent_set),
        'input_shape': x.shape,
        'n_parent_sets': len(model_parent_sets),
        'logit_stats': {
            'min': float(jnp.min(logits)),
            'max': float(jnp.max(logits)),
            'mean': float(jnp.mean(logits)),
            'std': float(jnp.std(logits))
        },
        'probability_stats': {
            'min': float(jnp.min(probabilities)),
            'max': float(jnp.max(probabilities)),
            'entropy': float(-jnp.sum(probabilities * jnp.log(probabilities + 1e-8)))
        },
        'true_parent_set_found': true_idx is not None,
        'true_parent_set_index': int(true_idx) if true_idx is not None else None,
        'true_parent_set_logit': float(logits[true_idx]) if true_idx is not None else None,
        'true_parent_set_probability': float(probabilities[true_idx]) if true_idx is not None else None,
        'model_output': output
    }
    
    return debug_info


def debug_sample_conversion(
    sample: pyr.PMap,
    sample_idx: int,
    avici_data: jnp.ndarray,
    variable_order: VariableOrder,
    target_variable: str
) -> Dict[str, Any]:
    """
    Debug the conversion of a specific sample for detailed inspection.
    
    Args:
        sample: Original Sample object
        sample_idx: Index of sample in the batch
        avici_data: Converted AVICI data tensor
        variable_order: Variable order used in conversion
        target_variable: Target variable name
        
    Returns:
        Dictionary with detailed debug information
    """
    print(f"\n🔍 Sample Conversion Debug")
    print(f"Sample index: {sample_idx}")
    print(f"Target variable: '{target_variable}'")
    
    # Extract original sample information
    original_values = dict(sample['values'])
    original_intervention_type = sample['intervention_type']
    original_intervention_targets = set(sample['intervention_targets'])
    
    print(f"\n📝 Original Sample:")
    print(f"Values: {original_values}")
    print(f"Intervention type: {original_intervention_type}")
    print(f"Intervention targets: {sorted(original_intervention_targets)}")
    
    # Extract converted information
    converted_values = {}
    converted_interventions = {}
    converted_targets = {}
    
    for j, var_name in enumerate(variable_order):
        converted_values[var_name] = float(avici_data[sample_idx, j, 0])
        converted_interventions[var_name] = float(avici_data[sample_idx, j, 1])
        converted_targets[var_name] = float(avici_data[sample_idx, j, 2])
    
    print(f"\n📊 Converted Sample:")
    print(f"Values: {converted_values}")
    print(f"Intervention indicators: {converted_interventions}")
    print(f"Target indicators: {converted_targets}")
    
    # Check consistency
    intervention_consistency = set()
    for var_name in variable_order:
        if converted_interventions[var_name] > 0.5:
            intervention_consistency.add(var_name)
    
    target_consistency = None
    for var_name in variable_order:
        if converted_targets[var_name] > 0.5:
            target_consistency = var_name
            break
    
    print(f"\n✅ Consistency Checks:")
    print(f"Intervention targets match: {intervention_consistency == original_intervention_targets}")
    print(f"Target variable match: {target_consistency == target_variable}")
    
    if intervention_consistency != original_intervention_targets:
        print(f"  Expected: {sorted(original_intervention_targets)}")
        print(f"  Found: {sorted(intervention_consistency)}")
    
    if target_consistency != target_variable:
        print(f"  Expected target: '{target_variable}'")
        print(f"  Found target: '{target_consistency}'")
    
    debug_info = {
        'sample_index': sample_idx,
        'target_variable': target_variable,
        'original': {
            'values': original_values,
            'intervention_type': original_intervention_type,
            'intervention_targets': sorted(original_intervention_targets)
        },
        'converted': {
            'values': converted_values,
            'intervention_indicators': converted_interventions,
            'target_indicators': converted_targets
        },
        'consistency_checks': {
            'intervention_targets_match': intervention_consistency == original_intervention_targets,
            'target_variable_match': target_consistency == target_variable,
            'expected_target': target_variable,
            'detected_target': target_consistency,
            'expected_interventions': sorted(original_intervention_targets),
            'detected_interventions': sorted(intervention_consistency)
        }
    }
    
    return debug_info


def debug_logits_and_probabilities(
    logits: jnp.ndarray, 
    parent_sets: List[FrozenSet[str]], 
    target_variable: str
) -> jnp.ndarray:
    """
    Debug logit ranges and probability distribution.
    
    Args:
        logits: Model logits for parent sets
        parent_sets: List of parent sets corresponding to logits
        target_variable: Target variable name
        
    Returns:
        Computed probabilities from logits
    """
    print(f"\n🔍 Logits and Probabilities Debug")
    print(f"Target: '{target_variable}'")
    print(f"Number of parent sets: {len(parent_sets)}")
    print(f"Logits shape: {logits.shape}")
    
    # Logit analysis
    logit_min, logit_max = jnp.min(logits), jnp.max(logits)
    logit_range = logit_max - logit_min
    logit_mean, logit_std = jnp.mean(logits), jnp.std(logits)
    
    print(f"\n📊 Logit Statistics:")
    print(f"Range: [{logit_min:.3f}, {logit_max:.3f}] (span: {logit_range:.3f})")
    print(f"Mean: {logit_mean:.3f}, Std: {logit_std:.3f}")
    
    if logit_range > 10:
        print(f"⚠️ WARNING: Large logit range ({logit_range:.1f}) may cause numerical issues!")
    
    # Convert to probabilities
    probabilities = jax.nn.softmax(logits)
    prob_min, prob_max = jnp.min(probabilities), jnp.max(probabilities)
    prob_entropy = -jnp.sum(probabilities * jnp.log(probabilities + 1e-8))
    
    print(f"\n📈 Probability Statistics:")
    print(f"Range: [{prob_min:.6f}, {prob_max:.6f}]")
    print(f"Entropy: {prob_entropy:.3f} (max: {jnp.log(len(parent_sets)):.3f})")
    print(f"Max probability: {prob_max:.6f} (1/n = {1.0/len(parent_sets):.6f})")
    
    # Special analysis for empty set
    empty_set_info = None
    for i, ps in enumerate(parent_sets):
        if len(ps) == 0:
            empty_set_info = {
                'index': i,
                'logit': float(logits[i]),
                'probability': float(probabilities[i])
            }
            print(f"\n∅ Empty Set Analysis:")
            print(f"Position: {i}")
            print(f"Logit: {logits[i]:.3f} (rank: {jnp.sum(logits > logits[i]) + 1})")
            print(f"Probability: {probabilities[i]:.6f}")
            break
    
    if empty_set_info is None:
        print(f"\n⚠️ No empty set found in parent sets!")
    
    # Top and bottom predictions
    sorted_indices = jnp.argsort(logits)[::-1]
    
    print(f"\n🏆 Top 3 Parent Sets:")
    for rank, idx in enumerate(sorted_indices[:3]):
        ps = parent_sets[idx]
        ps_str = set(ps) if ps else "{}"
        print(f"  {rank+1}: {ps_str} -> logit: {logits[idx]:.3f}, prob: {probabilities[idx]:.6f}")
    
    print(f"\n👇 Bottom 3 Parent Sets:")
    for rank, idx in enumerate(sorted_indices[-3:]):
        ps = parent_sets[idx]
        ps_str = set(ps) if ps else "{}"
        print(f"  {len(sorted_indices) - 2 + rank}: {ps_str} -> logit: {logits[idx]:.3f}, prob: {probabilities[idx]:.6f}")
    
    return probabilities
