"""Data analysis utilities (production-safe)."""

import jax.numpy as jnp
import pyrsistent as pyr
from typing import Dict, List, Optional, Any

# Type aliases
SampleList = List[pyr.PMap]
VariableOrder = List[str]


def analyze_avici_data(
    avici_data: jnp.ndarray,
    variable_order: VariableOrder
) -> Dict[str, Any]:
    """
    Analyze AVICI data and return structured statistics.
    
    Args:
        avici_data: AVICI data tensor [N, d, 3]
        variable_order: Variable order used in conversion
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    n_samples, n_vars, n_channels = avici_data.shape
    
    # Extract channels
    values = avici_data[:, :, 0]
    interventions = avici_data[:, :, 1]
    targets = avici_data[:, :, 2]
    
    # Compute value statistics
    values_stats = {
        'mean': float(jnp.mean(values)),
        'std': float(jnp.std(values)),
        'min': float(jnp.min(values)),
        'max': float(jnp.max(values)),
        'median': float(jnp.median(values)),
        'per_variable_means': [float(jnp.mean(values[:, i])) for i in range(n_vars)],
        'per_variable_stds': [float(jnp.std(values[:, i])) for i in range(n_vars)]
    }
    
    # Compute intervention statistics
    intervention_stats = {
        'total_interventions': int(jnp.sum(interventions)),
        'samples_with_interventions': int(jnp.sum(jnp.any(interventions, axis=1))),
        'intervention_rate': float(jnp.mean(jnp.any(interventions, axis=1))),
        'variables_intervened': [
            var for i, var in enumerate(variable_order) 
            if jnp.sum(interventions[:, i]) > 0
        ],
        'per_variable_intervention_counts': [
            int(jnp.sum(interventions[:, i])) for i in range(n_vars)
        ]
    }
    
    # Compute target statistics
    target_variable_idx = int(jnp.argmax(jnp.sum(targets, axis=0)))
    target_variable = variable_order[target_variable_idx]
    
    target_stats = {
        'target_variable': target_variable,
        'target_variable_index': target_variable_idx,
        'target_indicator_sum': float(jnp.sum(targets)),
        'expected_target_sum': float(n_samples),  # Should equal n_samples
        'target_consistency': float(jnp.sum(targets)) == float(n_samples)
    }
    
    # Overall structure analysis
    structure_analysis = {
        'shape': avici_data.shape,
        'n_samples': n_samples,
        'n_variables': n_vars,
        'n_channels': n_channels,
        'variable_order': variable_order,
        'data_type': str(avici_data.dtype),
        'has_nan_values': bool(jnp.any(jnp.isnan(avici_data))),
        'has_inf_values': bool(jnp.any(jnp.isinf(avici_data))),
        'all_finite': bool(jnp.all(jnp.isfinite(avici_data)))
    }
    
    return {
        'structure': structure_analysis,
        'values': values_stats,
        'interventions': intervention_stats,
        'targets': target_stats
    }


def compare_conversions(
    conversion1: jnp.ndarray,
    conversion2: jnp.ndarray,
    variable_order: VariableOrder,
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compare two AVICI conversions for differences.
    
    Args:
        conversion1: First AVICI conversion [N, d, 3]
        conversion2: Second AVICI conversion [N, d, 3]
        variable_order: Variable order used in conversions
        labels: Labels for the two conversions (for reporting)
        
    Returns:
        Dictionary with detailed comparison results
    """
    if labels is None:
        labels = ["Conversion 1", "Conversion 2"]
    
    # Basic shape comparison
    shape_match = conversion1.shape == conversion2.shape
    
    if not shape_match:
        return {
            'labels': labels,
            'shape_match': False,
            'shapes': [conversion1.shape, conversion2.shape],
            'error': 'Shape mismatch prevents detailed comparison'
        }
    
    # Channel-wise differences
    values_diff = jnp.mean(jnp.abs(conversion1[:, :, 0] - conversion2[:, :, 0]))
    intervention_diff = jnp.mean(jnp.abs(conversion1[:, :, 1] - conversion2[:, :, 1]))
    target_diff = jnp.mean(jnp.abs(conversion1[:, :, 2] - conversion2[:, :, 2]))
    
    # Exact matches for binary channels
    intervention_match = jnp.allclose(conversion1[:, :, 1], conversion2[:, :, 1])
    target_match = jnp.allclose(conversion1[:, :, 2], conversion2[:, :, 2])
    
    # Per-variable analysis
    per_variable_diffs = []
    for i, var_name in enumerate(variable_order):
        var_values_diff = jnp.mean(jnp.abs(conversion1[:, i, 0] - conversion2[:, i, 0]))
        var_intervention_diff = jnp.mean(jnp.abs(conversion1[:, i, 1] - conversion2[:, i, 1]))
        var_target_diff = jnp.mean(jnp.abs(conversion1[:, i, 2] - conversion2[:, i, 2]))
        
        per_variable_diffs.append({
            'variable': var_name,
            'values_diff': float(var_values_diff),
            'intervention_diff': float(var_intervention_diff),
            'target_diff': float(var_target_diff)
        })
    
    # Statistical comparisons
    statistical_comparison = {
        'values_correlation': float(jnp.corrcoef(
            conversion1[:, :, 0].flatten(),
            conversion2[:, :, 0].flatten()
        )[0, 1]),
        'max_absolute_difference': float(jnp.max(jnp.abs(conversion1 - conversion2))),
        'mean_squared_error': float(jnp.mean((conversion1 - conversion2) ** 2)),
        'relative_error': float(jnp.mean(jnp.abs(conversion1 - conversion2) / (jnp.abs(conversion1) + 1e-8)))
    }
    
    return {
        'labels': labels,
        'shape_match': shape_match,
        'shapes': [conversion1.shape, conversion2.shape],
        'channel_differences': {
            'values': float(values_diff),
            'interventions': float(intervention_diff),
            'targets': float(target_diff)
        },
        'binary_channel_matches': {
            'interventions': intervention_match,
            'targets': target_match
        },
        'per_variable_analysis': per_variable_diffs,
        'statistical_comparison': statistical_comparison,
        'individual_analyses': {
            labels[0]: analyze_avici_data(conversion1, variable_order),
            labels[1]: analyze_avici_data(conversion2, variable_order)
        }
    }


def compute_data_quality_metrics(
    avici_data: jnp.ndarray,
    variable_order: VariableOrder
) -> Dict[str, Any]:
    """
    Compute data quality metrics for AVICI data.
    
    Args:
        avici_data: AVICI data tensor [N, d, 3]
        variable_order: Variable order used in conversion
        
    Returns:
        Dictionary with data quality assessment
    """
    n_samples, n_vars, _ = avici_data.shape
    
    # Extract channels
    values = avici_data[:, :, 0]
    interventions = avici_data[:, :, 1]
    targets = avici_data[:, :, 2]
    
    # Value quality metrics
    value_quality = {
        'standardization_quality': {
            'mean_close_to_zero': float(jnp.abs(jnp.mean(values))) < 0.1,
            'std_close_to_one': abs(float(jnp.std(values)) - 1.0) < 0.1,
            'per_variable_means': [float(jnp.mean(values[:, i])) for i in range(n_vars)],
            'per_variable_stds': [float(jnp.std(values[:, i])) for i in range(n_vars)]
        },
        'outlier_detection': {
            'values_beyond_3_sigma': int(jnp.sum(jnp.abs(values) > 3.0)),
            'outlier_rate': float(jnp.mean(jnp.abs(values) > 3.0))
        },
        'distribution_properties': {
            'skewness_magnitude': float(jnp.mean(jnp.abs(
                jnp.mean((values - jnp.mean(values)) ** 3, axis=0) / (jnp.std(values, axis=0) ** 3)
            ))),
            'kurtosis_magnitude': float(jnp.mean(jnp.abs(
                jnp.mean((values - jnp.mean(values)) ** 4, axis=0) / (jnp.std(values, axis=0) ** 4) - 3
            )))
        }
    }
    
    # Binary channel quality
    binary_quality = {
        'intervention_channel': {
            'is_binary': bool(jnp.all(jnp.isin(interventions, jnp.array([0.0, 1.0])))),
            'unique_values': list(float(x) for x in jnp.unique(interventions)),
            'sparsity': float(jnp.mean(interventions == 0.0))
        },
        'target_channel': {
            'is_binary': bool(jnp.all(jnp.isin(targets, jnp.array([0.0, 1.0])))),
            'unique_values': list(float(x) for x in jnp.unique(targets)),
            'exactly_one_per_sample': bool(jnp.all(jnp.sum(targets, axis=1) == 1.0))
        }
    }
    
    # Coverage analysis
    coverage_analysis = {
        'intervention_coverage': {
            'variables_never_intervened': [
                variable_order[i] for i in range(n_vars)
                if jnp.sum(interventions[:, i]) == 0
            ],
            'variables_always_intervened': [
                variable_order[i] for i in range(n_vars)
                if jnp.sum(interventions[:, i]) == n_samples
            ],
            'intervention_balance': [
                float(jnp.mean(interventions[:, i])) for i in range(n_vars)
            ]
        }
    }
    
    return {
        'value_quality': value_quality,
        'binary_quality': binary_quality,
        'coverage_analysis': coverage_analysis,
        'overall_quality_score': _compute_overall_quality_score(
            value_quality, binary_quality, coverage_analysis
        )
    }


def _compute_overall_quality_score(
    value_quality: Dict,
    binary_quality: Dict,
    coverage_analysis: Dict
) -> float:
    """
    Compute an overall quality score from quality metrics.
    
    Returns a score between 0 and 1, where 1 is perfect quality.
    """
    score = 0.0
    total_weight = 0.0
    
    # Standardization quality (weight: 0.3)
    if value_quality['standardization_quality']['mean_close_to_zero']:
        score += 0.1
    if value_quality['standardization_quality']['std_close_to_one']:
        score += 0.1
    # Penalty for high outlier rate
    outlier_penalty = min(0.1, value_quality['outlier_detection']['outlier_rate'])
    score += 0.1 - outlier_penalty
    total_weight += 0.3
    
    # Binary channel quality (weight: 0.4)
    if binary_quality['intervention_channel']['is_binary']:
        score += 0.2
    if binary_quality['target_channel']['is_binary'] and \
       binary_quality['target_channel']['exactly_one_per_sample']:
        score += 0.2
    total_weight += 0.4
    
    # Coverage quality (weight: 0.3)
    never_intervened = len(coverage_analysis['intervention_coverage']['variables_never_intervened'])
    always_intervened = len(coverage_analysis['intervention_coverage']['variables_always_intervened'])
    total_vars = len(coverage_analysis['intervention_coverage']['intervention_balance'])
    
    # Penalty for poor intervention coverage
    coverage_penalty = (never_intervened + always_intervened) / (2 * total_vars) * 0.3
    score += 0.3 - coverage_penalty
    total_weight += 0.3
    
    return score / total_weight if total_weight > 0 else 0.0
