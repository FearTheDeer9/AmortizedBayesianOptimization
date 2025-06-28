"""
Statistical significance testing for ACBO evaluation.

This module provides pure functions for statistical hypothesis testing
and confidence interval computation following functional principles.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as onp
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from scipy import stats as scipy_stats
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StatisticalTestResult:
    """Immutable statistical test result."""
    test_name: str
    statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    is_significant: bool
    alpha: float
    sample_size: int


@dataclass(frozen=True)
class BootstrapResult:
    """Immutable bootstrap analysis result."""
    point_estimate: float
    confidence_interval: Tuple[float, float]
    bootstrap_samples: Tuple[float, ...]
    bias: float
    standard_error: float


# Pure statistical test functions
def paired_t_test(
    sample1: jnp.ndarray,
    sample2: jnp.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> StatisticalTestResult:
    """
    Perform paired t-test for comparing two related samples.
    
    Args:
        sample1: First sample (e.g., ACBO results)
        sample2: Second sample (e.g., baseline results)
        alpha: Significance level
        alternative: Alternative hypothesis ("two-sided", "greater", "less")
        
    Returns:
        StatisticalTestResult with test results
    """
    if len(sample1) != len(sample2):
        raise ValueError("Sample sizes must be equal for paired t-test")
    
    # Compute differences
    differences = sample1 - sample2
    n = len(differences)
    
    if n < 2:
        raise ValueError("Need at least 2 paired observations")
    
    # Compute test statistics
    mean_diff = jnp.mean(differences)
    std_diff = jnp.std(differences, ddof=1)
    se_diff = std_diff / jnp.sqrt(n)
    
    t_statistic = mean_diff / (se_diff + 1e-8)
    
    # Convert to numpy for scipy.stats
    t_stat_np = float(t_statistic)
    df = n - 1
    
    # Compute p-value using scipy
    if alternative == "two-sided":
        p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat_np), df))
    elif alternative == "greater":
        p_value = 1 - scipy_stats.t.cdf(t_stat_np, df)
    elif alternative == "less":
        p_value = scipy_stats.t.cdf(t_stat_np, df)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    
    # Confidence interval
    t_critical = scipy_stats.t.ppf(1 - alpha/2, df)
    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff
    
    # Effect size (Cohen's d for paired samples)
    effect_size = float(mean_diff / (std_diff + 1e-8))
    
    return StatisticalTestResult(
        test_name="paired_t_test",
        statistic=t_stat_np,
        p_value=p_value,
        confidence_interval=(float(ci_lower), float(ci_upper)),
        effect_size=effect_size,
        is_significant=p_value < alpha,
        alpha=alpha,
        sample_size=n
    )


def wilcoxon_signed_rank_test(
    sample1: jnp.ndarray,
    sample2: jnp.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> StatisticalTestResult:
    """
    Perform Wilcoxon signed-rank test (non-parametric paired test).
    
    Args:
        sample1: First sample
        sample2: Second sample  
        alpha: Significance level
        alternative: Alternative hypothesis
        
    Returns:
        StatisticalTestResult with test results
    """
    if len(sample1) != len(sample2):
        raise ValueError("Sample sizes must be equal for Wilcoxon signed-rank test")
    
    # Convert to numpy for scipy
    sample1_np = onp.array(sample1)
    sample2_np = onp.array(sample2)
    
    # Perform Wilcoxon test
    try:
        statistic, p_value = scipy_stats.wilcoxon(
            sample1_np, sample2_np, alternative=alternative
        )
    except ValueError as e:
        # Handle case where all differences are zero
        logger.warning(f"Wilcoxon test failed: {e}")
        return StatisticalTestResult(
            test_name="wilcoxon_signed_rank_test",
            statistic=0.0,
            p_value=1.0,
            confidence_interval=(0.0, 0.0),
            effect_size=0.0,
            is_significant=False,
            alpha=alpha,
            sample_size=len(sample1)
        )
    
    # Compute effect size (rank-biserial correlation)
    differences = sample1 - sample2
    n_pos = jnp.sum(differences > 0)
    n_neg = jnp.sum(differences < 0)
    effect_size = float((n_pos - n_neg) / (n_pos + n_neg + 1e-8))
    
    # Simplified confidence interval (would need more sophisticated computation)
    median_diff = float(jnp.median(differences))
    ci_width = 1.96 * jnp.std(differences) / jnp.sqrt(len(differences))
    
    return StatisticalTestResult(
        test_name="wilcoxon_signed_rank_test",
        statistic=float(statistic),
        p_value=float(p_value),
        confidence_interval=(median_diff - ci_width, median_diff + ci_width),
        effect_size=effect_size,
        is_significant=p_value < alpha,
        alpha=alpha,
        sample_size=len(sample1)
    )


def bootstrap_confidence_interval(
    data: jnp.ndarray,
    statistic_fn: callable = jnp.mean,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    key: random.PRNGKey = None
) -> BootstrapResult:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: Input data array
        statistic_fn: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        key: Random key for reproducibility
        
    Returns:
        BootstrapResult with bootstrap analysis
    """
    if key is None:
        key = random.PRNGKey(42)
    
    n = len(data)
    if n == 0:
        raise ValueError("Data array cannot be empty")
    
    # Original statistic
    original_stat = statistic_fn(data)
    
    # Bootstrap sampling
    bootstrap_stats = []
    for i in range(n_bootstrap):
        key, subkey = random.split(key)
        
        # Sample with replacement
        indices = random.choice(subkey, n, shape=(n,), replace=True)
        bootstrap_sample = data[indices]
        
        # Compute statistic on bootstrap sample
        bootstrap_stat = statistic_fn(bootstrap_sample)
        bootstrap_stats.append(float(bootstrap_stat))
    
    bootstrap_stats = jnp.array(bootstrap_stats)
    
    # Confidence interval using percentile method
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    ci_lower = jnp.percentile(bootstrap_stats, lower_percentile)
    ci_upper = jnp.percentile(bootstrap_stats, upper_percentile)
    
    # Bootstrap bias and standard error
    bias = jnp.mean(bootstrap_stats) - original_stat
    standard_error = jnp.std(bootstrap_stats)
    
    return BootstrapResult(
        point_estimate=float(original_stat),
        confidence_interval=(float(ci_lower), float(ci_upper)),
        bootstrap_samples=tuple(bootstrap_stats),
        bias=float(bias),
        standard_error=float(standard_error)
    )


def compute_effect_size(
    sample1: jnp.ndarray,
    sample2: jnp.ndarray,
    effect_type: str = "cohen_d"
) -> float:
    """
    Compute effect size between two samples.
    
    Args:
        sample1: First sample
        sample2: Second sample
        effect_type: Type of effect size ("cohen_d", "glass_delta", "hedges_g")
        
    Returns:
        Effect size value
    """
    mean1 = jnp.mean(sample1)
    mean2 = jnp.mean(sample2)
    mean_diff = mean1 - mean2
    
    if effect_type == "cohen_d":
        # Cohen's d: standardized mean difference
        std1 = jnp.std(sample1, ddof=1)
        std2 = jnp.std(sample2, ddof=1)
        pooled_std = jnp.sqrt(((len(sample1) - 1) * std1**2 + (len(sample2) - 1) * std2**2) / 
                             (len(sample1) + len(sample2) - 2))
        effect_size = mean_diff / (pooled_std + 1e-8)
        
    elif effect_type == "glass_delta":
        # Glass's delta: uses control group standard deviation
        std2 = jnp.std(sample2, ddof=1)
        effect_size = mean_diff / (std2 + 1e-8)
        
    elif effect_type == "hedges_g":
        # Hedges' g: bias-corrected Cohen's d
        std1 = jnp.std(sample1, ddof=1)
        std2 = jnp.std(sample2, ddof=1)
        pooled_std = jnp.sqrt(((len(sample1) - 1) * std1**2 + (len(sample2) - 1) * std2**2) / 
                             (len(sample1) + len(sample2) - 2))
        cohen_d = mean_diff / (pooled_std + 1e-8)
        
        # Bias correction factor
        df = len(sample1) + len(sample2) - 2
        correction = 1 - (3 / (4 * df - 1))
        effect_size = cohen_d * correction
        
    else:
        raise ValueError(f"Unknown effect type: {effect_type}")
    
    return float(effect_size)


def multiple_comparisons_correction(
    p_values: jnp.ndarray,
    method: str = "bonferroni"
) -> jnp.ndarray:
    """
    Apply multiple comparisons correction to p-values.
    
    Args:
        p_values: Array of p-values
        method: Correction method ("bonferroni", "holm", "fdr_bh")
        
    Returns:
        Corrected p-values
    """
    p_values = jnp.array(p_values)
    n = len(p_values)
    
    if method == "bonferroni":
        # Bonferroni correction
        corrected = p_values * n
        return jnp.clip(corrected, 0.0, 1.0)
        
    elif method == "holm":
        # Holm-Bonferroni method
        sorted_indices = jnp.argsort(p_values)
        corrected = jnp.zeros_like(p_values)
        
        for i, idx in enumerate(sorted_indices):
            correction_factor = n - i
            corrected = corrected.at[idx].set(p_values[idx] * correction_factor)
        
        return jnp.clip(corrected, 0.0, 1.0)
        
    elif method == "fdr_bh":
        # Benjamini-Hochberg FDR control
        sorted_indices = jnp.argsort(p_values)
        corrected = jnp.zeros_like(p_values)
        
        for i, idx in enumerate(sorted_indices):
            correction_factor = n / (i + 1)
            corrected = corrected.at[idx].set(p_values[idx] * correction_factor)
        
        return jnp.clip(corrected, 0.0, 1.0)
    
    else:
        raise ValueError(f"Unknown correction method: {method}")


def power_analysis(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    test_type: str = "two_sample_t"
) -> int:
    """
    Compute required sample size for given statistical power.
    
    Args:
        effect_size: Expected effect size
        alpha: Type I error rate
        power: Desired statistical power
        test_type: Type of statistical test
        
    Returns:
        Required sample size per group
    """
    if test_type == "two_sample_t":
        # Simplified power analysis for two-sample t-test
        # This is an approximation - in practice would use more sophisticated methods
        z_alpha = scipy_stats.norm.ppf(1 - alpha/2)
        z_beta = scipy_stats.norm.ppf(power)
        
        n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return int(jnp.ceil(n_per_group))
    
    else:
        raise ValueError(f"Unknown test type: {test_type}")


__all__ = [
    'StatisticalTestResult',
    'BootstrapResult',
    'paired_t_test',
    'wilcoxon_signed_rank_test',
    'bootstrap_confidence_interval',
    'compute_effect_size',
    'multiple_comparisons_correction',
    'power_analysis'
]