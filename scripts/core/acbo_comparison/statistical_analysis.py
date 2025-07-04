"""
Statistical Analysis for ACBO Comparison Framework

This module provides statistical testing and analysis capabilities for comparing
ACBO methods. It includes significance testing, effect size computation, and
confidence interval estimation.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as onp
from dataclasses import dataclass

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of statistical comparison between two methods."""
    method1_name: str
    method2_name: str
    method1_mean: float
    method1_std: float
    method2_mean: float
    method2_std: float
    mean_difference: float
    t_statistic: float
    p_value: float
    significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]
    sample_sizes: Tuple[int, int]


class StatisticalAnalyzer:
    """Analyzer for statistical comparison of ACBO methods."""
    
    def __init__(self, significance_level: float = 0.05, 
                 correction_method: str = "bonferroni"):
        """
        Initialize statistical analyzer.
        
        Args:
            significance_level: Alpha level for significance testing
            correction_method: Method for multiple comparisons correction
        """
        self.significance_level = significance_level
        self.correction_method = correction_method
        
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available - statistical tests will be limited")
    
    def compare_all_methods(self, method_results: Dict[str, List[Dict[str, Any]]], 
                          metric_name: str = 'target_improvement') -> Dict[str, ComparisonResult]:
        """
        Compare all method pairs statistically.
        
        Args:
            method_results: Results for all methods
            metric_name: Metric to compare (default: target_improvement)
            
        Returns:
            Dictionary of pairwise comparison results
        """
        comparisons = {}
        method_names = list(method_results.keys())
        
        # Perform pairwise comparisons
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names[i+1:], i+1):
                
                # Extract values for comparison
                values1 = self._extract_metric_values(method_results[method1], metric_name)
                values2 = self._extract_metric_values(method_results[method2], metric_name)
                
                if len(values1) > 1 and len(values2) > 1:
                    comparison = self._compare_two_methods(
                        method1, values1, method2, values2
                    )
                    comparisons[f"{method1}_vs_{method2}"] = comparison
                else:
                    logger.warning(f"Insufficient data for comparison: {method1} vs {method2}")
        
        # Apply multiple comparisons correction
        if len(comparisons) > 1:
            comparisons = self._apply_multiple_comparisons_correction(comparisons)
        
        return comparisons
    
    def compute_summary_statistics(self, method_results: Dict[str, List[Dict[str, Any]]], 
                                 metrics: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Compute summary statistics for all methods.
        
        Args:
            method_results: Results for all methods
            metrics: List of metrics to analyze
            
        Returns:
            Summary statistics for each method
        """
        if metrics is None:
            metrics = ['target_improvement', 'structure_accuracy', 'sample_efficiency', 'convergence_steps']
        
        summary_stats = {}
        
        for method_name, results in method_results.items():
            method_stats = {'method_name': method_name}
            
            for metric in metrics:
                values = self._extract_metric_values(results, metric)
                
                if values:
                    stats_dict = self._compute_descriptive_statistics(values)
                    # Prefix with metric name
                    for stat_name, stat_value in stats_dict.items():
                        method_stats[f"{metric}_{stat_name}"] = stat_value
                else:
                    # Fill with zeros if no data
                    for stat_name in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']:
                        method_stats[f"{metric}_{stat_name}"] = 0.0
                    method_stats[f"{metric}_count"] = 0
            
            summary_stats[method_name] = method_stats
        
        return summary_stats
    
    def perform_anova(self, method_results: Dict[str, List[Dict[str, Any]]], 
                     metric_name: str = 'target_improvement') -> Dict[str, Any]:
        """
        Perform one-way ANOVA to test for differences between methods.
        
        Args:
            method_results: Results for all methods
            metric_name: Metric to analyze
            
        Returns:
            ANOVA results
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available - cannot perform ANOVA")
            return {'error': 'scipy not available'}
        
        # Extract values for all methods
        all_values = []
        method_names = []
        
        for method_name, results in method_results.items():
            values = self._extract_metric_values(results, metric_name)
            if values:
                all_values.append(values)
                method_names.append(method_name)
        
        if len(all_values) < 2:
            return {'error': 'Need at least 2 methods with data for ANOVA'}
        
        try:
            # Perform one-way ANOVA
            f_statistic, p_value = stats.f_oneway(*all_values)
            
            # Compute effect size (eta-squared)
            ss_between = sum(len(group) * (onp.mean(group) - onp.mean(onp.concatenate(all_values)))**2 
                           for group in all_values)
            ss_total = sum((onp.array(group) - onp.mean(onp.concatenate(all_values)))**2 
                          for group in all_values for value in group)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
            
            return {
                'metric': metric_name,
                'methods': method_names,
                'f_statistic': float(f_statistic),
                'p_value': float(p_value),
                'significant': p_value < self.significance_level,
                'eta_squared': float(eta_squared),
                'sample_sizes': [len(values) for values in all_values],
                'group_means': [float(onp.mean(values)) for values in all_values]
            }
            
        except Exception as e:
            logger.error(f"ANOVA failed: {e}")
            return {'error': str(e)}
    
    def compute_effect_sizes(self, method_results: Dict[str, List[Dict[str, Any]]], 
                           metric_name: str = 'target_improvement') -> Dict[str, float]:
        """
        Compute effect sizes for all method pairs.
        
        Args:
            method_results: Results for all methods
            metric_name: Metric to analyze
            
        Returns:
            Effect sizes for all pairs
        """
        effect_sizes = {}
        method_names = list(method_results.keys())
        
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names[i+1:], i+1):
                
                values1 = self._extract_metric_values(method_results[method1], metric_name)
                values2 = self._extract_metric_values(method_results[method2], metric_name)
                
                if values1 and values2:
                    effect_size = self._compute_cohens_d(values1, values2)
                    effect_sizes[f"{method1}_vs_{method2}"] = effect_size
        
        return effect_sizes
    
    def _extract_metric_values(self, results: List[Dict[str, Any]], metric_name: str) -> List[float]:
        """Extract metric values from results list."""
        values = []
        for result in results:
            if result and metric_name in result:
                value = result[metric_name]
                if isinstance(value, (int, float)) and not onp.isnan(value):
                    values.append(float(value))
        return values
    
    def _compare_two_methods(self, method1_name: str, values1: List[float], 
                           method2_name: str, values2: List[float]) -> ComparisonResult:
        """Compare two methods using t-test."""
        
        # Compute descriptive statistics
        mean1, std1 = float(onp.mean(values1)), float(onp.std(values1, ddof=1))
        mean2, std2 = float(onp.mean(values2)), float(onp.std(values2, ddof=1))
        mean_diff = mean1 - mean2
        
        # Perform t-test
        if SCIPY_AVAILABLE and len(values1) > 1 and len(values2) > 1:
            try:
                t_stat, p_val = stats.ttest_ind(values1, values2)
                t_stat, p_val = float(t_stat), float(p_val)
                
                # Compute confidence interval for mean difference
                pooled_std = onp.sqrt(((len(values1) - 1) * std1**2 + (len(values2) - 1) * std2**2) / 
                                    (len(values1) + len(values2) - 2))
                se_diff = pooled_std * onp.sqrt(1/len(values1) + 1/len(values2))
                df = len(values1) + len(values2) - 2
                t_critical = stats.t.ppf(1 - self.significance_level/2, df)
                
                ci_lower = mean_diff - t_critical * se_diff
                ci_upper = mean_diff + t_critical * se_diff
                confidence_interval = (float(ci_lower), float(ci_upper))
                
            except Exception as e:
                logger.warning(f"T-test failed for {method1_name} vs {method2_name}: {e}")
                t_stat, p_val = 0.0, 1.0
                confidence_interval = (mean_diff, mean_diff)
        else:
            t_stat, p_val = 0.0, 1.0
            confidence_interval = (mean_diff, mean_diff)
        
        # Compute effect size (Cohen's d)
        effect_size = self._compute_cohens_d(values1, values2)
        
        return ComparisonResult(
            method1_name=method1_name,
            method2_name=method2_name,
            method1_mean=mean1,
            method1_std=std1,
            method2_mean=mean2,
            method2_std=std2,
            mean_difference=mean_diff,
            t_statistic=t_stat,
            p_value=p_val,
            significant=p_val < self.significance_level,
            effect_size=effect_size,
            confidence_interval=confidence_interval,
            sample_sizes=(len(values1), len(values2))
        )
    
    def _compute_descriptive_statistics(self, values: List[float]) -> Dict[str, float]:
        """Compute descriptive statistics for a list of values."""
        if not values:
            return {stat: 0.0 for stat in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'count']}
        
        values_array = onp.array(values)
        
        return {
            'mean': float(onp.mean(values_array)),
            'std': float(onp.std(values_array, ddof=1)) if len(values) > 1 else 0.0,
            'min': float(onp.min(values_array)),
            'max': float(onp.max(values_array)),
            'median': float(onp.median(values_array)),
            'q25': float(onp.percentile(values_array, 25)),
            'q75': float(onp.percentile(values_array, 75)),
            'count': len(values)
        }
    
    def _compute_cohens_d(self, values1: List[float], values2: List[float]) -> float:
        """Compute Cohen's d effect size."""
        if not values1 or not values2:
            return 0.0
        
        mean1, mean2 = onp.mean(values1), onp.mean(values2)
        
        if len(values1) == 1 and len(values2) == 1:
            return 0.0
        
        # Pooled standard deviation
        n1, n2 = len(values1), len(values2)
        if n1 > 1 and n2 > 1:
            var1 = onp.var(values1, ddof=1)
            var2 = onp.var(values2, ddof=1)
            pooled_std = onp.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        elif n1 > 1:
            pooled_std = onp.std(values1, ddof=1)
        elif n2 > 1:
            pooled_std = onp.std(values2, ddof=1)
        else:
            return 0.0
        
        if pooled_std == 0:
            return 0.0
        
        return float((mean1 - mean2) / pooled_std)
    
    def _apply_multiple_comparisons_correction(self, comparisons: Dict[str, ComparisonResult]) -> Dict[str, ComparisonResult]:
        """Apply multiple comparisons correction to p-values."""
        if not SCIPY_AVAILABLE or len(comparisons) <= 1:
            return comparisons
        
        p_values = [comp.p_value for comp in comparisons.values()]
        comparison_names = list(comparisons.keys())
        
        try:
            if self.correction_method == "bonferroni":
                corrected_p_values = [p * len(p_values) for p in p_values]
                corrected_p_values = [min(1.0, p) for p in corrected_p_values]  # Cap at 1.0
            
            elif self.correction_method == "holm":
                # Holm-Bonferroni correction
                sorted_indices = onp.argsort(p_values)
                corrected_p_values = [0.0] * len(p_values)
                
                for i, idx in enumerate(sorted_indices):
                    corrected_p = p_values[idx] * (len(p_values) - i)
                    if i > 0:
                        corrected_p = max(corrected_p, corrected_p_values[sorted_indices[i-1]])
                    corrected_p_values[idx] = min(1.0, corrected_p)
            
            else:
                logger.warning(f"Unknown correction method: {self.correction_method}")
                corrected_p_values = p_values
            
            # Update comparison results with corrected p-values
            corrected_comparisons = {}
            for i, (name, comparison) in enumerate(comparisons.items()):
                corrected_comparison = ComparisonResult(
                    method1_name=comparison.method1_name,
                    method2_name=comparison.method2_name,
                    method1_mean=comparison.method1_mean,
                    method1_std=comparison.method1_std,
                    method2_mean=comparison.method2_mean,
                    method2_std=comparison.method2_std,
                    mean_difference=comparison.mean_difference,
                    t_statistic=comparison.t_statistic,
                    p_value=corrected_p_values[i],
                    significant=corrected_p_values[i] < self.significance_level,
                    effect_size=comparison.effect_size,
                    confidence_interval=comparison.confidence_interval,
                    sample_sizes=comparison.sample_sizes
                )
                corrected_comparisons[name] = corrected_comparison
            
            logger.info(f"Applied {self.correction_method} correction to {len(comparisons)} comparisons")
            return corrected_comparisons
            
        except Exception as e:
            logger.error(f"Multiple comparisons correction failed: {e}")
            return comparisons
    
    def generate_summary_report(self, method_results: Dict[str, List[Dict[str, Any]]], 
                              output_file: Optional[str] = None) -> str:
        """Generate a comprehensive statistical summary report."""
        
        report_lines = []
        report_lines.append("ACBO Methods Statistical Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Summary statistics
        summary_stats = self.compute_summary_statistics(method_results)
        report_lines.append("Summary Statistics:")
        report_lines.append("-" * 20)
        
        for method_name, stats in summary_stats.items():
            report_lines.append(f"\n{method_name}:")
            for metric in ['target_improvement', 'structure_accuracy', 'sample_efficiency']:
                mean_val = stats.get(f'{metric}_mean', 0.0)
                std_val = stats.get(f'{metric}_std', 0.0)
                count_val = stats.get(f'{metric}_count', 0)
                report_lines.append(f"  {metric}: {mean_val:.4f} ± {std_val:.4f} (n={count_val})")
        
        # Pairwise comparisons
        comparisons = self.compare_all_methods(method_results)
        if comparisons:
            report_lines.append("\n\nPairwise Comparisons:")
            report_lines.append("-" * 22)
            
            for comp_name, comp_result in comparisons.items():
                report_lines.append(f"\n{comp_name}:")
                report_lines.append(f"  Mean difference: {comp_result.mean_difference:.4f}")
                report_lines.append(f"  P-value: {comp_result.p_value:.4f}")
                report_lines.append(f"  Significant: {comp_result.significant}")
                report_lines.append(f"  Effect size (Cohen's d): {comp_result.effect_size:.4f}")
                report_lines.append(f"  95% CI: ({comp_result.confidence_interval[0]:.4f}, {comp_result.confidence_interval[1]:.4f})")
        
        # ANOVA
        anova_result = self.perform_anova(method_results)
        if 'error' not in anova_result:
            report_lines.append("\n\nOne-way ANOVA:")
            report_lines.append("-" * 15)
            report_lines.append(f"F-statistic: {anova_result['f_statistic']:.4f}")
            report_lines.append(f"P-value: {anova_result['p_value']:.4f}")
            report_lines.append(f"Significant: {anova_result['significant']}")
            report_lines.append(f"Effect size (eta²): {anova_result['eta_squared']:.4f}")
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_text)
                logger.info(f"Statistical report saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        
        return report_text