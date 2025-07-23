#!/usr/bin/env python3
"""
WandB Metrics Extractor for Causal Bayesian Optimization

This script extracts F1 scores and P(Parents|Data) metrics from WandB runs
and creates easy-to-understand visualizations.
"""

import sys
import os
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("WandB not available")

class WandBMetricsExtractor:
    """Extract and visualize F1 scores and P(Parents|Data) from WandB runs."""
    
    def __init__(self, project_name: str = "causal_bayes_opt", entity: Optional[str] = None):
        """
        Initialize the metrics extractor.
        
        Args:
            project_name: WandB project name
            entity: WandB entity (username/team)
        """
        self.project_name = project_name
        self.entity = entity
        self.api = wandb.Api() if WANDB_AVAILABLE else None
        
    def get_recent_runs(self, limit: int = 10) -> List[Any]:
        """Get recent runs from WandB."""
        if not self.api:
            return []
        
        try:
            runs = self.api.runs(
                f"{self.entity}/{self.project_name}" if self.entity else self.project_name,
                filters={"tags": {"$in": ["acbo_comparison", "causal_discovery"]}},
                order="-created_at"
            )
            return list(runs)[:limit]
        except Exception as e:
            logger.error(f"Failed to get runs: {e}")
            return []
    
    def extract_metrics_from_run(self, run: Any) -> Dict[str, Any]:
        """Extract F1 scores and P(Parents|Data) from a single run."""
        if not run:
            return {}
        
        try:
            # Get run history
            history = run.history()
            
            # Extract step-by-step metrics
            methods = self._identify_methods(history)
            extracted_data = {}
            
            for method in methods:
                method_data = {
                    'steps': [],
                    'f1_scores': [],
                    'parent_probabilities': [],
                    'target_values': [],
                    'shd_values': [],
                    'uncertainties': []
                }
                
                # Extract metrics for this method
                for _, row in history.iterrows():
                    if f"{method}/f1_score" in row and not pd.isna(row[f"{method}/f1_score"]):
                        method_data['steps'].append(row.get('intervention_step', len(method_data['steps'])))
                        method_data['f1_scores'].append(row[f"{method}/f1_score"])
                        method_data['parent_probabilities'].append(
                            row.get(f"{method}/parent_probability", 
                                   row.get(f"{method}/true_parent_likelihood", 0.0))
                        )
                        method_data['target_values'].append(row.get(f"{method}/target_value", 0.0))
                        method_data['shd_values'].append(row.get(f"{method}/shd", 0.0))
                        method_data['uncertainties'].append(row.get(f"{method}/uncertainty", 0.0))
                
                extracted_data[method] = method_data
            
            return {
                'run_id': run.id,
                'run_name': run.name,
                'created_at': run.created_at,
                'config': run.config,
                'metrics': extracted_data
            }
            
        except Exception as e:
            logger.error(f"Failed to extract metrics from run {run.id}: {e}")
            return {}
    
    def _identify_methods(self, history: pd.DataFrame) -> List[str]:
        """Identify method names from history columns."""
        methods = set()
        
        for column in history.columns:
            if "/f1_score" in column:
                method = column.split("/f1_score")[0]
                methods.add(method)
        
        return sorted(list(methods))
    
    def create_f1_score_plot(self, metrics_data: Dict[str, Any], 
                            save_path: Optional[str] = None) -> str:
        """Create F1 score comparison plot."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        has_data = False
        for i, (method, data) in enumerate(metrics_data.items()):
            if data.get('f1_scores') and len(data['f1_scores']) > 0:
                ax.plot(data['steps'], data['f1_scores'], 
                       label=method, linewidth=2, color=colors[i % len(colors)],
                       marker='o', markersize=4)
                has_data = True
        
        if not has_data:
            # Create placeholder plot with message
            ax.text(0.5, 0.5, 'No F1 score data available\n(Baseline methods may not track structure metrics)', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
        
        ax.set_xlabel('Intervention Step')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score by Step (Structure Recovery Performance)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add horizontal line at 0.5 for reference
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"F1 score plot saved to {save_path}")
        
        return save_path or "f1_score_plot.png"
    
    def create_parent_probability_plot(self, metrics_data: Dict[str, Any], 
                                     save_path: Optional[str] = None) -> str:
        """Create P(Parents|Data) comparison plot."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        has_data = False
        for i, (method, data) in enumerate(metrics_data.items()):
            if data.get('parent_probabilities') and len(data['parent_probabilities']) > 0:
                ax.plot(data['steps'], data['parent_probabilities'], 
                       label=method, linewidth=2, color=colors[i % len(colors)],
                       marker='s', markersize=4)
                has_data = True
        
        if not has_data:
            # Create placeholder plot with message
            ax.text(0.5, 0.5, 'No P(Parents|Data) metrics available\n(Baseline methods may not track parent probabilities)', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
        
        ax.set_xlabel('Intervention Step')
        ax.set_ylabel('P(True Parents | Data)')
        ax.set_title('Probability of True Parents Given Data by Step')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add horizontal line at 0.5 for reference
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Parent probability plot saved to {save_path}")
        
        return save_path or "parent_probability_plot.png"
    
    def create_combined_dashboard(self, metrics_data: Dict[str, Any], 
                                save_path: Optional[str] = None) -> str:
        """Create combined dashboard with F1 scores and P(Parents|Data)."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # F1 Score plot
        for i, (method, data) in enumerate(metrics_data.items()):
            if data['f1_scores']:
                ax1.plot(data['steps'], data['f1_scores'], 
                        label=method, linewidth=2, color=colors[i % len(colors)],
                        marker='o', markersize=3)
        
        ax1.set_xlabel('Intervention Step')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('F1 Score by Step (Structure Recovery)')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # P(Parents|Data) plot
        for i, (method, data) in enumerate(metrics_data.items()):
            if data['parent_probabilities']:
                ax2.plot(data['steps'], data['parent_probabilities'], 
                        label=method, linewidth=2, color=colors[i % len(colors)],
                        marker='s', markersize=3)
        
        ax2.set_xlabel('Intervention Step')
        ax2.set_ylabel('P(True Parents | Data)')
        ax2.set_title('P(Parents|Data) by Step')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Target value plot
        for i, (method, data) in enumerate(metrics_data.items()):
            if data['target_values']:
                ax3.plot(data['steps'], data['target_values'], 
                        label=method, linewidth=2, color=colors[i % len(colors)],
                        marker='^', markersize=3)
        
        ax3.set_xlabel('Intervention Step')
        ax3.set_ylabel('Target Value')
        ax3.set_title('Target Value by Step (Optimization)')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # SHD plot (lower is better)
        for i, (method, data) in enumerate(metrics_data.items()):
            if data['shd_values']:
                ax4.plot(data['steps'], data['shd_values'], 
                        label=method, linewidth=2, color=colors[i % len(colors)],
                        marker='d', markersize=3)
        
        ax4.set_xlabel('Intervention Step')
        ax4.set_ylabel('Structural Hamming Distance')
        ax4.set_title('SHD by Step (Structure Distance, Lower is Better)')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Combined dashboard saved to {save_path}")
        
        return save_path or "combined_dashboard.png"
    
    def generate_summary_report(self, metrics_data: Dict[str, Any]) -> str:
        """Generate text summary report."""
        report = []
        report.append("🎯 Causal Bayesian Optimization - Results Summary")
        report.append("=" * 60)
        
        has_f1_data = False
        has_parent_prob_data = False
        
        for method, data in metrics_data.items():
            report.append(f"\n📊 {method}:")
            
            if data.get('f1_scores') and len(data['f1_scores']) > 0:
                has_f1_data = True
                final_f1 = data['f1_scores'][-1]
                max_f1 = max(data['f1_scores'])
                avg_f1 = sum(data['f1_scores']) / len(data['f1_scores'])
                
                report.append(f"  F1 Score: Final={final_f1:.3f}, Max={max_f1:.3f}, Avg={avg_f1:.3f}")
            else:
                report.append(f"  F1 Score: No data available")
            
            if data.get('parent_probabilities') and len(data['parent_probabilities']) > 0:
                has_parent_prob_data = True
                final_prob = data['parent_probabilities'][-1]
                max_prob = max(data['parent_probabilities'])
                avg_prob = sum(data['parent_probabilities']) / len(data['parent_probabilities'])
                
                report.append(f"  P(Parents|Data): Final={final_prob:.3f}, Max={max_prob:.3f}, Avg={avg_prob:.3f}")
            else:
                report.append(f"  P(Parents|Data): No data available")
            
            if data.get('target_values') and len(data['target_values']) > 0:
                final_target = data['target_values'][-1]
                max_target = max(data['target_values'])
                improvement = final_target - data['target_values'][0] if len(data['target_values']) > 1 else 0
                
                report.append(f"  Target Value: Final={final_target:.3f}, Max={max_target:.3f}, Improvement={improvement:+.3f}")
            else:
                report.append(f"  Target Value: No data available")
        
        # Find best performing method (with error handling)
        if has_f1_data:
            best_f1_method = max(metrics_data.items(), 
                               key=lambda x: max(x[1].get('f1_scores', [0])) if x[1].get('f1_scores') else 0)
            if best_f1_method[1].get('f1_scores'):
                report.append(f"\n🏆 Best Performers:")
                report.append(f"  Best F1 Score: {best_f1_method[0]} ({max(best_f1_method[1]['f1_scores']):.3f})")
        
        if has_parent_prob_data:
            best_prob_method = max(metrics_data.items(), 
                                 key=lambda x: max(x[1].get('parent_probabilities', [0])) if x[1].get('parent_probabilities') else 0)
            if best_prob_method[1].get('parent_probabilities'):
                if not has_f1_data:
                    report.append(f"\n🏆 Best Performers:")
                report.append(f"  Best P(Parents|Data): {best_prob_method[0]} ({max(best_prob_method[1]['parent_probabilities']):.3f})")
        
        if not has_f1_data and not has_parent_prob_data:
            report.append(f"\n⚠️ No structure learning metrics available")
            report.append(f"  Baseline methods may not track F1 scores or parent probabilities")
            report.append(f"  Consider running with enriched policy methods for structure metrics")
        
        return "\n".join(report)
    
    def extract_and_visualize_latest_run(self, output_dir: str = "wandb_extracted_metrics", debug: bool = False) -> None:
        """Extract and visualize metrics from the latest run."""
        if not WANDB_AVAILABLE:
            logger.error("WandB not available")
            return
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get latest run
        runs = self.get_recent_runs(limit=1)
        if not runs:
            logger.error("No runs found")
            return
        
        latest_run = runs[0]
        logger.info(f"Extracting metrics from run: {latest_run.name}")
        
        # Debug mode: show available metrics
        if debug:
            print("\n🔍 DEBUG: Available metrics in run:")
            history = latest_run.history()
            print(f"  Columns: {list(history.columns)[:20]}...")  # Show first 20 columns
            print(f"  Total columns: {len(history.columns)}")
            print(f"  Rows: {len(history)}")
            
            # Show which methods have F1/parent prob metrics
            f1_cols = [col for col in history.columns if 'f1' in col.lower()]
            parent_cols = [col for col in history.columns if 'parent' in col.lower() or 'likelihood' in col.lower()]
            print(f"\n  F1 score columns found: {f1_cols[:10]}...")
            print(f"  Parent probability columns found: {parent_cols[:10]}...")
        
        # Extract metrics
        run_data = self.extract_metrics_from_run(latest_run)
        if not run_data.get('metrics'):
            logger.error("No metrics found in run")
            print("\n⚠️ No metrics could be extracted. This might be because:")
            print("  1. The run hasn't logged any metrics yet")
            print("  2. The metric names don't match expected patterns")
            print("  3. The run failed before logging metrics")
            print("\nTry running with debug=True to see available metrics")
            return
        
        metrics_data = run_data['metrics']
        
        # Check if we have any structure learning metrics
        has_structure_metrics = any(
            (method_data.get('f1_scores') or method_data.get('parent_probabilities'))
            for method_data in metrics_data.values()
        )
        
        if not has_structure_metrics:
            print("\n⚠️ No structure learning metrics (F1 scores or parent probabilities) found.")
            print("This is expected for baseline methods that don't track structure recovery.")
            print("The enriched policy methods should provide these metrics.")
        
        # Create visualizations
        f1_plot_path = output_path / "f1_scores_by_step.png"
        parent_plot_path = output_path / "parent_probabilities_by_step.png"
        dashboard_path = output_path / "combined_dashboard.png"
        
        self.create_f1_score_plot(metrics_data, str(f1_plot_path))
        self.create_parent_probability_plot(metrics_data, str(parent_plot_path))
        self.create_combined_dashboard(metrics_data, str(dashboard_path))
        
        # Generate summary report
        summary = self.generate_summary_report(metrics_data)
        summary_path = output_path / "summary_report.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print(summary)
        print(f"\n📁 Outputs saved to: {output_path}")
        print(f"  📊 F1 scores: {f1_plot_path}")
        print(f"  📊 P(Parents|Data): {parent_plot_path}")
        print(f"  📊 Combined dashboard: {dashboard_path}")
        print(f"  📝 Summary report: {summary_path}")


def main():
    """Main function to run the metrics extractor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract and visualize WandB metrics")
    parser.add_argument("--project", default="causal_bayes_opt", help="WandB project name")
    parser.add_argument("--entity", help="WandB entity (username/team)")
    parser.add_argument("--output", default="wandb_extracted_metrics", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to show available metrics")
    
    args = parser.parse_args()
    
    if not WANDB_AVAILABLE:
        print("❌ WandB not available. Please install with: pip install wandb")
        return
    
    extractor = WandBMetricsExtractor(args.project, args.entity)
    extractor.extract_and_visualize_latest_run(args.output, debug=args.debug)


if __name__ == "__main__":
    main()