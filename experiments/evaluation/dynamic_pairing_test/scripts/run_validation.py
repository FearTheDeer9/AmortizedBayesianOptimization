#!/usr/bin/env python3
"""
Main script to run dynamic pairing validation.

This script validates the three core functionalities:
1. Dynamic pairing creation
2. Untrained model usage  
3. Trained model loading
"""

import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any

# Add paths
import sys
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from experiments.evaluation.dynamic_pairing_test.src.validation_runner import ValidationRunner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Run dynamic pairing validation')
    parser.add_argument('--config', type=Path,
                       default=Path(__file__).parent.parent / 'configs' / 'validation_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', choices=['quick', 'standard', 'comprehensive'],
                       default='standard',
                       help='Validation mode')
    parser.add_argument('--output-dir', type=Path,
                       help='Output directory (default: results/validation_[timestamp])')
    parser.add_argument('--pairing', type=str,
                       help='Test specific pairing only')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(__file__).parent.parent / 'results' / f'validation_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Get experiments directory
    experiments_dir = Path(__file__).parent.parent.parent
    logger.info(f"Experiments directory: {experiments_dir}")
    
    # Create validation runner
    runner = ValidationRunner(config)
    
    # Run validation
    if args.pairing:
        logger.info(f"Testing specific pairing: {args.pairing}")
        # TODO: Implement specific pairing test
        logger.warning("Specific pairing testing not yet implemented")
        return
    else:
        logger.info(f"Running {args.mode} validation")
        results = runner.run_comprehensive_validation(experiments_dir)
    
    # Export results
    runner.export_results(results, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if 'summary' in results:
        summary = results['summary']
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Pairing Success Rate: {summary['pairing_success_rate']:.1%}")
        print(f"All Functionalities Working: {summary['all_functionalities_working']}")
    
    if 'functionality_tests' in results:
        func_tests = results['functionality_tests']
        print(f"\nFunctionality Test Results:")
        print(f"  Dynamic Pairing: {'PASS' if func_tests.get('dynamic_pairing', {}).get('pairing_creation', False) else 'FAIL'}")
        print(f"  Untrained Models: {'PASS' if func_tests.get('untrained_models', {}).get('untrained_policy', False) else 'FAIL'}")
        
        if 'trained_model_loading' in func_tests:
            tl_success = sum(1 for v in func_tests['trained_model_loading'].values() if v)
            tl_total = len(func_tests['trained_model_loading'])
            print(f"  Trained Loading: {tl_success}/{tl_total} checkpoints")
    
    print(f"\nDetailed results saved to: {output_dir}")
    print("=" * 60)
    
    # Set exit code based on results
    if results.get('summary', {}).get('overall_status') == 'PASS':
        logger.info("Validation PASSED")
        exit(0)
    else:
        logger.error("Validation FAILED")
        exit(1)


if __name__ == "__main__":
    main()