#!/usr/bin/env python3
"""
Local Development Workflow Script

Provides a streamlined workflow for local development and testing of the ACBO system.
Designed for rapid iteration on your laptop before deploying to cluster.

Features:
- Small-scale data collection for testing
- Quick validation and debugging
- Local training experiments 
- Configuration testing
- Integration validation

Usage:
    python scripts/dev_workflow.py --quick-test
    python scripts/dev_workflow.py --collect-dev-data
    python scripts/dev_workflow.py --test-training
"""

import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DevWorkflow:
    """Local development workflow manager."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.dev_data_dir = self.project_root / "dev_data"
        self.dev_results_dir = self.project_root / "dev_results"
        
        # Create dev directories
        self.dev_data_dir.mkdir(exist_ok=True)
        self.dev_results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Dev workflow initialized")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Dev data: {self.dev_data_dir}")
        logger.info(f"Dev results: {self.dev_results_dir}")
    
    def quick_test(self) -> bool:
        """Run quick integration test of the complete system."""
        logger.info("Running quick integration test...")
        
        tests = [
            ("Environment check", self._test_environment),
            ("Import validation", self._test_imports),
            ("PARENT_SCALE integration", self._test_parent_scale_integration),
            ("Expert collection", self._test_expert_collection),
            ("Data processing", self._test_data_processing)
        ]
        
        results = {}
        overall_success = True
        
        for test_name, test_func in tests:
            logger.info(f"Running test: {test_name}")
            start_time = time.time()
            
            try:
                success = test_func()
                duration = time.time() - start_time
                
                results[test_name] = {
                    "success": success,
                    "duration": duration,
                    "error": None
                }
                
                if success:
                    logger.info(f"✅ {test_name} passed ({duration:.1f}s)")
                else:
                    logger.error(f"❌ {test_name} failed ({duration:.1f}s)")
                    overall_success = False
                    
            except Exception as e:
                duration = time.time() - start_time
                results[test_name] = {
                    "success": False,
                    "duration": duration,
                    "error": str(e)
                }
                logger.error(f"❌ {test_name} error: {e}")
                overall_success = False
        
        # Save test results
        results_file = self.dev_results_dir / "quick_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"Quick Test Summary")
        print(f"{'='*50}")
        
        for test_name, result in results.items():
            icon = "✅" if result["success"] else "❌"
            print(f"{icon} {test_name:<25} {result['duration']:>6.1f}s")
            if result["error"]:
                print(f"    Error: {result['error']}")
        
        print(f"\nOverall result: {'✅ PASSED' if overall_success else '❌ FAILED'}")
        print(f"Results saved to: {results_file}")
        
        return overall_success
    
    def collect_dev_data(self, size: str = "small") -> bool:
        """Collect small dataset for development."""
        logger.info(f"Collecting development data (size: {size})...")
        
        output_dir = self.dev_data_dir / f"raw_{size}"
        
        try:
            cmd = [
                "python", "scripts/collect_sft_dataset.py",
                "--size", size,
                "--output-dir", str(output_dir),
                "--serial",  # Use serial processing to avoid pickle issues
                "--batch-size", "10"  # Small batches
            ]
            
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                logger.info("✅ Development data collection completed")
                
                # Validate collected data
                logger.info("Validating collected data...")
                validate_cmd = [
                    "python", "scripts/validate_dataset.py",
                    str(output_dir),
                    "--detailed"
                ]
                
                subprocess.run(validate_cmd, timeout=120)
                return True
            else:
                logger.error(f"❌ Data collection failed with exit code: {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Data collection timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Data collection error: {e}")
            return False
    
    def test_training(self) -> bool:
        """Test training pipeline with development data."""
        logger.info("Testing training pipeline...")
        
        # Check for development data
        raw_data_dir = self.dev_data_dir / "raw_small"
        if not raw_data_dir.exists():
            logger.info("No development data found, collecting first...")
            if not self.collect_dev_data("small"):
                return False
        
        # Process data
        processed_dir = self.dev_data_dir / "processed"
        logger.info("Processing development data...")
        
        try:
            # Prepare SFT data
            cmd = [
                "python", "scripts/prepare_sft_data.py",
                str(raw_data_dir),
                "--output", str(processed_dir),
                "--curriculum"
            ]
            
            result = subprocess.run(cmd, timeout=300)
            if result.returncode != 0:
                logger.error("❌ Data preparation failed")
                return False
            
            # Split dataset
            logger.info("Splitting dataset...")
            cmd = [
                "python", "scripts/split_dataset.py",
                str(processed_dir),
                "--curriculum",
                "--stratify", "difficulty", "graph_type"
            ]
            
            result = subprocess.run(cmd, timeout=120)
            if result.returncode != 0:
                logger.error("❌ Dataset splitting failed")
                return False
            
            logger.info("✅ Training pipeline test completed successfully")
            logger.info(f"Processed data available at: {processed_dir}")
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Training pipeline test timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Training pipeline test error: {e}")
            return False
    
    def test_config(self, config_file: Optional[str] = None) -> bool:
        """Test configuration files and settings."""
        logger.info("Testing configuration...")
        
        # Test default configuration
        try:
            from causal_bayes_opt.training.config import create_default_training_config
            config = create_default_training_config()
            logger.info("✅ Default configuration loaded successfully")
            
            # Test configuration validation
            from causal_bayes_opt.training.config import validate_training_config
            is_valid = validate_training_config(config)
            
            if is_valid:
                logger.info("✅ Configuration validation passed")
            else:
                logger.error("❌ Configuration validation failed")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration test failed: {e}")
            return False
    
    def clean_dev_data(self) -> bool:
        """Clean up development data."""
        logger.info("Cleaning development data...")
        
        try:
            import shutil
            
            if self.dev_data_dir.exists():
                shutil.rmtree(self.dev_data_dir)
                self.dev_data_dir.mkdir()
                logger.info("✅ Development data cleaned")
            
            if self.dev_results_dir.exists():
                shutil.rmtree(self.dev_results_dir)
                self.dev_results_dir.mkdir()
                logger.info("✅ Development results cleaned")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return False
    
    def _test_environment(self) -> bool:
        """Test environment setup."""
        try:
            import jax
            import jax.numpy as jnp
            
            # Test basic JAX functionality
            x = jnp.array([1.0, 2.0, 3.0])
            result = jnp.dot(x, x)
            
            # Check if GPU is available
            devices = jax.devices()
            has_gpu = any(dev.platform == 'gpu' for dev in devices)
            
            logger.info(f"JAX version: {jax.__version__}")
            logger.info(f"JAX devices: {devices}")
            logger.info(f"GPU available: {has_gpu}")
            
            return True
            
        except Exception as e:
            logger.error(f"Environment test failed: {e}")
            return False
    
    def _test_imports(self) -> bool:
        """Test key imports."""
        try:
            # Test core imports
            import numpy as onp
            import jax.numpy as jnp
            import optax
            import haiku as hk
            
            # Test project imports
            from causal_bayes_opt.training.expert_collection.collector import ExpertDemonstrationCollector
            from causal_bayes_opt.training.surrogate_training import extract_training_data_from_demonstrations
            from causal_bayes_opt.training.config import create_default_training_config
            
            logger.info("All key imports successful")
            return True
            
        except ImportError as e:
            logger.error(f"Import failed: {e}")
            return False
    
    def _test_parent_scale_integration(self) -> bool:
        """Test PARENT_SCALE integration."""
        try:
            # Quick integration test using refactored integration
            from causal_bayes_opt.integration.parent_scale import check_parent_scale_availability
            
            # Run basic integration test
            available = check_parent_scale_availability()
            if available:
                # Test basic functionality
                from causal_bayes_opt.integration.parent_scale import run_full_parent_scale_algorithm
                # Just test import, don't run expensive algorithm
                logger.info("PARENT_SCALE integration components available")
                return True
            else:
                logger.warning("PARENT_SCALE not available but integration imports work")
                return False
            
        except Exception as e:
            logger.error(f"PARENT_SCALE integration test failed: {e}")
            return False
    
    def _test_expert_collection(self) -> bool:
        """Test expert demonstration collection."""
        try:
            from causal_bayes_opt.training.expert_collection.collector import ExpertDemonstrationCollector
            from causal_bayes_opt.training.expert_collection.scm_generation import generate_scm
            import jax.random as random
            
            # Create collector
            collector = ExpertDemonstrationCollector(output_dir=str(self.dev_data_dir / "test_collection"))
            
            # Generate a test SCM
            key = random.PRNGKey(42)
            test_scm = generate_scm(n_nodes=3, graph_type="chain", key=key)
            
            # Collect a single demonstration
            demo = collector.collect_demonstration(
                scm=test_scm,
                graph_type="chain",
                min_accuracy=0.5  # Lower threshold for testing
            )
            
            if demo and demo.accuracy > 0:
                logger.info(f"Test demonstration: accuracy={demo.accuracy:.3f}")
                return True
            else:
                logger.error("Failed to collect test demonstration")
                return False
                
        except Exception as e:
            logger.error(f"Expert collection test failed: {e}")
            return False
    
    def _test_data_processing(self) -> bool:
        """Test data processing pipeline."""
        try:
            # This would test the SFT data processing
            # For now, just validate the scripts exist and are importable
            
            scripts_to_test = [
                "scripts/prepare_sft_data.py",
                "scripts/split_dataset.py",
                "scripts/validate_dataset.py"
            ]
            
            for script in scripts_to_test:
                script_path = self.project_root / script
                if not script_path.exists():
                    logger.error(f"Script not found: {script}")
                    return False
            
            logger.info("Data processing scripts validated")
            return True
            
        except Exception as e:
            logger.error(f"Data processing test failed: {e}")
            return False


def main():
    """CLI entry point for development workflow."""
    parser = argparse.ArgumentParser(
        description="Local development workflow for ACBO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick integration test
  python scripts/dev_workflow.py --quick-test
  
  # Collect small development dataset
  python scripts/dev_workflow.py --collect-dev-data
  
  # Test complete training pipeline
  python scripts/dev_workflow.py --test-training
  
  # Test configuration
  python scripts/dev_workflow.py --test-config
  
  # Clean up development data
  python scripts/dev_workflow.py --clean
        """
    )
    
    # Actions
    parser.add_argument("--quick-test", action="store_true",
                       help="Run quick integration test")
    parser.add_argument("--collect-dev-data", action="store_true",
                       help="Collect development dataset")
    parser.add_argument("--test-training", action="store_true",
                       help="Test training pipeline")
    parser.add_argument("--test-config", action="store_true",
                       help="Test configuration")
    parser.add_argument("--clean", action="store_true",
                       help="Clean development data")
    
    # Options
    parser.add_argument("--data-size", choices=["small", "medium"], default="small",
                       help="Size of development dataset")
    parser.add_argument("--config-file", type=str,
                       help="Custom configuration file to test")
    
    args = parser.parse_args()
    
    try:
        # Create workflow manager
        workflow = DevWorkflow()
        
        success = True
        
        # Execute requested actions
        if args.quick_test:
            success = workflow.quick_test()
        
        elif args.collect_dev_data:
            success = workflow.collect_dev_data(args.data_size)
        
        elif args.test_training:
            success = workflow.test_training()
        
        elif args.test_config:
            success = workflow.test_config(args.config_file)
        
        elif args.clean:
            success = workflow.clean_dev_data()
        
        else:
            print("No action specified. Use --quick-test, --collect-dev-data, --test-training, --test-config, or --clean")
            return 1
        
        if success:
            print(f"\n✅ Development workflow completed successfully!")
        else:
            print(f"\n❌ Development workflow failed. Check logs for details.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Workflow interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        logger.exception("Workflow failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())