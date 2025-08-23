"""
Validation runner for dynamic pairing test experiment.

This module validates that all model combinations can be created,
loaded, and executed successfully.
"""

import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import yaml
from dataclasses import dataclass

# Add paths for imports
import sys
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from experiments.evaluation.core.pairing_manager import (
    PairingManager, PairingConfig, ModelSpec, ModelType, create_standard_pairings
)
from experiments.evaluation.core.model_loader import ModelLoader
from experiments.evaluation.core.performance_utils import PerformanceProfiler, quick_profile
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.data_structures.scm import get_variables, get_target
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a single pairing."""
    pairing_name: str
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0
    memory_usage: float = 0.0
    interventions_completed: int = 0
    additional_info: Dict[str, Any] = None


class ValidationRunner:
    """Runs validation tests for dynamic pairing system."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validation runner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.exp_config = config['experiment']
        self.scm_config = config['scm_generation'] 
        self.data_config = config['data_generation']
        self.test_config = config.get('test_configuration', {})
        
        self.scm_factory = VariableSCMFactory(seed=self.exp_config['seed'])
        self.profiler = PerformanceProfiler()
        self.results = []
        
        logger.info(f"Initialized ValidationRunner for {self.exp_config['name']}")
    
    def create_test_scm(self, size: int) -> Dict[str, Any]:
        """Create a simple test SCM."""
        return self.scm_factory.create_variable_scm(
            num_variables=size,
            structure_type="random",
            edge_density=self.scm_config['edge_density']
        )
    
    def validate_checkpoint_loading(self, experiments_dir: Path) -> Dict[str, bool]:
        """
        Validate that all checkpoints can be loaded.
        
        Args:
            experiments_dir: Base experiments directory
            
        Returns:
            Dictionary mapping checkpoint paths to success status
        """
        logger.info("Validating checkpoint loading...")
        
        manager = PairingManager()
        discovered = manager.discover_all_checkpoints(experiments_dir)
        
        validation_results = {}
        
        # Test joint training checkpoints
        for checkpoint_dir in discovered['joint_training']:
            policy_path = checkpoint_dir / 'policy.pkl'
            surrogate_path = checkpoint_dir / 'surrogate.pkl'
            
            # Test policy loading
            try:
                policy_info = ModelLoader.verify_checkpoint(policy_path)
                validation_results[str(policy_path)] = policy_info.get('exists', False) and 'error' not in policy_info
            except Exception as e:
                validation_results[str(policy_path)] = False
                logger.error(f"Policy checkpoint validation failed {policy_path}: {e}")
            
            # Test surrogate loading
            try:
                surrogate_info = ModelLoader.verify_checkpoint(surrogate_path)
                validation_results[str(surrogate_path)] = surrogate_info.get('exists', False) and 'error' not in surrogate_info
            except Exception as e:
                validation_results[str(surrogate_path)] = False
                logger.error(f"Surrogate checkpoint validation failed {surrogate_path}: {e}")
        
        # Test policy-only checkpoints
        for policy_path in discovered['policy_only'][:5]:  # Limit for testing
            try:
                policy_info = ModelLoader.verify_checkpoint(policy_path)
                validation_results[str(policy_path)] = policy_info.get('exists', False) and 'error' not in policy_info
            except Exception as e:
                validation_results[str(policy_path)] = False
                logger.error(f"Policy-only checkpoint validation failed {policy_path}: {e}")
        
        # Test surrogate-only checkpoints  
        for surrogate_path in discovered['surrogate_only'][:5]:  # Limit for testing
            try:
                surrogate_info = ModelLoader.verify_checkpoint(surrogate_path)
                validation_results[str(surrogate_path)] = surrogate_info.get('exists', False) and 'error' not in surrogate_info
            except Exception as e:
                validation_results[str(surrogate_path)] = False
                logger.error(f"Surrogate-only checkpoint validation failed {surrogate_path}: {e}")
        
        success_rate = sum(validation_results.values()) / len(validation_results) if validation_results else 0
        logger.info(f"Checkpoint loading validation: {success_rate:.1%} success rate")
        
        return validation_results
    
    def validate_untrained_creation(self) -> Dict[str, bool]:
        """
        Validate creation of untrained models.
        
        Returns:
            Dictionary mapping model types to success status
        """
        logger.info("Validating untrained model creation...")
        
        validation_results = {}
        
        # Test untrained policy creation
        try:
            with quick_profile("untrained_policy_creation") as metrics:
                policy_fn = ModelLoader.create_untrained_policy(
                    architecture='simple_permutation_invariant',
                    hidden_dim=256,
                    seed=42
                )
            validation_results['untrained_policy'] = True
            logger.info(f"Untrained policy creation: {metrics.wall_time:.3f}s")
        except Exception as e:
            validation_results['untrained_policy'] = False
            logger.error(f"Untrained policy creation failed: {e}")
        
        # Test untrained surrogate creation
        try:
            with quick_profile("untrained_surrogate_creation") as metrics:
                variables = ['X1', 'X2', 'X3', 'X4', 'Y']
                surrogate_fn, config = ModelLoader.create_untrained_surrogate(
                    variables=variables,
                    target_variable='Y',
                    seed=42
                )
            validation_results['untrained_surrogate'] = True
            logger.info(f"Untrained surrogate creation: {metrics.wall_time:.3f}s")
        except Exception as e:
            validation_results['untrained_surrogate'] = False
            logger.error(f"Untrained surrogate creation failed: {e}")
        
        return validation_results
    
    def validate_single_pairing(self, 
                               pairing: PairingConfig,
                               test_scm: Dict[str, Any]) -> ValidationResult:
        """
        Validate a single pairing by running a mini-experiment.
        
        Args:
            pairing: Pairing configuration to test
            test_scm: Test SCM to use
            
        Returns:
            ValidationResult with success status and metrics
        """
        start_time = time.time()
        start_memory = self.profiler._get_memory_usage()
        
        try:
            logger.debug(f"Validating pairing: {pairing.name}")
            
            # Get SCM info
            variables = list(get_variables(test_scm))
            target_var = get_target(test_scm)
            
            # Load policy
            if pairing.policy_spec.model_type == ModelType.TRAINED:
                acquisition_fn = ModelLoader.load_policy(
                    pairing.policy_spec.checkpoint_path,
                    seed=pairing.policy_spec.seed
                )
            elif pairing.policy_spec.model_type == ModelType.UNTRAINED:
                acquisition_fn = ModelLoader.create_untrained_policy(
                    architecture=pairing.policy_spec.architecture or 'simple_permutation_invariant',
                    hidden_dim=pairing.policy_spec.hidden_dim or 256,
                    seed=pairing.policy_spec.seed
                )
            elif pairing.policy_spec.model_type == ModelType.RANDOM:
                acquisition_fn = ModelLoader.load_baseline('random', seed=pairing.policy_spec.seed)
            elif pairing.policy_spec.model_type == ModelType.ORACLE:
                acquisition_fn = ModelLoader.load_baseline('oracle', scm=test_scm)
            else:
                raise ValueError(f"Unknown policy type: {pairing.policy_spec.model_type}")
            
            # Load surrogate if specified
            surrogate_model = None
            if pairing.surrogate_spec.model_type == ModelType.TRAINED:
                surrogate_params, surrogate_arch = ModelLoader.load_surrogate(
                    pairing.surrogate_spec.checkpoint_path
                )
                surrogate_model = (surrogate_params, surrogate_arch)
            elif pairing.surrogate_spec.model_type == ModelType.UNTRAINED:
                surrogate_fn, surrogate_config = ModelLoader.create_untrained_surrogate(
                    variables=variables,
                    target_variable=target_var,
                    seed=pairing.surrogate_spec.seed
                )
                surrogate_model = (surrogate_fn, surrogate_config)
            
            # Create minimal test buffer
            buffer = ExperienceBuffer()
            obs_samples = sample_from_linear_scm(test_scm, n_samples=20, seed=42)
            for sample in obs_samples:
                buffer.add_observation(sample)
            
            # Test inference
            interventions_completed = 0
            for i in range(min(5, self.data_config['n_interventions'])):
                # Convert buffer to tensor
                tensor, _ = buffer_to_three_channel_tensor(buffer, target_var)
                
                # Test acquisition function
                intervention_dict = acquisition_fn(tensor, None, target_var, variables)
                
                # Validate output format
                if not isinstance(intervention_dict, dict):
                    raise ValueError(f"Invalid intervention output type: {type(intervention_dict)}")
                if 'targets' not in intervention_dict or 'values' not in intervention_dict:
                    raise ValueError(f"Missing required keys in intervention output")
                
                interventions_completed += 1
            
            # Success metrics
            end_time = time.time()
            end_memory = self.profiler._get_memory_usage()
            
            return ValidationResult(
                pairing_name=pairing.name,
                success=True,
                execution_time=end_time - start_time,
                memory_usage=end_memory - start_memory,
                interventions_completed=interventions_completed,
                additional_info={
                    'policy_type': pairing.policy_spec.model_type.value,
                    'surrogate_type': pairing.surrogate_spec.model_type.value,
                    'variables_count': len(variables)
                }
            )
            
        except Exception as e:
            end_time = time.time()
            error_details = traceback.format_exc()
            logger.error(f"Pairing validation failed for '{pairing.name}': {e}")
            logger.debug(f"Full error traceback:\n{error_details}")
            
            return ValidationResult(
                pairing_name=pairing.name,
                success=False,
                error_message=str(e),
                execution_time=end_time - start_time,
                additional_info={'error_traceback': error_details}
            )
    
    def run_validation(self, 
                      experiments_dir: Path,
                      mode: str = 'standard') -> Dict[str, Any]:
        """
        Run validation for all pairings.
        
        Args:
            experiments_dir: Base experiments directory
            mode: 'quick', 'standard', or 'comprehensive'
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Running validation in {mode} mode")
        
        # Adjust config based on mode
        if mode == 'quick':
            test_config = self.config.get('quick_test', {})
            sizes = test_config.get('sizes', [5])
            n_scms = test_config.get('n_scms_per_size', 1)
        elif mode == 'comprehensive':
            test_config = self.config.get('comprehensive_test', {})
            sizes = test_config.get('sizes', [5, 10, 20])
            n_scms = test_config.get('n_scms_per_size', 3)
        else:
            sizes = self.scm_config['sizes']
            n_scms = self.scm_config['n_scms_per_size']
        
        # Create test SCMs
        test_scms = []
        for size in sizes[:2]:  # Limit for validation
            scm = self.create_test_scm(size)
            test_scms.append((size, scm))
        
        # Create pairings
        manager = create_standard_pairings(experiments_dir)
        pairings = manager.get_pairings()
        
        logger.info(f"Testing {len(pairings)} pairings on {len(test_scms)} SCMs")
        
        # Run validation for each pairing
        all_results = []
        successful_pairings = 0
        
        for pairing in pairings:
            logger.info(f"Testing pairing: {pairing.name}")
            
            # Test on first SCM (enough for validation)
            if test_scms:
                size, test_scm = test_scms[0]
                result = self.validate_single_pairing(pairing, test_scm)
                all_results.append(result)
                
                if result.success:
                    successful_pairings += 1
                    logger.info(f"✓ {pairing.name}: {result.execution_time:.2f}s")
                else:
                    logger.warning(f"✗ {pairing.name}: {result.error_message}")
        
        # Summary statistics
        success_rate = successful_pairings / len(pairings) if pairings else 0
        
        validation_summary = {
            'total_pairings': len(pairings),
            'successful_pairings': successful_pairings,
            'failed_pairings': len(pairings) - successful_pairings,
            'success_rate': success_rate,
            'validation_results': all_results,
            'test_scms_used': len(test_scms),
            'mode': mode
        }
        
        logger.info(f"Validation complete: {success_rate:.1%} success rate "
                   f"({successful_pairings}/{len(pairings)} pairings)")
        
        return validation_summary
    
    def test_specific_functionality(self, experiments_dir: Path) -> Dict[str, Dict[str, bool]]:
        """
        Test the three specific functionalities requested.
        
        Args:
            experiments_dir: Base experiments directory
            
        Returns:
            Dictionary with test results for each functionality
        """
        functionality_results = {
            'dynamic_pairing': {},
            'untrained_models': {},
            'trained_model_loading': {}
        }
        
        # 1. Test dynamic pairing creation
        logger.info("Testing dynamic pairing creation...")
        try:
            manager = PairingManager()
            
            # Test different pairing creation methods
            manager.add_baseline_pairing("Test Random", "random")
            manager.add_untrained_policy_pairing("Test Untrained", "simple_permutation_invariant")
            
            # Try to add joint checkpoint if available
            joint_dir = experiments_dir / 'joint-training' / 'checkpoints'
            if joint_dir.exists():
                for subdir in list(joint_dir.iterdir())[:1]:  # Test one
                    if subdir.is_dir():
                        manager.add_joint_training_pairings(subdir, "Test Joint")
                        break
            
            functionality_results['dynamic_pairing']['pairing_creation'] = True
            functionality_results['dynamic_pairing']['pairing_count'] = len(manager.get_pairings())
            
        except Exception as e:
            functionality_results['dynamic_pairing']['pairing_creation'] = False
            functionality_results['dynamic_pairing']['error'] = str(e)
            logger.error(f"Dynamic pairing creation failed: {e}")
        
        # 2. Test untrained model creation
        logger.info("Testing untrained model creation...")
        untrained_results = self.validate_untrained_creation()
        functionality_results['untrained_models'] = untrained_results
        
        # 3. Test trained model loading
        logger.info("Testing trained model loading...")
        checkpoint_results = self.validate_checkpoint_loading(experiments_dir)
        functionality_results['trained_model_loading'] = checkpoint_results
        
        return functionality_results
    
    def run_comprehensive_validation(self, experiments_dir: Path) -> Dict[str, Any]:
        """
        Run comprehensive validation of all functionality.
        
        Args:
            experiments_dir: Base experiments directory
            
        Returns:
            Complete validation results
        """
        logger.info("Starting comprehensive validation...")
        
        # Test specific functionalities
        functionality_results = self.test_specific_functionality(experiments_dir)
        
        # Run pairing validation
        pairing_results = self.run_validation(experiments_dir, mode='standard')
        
        # Combine results
        comprehensive_results = {
            'functionality_tests': functionality_results,
            'pairing_validation': pairing_results,
            'summary': {
                'all_functionalities_working': all(
                    all(result.values()) if isinstance(result, dict) else result
                    for result in functionality_results.values()
                ),
                'pairing_success_rate': pairing_results['success_rate'],
                'overall_status': 'PASS' if (
                    pairing_results['success_rate'] > 0.8 and
                    functionality_results['dynamic_pairing'].get('pairing_creation', False) and
                    functionality_results['untrained_models'].get('untrained_policy', False)
                ) else 'FAIL'
            }
        }
        
        return comprehensive_results
    
    def export_results(self, results: Dict[str, Any], output_dir: Path) -> None:
        """
        Export validation results.
        
        Args:
            results: Validation results
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save functionality test results
        if 'functionality_tests' in results:
            with open(output_dir / 'functionality_tests.json', 'w') as f:
                import json
                json.dump(results['functionality_tests'], f, indent=2, default=str)
        
        # Save pairing validation results
        if 'pairing_validation' in results:
            pairing_data = []
            for result in results['pairing_validation'].get('validation_results', []):
                pairing_data.append({
                    'pairing_name': result.pairing_name,
                    'success': result.success,
                    'error_message': result.error_message,
                    'execution_time': result.execution_time,
                    'memory_usage': result.memory_usage,
                    'interventions_completed': result.interventions_completed
                })
            
            df = pd.DataFrame(pairing_data)
            df.to_csv(output_dir / 'pairing_validation.csv', index=False)
        
        # Generate summary report
        self.generate_validation_report(results, output_dir)
        
        logger.info(f"Validation results exported to {output_dir}")
    
    def generate_validation_report(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Generate human-readable validation report."""
        lines = ["=" * 80]
        lines.append("DYNAMIC PAIRING VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Overall status
        if 'summary' in results:
            status = results['summary']['overall_status']
            lines.append(f"OVERALL STATUS: {status}")
            lines.append("")
        
        # Functionality tests
        if 'functionality_tests' in results:
            lines.append("FUNCTIONALITY TEST RESULTS")
            lines.append("-" * 40)
            
            func_tests = results['functionality_tests']
            
            lines.append("1. Dynamic Pairing Creation:")
            if 'dynamic_pairing' in func_tests:
                dp_result = func_tests['dynamic_pairing']
                success = dp_result.get('pairing_creation', False)
                count = dp_result.get('pairing_count', 0)
                lines.append(f"   Status: {'PASS' if success else 'FAIL'}")
                if success:
                    lines.append(f"   Created: {count} pairings")
            
            lines.append("\n2. Untrained Model Creation:")
            if 'untrained_models' in func_tests:
                ut_result = func_tests['untrained_models']
                policy_success = ut_result.get('untrained_policy', False)
                surrogate_success = ut_result.get('untrained_surrogate', False)
                lines.append(f"   Untrained Policy: {'PASS' if policy_success else 'FAIL'}")
                lines.append(f"   Untrained Surrogate: {'PASS' if surrogate_success else 'FAIL'}")
            
            lines.append("\n3. Trained Model Loading:")
            if 'trained_model_loading' in func_tests:
                tl_result = func_tests['trained_model_loading']
                success_count = sum(1 for v in tl_result.values() if v)
                total_count = len(tl_result)
                success_rate = success_count / total_count if total_count > 0 else 0
                lines.append(f"   Success Rate: {success_rate:.1%} ({success_count}/{total_count})")
        
        # Pairing validation
        if 'pairing_validation' in results:
            lines.append("\nPAIRING VALIDATION RESULTS")
            lines.append("-" * 40)
            
            pv_result = results['pairing_validation']
            total = pv_result.get('total_pairings', 0)
            successful = pv_result.get('successful_pairings', 0)
            success_rate = pv_result.get('success_rate', 0)
            
            lines.append(f"Total pairings tested: {total}")
            lines.append(f"Successful pairings: {successful}")
            lines.append(f"Success rate: {success_rate:.1%}")
            
            # List successful pairings
            if 'validation_results' in pv_result:
                successful_names = [r.pairing_name for r in pv_result['validation_results'] if r.success]
                failed_names = [r.pairing_name for r in pv_result['validation_results'] if not r.success]
                
                if successful_names:
                    lines.append(f"\nSuccessful pairings:")
                    for name in successful_names:
                        lines.append(f"  ✓ {name}")
                
                if failed_names:
                    lines.append(f"\nFailed pairings:")
                    for name in failed_names:
                        lines.append(f"  ✗ {name}")
        
        lines.append("\n" + "=" * 80)
        
        # Write report
        with open(output_dir / 'validation_report.txt', 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Validation report saved to {output_dir / 'validation_report.txt'}")