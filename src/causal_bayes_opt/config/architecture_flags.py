"""
Architecture feature flags for A/B testing new implementations.

This module provides configuration flags for switching between discrete/continuous
parent set models and legacy/enriched policy architectures, enabling safe rollout
and performance comparison of architectural improvements.
"""

import enum
from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ParentSetModelType(enum.Enum):
    """Parent set model architecture types."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class PolicyArchitectureType(enum.Enum):
    """Policy network architecture types."""
    LEGACY = "legacy"              # Post-transformer feature concatenation
    ENRICHED = "enriched"          # Enriched transformer input


@dataclass
class ArchitectureConfig:
    """
    Configuration for architecture selection and A/B testing.
    
    This configuration allows switching between old and new implementations
    while maintaining backward compatibility and enabling performance comparison.
    """
    
    # Parent set model configuration
    parent_set_model_type: ParentSetModelType = ParentSetModelType.DISCRETE
    parent_set_config: Dict[str, Any] = None
    
    # Policy architecture configuration
    policy_architecture_type: PolicyArchitectureType = PolicyArchitectureType.LEGACY
    policy_config: Dict[str, Any] = None
    
    # A/B testing configuration
    enable_side_by_side_validation: bool = False
    validation_sample_rate: float = 0.1  # Fraction of samples to validate with both
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    log_architecture_metrics: bool = True
    
    # Compatibility settings
    maintain_legacy_interface: bool = True
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.parent_set_config is None:
            self.parent_set_config = self._get_default_parent_set_config()
        
        if self.policy_config is None:
            self.policy_config = self._get_default_policy_config()
    
    def _get_default_parent_set_config(self) -> Dict[str, Any]:
        """Get default configuration for parent set model."""
        if self.parent_set_model_type == ParentSetModelType.DISCRETE:
            return {
                'layers': 8,
                'dim': 128,
                'key_size': 32,
                'num_heads': 8,
                'widening_factor': 4,
                'dropout': 0.1,
                'max_parent_size': 3
            }
        else:  # CONTINUOUS
            return {
                'hidden_dim': 128,
                'num_layers': 4,
                'num_heads': 8,
                'key_size': 32,
                'dropout': 0.1,
                'use_acyclicity_constraint': True,
                'acyclicity_penalty_weight': 1.0
            }
    
    def _get_default_policy_config(self) -> Dict[str, Any]:
        """Get default configuration for policy architecture."""
        if self.policy_architecture_type == PolicyArchitectureType.LEGACY:
            return {
                'hidden_dim': 128,
                'num_layers': 4,
                'num_heads': 8,
                'dropout': 0.1
            }
        else:  # ENRICHED
            return {
                'num_layers': 4,
                'num_heads': 8,
                'hidden_dim': 128,
                'key_size': 32,
                'widening_factor': 4,
                'dropout': 0.1,
                'policy_intermediate_dim': 64
            }
    
    def validate(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate enum values
            if not isinstance(self.parent_set_model_type, ParentSetModelType):
                logger.error(f"Invalid parent_set_model_type: {self.parent_set_model_type}")
                return False
            
            if not isinstance(self.policy_architecture_type, PolicyArchitectureType):
                logger.error(f"Invalid policy_architecture_type: {self.policy_architecture_type}")
                return False
            
            # Validate numeric parameters
            if not (0.0 <= self.validation_sample_rate <= 1.0):
                logger.error(f"validation_sample_rate must be in [0, 1], got {self.validation_sample_rate}")
                return False
            
            # Validate configurations
            if not self._validate_parent_set_config():
                return False
            
            if not self._validate_policy_config():
                return False
            
            logger.info("✓ Architecture configuration is valid")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def _validate_parent_set_config(self) -> bool:
        """Validate parent set model configuration."""
        config = self.parent_set_config
        
        if self.parent_set_model_type == ParentSetModelType.DISCRETE:
            required_keys = ['layers', 'dim', 'max_parent_size']
            for key in required_keys:
                if key not in config:
                    logger.error(f"Missing required discrete config key: {key}")
                    return False
                if config[key] <= 0:
                    logger.error(f"Invalid value for {key}: {config[key]}")
                    return False
        
        else:  # CONTINUOUS
            required_keys = ['hidden_dim', 'num_layers']
            for key in required_keys:
                if key not in config:
                    logger.error(f"Missing required continuous config key: {key}")
                    return False
                if config[key] <= 0:
                    logger.error(f"Invalid value for {key}: {config[key]}")
                    return False
        
        return True
    
    def _validate_policy_config(self) -> bool:
        """Validate policy architecture configuration."""
        config = self.policy_config
        
        required_keys = ['hidden_dim', 'num_layers']
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required policy config key: {key}")
                return False
            if config[key] <= 0:
                logger.error(f"Invalid value for {key}: {config[key]}")
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'parent_set_model_type': self.parent_set_model_type.value,
            'parent_set_config': self.parent_set_config,
            'policy_architecture_type': self.policy_architecture_type.value,
            'policy_config': self.policy_config,
            'enable_side_by_side_validation': self.enable_side_by_side_validation,
            'validation_sample_rate': self.validation_sample_rate,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'log_architecture_metrics': self.log_architecture_metrics,
            'maintain_legacy_interface': self.maintain_legacy_interface
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ArchitectureConfig':
        """Create configuration from dictionary."""
        return cls(
            parent_set_model_type=ParentSetModelType(config_dict['parent_set_model_type']),
            parent_set_config=config_dict['parent_set_config'],
            policy_architecture_type=PolicyArchitectureType(config_dict['policy_architecture_type']),
            policy_config=config_dict['policy_config'],
            enable_side_by_side_validation=config_dict.get('enable_side_by_side_validation', False),
            validation_sample_rate=config_dict.get('validation_sample_rate', 0.1),
            enable_performance_monitoring=config_dict.get('enable_performance_monitoring', True),
            log_architecture_metrics=config_dict.get('log_architecture_metrics', True),
            maintain_legacy_interface=config_dict.get('maintain_legacy_interface', True)
        )


# Predefined configurations for common use cases
LEGACY_CONFIG = ArchitectureConfig(
    parent_set_model_type=ParentSetModelType.DISCRETE,
    policy_architecture_type=PolicyArchitectureType.LEGACY,
    enable_side_by_side_validation=False
)

CONTINUOUS_ONLY_CONFIG = ArchitectureConfig(
    parent_set_model_type=ParentSetModelType.CONTINUOUS,
    policy_architecture_type=PolicyArchitectureType.LEGACY,
    enable_side_by_side_validation=False
)

ENRICHED_ONLY_CONFIG = ArchitectureConfig(
    parent_set_model_type=ParentSetModelType.DISCRETE,
    policy_architecture_type=PolicyArchitectureType.ENRICHED,
    enable_side_by_side_validation=False
)

FULL_ENHANCED_CONFIG = ArchitectureConfig(
    parent_set_model_type=ParentSetModelType.CONTINUOUS,
    policy_architecture_type=PolicyArchitectureType.ENRICHED,
    enable_side_by_side_validation=False
)

AB_TEST_CONFIG = ArchitectureConfig(
    parent_set_model_type=ParentSetModelType.CONTINUOUS,
    policy_architecture_type=PolicyArchitectureType.ENRICHED,
    enable_side_by_side_validation=True,
    validation_sample_rate=0.2,
    enable_performance_monitoring=True,
    log_architecture_metrics=True
)


class ArchitectureFlagManager:
    """Manager for architecture feature flags and A/B testing."""
    
    def __init__(self, config: ArchitectureConfig):
        self.config = config
        self.metrics = {}
        
        if not config.validate():
            raise ValueError("Invalid architecture configuration")
    
    def should_use_continuous_parent_model(self) -> bool:
        """Check if continuous parent set model should be used."""
        return self.config.parent_set_model_type == ParentSetModelType.CONTINUOUS
    
    def should_use_enriched_policy(self) -> bool:
        """Check if enriched policy architecture should be used."""
        return self.config.policy_architecture_type == PolicyArchitectureType.ENRICHED
    
    def should_run_side_by_side_validation(self, sample_id: Optional[str] = None) -> bool:
        """
        Check if side-by-side validation should be run for this sample.
        
        Args:
            sample_id: Optional sample identifier for deterministic sampling
            
        Returns:
            True if side-by-side validation should be run
        """
        if not self.config.enable_side_by_side_validation:
            return False
        
        # Simple sampling based on rate
        import random
        if sample_id is not None:
            # Deterministic sampling based on sample_id hash
            random.seed(hash(sample_id) % 1000000)
        
        return random.random() < self.config.validation_sample_rate
    
    def log_architecture_metric(self, metric_name: str, value: float, architecture_type: str):
        """
        Log performance metric for architecture comparison.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            architecture_type: Architecture type (e.g., 'legacy', 'continuous', 'enriched')
        """
        if not self.config.log_architecture_metrics:
            return
        
        if architecture_type not in self.metrics:
            self.metrics[architecture_type] = {}
        
        if metric_name not in self.metrics[architecture_type]:
            self.metrics[architecture_type][metric_name] = []
        
        self.metrics[architecture_type][metric_name].append(value)
        
        logger.info(f"Architecture metric - {architecture_type}.{metric_name}: {value}")
    
    def get_performance_comparison(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance comparison between architectures.
        
        Returns:
            Dictionary with mean metrics for each architecture type
        """
        comparison = {}
        
        for arch_type, metrics in self.metrics.items():
            comparison[arch_type] = {}
            for metric_name, values in metrics.items():
                if values:
                    comparison[arch_type][metric_name] = {
                        'mean': sum(values) / len(values),
                        'count': len(values),
                        'min': min(values),
                        'max': max(values)
                    }
        
        return comparison
    
    def print_performance_summary(self):
        """Print performance comparison summary."""
        comparison = self.get_performance_comparison()
        
        print("\n=== Architecture Performance Comparison ===")
        for arch_type, metrics in comparison.items():
            print(f"\n{arch_type.upper()} Architecture:")
            for metric_name, stats in metrics.items():
                print(f"  {metric_name}: {stats['mean']:.4f} (±{(stats['max']-stats['min'])/2:.4f}, n={stats['count']})")
        print("=" * 45)


def get_architecture_config_from_env() -> ArchitectureConfig:
    """
    Get architecture configuration from environment variables.
    
    Environment variables:
    - ACBO_PARENT_SET_MODEL: 'discrete' or 'continuous'
    - ACBO_POLICY_ARCHITECTURE: 'legacy' or 'enriched'
    - ACBO_ENABLE_AB_TEST: 'true' or 'false'
    - ACBO_AB_TEST_RATE: float between 0 and 1
    
    Returns:
        ArchitectureConfig based on environment variables
    """
    import os
    
    parent_set_type = os.getenv('ACBO_PARENT_SET_MODEL', 'discrete')
    policy_type = os.getenv('ACBO_POLICY_ARCHITECTURE', 'legacy')
    enable_ab_test = os.getenv('ACBO_ENABLE_AB_TEST', 'false').lower() == 'true'
    ab_test_rate = float(os.getenv('ACBO_AB_TEST_RATE', '0.1'))
    
    try:
        parent_set_model_type = ParentSetModelType(parent_set_type)
        policy_architecture_type = PolicyArchitectureType(policy_type)
    except ValueError as e:
        logger.warning(f"Invalid environment configuration: {e}, using defaults")
        return LEGACY_CONFIG
    
    config = ArchitectureConfig(
        parent_set_model_type=parent_set_model_type,
        policy_architecture_type=policy_architecture_type,
        enable_side_by_side_validation=enable_ab_test,
        validation_sample_rate=ab_test_rate,
        enable_performance_monitoring=True,
        log_architecture_metrics=True
    )
    
    if config.validate():
        return config
    else:
        logger.warning("Environment configuration validation failed, using defaults")
        return LEGACY_CONFIG