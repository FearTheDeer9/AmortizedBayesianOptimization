"""
Dynamic pairing system for flexible model combinations.

This module provides a system for creating arbitrary combinations of
policy and surrogate models, including trained, untrained, and baseline models.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of models that can be used in pairings."""
    TRAINED = "trained"
    UNTRAINED = "untrained"
    RANDOM = "random"
    ORACLE = "oracle"
    NONE = "none"


@dataclass
class ModelSpec:
    """Specification for a single model (policy or surrogate)."""
    model_type: ModelType
    checkpoint_path: Optional[Path] = None
    architecture: Optional[str] = None
    hidden_dim: Optional[int] = None
    use_fixed_std: Optional[bool] = None
    fixed_std: Optional[float] = None
    seed: int = 42
    
    def __post_init__(self):
        """Validate model specification."""
        if self.model_type == ModelType.TRAINED and self.checkpoint_path is None:
            raise ValueError("Trained models require checkpoint_path")
        if self.checkpoint_path and not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")


@dataclass
class PairingConfig:
    """Configuration for a policy-surrogate pairing."""
    name: str
    policy_spec: ModelSpec
    surrogate_spec: ModelSpec
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate pairing configuration."""
        # Oracle policy requires no surrogate (it has perfect knowledge)
        if self.policy_spec.model_type == ModelType.ORACLE:
            if self.surrogate_spec.model_type != ModelType.NONE:
                logger.warning(f"Oracle policy in '{self.name}' paired with surrogate - surrogate will be ignored")
        
        # Some combinations might not make sense but we allow them for flexibility
        logger.debug(f"Created pairing '{self.name}': "
                    f"Policy={self.policy_spec.model_type.value}, "
                    f"Surrogate={self.surrogate_spec.model_type.value}")


class PairingManager:
    """Manages creation and validation of model pairings."""
    
    def __init__(self, base_seed: int = 42):
        """
        Initialize pairing manager.
        
        Args:
            base_seed: Base seed for deterministic model creation
        """
        self.base_seed = base_seed
        self.pairings = []
        
    def add_pairing(self, pairing: PairingConfig) -> None:
        """
        Add a pairing configuration.
        
        Args:
            pairing: Pairing configuration to add
        """
        # Validate pairing
        try:
            pairing.__post_init__()
            self.pairings.append(pairing)
            logger.info(f"Added pairing: {pairing.name}")
        except Exception as e:
            logger.error(f"Failed to add pairing '{pairing.name}': {e}")
            raise
    
    def add_trained_policy_pairing(self,
                                  name: str,
                                  policy_checkpoint: Path,
                                  surrogate_checkpoint: Optional[Path] = None,
                                  description: str = None) -> None:
        """
        Add pairing with trained policy.
        
        Args:
            name: Pairing name
            policy_checkpoint: Path to trained policy
            surrogate_checkpoint: Optional path to trained surrogate
            description: Optional description
        """
        policy_spec = ModelSpec(
            model_type=ModelType.TRAINED,
            checkpoint_path=policy_checkpoint,
            seed=self.base_seed
        )
        
        if surrogate_checkpoint:
            surrogate_spec = ModelSpec(
                model_type=ModelType.TRAINED,
                checkpoint_path=surrogate_checkpoint,
                seed=self.base_seed
            )
        else:
            surrogate_spec = ModelSpec(model_type=ModelType.NONE)
        
        pairing = PairingConfig(
            name=name,
            policy_spec=policy_spec,
            surrogate_spec=surrogate_spec,
            description=description
        )
        self.add_pairing(pairing)
    
    def add_untrained_policy_pairing(self,
                                   name: str,
                                   architecture: str = 'simple_permutation_invariant',
                                   hidden_dim: int = 256,
                                   use_fixed_std: bool = True,
                                   fixed_std: float = 0.5,
                                   surrogate_checkpoint: Optional[Path] = None,
                                   description: str = None) -> None:
        """
        Add pairing with untrained policy.
        
        Args:
            name: Pairing name
            architecture: Policy architecture
            hidden_dim: Hidden dimension size
            use_fixed_std: Whether to use fixed std
            fixed_std: Fixed std value
            surrogate_checkpoint: Optional trained surrogate
            description: Optional description
        """
        policy_spec = ModelSpec(
            model_type=ModelType.UNTRAINED,
            architecture=architecture,
            hidden_dim=hidden_dim,
            use_fixed_std=use_fixed_std,
            fixed_std=fixed_std,
            seed=self.base_seed
        )
        
        if surrogate_checkpoint:
            surrogate_spec = ModelSpec(
                model_type=ModelType.TRAINED,
                checkpoint_path=surrogate_checkpoint,
                seed=self.base_seed
            )
        else:
            surrogate_spec = ModelSpec(model_type=ModelType.NONE)
        
        pairing = PairingConfig(
            name=name,
            policy_spec=policy_spec,
            surrogate_spec=surrogate_spec,
            description=description
        )
        self.add_pairing(pairing)
    
    def add_baseline_pairing(self,
                           name: str,
                           baseline_type: str,
                           surrogate_checkpoint: Optional[Path] = None,
                           description: str = None) -> None:
        """
        Add baseline pairing (random or oracle).
        
        Args:
            name: Pairing name
            baseline_type: 'random' or 'oracle'
            surrogate_checkpoint: Optional surrogate to pair with
            description: Optional description
        """
        if baseline_type == 'random':
            policy_type = ModelType.RANDOM
        elif baseline_type == 'oracle':
            policy_type = ModelType.ORACLE
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")
        
        policy_spec = ModelSpec(
            model_type=policy_type,
            seed=self.base_seed
        )
        
        if surrogate_checkpoint:
            surrogate_spec = ModelSpec(
                model_type=ModelType.TRAINED,
                checkpoint_path=surrogate_checkpoint,
                seed=self.base_seed
            )
        else:
            surrogate_spec = ModelSpec(model_type=ModelType.NONE)
        
        pairing = PairingConfig(
            name=name,
            policy_spec=policy_spec,
            surrogate_spec=surrogate_spec,
            description=description
        )
        self.add_pairing(pairing)
    
    def add_joint_training_pairings(self, checkpoint_dir: Path, prefix: str = "Joint") -> None:
        """
        Add pairings from joint training checkpoints.
        
        Args:
            checkpoint_dir: Directory containing policy.pkl and surrogate.pkl
            prefix: Prefix for pairing names
        """
        policy_path = checkpoint_dir / 'policy.pkl'
        surrogate_path = checkpoint_dir / 'surrogate.pkl'
        
        if policy_path.exists() and surrogate_path.exists():
            self.add_trained_policy_pairing(
                name=f"{prefix} ({checkpoint_dir.name})",
                policy_checkpoint=policy_path,
                surrogate_checkpoint=surrogate_path,
                description=f"Joint trained models from {checkpoint_dir}"
            )
        else:
            logger.warning(f"Joint checkpoint incomplete in {checkpoint_dir}")
    
    def discover_all_checkpoints(self, experiments_dir: Path) -> Dict[str, List[Path]]:
        """
        Discover all available checkpoints in experiments directory.
        
        Args:
            experiments_dir: Base experiments directory
            
        Returns:
            Dictionary mapping checkpoint types to lists of paths
        """
        checkpoints = {
            'joint_training': [],
            'policy_only': [],
            'surrogate_only': []
        }
        
        # Joint training checkpoints
        joint_dir = experiments_dir / 'joint-training' / 'checkpoints'
        if joint_dir.exists():
            for subdir in joint_dir.iterdir():
                if subdir.is_dir() and (subdir / 'policy.pkl').exists():
                    checkpoints['joint_training'].append(subdir)
        
        # Policy-only checkpoints
        policy_dir = experiments_dir / 'policy-only-training' / 'checkpoints'
        if policy_dir.exists():
            for subdir in policy_dir.rglob('**/policy.pkl'):
                checkpoints['policy_only'].append(subdir)
        
        # Surrogate-only checkpoints
        surrogate_dir = experiments_dir / 'surrogate-only-training' / 'scripts' / 'checkpoints'
        if surrogate_dir.exists():
            for subdir in surrogate_dir.rglob('**/best_model.pkl'):
                checkpoints['surrogate_only'].append(subdir)
        
        logger.info(f"Discovered checkpoints: {len(checkpoints['joint_training'])} joint, "
                   f"{len(checkpoints['policy_only'])} policy-only, "
                   f"{len(checkpoints['surrogate_only'])} surrogate-only")
        
        return checkpoints
    
    def create_comprehensive_pairings(self, experiments_dir: Path) -> None:
        """
        Create comprehensive set of pairings from all available checkpoints.
        
        Args:
            experiments_dir: Base experiments directory
        """
        checkpoints = self.discover_all_checkpoints(experiments_dir)
        
        # Add baseline pairings
        self.add_baseline_pairing("Random", "random", description="Random intervention baseline")
        self.add_baseline_pairing("Oracle", "oracle", description="Perfect knowledge baseline")
        
        # Add untrained policy pairings
        self.add_untrained_policy_pairing(
            "Untrained Policy", 
            description="Randomly initialized policy"
        )
        
        # Add joint training pairings
        for joint_checkpoint in checkpoints['joint_training']:
            self.add_joint_training_pairings(joint_checkpoint, "Joint")
        
        # Add mixed pairings (trained policy + untrained surrogate, etc.)
        for policy_checkpoint in checkpoints['policy_only'][:3]:  # Limit for testing
            self.add_trained_policy_pairing(
                f"Trained Policy Only ({policy_checkpoint.parent.name})",
                policy_checkpoint,
                description=f"Trained policy from {policy_checkpoint.parent}"
            )
        
        logger.info(f"Created {len(self.pairings)} total pairings")
    
    def get_pairings(self) -> List[PairingConfig]:
        """Get all configured pairings."""
        return self.pairings.copy()
    
    def get_pairing_by_name(self, name: str) -> Optional[PairingConfig]:
        """Get specific pairing by name."""
        for pairing in self.pairings:
            if pairing.name == name:
                return pairing
        return None
    
    def validate_all_pairings(self) -> Dict[str, bool]:
        """
        Validate all pairings can be loaded.
        
        Returns:
            Dictionary mapping pairing names to validation success
        """
        validation_results = {}
        
        for pairing in self.pairings:
            try:
                # Check if checkpoints exist
                if pairing.policy_spec.checkpoint_path:
                    if not pairing.policy_spec.checkpoint_path.exists():
                        validation_results[pairing.name] = False
                        continue
                
                if pairing.surrogate_spec.checkpoint_path:
                    if not pairing.surrogate_spec.checkpoint_path.exists():
                        validation_results[pairing.name] = False
                        continue
                
                validation_results[pairing.name] = True
                
            except Exception as e:
                logger.error(f"Validation failed for '{pairing.name}': {e}")
                validation_results[pairing.name] = False
        
        return validation_results
    
    def summary(self) -> str:
        """Generate summary of all pairings."""
        lines = ["=" * 60]
        lines.append("PAIRING MANAGER SUMMARY")
        lines.append("=" * 60)
        
        for i, pairing in enumerate(self.pairings, 1):
            lines.append(f"\n{i}. {pairing.name}")
            lines.append(f"   Policy: {pairing.policy_spec.model_type.value}")
            if pairing.policy_spec.checkpoint_path:
                lines.append(f"     Checkpoint: {pairing.policy_spec.checkpoint_path}")
            lines.append(f"   Surrogate: {pairing.surrogate_spec.model_type.value}")
            if pairing.surrogate_spec.checkpoint_path:
                lines.append(f"     Checkpoint: {pairing.surrogate_spec.checkpoint_path}")
            if pairing.description:
                lines.append(f"   Description: {pairing.description}")
        
        lines.append(f"\nTotal pairings: {len(self.pairings)}")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def create_standard_pairings(experiments_dir: Path, base_seed: int = 42) -> PairingManager:
    """
    Create a standard set of pairings for evaluation.
    
    Args:
        experiments_dir: Base experiments directory
        base_seed: Base random seed
        
    Returns:
        PairingManager with standard pairings
    """
    manager = PairingManager(base_seed)
    manager.create_comprehensive_pairings(experiments_dir)
    return manager


def create_ablation_pairings(experiments_dir: Path, base_seed: int = 42) -> PairingManager:
    """
    Create pairings specifically for ablation studies.
    
    Args:
        experiments_dir: Base experiments directory
        base_seed: Base random seed
        
    Returns:
        PairingManager with ablation-focused pairings
    """
    manager = PairingManager(base_seed)
    
    # Find a representative trained checkpoint for reference architecture
    checkpoints = manager.discover_all_checkpoints(experiments_dir)
    reference_checkpoint = None
    if checkpoints['joint_training']:
        reference_checkpoint = checkpoints['joint_training'][0] / 'policy.pkl'
    elif checkpoints['policy_only']:
        reference_checkpoint = checkpoints['policy_only'][0]
    
    # Add ablation-specific pairings
    manager.add_baseline_pairing("Random", "random", description="Random baseline")
    manager.add_baseline_pairing("Oracle", "oracle", description="Oracle baseline")
    
    # Untrained models with different architectures
    architectures = ['simple_permutation_invariant', 'attention_based']
    for arch in architectures:
        try:
            manager.add_untrained_policy_pairing(
                f"Untrained {arch.title()}",
                architecture=arch,
                description=f"Untrained policy with {arch} architecture"
            )
        except Exception as e:
            logger.warning(f"Could not create untrained pairing for {arch}: {e}")
    
    # If we have a reference checkpoint, create architecture-matched untrained
    if reference_checkpoint:
        try:
            manager.add_pairing(PairingConfig(
                name="Untrained (Architecture Matched)",
                policy_spec=ModelSpec(
                    model_type=ModelType.UNTRAINED,
                    seed=base_seed
                ),
                surrogate_spec=ModelSpec(model_type=ModelType.NONE),
                description="Untrained policy matching trained architecture"
            ))
        except Exception as e:
            logger.warning(f"Could not create architecture-matched untrained: {e}")
    
    return manager


def create_transfer_pairings(experiments_dir: Path, base_seed: int = 42) -> PairingManager:
    """
    Create pairings for transfer learning experiments.
    
    Args:
        experiments_dir: Base experiments directory
        base_seed: Base random seed
        
    Returns:
        PairingManager with transfer-focused pairings
    """
    manager = PairingManager(base_seed)
    
    # Add models trained on different distributions/sizes
    checkpoints = manager.discover_all_checkpoints(experiments_dir)
    
    # Add representative checkpoints from different training paradigms
    if checkpoints['joint_training']:
        for i, joint_checkpoint in enumerate(checkpoints['joint_training'][:3]):
            manager.add_joint_training_pairings(
                joint_checkpoint, 
                f"Joint-{i+1}"
            )
    
    if checkpoints['policy_only']:
        for i, policy_checkpoint in enumerate(checkpoints['policy_only'][:3]):
            manager.add_trained_policy_pairing(
                f"Policy-Only-{i+1}",
                policy_checkpoint,
                description=f"Policy-only training checkpoint {i+1}"
            )
    
    return manager