# Causal Bayes Opt Interface Specifications

## Core Data Structures

All core data structures use **pyrsistent** for immutability and functional programming patterns.

### SCM (Structural Causal Model)
```python
import pyrsistent as pyr
from typing import FrozenSet, Dict, Callable, Any

# SCM is a pyrsistent PMap with these keys:
SCM = pyr.PMap[str, Any]

def create_scm(
    variables: FrozenSet[str],
    edges: FrozenSet[tuple[str, str]], 
    target: str,
    mechanisms: Dict[str, Callable],
    mechanism_names: Dict[str, str] = None
) -> SCM:
    """Factory function for creating validated SCMs."""

# SCM structure:
# {
#   'variables': frozenset of variable names,
#   'edges': frozenset of (parent, child) tuples,
#   'target': str target variable name,
#   'mechanisms': dict mapping variables to mechanism functions,
#   'mechanism_names': dict mapping variables to mechanism type names
# }
```

### Sample
```python
# Sample is a pyrsistent PMap with these keys:
Sample = pyr.PMap[str, Any]

def create_sample(
    values: Dict[str, Any],
    intervention_type: str = "observational",
    intervention_targets: FrozenSet[str] = frozenset(),
    metadata: Dict[str, Any] = None
) -> Sample:
    """Factory function for creating samples."""

# Sample structure:
# {
#   'values': pyr.pmap of variable -> value,
#   'intervention_type': str ("observational", "perfect", etc.),
#   'intervention_targets': frozenset of intervened variables,
#   'metadata': pyr.pmap of additional info
# }
```

### Interventions
```python
# Interventions are pyrsistent PMaps, no dedicated class
Intervention = pyr.PMap[str, Any]

def create_3_intervention(
    targets: FrozenSet[str],
    values: Dict[str, float],
    metadata: Dict[str, Any] = None
) -> Intervention:
    """Create perfect intervention specification."""

def apply_intervention(scm: SCM, intervention: Intervention) -> SCM:
    """Apply intervention to SCM using registry pattern."""

# Registry for intervention handlers
@register_intervention_handler("perfect")
def handle_perfect_intervention(scm: SCM, intervention: Intervention) -> SCM:
    """Handler for perfect interventions."""
```

### ExperienceBuffer
```python
from dataclasses import dataclass

@dataclass
class BufferStatistics:
    """Statistics computed from buffer contents."""
    total_samples: int
    observational_count: int
    interventional_count: int
    unique_intervention_targets: FrozenSet[str]
    intervention_types: Dict[str, int]

class ExperienceBuffer:
    """Mutable, performance-optimized buffer with sophisticated indexing."""
    
    def __init__(self):
        # Core storage
        self._samples: List[Sample] = []
        
        # Performance indices
        self._obs_by_variables: Dict[FrozenSet[str], List[int]] = {}
        self._int_by_targets: Dict[FrozenSet[str], List[int]] = {}
        self._int_by_type: Dict[str, List[int]] = {}
        
        # Statistics cache
        self._stats: Optional[BufferStatistics] = None
    
    def add_sample(self, sample: Sample) -> None:
        """Add sample and update indices."""
    
    def get_samples(self) -> List[Sample]:
        """Get all samples."""
    
    def filter_by_variables(self, variables: FrozenSet[str]) -> List[Sample]:
        """Get samples containing specified variables."""
    
    def filter_interventions_by_targets(self, targets: FrozenSet[str]) -> List[Sample]:
        """Get interventional samples targeting specified variables."""
    
    def get_statistics(self) -> BufferStatistics:
        """Get cached buffer statistics."""
    
    def sample_batch(self, size: int, intervention_ratio: float = 0.5) -> List[Sample]:
        """Sample balanced batch of observations and interventions."""
```

## Actual Module Interfaces

### data_structures.scm
```python
def create_scm(
    variables: FrozenSet[str],
    edges: FrozenSet[tuple[str, str]], 
    target: str,
    mechanisms: Dict[str, Callable],
    mechanism_names: Dict[str, str] = None
) -> SCM:
    """Factory function for creating validated SCMs."""

def get_variables(scm: SCM) -> FrozenSet[str]:
    """Get all variables in SCM."""

def get_edges(scm: SCM) -> FrozenSet[tuple[str, str]]:
    """Get all edges in SCM."""

def get_target(scm: SCM) -> str:
    """Get target variable."""

def get_mechanisms(scm: SCM) -> Dict[str, Callable]:
    """Get mechanism functions."""

def get_parents(scm: SCM, variable: str) -> FrozenSet[str]:
    """Get parent variables of a node."""

def get_children(scm: SCM, variable: str) -> FrozenSet[str]:
    """Get child variables of a node."""

def topological_sort(scm: SCM) -> List[str]:
    """Get topologically sorted variable ordering."""

def validate_scm(scm: SCM) -> None:
    """Validate SCM consistency, raises ValueError if invalid."""
```

### interventions.registry
```python
# Intervention registry
InterventionHandler = Callable[[SCM, Intervention], SCM]

def register_intervention_handler(intervention_type: str):
    """Decorator to register intervention handler."""

def apply_intervention(scm: SCM, intervention: Intervention) -> SCM:
    """Apply intervention using registered handler."""

def create_perfect_intervention(
    targets: FrozenSet[str],
    values: Dict[str, float],
    metadata: Dict[str, Any] = None
) -> Intervention:
    """Create perfect intervention specification."""
```

### mechanisms.base
```python
from abc import ABC, abstractmethod

class Mechanism(ABC):
    """Base class for all causal mechanisms."""
    
    @abstractmethod
    def sample(self, parent_values: Dict[str, float], key: jax.Array) -> float:
        """Sample from mechanism given parent values and random key."""
```

### mechanisms.linear
```python
class LinearMechanism(Mechanism):
    """Linear mechanism: Y = sum(coef * parent) + intercept + noise."""
    
    def __init__(
        self,
        parents: List[str],
        coefficients: Dict[str, float],
        intercept: float = 0.0,
        noise_scale: float = 1.0
    ):
        """Initialize linear mechanism."""
    
    def sample(self, parent_values: Dict[str, float], key: jax.Array) -> float:
        """Sample from linear mechanism."""

def create_linear_mechanism(
    parents: List[str],
    coefficients: Dict[str, float],
    intercept: float = 0.0,
    noise_scale: float = 1.0
) -> LinearMechanism:
    """Factory for linear mechanisms."""
```

### environments.sampling
```python
def sample_with_intervention(
    scm: SCM,
    intervention: Intervention,
    n_samples: int = 1,
    key: jax.Array = None
) -> List[Sample]:
    """Sample from SCM under intervention."""

def sample_observational(
    scm: SCM,
    n_samples: int = 1,
    key: jax.Array = None
) -> List[Sample]:
    """Sample observational data from SCM."""

def generate_mixed_dataset(
    scm: SCM,
    n_observational: int,
    n_interventional: int,
    intervention_targets: List[str] = None,
    key: jax.Array = None
) -> List[Sample]:
    """Generate mixed observational/interventional dataset."""
```

### avici_integration.core.conversion
```python
def samples_to_avici_data(
    samples: List[Sample],
    variable_order: List[str],
    target_variable: str = None
) -> jnp.ndarray:
    """Convert samples to AVICI format [N, d, 3]."""

def standardize_avici_data(data: jnp.ndarray) -> jnp.ndarray:
    """Standardize AVICI data format."""

def validate_avici_data(data: jnp.ndarray, variable_order: List[str]) -> None:
    """Validate AVICI data format and consistency."""
```

### avici_integration.parent_set.model
```python
class ParentSetPredictionModel:
    """Neural network for predicting parent set posteriors."""
    
    def __init__(
        self,
        n_vars: int,
        max_parents: int = 5,
        hidden_dim: int = 128,
        n_layers: int = 8
    ):
        """Initialize parent set model."""
    
    def predict_posterior(
        self,
        data: jnp.ndarray,
        target_idx: int
    ) -> jnp.ndarray:
        """Predict parent set posterior for target variable."""

    def train_step(
        self,
        data: jnp.ndarray,
        targets: jnp.ndarray,
        learning_rate: float = 1e-3
    ) -> Dict[str, float]:
        """Single training step."""
```

### acquisition.state
```python
@dataclass
class AcquisitionState:
    """Rich state representation for acquisition decisions."""
    
    parent_posterior: jnp.ndarray
    buffer_statistics: BufferStatistics
    optimization_target: str
    best_target_value: float
    intervention_history: List[Sample]
    uncertainty_bits: float
    marginal_parent_probs: Dict[str, float]
    
    def get_optimization_progress(self) -> float:
        """Compute optimization progress metric."""
    
    def get_exploration_coverage(self) -> float:
        """Compute exploration coverage metric."""
    
    def summary(self) -> Dict[str, Any]:
        """Get state summary for logging."""

def create_acquisition_state(
    samples: List[Sample],
    parent_posterior: jnp.ndarray,
    target_variable: str,
    variable_order: List[str]
) -> AcquisitionState:
    """Factory for acquisition state."""
```

### acquisition.rewards
```python
def compute_reward(
    state: AcquisitionState,
    intervention: Intervention,
    outcome: Sample,
    next_state: AcquisitionState,
    alpha: float = 0.5
) -> float:
    """Compute multi-objective verifiable reward."""

def structure_learning_reward(
    current_posterior: jnp.ndarray,
    next_posterior: jnp.ndarray
) -> float:
    """Information gain reward for structure learning."""

def optimization_reward(
    current_best: float,
    outcome_value: float,
    target_variable: str
) -> float:
    """Reward for target optimization."""
```

### acquisition.policy
```python
class AcquisitionPolicyNetwork:
    """Neural network for intervention selection."""
    
    def __init__(
        self,
        n_vars: int,
        hidden_dim: int = 128,
        n_layers: int = 4
    ):
        """Initialize policy network."""
    
    def select_intervention(
        self,
        state: AcquisitionState,
        exploration_rate: float = 0.1,
        key: jax.Array = None
    ) -> Intervention:
        """Select intervention given current state."""
```

### acquisition.grpo ✅
```python
@dataclass
class GRPOConfig:
    """Configuration for GRPO algorithm following DeepSeek literature."""
    group_size: int = 64
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    kl_penalty_coeff: float = 0.1
    max_grad_norm: float = 1.0
    learning_rate: float = 3e-4

@dataclass
class GRPOUpdate:
    """Results from a GRPO update step."""
    policy_loss: float
    entropy_loss: float
    kl_penalty: float
    total_loss: float
    grad_norm: float
    group_baseline: float
    mean_reward: float
    reward_std: float
    mean_advantage: float
    advantage_std: float
    mean_entropy: float
    approx_kl: float

def create_grpo_trainer(
    policy_network: Any,
    config: GRPOConfig
) -> Tuple[Callable, Callable]:
    """Create GRPO training infrastructure."""

def collect_grpo_batch(
    policy_network: Any,
    params: Any,
    states: List[AcquisitionState],
    scms: List[pyr.PMap],
    surrogate_model: Any,
    surrogate_params: Any,
    config: GRPOConfig,
    reward_config: pyr.PMap,
    key: jax.Array
) -> Dict[str, Any]:
    """Collect batch of experience for GRPO training."""

def create_grpo_batch_from_samples(
    samples: List[Tuple[AcquisitionState, pyr.PMap, float, float]],
    config: GRPOConfig
) -> Dict[str, Any]:
    """Create GRPO training batch from pre-collected samples."""
```

## Type Aliases and Constants

```python
# Core types
SCM = pyr.PMap[str, Any]
Sample = pyr.PMap[str, Any] 
Intervention = pyr.PMap[str, Any]

# Convenience aliases
VariableName = str
ParentSet = FrozenSet[str]
VariableValue = float
NodeId = str
InterventionValue = float
```