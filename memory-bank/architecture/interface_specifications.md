# ACBO Key Interface Specifications

## Core Data Structures

### SCM (Structural Causal Model)
```python
from dataclasses import dataclass
from typing import FrozenSet, Mapping, Callable, Any
import pyrsistent as pyr

@dataclass(frozen=True)
class SCM:
    """Immutable representation of a Structural Causal Model."""
    variables: FrozenSet[str]
    edges: FrozenSet[tuple[str, str]]  # (parent, child) pairs
    target: str
    mechanisms: pyr.PMap[str, Callable]  # variable -> mechanism function
    
    def __post_init__(self):
        """Validates SCM consistency (no cycles, valid target, etc.)"""
```

### Sample
```python
@dataclass(frozen=True)
class Sample:
    """Immutable representation of variable assignments."""
    values: pyr.PMap[str, Any]  # variable -> value
    intervention: 'Intervention | None' = None
    metadata: pyr.PMap[str, Any] = pyr.m()  # timestamp, noise_seed, etc.
```

### Intervention
```python
@dataclass(frozen=True)
class Intervention:
    """Immutable intervention specification."""
    type: str  # "perfect", "imperfect", "soft"
    variables: FrozenSet[str]
    parameters: pyr.PMap[str, Any]  # type-specific parameters
    
    def apply(self, scm: SCM) -> SCM:
        """Apply intervention to SCM using registry pattern."""
```

### ExperienceBuffer
```python
class ExperienceBuffer:
    """Mutable, append-only buffer for storing intervention-outcome pairs."""
    
    def __init__(self):
        self._observations: list[Sample] = []
        self._interventions: list[tuple[Intervention, Sample]] = []
        self._indices = {}  # for fast querying
    
    def add_observation(self, sample: Sample) -> None:
        """Add observational data."""
    
    def add_intervention(self, intervention: Intervention, outcome: Sample) -> None:
        """Add interventional data."""
    
    def get_observations(self) -> list[Sample]:
        """Get all observational samples."""
    
    def get_interventions(self) -> list[tuple[Intervention, Sample]]:
        """Get all intervention-outcome pairs."""
    
    def filter_by_variables(self, variables: FrozenSet[str]) -> 'ExperienceBuffer':
        """Create filtered view of buffer."""
```

## Core Module Interfaces

### acbo.core.scm
```python
def create_scm(
    variables: FrozenSet[str],
    edges: FrozenSet[tuple[str, str]], 
    target: str,
    mechanisms: Mapping[str, Callable]
) -> SCM:
    """Factory function for creating validated SCMs."""

def get_parents(scm: SCM, variable: str) -> FrozenSet[str]:
    """Get parent variables of a node."""

def get_children(scm: SCM, variable: str) -> FrozenSet[str]:
    """Get child variables of a node."""

def get_ancestors(scm: SCM, variable: str) -> FrozenSet[str]:
    """Get all ancestor variables of a node."""

def get_descendants(scm: SCM, variable: str) -> FrozenSet[str]:
    """Get all descendant variables of a node."""

def topological_order(scm: SCM) -> list[str]:
    """Get topologically sorted variable ordering."""

def validate_scm(scm: SCM) -> list[str]:
    """Validate SCM consistency, return list of issues."""
```

### acbo.core.intervention
```python
# Intervention registry
InterventionHandler = Callable[[SCM, Intervention], SCM]
_intervention_registry: dict[str, InterventionHandler] = {}

def register_intervention_handler(intervention_type: str, handler: InterventionHandler) -> None:
    """Register handler for intervention type."""

def apply_intervention(scm: SCM, intervention: Intervention) -> SCM:
    """Apply intervention using registered handler."""

# Factory functions
def perfect_intervention(variables: FrozenSet[str], values: Mapping[str, Any]) -> Intervention:
    """Create perfect intervention."""

def soft_intervention(variable: str, strength: float, target_value: Any) -> Intervention:
    """Create soft intervention."""

def imperfect_intervention(
    variable: str, 
    noise_scale: float, 
    target_value: Any
) -> Intervention:
    """Create imperfect intervention."""
```

## Mechanisms Module Interfaces

### acbo.mechanisms.factories
```python
def linear_mechanism(
    coefficients: Mapping[str, float],
    intercept: float = 0.0,
    noise_scale: float = 1.0
) -> Callable:
    """Create linear mechanism: Y = sum(coef * X) + intercept + noise."""

def nonlinear_mechanism(
    function: Callable[[Mapping[str, Any]], float],
    noise_scale: float = 1.0
) -> Callable:
    """Create nonlinear mechanism with custom function."""

def polynomial_mechanism(
    variable: str,
    degree: int,
    coefficients: list[float],
    noise_scale: float = 1.0
) -> Callable:
    """Create polynomial mechanism."""
```

## Sampling Module Interfaces

### acbo.sampling.observational
```python
def sample_from_scm(scm: SCM, num_samples: int = 1, seed: int | None = None) -> list[Sample]:
    """Generate observational samples from SCM."""

def sample_single(scm: SCM, seed: int | None = None) -> Sample:
    """Generate single observational sample."""
```

### acbo.sampling.interventional
```python
def sample_with_intervention(
    scm: SCM, 
    intervention: Intervention, 
    num_samples: int = 1,
    seed: int | None = None
) -> list[Sample]:
    """Generate samples under intervention."""
```

### acbo.sampling.batch
```python
def generate_observational_batch(
    scm: SCM,
    batch_size: int,
    seed: int | None = None
) -> list[Sample]:
    """Generate batch of observational data."""

def generate_intervention_batch(
    scm: SCM,
    interventions: list[Intervention],
    samples_per_intervention: int,
    seed: int | None = None
) -> list[tuple[Intervention, list[Sample]]]:
    """Generate batch of interventional data."""
```

## Surrogate Model Interfaces

### acbo.surrogate.posterior
```python
@dataclass(frozen=True)
class ParentSetPosterior:
    """Immutable posterior over parent sets."""
    target_variable: str
    posterior_probs: pyr.PMap[FrozenSet[str], float]  # parent_set -> probability
    uncertainty: float
    metadata: pyr.PMap[str, Any] = pyr.m()

def get_most_likely_parents(posterior: ParentSetPosterior, top_k: int = 1) -> list[FrozenSet[str]]:
    """Get top-k most likely parent sets."""

def compute_uncertainty(posterior: ParentSetPosterior) -> float:
    """Compute entropy-based uncertainty measure."""
```

### acbo.surrogate.encoding
```python
def encode_data(buffer: ExperienceBuffer, config: pyr.PMap) -> tuple[Any, pyr.PMap]:
    """Encode buffer data for neural network input."""

def decode_posterior(
    encoded_output: Any, 
    metadata: pyr.PMap, 
    target: str
) -> ParentSetPosterior:
    """Decode neural network output to posterior."""
```

### acbo.surrogate.avici_adapter
```python
def initialize_surrogate_model(
    buffer: ExperienceBuffer,
    target: str,
    config: pyr.PMap
) -> Any:  # PyTorch model
    """Initialize AVICI-based surrogate model."""

def update_surrogate_model(
    model: Any,
    buffer: ExperienceBuffer,
    learning_rate: float,
    num_epochs: int
) -> Any:
    """Update surrogate model with new data."""
```

## Acquisition Model Interfaces

### acbo.acquisition.policy
```python
@dataclass(frozen=True)
class State:
    """Immutable state representation for RL."""
    posterior: ParentSetPosterior
    buffer: ExperienceBuffer
    best_value: float
    metadata: pyr.PMap[str, Any] = pyr.m()

def initialize_policy_network(state_dim: int, action_dim: int, config: pyr.PMap) -> Any:
    """Initialize policy network."""

def select_intervention(
    state: State, 
    policy_network: Any, 
    exploration_rate: float = 0.1
) -> Intervention:
    """Select intervention using policy network."""
```

### acbo.acquisition.rewards
```python
def compute_reward(
    state: State,
    intervention: Intervention,
    outcome: Sample,
    next_posterior: ParentSetPosterior,
    config: pyr.PMap
) -> float:
    """Compute verifiable reward combining optimization and structure discovery."""

def optimization_reward(
    state: State, 
    outcome: Sample, 
    target: str
) -> float:
    """Reward based on target variable improvement."""

def structure_discovery_reward(
    current_posterior: ParentSetPosterior,
    next_posterior: ParentSetPosterior
) -> float:
    """Reward based on information gain in structure."""
```

### acbo.acquisition.grpo
```python
def update_policy_grpo(
    policy_network: Any,
    trajectories: list[tuple[State, Intervention, float, State]],
    group_size: int,
    learning_rate: float,
    kl_coeff: float
) -> Any:
    """Update policy using GRPO algorithm."""
```

## Training Module Interfaces

### acbo.training.pipeline
```python
def initialize_models(
    buffer: ExperienceBuffer,
    target: str,
    config: pyr.PMap,
    seed: int = 42
) -> tuple[Any, Any]:  # (surrogate_params, acquisition_params)
    """Initialize both surrogate and acquisition models."""

def training_step(
    surrogate_model: Any,
    acquisition_model: Any,
    scm: SCM,
    buffer: ExperienceBuffer,
    config: pyr.PMap
) -> tuple[Any, Any, ExperienceBuffer]:
    """Single training step of the complete pipeline."""

def train_system(
    scm: SCM,
    initial_buffer: ExperienceBuffer,
    config: pyr.PMap,
    num_steps: int
) -> tuple[Any, Any, ExperienceBuffer]:
    """Train complete ACBO system."""
```

## Evaluation Module Interfaces

### acbo.evaluation.posterior_metrics
```python
def evaluate_posterior_accuracy(
    posterior: ParentSetPosterior,
    true_parents: FrozenSet[str]
) -> pyr.PMap[str, float]:
    """Evaluate posterior accuracy against ground truth."""

def compute_calibration_metrics(
    posteriors: list[ParentSetPosterior],
    true_parent_sets: list[FrozenSet[str]]
) -> pyr.PMap[str, float]:
    """Compute calibration metrics for uncertainty quantification."""
```

### acbo.evaluation.optimization_metrics
```python
def evaluate_optimization_performance(
    buffer: ExperienceBuffer,
    target: str,
    true_optimum: float | None = None
) -> pyr.PMap[str, float]:
    """Evaluate optimization performance metrics."""
```

## Type Aliases and Constants

```python
# Common type aliases
VariableName = str
EdgeSet = FrozenSet[tuple[str, str]]
ParentSet = FrozenSet[str]
VariableValue = float | int | str | bool

# Configuration schemas
SurrogateConfig = pyr.PMap[str, Any]
AcquisitionConfig = pyr.PMap[str, Any]
TrainingConfig = pyr.PMap[str, Any]
```