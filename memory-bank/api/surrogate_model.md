# Surrogate Model API

## Overview

The surrogate model estimates posterior distributions over parent sets for target variables using amortized inference techniques based on AVICI.

## Core Data Structures

```python
@dataclass(frozen=True)
class SurrogateParams:
    """Immutable parameters for surrogate model"""
    encoder_params: Any
    decoder_params: Any
    
@dataclass(frozen=True)
class ParentSetPosterior:
    target: str
    parent_sets: Mapping[FrozenSet[str], float]  # Parent set -> probability
## Core Functions
### encode_data
pythondef encode_data(
    buffer: ExperienceBuffer,
    params: SurrogateParams
) -> Tuple[np.ndarray, Dict]:
    """
    Encode data into latent representation
    
    Args:
        buffer: Contains observational and interventional data
        params: Model parameters for the encoder
        
    Returns:
        encoded_data: Latent representation
        metadata: Additional information for decoding
    """
### decode_posterior
pythondef decode_posterior(
    encoded_data: np.ndarray, 
    metadata: Dict, 
    target: str,
    params: SurrogateParams
) -> ParentSetPosterior:
    """
    Decode latent representation to parent set posterior
    
    Args:
        encoded_data: Latent representation from encoder
        metadata: Additional information from encoding process
        target: Target variable for which to infer parents
        params: Model parameters for the decoder
        
    Returns:
        ParentSetPosterior: Posterior distribution over possible parent sets
    """
### surrogate_loss
pythondef surrogate_loss(
    params: SurrogateParams,
    buffer: ExperienceBuffer,
    true_parent_set: Optional[FrozenSet[str]] = None
) -> float:
    """
    Compute loss for surrogate model training
    
    Args:
        params: Model parameters
        buffer: Experience buffer containing data
        true_parent_set: Optional ground truth for supervised training
        
    Returns:
        loss: Combined reconstruction and KL divergence loss
    """
### update_surrogate
pythondef update_surrogate(
    params: SurrogateParams,
    buffer: ExperienceBuffer,
    learning_rate: float,
    true_parent_set: Optional[FrozenSet[str]] = None
) -> SurrogateParams:
    """
    Update surrogate model parameters with gradient descent
    
    Args:
        params: Current model parameters
        buffer: Experience buffer containing data
        learning_rate: Learning rate for parameter updates
        true_parent_set: Optional ground truth for supervised training
        
    Returns:
        SurrogateParams: Updated model parameters
    """
## AVICI Integration
pythondef avici_to_surrogate_adapter(
    avici_model: 'avici.InferenceModel', 
    target: str
) -> SurrogateParams:
    """
    Adapt AVICI model for our surrogate model
    
    Args:
        avici_model: Pretrained AVICI model
        target: Target variable
        
    Returns:
        SurrogateParams: Initialized surrogate model parameters
    """