# Data Format Abstraction Layer

## Overview

This document outlines a critical abstraction layer missing from the current ACBO architecture: **data format abstraction**. While the codebase has excellent abstractions for behavior (interventions, rewards, training), it lacks abstraction for data representation, limiting the ability to experiment with alternative model architectures.

## Current State: Format Coupling Problem

### Hardcoded AVICI Format `[N, d, 3]`

The current architecture is tightly coupled to the AVICI data format:

```python
# Pervasive throughout codebase (43+ occurrences)
x: jnp.ndarray  # Always expected to be [N, d, 3]

# Channel semantics hardcoded
x[:, :, 0]  # Variable values (standardized)
x[:, :, 1]  # Intervention indicators (binary: 0/1) 
x[:, :, 2]  # Target indicators (binary: 0/1, exactly one per sample)
```

### Coupling Points

**Model Input Layers**:
```python
# All models assume [N, d, 3] input
hk.Linear(self.config.dim)(x)  # x must be [N, d, 3]
```

**Validation Functions**:
```python
# Hard validation enforcement
if avici_data.ndim != 3:
    raise ValueError("AVICI data must be 3D tensor")
```

**Training Infrastructure**:
```python
# Training batches hardcoded
TrainingBatchJAX.observational_data: jnp.ndarray  # [batch_size, N, d, 3]
```

### Limitations for Alternative Architectures

**Graph Neural Networks** would need:
- Node features: `[N, node_features]`
- Edge features: `[E, edge_features]`
- Adjacency matrices: `[N, N]` or edge indices: `[2, E]`

**Mixture of Experts** might prefer:
- Flattened input: `[N, d*3]` or `[N, feature_vector]`

**Autoregressive Models** would need:
- Sequential format: `[N, sequence_length]`
- Variable ordering information

**Alternative Architectures** might expect:
- Different channel arrangements
- Continuous vs discrete representations
- Multi-modal inputs (graphs + tensors)

## Proposed Abstraction Layer

### 1. Format Converter Protocol

```python
from typing import Protocol, Any, Dict, Tuple
import jax.numpy as jnp

class FormatConverter(Protocol):
    """Protocol for converting between different data formats."""
    
    def from_avici_format(
        self, 
        avici_data: jnp.ndarray,  # [N, d, 3]
        metadata: Dict[str, Any]
    ) -> Any:
        """Convert from AVICI format to target format."""
        ...
    
    def to_avici_format(
        self, 
        target_data: Any,
        metadata: Dict[str, Any]
    ) -> jnp.ndarray:
        """Convert from target format back to AVICI format."""
        ...
    
    def get_format_spec(self) -> Dict[str, Any]:
        """Return specification of the target format."""
        ...
```

### 2. Specific Format Converters

**GNN Format Converter**:
```python
@dataclass
class GNNFormatConverter:
    """Convert AVICI format to GNN-compatible format."""
    
    def from_avici_format(
        self, 
        avici_data: jnp.ndarray,  # [N, d, 3]
        metadata: Dict[str, Any]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Convert to (node_features, adjacency_matrix) format."""
        
        # Extract node features (values only)
        node_features = avici_data[:, :, 0]  # [N, d]
        
        # Infer/estimate adjacency matrix from intervention patterns
        # or use ground truth if available in metadata
        adjacency = self._infer_adjacency(avici_data, metadata)  # [d, d]
        
        return node_features, adjacency
    
    def to_avici_format(
        self, 
        target_data: Tuple[jnp.ndarray, jnp.ndarray],
        metadata: Dict[str, Any]
    ) -> jnp.ndarray:
        """Convert GNN output back to AVICI format."""
        node_features, adjacency = target_data
        
        # Reconstruct intervention and target channels
        intervention_channel = self._reconstruct_interventions(metadata)
        target_channel = self._reconstruct_targets(metadata)
        
        return jnp.stack([node_features, intervention_channel, target_channel], axis=-1)
```

**Flattened Format Converter**:
```python
@dataclass
class FlattenedFormatConverter:
    """Convert AVICI format to flattened vector format."""
    
    def from_avici_format(
        self, 
        avici_data: jnp.ndarray,  # [N, d, 3]
        metadata: Dict[str, Any]
    ) -> jnp.ndarray:
        """Flatten to [N, d*3] format."""
        return avici_data.reshape(avici_data.shape[0], -1)  # [N, d*3]
    
    def to_avici_format(
        self, 
        target_data: jnp.ndarray,  # [N, d*3]
        metadata: Dict[str, Any]
    ) -> jnp.ndarray:
        """Reshape back to AVICI format."""
        d = metadata['num_variables']
        return target_data.reshape(target_data.shape[0], d, 3)
```

### 3. Format Registry

```python
class DataFormatRegistry:
    """Registry for data format converters."""
    
    def __init__(self):
        self._converters: Dict[str, FormatConverter] = {}
        
        # Register default converters
        self.register_converter("avici", IdentityConverter())
        self.register_converter("gnn", GNNFormatConverter())
        self.register_converter("flattened", FlattenedFormatConverter())
        self.register_converter("adjacency", AdjacencyFormatConverter())
    
    def register_converter(self, format_name: str, converter: FormatConverter):
        """Register a new format converter."""
        self._converters[format_name] = converter
    
    def convert_for_model(
        self, 
        avici_data: jnp.ndarray,
        target_format: str,
        metadata: Dict[str, Any]
    ) -> Any:
        """Convert data to target format."""
        if target_format not in self._converters:
            raise ValueError(f"Unknown format: {target_format}")
        
        converter = self._converters[target_format]
        return converter.from_avici_format(avici_data, metadata)
    
    def convert_from_model(
        self,
        target_data: Any,
        source_format: str,
        metadata: Dict[str, Any]
    ) -> jnp.ndarray:
        """Convert data back to AVICI format."""
        if source_format not in self._converters:
            raise ValueError(f"Unknown format: {source_format}")
        
        converter = self._converters[source_format]
        return converter.to_avici_format(target_data, metadata)

# Global registry instance
format_registry = DataFormatRegistry()
```

### 4. Model Format Adapter

```python
class ModelFormatAdapter:
    """Adapter that handles format conversion for models."""
    
    def __init__(
        self,
        model: Any,
        input_format: str,
        output_format: str = "avici"
    ):
        self.model = model
        self.input_format = input_format
        self.output_format = output_format
    
    def __call__(
        self, 
        x: jnp.ndarray,  # Always AVICI format input
        variable_order: List[str],
        target_variable: str,
        is_training: bool = True
    ) -> Dict[str, Any]:
        """Apply model with format conversion."""
        
        # Prepare metadata for conversion
        metadata = {
            'num_variables': len(variable_order),
            'variable_order': variable_order,
            'target_variable': target_variable,
            'is_training': is_training
        }
        
        # Convert input to model's expected format
        model_input = format_registry.convert_for_model(
            x, self.input_format, metadata
        )
        
        # Apply model
        model_output = self.model(model_input, **metadata)
        
        # Convert output back to standard format if needed
        if self.output_format != "avici":
            # Handle different output formats
            pass
        
        return model_output
```

## Integration with Existing Architecture

### Enhanced Model Factory

```python
def create_parent_set_model(
    model_type: str = "jax_unified",
    input_format: str = "avici",
    **kwargs
) -> Any:
    """Enhanced factory with format support."""
    
    # Create base model
    if model_type == "jax_unified":
        base_model = create_jax_unified_parent_set_model(**kwargs)
    elif model_type == "gnn":
        base_model = create_gnn_parent_set_model(**kwargs)
    elif model_type == "moe":
        base_model = create_moe_parent_set_model(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Wrap with format adapter if needed
    if input_format != "avici":
        return ModelFormatAdapter(base_model, input_format)
    else:
        return base_model
```

### Configuration Integration

```python
@dataclass(frozen=True)
class ModelConfig:
    """Enhanced model configuration with format support."""
    
    # Existing fields
    model_type: str = "jax_unified"
    layers: int = 8
    dim: int = 128
    num_heads: int = 8
    
    # New format fields
    input_format: str = "avici"
    output_format: str = "avici"
    format_options: Dict[str, Any] = field(default_factory=dict)
```

## Alternative Architectures Enabled

### 1. Graph Neural Networks

```python
class GNNParentSetModel(hk.Module):
    """GNN-based parent set prediction model."""
    
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
    
    def __call__(
        self, 
        graph_input: Tuple[jnp.ndarray, jnp.ndarray],  # (node_features, adjacency)
        **kwargs
    ) -> Dict[str, Any]:
        """Apply GNN to graph-structured input."""
        node_features, adjacency = graph_input
        
        # GNN message passing
        updated_features = self.gnn_layers(node_features, adjacency)
        
        # Parent set prediction
        parent_set_logits = self.parent_set_head(updated_features)
        
        return {
            'parent_set_logits': parent_set_logits,
            'parent_sets': self.enumerate_parent_sets(parent_set_logits),
            'k': self.config.top_k
        }

# Usage with format adapter
gnn_model = create_parent_set_model(
    model_type="gnn",
    input_format="gnn",
    **gnn_config
)
```

### 2. Mixture of Experts

```python
class MoEParentSetModel(hk.Module):
    """Mixture of Experts for different SCM structures."""
    
    def __call__(
        self, 
        flattened_input: jnp.ndarray,  # [N, d*3]
        **kwargs
    ) -> Dict[str, Any]:
        """Apply MoE to flattened input."""
        
        # Router network
        routing_weights = self.router(flattened_input)
        
        # Expert networks
        expert_outputs = [
            expert(flattened_input) for expert in self.experts
        ]
        
        # Weighted combination
        combined_output = self.combine_experts(expert_outputs, routing_weights)
        
        return combined_output

# Usage
moe_model = create_parent_set_model(
    model_type="moe",
    input_format="flattened",
    n_experts=4
)
```

### 3. Autoregressive Models

```python
class AutoregressiveParentSetModel(hk.Module):
    """Autoregressive model for sequential variable ordering."""
    
    def __call__(
        self, 
        sequence_input: jnp.ndarray,  # [N, sequence_length]
        **kwargs
    ) -> Dict[str, Any]:
        """Apply autoregressive model to sequential input."""
        
        # LSTM/Transformer for sequence modeling
        sequence_features = self.sequence_model(sequence_input)
        
        # Parent set prediction for each position
        parent_set_logits = self.sequential_parent_head(sequence_features)
        
        return {
            'parent_set_logits': parent_set_logits,
            'parent_sets': self.decode_parent_sets(parent_set_logits),
            'k': self.config.top_k
        }
```

## Implementation Plan

### Phase 1: Core Abstraction Infrastructure (1-2 weeks)

**Tasks**:
1. Implement `FormatConverter` protocol
2. Create `DataFormatRegistry` with basic converters
3. Implement `ModelFormatAdapter` wrapper
4. Add format support to model factory functions

**Deliverables**:
- Format converter interfaces
- Registry implementation
- Basic format converters (identity, flattened)
- Enhanced model factory

### Phase 2: Specific Format Converters (2-3 weeks)

**Tasks**:
1. Implement GNN format converter
2. Implement adjacency matrix converter
3. Implement sequence format converter
4. Add format validation and error handling

**Deliverables**:
- GNN format converter with adjacency inference
- Adjacency matrix converter
- Sequence format converter
- Comprehensive format validation

### Phase 3: Model Integration (2-3 weeks)

**Tasks**:
1. Integrate format adapters into training pipeline
2. Update configuration system for format options
3. Add format-aware model wrappers
4. Implement format-specific optimizations

**Deliverables**:
- Training pipeline with format support
- Enhanced configuration system
- Format-aware model implementations
- Performance optimizations

### Phase 4: Alternative Architecture Examples (3-4 weeks)

**Tasks**:
1. Implement GNN parent set model
2. Implement MoE parent set model
3. Implement autoregressive parent set model
4. Add comprehensive testing and validation

**Deliverables**:
- Working GNN implementation
- Working MoE implementation
- Working autoregressive implementation
- Comprehensive test suite

## Benefits Analysis

### Comparison with Current Architecture

**Current State** (Hardcoded AVICI):
- ❌ **Single format**: Only `[N, d, 3]` supported
- ❌ **Architecture coupling**: Models must adapt to data format
- ❌ **Limited experimentation**: Cannot try fundamentally different architectures
- ✅ **Simple**: No format conversion overhead
- ✅ **Consistent**: Uniform data representation

**With Format Abstraction**:
- ✅ **Multiple formats**: Support for GNN, MoE, autoregressive, etc.
- ✅ **Architecture flexibility**: Models can specify preferred input format
- ✅ **Easy experimentation**: Swap model architectures without data pipeline changes
- ✅ **Backward compatible**: Existing models continue to work
- ❌ **Complexity**: Additional abstraction layer
- ❌ **Overhead**: Format conversion computational cost

### Research Flexibility Gained

**Model Architecture Experiments**:
```python
# Easy architecture comparison
for model_type in ["jax_unified", "gnn", "moe", "autoregressive"]:
    model = create_parent_set_model(
        model_type=model_type,
        input_format=get_optimal_format(model_type)
    )
    
    # Same training pipeline, different architectures
    results = train_and_evaluate(model, training_data)
```

**Format-Specific Optimizations**:
```python
# GNN with graph-specific optimizations
gnn_model = create_parent_set_model(
    model_type="gnn",
    input_format="gnn",
    format_options={
        'adjacency_threshold': 0.8,
        'edge_features': True,
        'graph_pooling': 'attention'
    }
)
```

## Relationship to Existing Abstractions

### Complementary Abstractions

**Current Excellent Abstractions**:
- ✅ **Behavior**: Interventions, rewards, training protocols
- ✅ **State**: Samples, SCMs, experience buffers
- ✅ **Algorithms**: Training pipelines, optimization

**Missing Abstraction (This Proposal)**:
- ❌ **Data Representation**: Format conversion, model input/output

### Integration Points

**With Intervention Registry**:
```python
# Different models might prefer different intervention representations
intervention_data = intervention_registry.apply_intervention(
    intervention_type="do_intervention",
    format=model.preferred_format
)
```

**With Experience Buffer**:
```python
# Buffer can serve data in multiple formats
buffer_data = experience_buffer.get_batch(
    batch_size=32,
    format=model.input_format
)
```

**With Training Pipeline**:
```python
# Training pipeline automatically handles format conversion
trainer = MasterTrainer(
    model=model,  # Any format-compatible model
    config=config  # Format-aware configuration
)
```

## Resource Requirements

### Development Time
- **Phase 1**: 1-2 weeks (core infrastructure)
- **Phase 2**: 2-3 weeks (format converters)
- **Phase 3**: 2-3 weeks (integration)
- **Phase 4**: 3-4 weeks (example implementations)
- **Total**: 8-12 weeks

### Computational Overhead
- **Format conversion**: 5-15% overhead depending on complexity
- **Memory usage**: 10-20% increase for format-specific storage
- **Training time**: Minimal impact (conversion done once per batch)

### Maintenance Burden
- **Additional abstractions**: Medium complexity increase
- **Testing requirements**: Comprehensive format validation needed
- **Documentation**: Extensive format specification documentation

## Success Metrics

### Quantitative Measures
- **Architecture flexibility**: Successfully support 3+ different model architectures
- **Performance overhead**: <20% computational overhead for format conversion
- **Backward compatibility**: 100% compatibility with existing models
- **Ease of use**: New architectures implementable in <1 week

### Qualitative Measures
- **Developer experience**: Easy to add new formats and architectures
- **Research velocity**: Faster experimentation with alternative approaches
- **Code maintainability**: Clean separation between data format and model logic
- **Extensibility**: Support for future format requirements

## Future Extensions

### Advanced Format Features
- **Multi-modal formats**: Combine tensors + graphs + sequences
- **Streaming formats**: Handle large datasets with online format conversion
- **Compressed formats**: Efficient storage and transmission
- **Distributed formats**: Format-aware distributed training

### Integration Opportunities
- **AutoML integration**: Automatic format selection based on model architecture
- **Performance optimization**: Format-specific JAX compilation
- **Visualization**: Format-aware debugging and interpretation tools
- **Export formats**: Easy conversion to standard ML formats (ONNX, etc.)

## Conclusion

The data format abstraction layer represents a critical missing piece in the ACBO architecture. While the current system has excellent abstractions for behavior and algorithms, the tight coupling to AVICI format limits architectural experimentation. 

Adding this abstraction layer would:
- **Enable architectural diversity** (GNNs, MoE, autoregressive models)
- **Maintain backward compatibility** with existing transformer-based models
- **Accelerate research** by making model architecture swapping trivial
- **Follow functional programming principles** with immutable data transformations

The implementation can be done incrementally, starting with core infrastructure and gradually adding format converters and example architectures. This investment in abstraction will pay dividends in research velocity and enable systematic exploration of alternative model architectures for causal discovery.

---

*This document provides a comprehensive roadmap for implementing data format abstraction in the ACBO system, enabling architectural experimentation while maintaining the existing system's strengths.*