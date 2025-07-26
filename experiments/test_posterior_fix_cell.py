# Quick test cell for Jupyter notebook to verify posterior collapse fix
# Copy this into a notebook cell and run it

import jax.numpy as jnp
from src.causal_bayes_opt.acquisition.enriched.state_enrichment import EnrichedHistoryBuilder
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.data_structures.sample import create_sample
import pyrsistent as pyr

# Create test state
buffer = ExperienceBuffer()
buffer.add_observation(create_sample(values={'X': 1.0, 'Y': -0.5, 'Z': 0.0}))
buffer.add_intervention(
    pyr.m(targets={'X'}, values={'X': 2.0}),
    create_sample(values={'X': 2.0, 'Y': 1.0, 'Z': 0.5})
)

state = type('MockState', (), {
    'buffer': buffer,
    'current_target': 'Z',
    'marginal_parent_probs': {'X': 0.8, 'Y': 0.3, 'Z': 0.1}
})()

# Build enriched history with 5 channels
builder = EnrichedHistoryBuilder(num_channels=5)
enriched_history, _ = builder.build_enriched_history(state)

# Check channels
recent = enriched_history[-1]
print(f"Shape: {recent.shape}")
print("\nChannel Analysis:")

channel_names = ["values", "interventions", "target", "parent_probs", "recency"]
for i, name in enumerate(channel_names):
    vals = recent[:, i]
    unique = len(jnp.unique(vals))
    print(f"  {name}: {vals} (unique: {unique})")
    
# Check if variables are differentiable
differentiable = not all(jnp.allclose(recent[i], recent[j]) 
                        for i in range(3) for j in range(i+1, 3))
print(f"\nVariables differentiable: {'YES ✅' if differentiable else 'NO ❌'}")