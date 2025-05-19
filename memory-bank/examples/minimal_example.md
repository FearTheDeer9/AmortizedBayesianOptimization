# Minimal Working Example

This example demonstrates a complete training cycle on a simple three-variable SCM.

## Setup

```python
import pyrsistent as pyr
from typing import Mapping, FrozenSet, Any
from dataclasses import dataclass

# Define basic SCM
variables = pyr.freeze({"X", "Y", "Z"})
edges = pyr.freeze({("X", "Y"), ("Z", "Y")})
target = "Y"

# Define mechanisms
def mechanism_x(noise):
    return noise

def mechanism_z(noise):
    return noise

def mechanism_y(values, noise):
    return 2 * values["X"] - 1.5 * values["Z"] + noise

mechanisms = pyr.m({
    "X": mechanism_x,
    "Z": mechanism_z,
    "Y": mechanism_y
})

# Create SCM
scm = SCM(
    variables=variables,
    edges=edges,
    target=target,
    mechanisms=mechanisms
)

# Generate observational data
buffer = ExperienceBuffer.empty()
for _ in range(100):
    sample = sample_from_scm(scm)
    buffer = buffer.add_observation(sample)

## Training Loop
# Initialize models
surrogate_params, acquisition_params = initialize_models(
    buffer=buffer,
    target="Y",
    random_seed=42
)

# Training loop
num_steps = 20
surrogate_lr = 1e-3
acquisition_lr = 1e-4

for step in range(num_steps):
    # Update surrogate model
    surrogate_params = update_surrogate(
        params=surrogate_params,
        buffer=buffer,
        learning_rate=surrogate_lr
    )
    
    # Compute posterior
    encoded_data, metadata = encode_data(buffer, surrogate_params)
    posterior = decode_posterior(encoded_data, metadata, target, surrogate_params)
    
    # Select intervention
    state = State(posterior=posterior, buffer=buffer, best_value=get_best_value(buffer))
    intervention_dist = policy_network(state, acquisition_params)
    intervention = sample_from_distribution(intervention_dist)
    
    # Apply intervention and observe outcome
    modified_scm = apply_intervention(scm, intervention)
    outcome = sample_from_scm(modified_scm)
    
    # Update buffer
    buffer = buffer.add_intervention(intervention, outcome)
    
    # Compute next state
    encoded_data_next, metadata_next = encode_data(buffer, surrogate_params)
    posterior_next = decode_posterior(encoded_data_next, metadata_next, target, surrogate_params)
    state_next = State(posterior=posterior_next, buffer=buffer, best_value=get_best_value(buffer))
    
    # Compute reward
    reward = compute_reward(state, intervention, outcome, posterior_next)
    
    # Collect trajectory
    trajectory = [(state, intervention, reward, state_next)]
    
    # Update acquisition model
    acquisition_params = update_acquisition(
        params=acquisition_params,
        trajectories=trajectory,
        learning_rate=acquisition_lr
    )
    
    # Print progress
    print(f"Step {step}: Best value = {get_best_value(buffer)}")
    print(f"Posterior: {format_posterior(posterior)}")
Evaluation
python# Evaluate posterior accuracy
true_parent_set = pyr.freeze({"X", "Z"})
posterior_metrics = evaluate_posterior(posterior, true_parent_set)
print(f"Posterior metrics: {posterior_metrics}")

# Evaluate intervention quality
intervention_metrics = evaluate_interventions(buffer, scm)
print(f"Intervention metrics: {intervention_metrics}")

# Evaluate optimization performance
optimization_metrics = evaluate_optimization(buffer, scm)
print(f"Optimization metrics: {optimization_metrics}")