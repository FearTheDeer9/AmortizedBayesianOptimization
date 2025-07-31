#!/usr/bin/env python3
"""Test to see variable order."""

import logging
logging.basicConfig(level=logging.INFO)

from src.causal_bayes_opt.data_structures.scm import get_variables
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm

# Create a fork SCM
scm = create_fork_scm(noise_scale=1.0)

print(f"SCM variables: {list(get_variables(scm))}")
print(f"Target: {scm.get('target')}")
parents = scm.get('parents')
if parents and 'Y' in parents:
    print(f"Parents of Y: {list(parents['Y'])}")

# Now create buffer and tensor
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm

# Sample data
buffer = ExperienceBuffer()
samples = sample_from_linear_scm(scm, 10, seed=42)
for sample in samples:
    buffer.add_observation(sample)

# Convert to tensor
tensor, var_order = buffer_to_three_channel_tensor(buffer, 'Y', standardize=True)
print(f"\nTensor variable order: {var_order}")
print(f"Tensor shape: {tensor.shape}")