#!/usr/bin/env python3
"""Quick test to see what's in the buffer observations."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm

# Create SCM and buffer
scm = create_fork_scm()
buffer = ExperienceBuffer()

# Sample observations
obs_samples = sample_from_linear_scm(scm, n_samples=5, seed=42)
print("Observation samples:")
for i, sample in enumerate(obs_samples):
    print(f"  Sample {i}: {sample}")
    buffer.add_observation(sample)

# Check what's in the buffer
print("\nBuffer observations:")
obs = buffer.get_observations()
print(f"  Number of observations: {len(obs)}")
if obs:
    print(f"  First observation: {obs[0]}")
    print(f"  Keys in observation: {list(obs[0].keys()) if obs else 'None'}")